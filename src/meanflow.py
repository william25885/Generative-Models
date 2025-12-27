"""
MeanFlow: Mean Flows for One-step Generative Modeling

Based on:
1. "Mean Flows for One-step Generative Modeling" (Geng et al., 2025)
   https://arxiv.org/pdf/2505.13447

2. "Improved Mean Flows" (2025)
   https://arxiv.org/pdf/2512.02012

Time convention (IMPORTANT):
- t=0: clean data (x)
- t=1: pure noise (ε)
- z_t = (1-t) * x + t * ε

Sampling (1-NFE):
- Start from ε ~ N(0, I) at t=1
- Predict u = model(ε, r=0, t=1)
- Output x = ε - u
"""

import torch
import torch.nn as nn
import numpy as np
from torch.func import jvp


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal Time Embedding for continuous time t ∈ [0, 1]."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        half_dim = dim // 2
        freqs = torch.exp(
            torch.arange(half_dim) * (-np.log(10000.0) / half_dim)
        )
        self.register_buffer('freqs', freqs)
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.view(-1, 1)
        freqs = self.freqs.view(1, -1)
        args = t * 100.0 * freqs
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return embedding


class MeanFlowResBlock(nn.Module):
    """Residual Block with LayerNorm for MeanFlow."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
            nn.SiLU(),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class MeanFlowNet(nn.Module):
    """MeanFlow Network that predicts average velocity u(z_t, r, t)."""
    
    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 256,
        time_embed_dim: int = 128,
        num_res_blocks: int = 4,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.time_embed_dim = time_embed_dim
        self.num_res_blocks = num_res_blocks
        
        self.time_embed_r = SinusoidalTimeEmbedding(time_embed_dim)
        self.time_embed_t = SinusoidalTimeEmbedding(time_embed_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(2 * time_embed_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.time_proj = nn.Linear(time_embed_dim, hidden_dim)
        self.res_blocks = nn.ModuleList([
            MeanFlowResBlock(hidden_dim) for _ in range(num_res_blocks)
        ])
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim),
        )
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(
        self, 
        z_t: torch.Tensor, 
        r: torch.Tensor, 
        t: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict average velocity u(z_t, r, t).
        
        Args:
            z_t: Current sample of shape (batch_size, input_dim)
            r: Start time of shape (batch_size,) in [0, 1]
            t: End time of shape (batch_size,) in [0, 1]
        
        Returns:
            u: Predicted average velocity of shape (batch_size, input_dim)
        """
        r_embed = self.time_embed_r(r)
        t_embed = self.time_embed_t(t)
        time_embed = torch.cat([r_embed, t_embed], dim=-1)
        time_hidden = self.time_mlp(time_embed)
        
        h = self.input_proj(z_t) + self.time_proj(time_hidden)
        for res_block in self.res_blocks:
            h = res_block(h)
        u = self.output_proj(h)
        
        return u


def sample_t_r(batch_size: int, device: torch.device, t_beta=None):
    """
    Sample (t, r) pairs for MeanFlow training.
    
    Args:
        batch_size: Number of samples
        device: Device to create tensors on
        t_beta: Optional tuple (a, b) for Beta distribution on t
    
    Returns:
        t: Time values in [0, 1]
        r: Start times in [0, t)
    """
    if t_beta is None:
        t = torch.rand(batch_size, device=device)
    else:
        a, b = t_beta
        t = torch.distributions.Beta(a, b).sample((batch_size,)).to(device)
    
    r = torch.rand(batch_size, device=device) * t
    return t, r


def compute_meanflow_training_step(
    model: MeanFlowNet,
    x: torch.Tensor,
    loss_type: str = "huber",
    t_beta=None,
) -> tuple[torch.Tensor, dict]:
    """
    Compute one training step for MeanFlow using JVP.
    
    Time convention:
    - t=0: clean data (x)
    - t=1: pure noise (ε)
    - z = (1-t) * x + t * ε
    - v = ε - x (velocity along the path)
    
    MeanFlow Identity:
    - u_target = v - (t-r) * du/dt
    
    Args:
        model: MeanFlowNet model
        x: Clean data batch of shape (batch_size, dim)
        loss_type: "mse" or "huber"
        t_beta: Optional tuple (a, b) for Beta distribution on t
    
    Returns:
        loss: Scalar loss
        metrics: Dictionary with additional metrics
    """
    B, D = x.shape
    device = x.device

    e = torch.randn_like(x)
    t, r = sample_t_r(B, device, t_beta=t_beta)

    t_col = t.view(-1, 1)
    z = (1 - t_col) * x + t_col * e
    v = e - x

    def fn(z_in, r_in, t_in):
        return model(z_in, r_in, t_in)

    u, dudt = jvp(
        fn,
        (z, r, t),
        (v, torch.zeros_like(r), torch.ones_like(t)),
    )

    tr = (t - r).view(-1, 1)
    u_tgt = v - tr * dudt
    u_tgt = u_tgt.detach()

    err = u - u_tgt

    if loss_type == "mse":
        loss = (err ** 2).mean()
    elif loss_type == "huber":
        loss = torch.nn.functional.smooth_l1_loss(u, u_tgt)
    else:
        raise ValueError("loss_type must be 'mse' or 'huber'")

    with torch.no_grad():
        mse = (err ** 2).mean().item()

    return loss, {"mse": mse, "t_mean": t.mean().item(), "r_mean": r.mean().item()}


@torch.no_grad()
def meanflow_sample_onestep(
    model: MeanFlowNet, 
    sample_size: int, 
    dim: int = 2, 
    device: str = "cuda"
) -> torch.Tensor:
    """
    Generate samples using 1-NFE.
    
    Sampling formula:
    - Start from ε ~ N(0, I) at t=1
    - u = model(ε, r=0, t=1)
    - x = ε - u
    
    Args:
        model: Trained MeanFlowNet
        sample_size: Number of samples to generate
        dim: Data dimension
        device: Device to use
    
    Returns:
        x: Generated samples of shape (sample_size, dim)
    """
    e = torch.randn(sample_size, dim, device=device)
    r = torch.zeros(sample_size, device=device)
    t = torch.ones(sample_size, device=device)
    u = model(e, r, t)
    x = e - u
    return x


class MeanFlowTrainer:
    """MeanFlow Trainer for training the MeanFlow model."""
    
    def __init__(
        self,
        model: MeanFlowNet,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        beta1: float = 0.9,
        beta2: float = 0.999,
        loss_type: str = "huber",
        t_beta=None,
        device: str = "cuda",
        use_wandb: bool = True,
        save_gif: bool = True,
    ):
        self.model = model.to(device)
        self.device = device
        self.use_wandb = use_wandb
        self.loss_type = loss_type
        self.t_beta = t_beta
        self.save_gif = save_gif
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(beta1, beta2),
        )
        
        self.global_step = 0
        self.gif_frames = []
        
        self.config = {
            "lr": lr,
            "weight_decay": weight_decay,
            "beta1": beta1,
            "beta2": beta2,
            "loss_type": loss_type,
            "t_beta": t_beta,
            "hidden_dim": model.hidden_dim,
            "time_embed_dim": model.time_embed_dim,
            "num_res_blocks": model.num_res_blocks,
        }
    
    def fit(self, dataloader: torch.utils.data.DataLoader, epochs: int = 5000, log_every: int = 100):
        from pathlib import Path
        import os
        from src.utils import init_wandb, log_wandb, finish_wandb
        
        if self.use_wandb:
            init_wandb(
                method="meanflow",
                run_name="meanflow-checkerboard",
                tags=["meanflow", "checkerboard", "flow-matching"],
                config={
                    "method": "MeanFlow",
                    "optimizer": "AdamW",
                    "epochs": epochs,
                    "batch_size": dataloader.batch_size,
                    **self.config,
                }
            )
        
        self.gif_frames = []
        target_dist = np.load(Path("data", "checkerboard.npy")).astype(np.float32)
        self.model.train()
        
        from tqdm import trange
        for epoch in trange(epochs, desc="Training MeanFlow"):
            epoch_loss = 0.0
            num_batches = 0
            
            for (x0,) in dataloader:
                x0 = x0.to(self.device)
                loss, metrics = compute_meanflow_training_step(
                    self.model, x0, 
                    loss_type=self.loss_type,
                    t_beta=self.t_beta
                )
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
                
                if self.use_wandb:
                    log_wandb({
                        "loss": loss.item(),
                        "mse": metrics["mse"],
                    }, step=self.global_step)
                
                self.global_step += 1
            
            avg_loss = epoch_loss / num_batches
            
            if (epoch + 1) % log_every == 0:
                eval_metrics = self._evaluate(target_dist)
                
                # Save frame for GIF
                if self.save_gif:
                    self._save_gif_frame(eval_metrics["generated_samples"], target_dist, epoch + 1)
                
                if self.use_wandb:
                    log_wandb({
                        "epoch": epoch + 1,
                        "avg_loss": avg_loss,
                        "energy_distance": eval_metrics["energy_distance"],
                        "wasserstein_distance": eval_metrics["wasserstein_distance"],
                    })
                
                print(f"\nEpoch {epoch + 1}/{epochs}")
                print(f"  Loss: {avg_loss:.4f}")
                print(f"  Energy Distance: {eval_metrics['energy_distance']:.4f}")
                print(f"  Wasserstein Distance: {eval_metrics['wasserstein_distance']:.4f}")
        
        os.makedirs("checkpoints", exist_ok=True)
        torch.save({
            "model": self.model.state_dict(),
            "model_config": {
                "input_dim": self.model.input_dim,
                "hidden_dim": self.model.hidden_dim,
                "time_embed_dim": self.model.time_embed_dim,
                "num_res_blocks": self.model.num_res_blocks,
            },
        }, Path("checkpoints", "meanflow.pth"))
        print(f"\n模型已保存到 checkpoints/meanflow.pth")
        
        # Compile and save GIF
        if self.save_gif and len(self.gif_frames) > 0:
            self._compile_gif()
        
        if self.use_wandb:
            finish_wandb()
    
    def _evaluate(self, target_dist: np.ndarray) -> dict:
        from src.metric import cal_energy_distance, cal_2_wasserstein_dist
        
        self.model.eval()
        with torch.no_grad():
            generated = self._sample(sample_size=5000)
        self.model.train()
        
        energy_dist = cal_energy_distance(generated, target_dist)
        wasserstein_dist = cal_2_wasserstein_dist(generated, target_dist)
        
        return {
            "energy_distance": energy_dist,
            "wasserstein_distance": wasserstein_dist,
            "generated_samples": generated,
        }
    
    @torch.no_grad()
    def _sample(self, sample_size: int = 5000) -> np.ndarray:
        """Generate samples using 1-NFE."""
        samples = meanflow_sample_onestep(
            self.model, sample_size, 
            dim=self.model.input_dim, 
            device=self.device
        )
        return samples.cpu().numpy()
    
    def _save_gif_frame(self, generated: np.ndarray, target: np.ndarray, epoch: int):
        """Save a frame for GIF animation showing generated vs target distribution."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        axes[0].scatter(generated[:, 0], generated[:, 1], s=1, alpha=0.5, c='blue')
        axes[0].set_xlim(-5, 5)
        axes[0].set_ylim(-5, 5)
        axes[0].set_title(f'Generated (Epoch {epoch})')
        axes[0].set_aspect('equal')
        axes[0].grid(True, alpha=0.3)
        
        axes[1].scatter(target[:, 0], target[:, 1], s=1, alpha=0.5, c='green')
        axes[1].set_xlim(-5, 5)
        axes[1].set_ylim(-5, 5)
        axes[1].set_title('Target (Checkerboard)')
        axes[1].set_aspect('equal')
        axes[1].grid(True, alpha=0.3)
        
        plt.suptitle(f'MeanFlow Training Progress - Epoch {epoch}', fontsize=14)
        plt.tight_layout()
        
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
        buf = buf.reshape((h, w, 4))
        rgb = np.roll(buf, -1, axis=2)[:, :, :3]
        
        self.gif_frames.append(rgb.copy())
        plt.close(fig)
    
    def _compile_gif(self):
        """Compile all saved frames into an animated GIF."""
        import os
        from pathlib import Path
        import imageio
        
        os.makedirs("results", exist_ok=True)
        gif_path = Path("results", "meanflow_training.gif")
        imageio.mimsave(str(gif_path), self.gif_frames, fps=5, loop=0)
        
        print(f"\n📽️ Saved training animation to {gif_path}")
        print(f"   Total frames: {len(self.gif_frames)}")
        
        if self.use_wandb:
            try:
                import wandb
                wandb.log({"training_animation": wandb.Video(str(gif_path), fps=5, format="gif")})
            except Exception as e:
                print(f"   Warning: Could not log GIF to wandb: {e}")


class MeanFlowPipeline:
    """MeanFlow Pipeline for 1-NFE generation."""
    
    def __init__(self, model: MeanFlowNet, device: str = "cuda"):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
    
    @torch.no_grad()
    def sample(self, sample_size: int = 5000) -> torch.Tensor:
        """Generate samples using 1-NFE."""
        return meanflow_sample_onestep(
            self.model, sample_size,
            dim=self.model.input_dim,
            device=self.device
        )


def load_meanflow_model(checkpoint_path: str, device: str = "cuda") -> MeanFlowNet:
    """Load a trained MeanFlow model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model_config = checkpoint.get("model_config", {})
    
    model = MeanFlowNet(
        input_dim=model_config.get("input_dim", 2),
        hidden_dim=model_config.get("hidden_dim", 256),
        time_embed_dim=model_config.get("time_embed_dim", 128),
        num_res_blocks=model_config.get("num_res_blocks", 4),
    )
    
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    
    return model

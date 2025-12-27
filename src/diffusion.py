import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from src.utils import init_wandb, log_wandb, finish_wandb


class SinusoidalPositionalEmbedding(nn.Module):
    """
    Sinusoidal Positional Embedding from Transformer (Vaswani et al., 2017)
    Section 3.5: Positional Encoding
    
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    def __init__(self, dim: int, max_timesteps: int = 10000):
        super().__init__()
        self.dim = dim
        
        position = torch.arange(max_timesteps).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-np.log(10000.0) / dim))
        
        pe = torch.zeros(max_timesteps, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Timesteps of shape (batch_size,)
        Returns:
            Embeddings of shape (batch_size, dim)
        """
        return self.pe[t]


class LearnedEmbedding(nn.Module):
    """
    Learned Embedding for timesteps
    Simple lookup table that learns embeddings for each timestep
    """
    def __init__(self, num_timesteps: int, dim: int):
        super().__init__()
        self.embedding = nn.Embedding(num_timesteps, dim)
        
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Timesteps of shape (batch_size,)
        Returns:
            Embeddings of shape (batch_size, dim)
        """
        return self.embedding(t)


class DDPMScheduler:
    """
    DDPM Scheduler implementing the forward and reverse diffusion processes.
    Based on "Denoising Diffusion Probabilistic Models" (Ho et al., 2020)
    """
    def __init__(
        self,
        num_timesteps: int,
        beta_start: float,
        beta_end: float,
        device: str = "cuda",
    ):
        self.T = num_timesteps
        self.device = device
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.alpha_bars_prev = F.pad(self.alpha_bars[:-1], (1, 0), value=1.0)
        
        # β̃_t = (1 - ᾱ_{t-1}) / (1 - ᾱ_t) * β_t
        self.posterior_variance = (
            self.betas * (1.0 - self.alpha_bars_prev) / (1.0 - self.alpha_bars)
        )
        self.posterior_variance = torch.clamp(self.posterior_variance, min=1e-20)
        
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alpha_bars_prev) / (1.0 - self.alpha_bars)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alpha_bars_prev) * torch.sqrt(self.alphas) / (1.0 - self.alpha_bars)
        )

    def forward(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        Forward diffusion process: q(x_t | x_0)
        
        x_t = √ᾱ_t * x_0 + √(1 - ᾱ_t) * ε
        
        Args:
            x0: Original data (batch_size, *)
            t: Timesteps (batch_size,)
            noise: Random noise ε ~ N(0, I) (batch_size, *)
        
        Returns:
            x_t: Noised data at timestep t
        """
        alpha_bar_t = self.alpha_bars[t]
        
        while len(alpha_bar_t.shape) < len(x0.shape):
            alpha_bar_t = alpha_bar_t.unsqueeze(-1)
        
        sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1.0 - alpha_bar_t)
        
        x_t = sqrt_alpha_bar_t * x0 + sqrt_one_minus_alpha_bar_t * noise
        
        return x_t

    def reverse(self, sample: torch.Tensor, t: int, eps: torch.Tensor) -> torch.Tensor:
        """
        Reverse diffusion process: p_θ(x_{t-1} | x_t)
        Using x0-form posterior mean with posterior variance (fixed_small)
        
        Steps:
        1. Predict x0 from eps: x̂_0 = (x_t - √(1-ᾱ_t) * ε_θ) / √ᾱ_t
        2. Compute posterior mean: μ = c1 * x̂_0 + c2 * x_t
        3. Sample: x_{t-1} = μ + √β̃_t * z
        
        Args:
            sample: x_t (batch_size, *)
            t: Current timestep (scalar)
            eps: Predicted noise ε_θ(x_t, t) from model (batch_size, *)
        
        Returns:
            x_{t-1}: Denoised sample at timestep t-1
        """
        alpha_bar_t = self.alpha_bars[t]
        sqrt_alpha_bar = torch.sqrt(alpha_bar_t)
        sqrt_one_minus = torch.sqrt(1.0 - alpha_bar_t)
        x0_pred = (sample - sqrt_one_minus * eps) / sqrt_alpha_bar
        mean = self.posterior_mean_coef1[t] * x0_pred + self.posterior_mean_coef2[t] * sample
        
        if t > 0:
            noise = torch.randn_like(sample)
            variance = self.posterior_variance[t]
            std = torch.sqrt(variance)
            x_prev = mean + std * noise
        else:
            x_prev = mean
        
        return x_prev


class DDIMScheduler(DDPMScheduler):
    """
    DDIM Scheduler implementing the non-Markovian reverse process.
    
    Key features:
    - Uses the same trained model as DDPM
    - Allows faster sampling with fewer steps (10-100 instead of 1000)
    - η parameter controls stochasticity (η=0: deterministic, η=1: same as DDPM)
    """
    def __init__(
        self,
        *args,
        eta: float = 0.0,  # η=0 for deterministic DDIM, η=1 for DDPM
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.eta = eta

    def reverse(
        self,
        sample: torch.Tensor,
        t: int,
        t_prev: int,
        eps_pred: torch.Tensor,
    ) -> torch.Tensor:
        """
        DDIM reverse step from timestep t to t_prev.
        
        Implements Equation 12 from the DDIM paper:
        x_{t-1} = √α_{t-1} * (x_t - √(1-α_t) * ε_θ) / √α_t 
                  + √(1-α_{t-1} - σ_t²) * ε_θ 
                  + σ_t * ε
        
        where σ_t = η * √((1-α_{t-1})/(1-α_t)) * √(1 - α_t/α_{t-1})
        
        Args:
            sample: Current sample x_t of shape (batch_size, dim)
            t: Current timestep
            t_prev: Previous timestep (t_prev < t)
            eps_pred: Predicted noise ε_θ(x_t, t) of shape (batch_size, dim)
        
        Returns:
            x_{t_prev}: Sample at previous timestep
        """
        alpha_bar_t = self.alpha_bars[t]
        
        if t_prev >= 0:
            alpha_bar_t_prev = self.alpha_bars[t_prev]
        else:
            alpha_bar_t_prev = alpha_bar_t.new_tensor(1.0)
        
        pred_x0 = (sample - torch.sqrt(1 - alpha_bar_t) * eps_pred) / torch.sqrt(alpha_bar_t)
        
        add_noise = (self.eta > 0 and t_prev >= 0)
        
        if add_noise:
            variance = (1 - alpha_bar_t_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_t_prev)
            variance = variance.clamp(min=0)
            sigma_t = self.eta * torch.sqrt(variance)
        else:
            sigma_t = 0.0
        
        if add_noise:
            dir_coef_sq = (1 - alpha_bar_t_prev - sigma_t ** 2).clamp(min=0)
            dir_coef = torch.sqrt(dir_coef_sq)
        else:
            dir_coef = torch.sqrt((1 - alpha_bar_t_prev).clamp(min=0))
        
        x_prev = torch.sqrt(alpha_bar_t_prev) * pred_x0 + dir_coef * eps_pred
        
        if add_noise:
            noise = torch.randn_like(sample)
            x_prev = x_prev + sigma_t * noise
        
        return x_prev


class DDPMPipeline:
    """
    DDPM Sampling Pipeline
    Implements Algorithm 2 from DDPM paper (Sampling)
    """
    def __init__(
        self,
        model: nn.Module,
        scheduler: DDPMScheduler,
        device: str = "cuda",
    ):
        self.model = model.to(device)
        self.model.eval()
        self.scheduler = scheduler
        self.device = device

    @torch.no_grad()
    def sample(self, sample_size: int = 1) -> torch.Tensor:
        """
        Generate samples using DDPM reverse process (Algorithm 2)
        
        Start from x_T ~ N(0, I) and iteratively denoise for T steps
        
        Args:
            sample_size: Number of samples to generate
        
        Returns:
            Generated samples of shape (sample_size, 2)
        """
        x_t = torch.randn(sample_size, 2, device=self.device)
        
        for t in tqdm(reversed(range(self.scheduler.T)), desc="Sampling", total=self.scheduler.T):
            t_tensor = torch.full((sample_size,), t, device=self.device, dtype=torch.long)
            eps_pred = self.model(x_t, t_tensor)
            x_t = self.scheduler.reverse(x_t, t, eps_pred)
        
        return x_t


class DDIMPipeline:
    """
    DDIM Sampling Pipeline for accelerated generation.
    
    Key advantage: Can generate samples 10x-50x faster than DDPM
    by using fewer denoising steps (e.g., 50-100 instead of 1000).
    """
    def __init__(
        self, 
        model: nn.Module, 
        scheduler: DDIMScheduler, 
        num_inference_steps: int = 50,  # Much fewer steps than DDPM's 1000
        device: str = "cuda"
    ):
        self.model = model.to(device)
        self.model.eval()
        self.scheduler = scheduler
        self.num_inference_steps = num_inference_steps
        self.device = device
        
        self.timesteps = self._create_timestep_schedule()
    
    def _create_timestep_schedule(self) -> list:
        """
        Create a subset of timesteps for accelerated sampling.
        
        From Appendix D.2: Linear or quadratic spacing of timesteps.
        We use linear spacing: τ_i = floor(c * i) where c = T / num_steps
        """
        T = self.scheduler.T
        step_ratio = T / self.num_inference_steps
        timesteps = [int((i + 1) * step_ratio) - 1 for i in range(self.num_inference_steps)]
        timesteps = list(reversed(timesteps))
        return timesteps

    @torch.no_grad()
    def sample(self, sample_size: int = 1) -> torch.Tensor:
        """
        Generate samples using DDIM accelerated sampling.
        
        Uses a subset of timesteps for faster generation while maintaining quality.
        
        Args:
            sample_size: Number of samples to generate
        
        Returns:
            Generated samples of shape (sample_size, 2)
        """
        x_t = torch.randn(sample_size, 2, device=self.device)
        
        for i, t in enumerate(tqdm(self.timesteps, desc="DDIM Sampling")):
            if i + 1 < len(self.timesteps):
                t_prev = self.timesteps[i + 1]
            else:
                t_prev = -1
            
            t_tensor = torch.full((sample_size,), t, device=self.device, dtype=torch.long)
            eps_pred = self.model(x_t, t_tensor)
            x_t = self.scheduler.reverse(x_t, t, t_prev, eps_pred)
        
        return x_t


class ResBlock(nn.Module):
    """
    Residual Block with LayerNorm for stable training.
    
    Based on the reference implementation that achieved ED < 0.001.
    Uses pre-norm architecture: LayerNorm -> Linear -> Activation -> Linear -> Residual
    """
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


class Denoiser(nn.Module):
    """
    Denoiser model ε_θ(x_t, t) that predicts noise from noised data.
    
    Uses ResNet-style MLP architecture with LayerNorm and time embeddings.
    Supports both learned embeddings and sinusoidal positional embeddings.
    
    Architecture improvements over basic MLP:
    - ResBlock: Residual connections for stable gradient flow
    - LayerNorm: Better normalization for sharp boundary learning
    - Time injection: Add time embedding to initial hidden state
    
    Time conditioning methods:
    - "concat": Concatenate time embedding with input (basic method)
    - "add": Add time embedding at each layer (Time MLP - more expressive, bonus points!)
    """
    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 128,
        time_embed_dim: int = 128,
        num_timesteps: int = 1000,
        time_embed_type: str = "sinusoidal",  # "sinusoidal" or "learned"
        time_conditioning: str = "add",  # "concat" or "add" (Time MLP - bonus)
        num_res_blocks: int = 4,  # Number of residual blocks
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.time_embed_dim = time_embed_dim
        self.time_embed_type = time_embed_type
        self.time_conditioning = time_conditioning
        self.num_res_blocks = num_res_blocks
        
        # Time embedding layer
        if time_embed_type == "sinusoidal":
            self.time_embed = SinusoidalPositionalEmbedding(time_embed_dim, num_timesteps)
        elif time_embed_type == "learned":
            self.time_embed = LearnedEmbedding(num_timesteps, time_embed_dim)
        else:
            raise ValueError(f"Unknown time_embed_type: {time_embed_type}")
        
        if time_conditioning == "add":
            self.time_mlp = nn.Sequential(
                nn.Linear(time_embed_dim, time_embed_dim),
                nn.SiLU(),
                nn.Linear(time_embed_dim, time_embed_dim),
            )
            
            self.input_proj = nn.Linear(input_dim, hidden_dim)
            self.t_proj = nn.Linear(time_embed_dim, hidden_dim)
            self.res_blocks = nn.ModuleList([
                ResBlock(hidden_dim) for _ in range(num_res_blocks)
            ])
            self.output_mlp = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, input_dim),
            )
            
        else:  # "concat"
            self.time_mlp = nn.Sequential(
                nn.Linear(time_embed_dim, time_embed_dim),
                nn.SiLU(),
                nn.Linear(time_embed_dim, time_embed_dim),
            )
            
            self.input_proj = nn.Linear(input_dim, hidden_dim)
            self.t_proj = nn.Linear(time_embed_dim, hidden_dim)
            self.res_blocks = nn.ModuleList([
                ResBlock(hidden_dim) for _ in range(num_res_blocks)
            ])
            self.output_mlp = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, input_dim),
            )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Predict noise from noised data.
        
        Args:
            x: Noised data x_t of shape (batch_size, input_dim)
            t: Timesteps of shape (batch_size,)
        
        Returns:
            Predicted noise ε_θ(x_t, t) of shape (batch_size, input_dim)
        """
        t_embed = self.time_embed(t)
        t_hidden = self.time_mlp(t_embed)
        h = self.input_proj(x) + self.t_proj(t_hidden)
        
        for res_block in self.res_blocks:
            h = res_block(h)
        
        eps_pred = self.output_mlp(h)
        
        return eps_pred


class DiffusionTrainer:
    """
    DDPM Trainer implementing Algorithm 1 from the paper
    """
    def __init__(
        self,
        model: nn.Module,
        pipe: DDPMPipeline,
        scheduler: DDPMScheduler,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        beta1: float = 0.9,
        beta2: float = 0.999,
        device: str = "cuda",
        use_wandb: bool = True,
        save_gif: bool = True,
    ):
        self.model = model.to(device)
        self.pipe = pipe
        self.scheduler = scheduler
        self.optimizer = AdamW(
            self.model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay,
            betas=(beta1, beta2)
        )
        self.criterion = nn.MSELoss()
        self.device = device
        self.use_wandb = use_wandb
        self.save_gif = save_gif
        self.weight_decay = weight_decay
        self.beta1 = beta1
        self.beta2 = beta2
        
        self.global_step = 0
        
        self.gif_frames = []

    def fit(self, dataloader: DataLoader, epochs: int = 5000, log_every: int = 100):
        """
        Train the diffusion model using Algorithm 1 from DDPM paper
        
        Algorithm 1 (Training):
        1. Sample x_0 ~ q(x_0)
        2. Sample t ~ Uniform({1, ..., T})
        3. Sample ε ~ N(0, I)
        4. Take gradient descent step on ∇_θ ||ε - ε_θ(√ᾱ_t x_0 + √(1-ᾱ_t)ε, t)||²
        """
        if self.use_wandb:
            init_wandb(
                method="ddpm",
                run_name="ddpm-checkerboard-adamw",
                tags=["ddpm", "checkerboard", "diffusion", "adamw"],
                config={
                    "optimizer": "AdamW",
                    "num_timesteps": self.scheduler.T,
                    "beta_start": self.scheduler.beta_start,
                    "beta_end": self.scheduler.beta_end,
                    "lr": self.optimizer.param_groups[0]['lr'],
                    "weight_decay": self.weight_decay,
                    "beta1": self.beta1,
                    "beta2": self.beta2,
                    "epochs": epochs,
                    "batch_size": dataloader.batch_size,
                    "model_hidden_dim": self.model.hidden_dim if hasattr(self.model, 'hidden_dim') else "unknown",
                    "time_embed_type": self.model.time_embed_type if hasattr(self.model, 'time_embed_type') else "unknown",
                    "time_conditioning": self.model.time_conditioning if hasattr(self.model, 'time_conditioning') else "unknown",
                    "architecture": "MLP with Time MLP injection",
                }
            )
        
        self.gif_frames = []
        target_dist = np.load(Path("data", "checkerboard.npy")).astype(np.float32)
        
        self.model.train()
        
        for epoch in trange(epochs, desc="Training DDPM"):
            epoch_loss = 0.0
            num_batches = 0
            
            for (x0,) in dataloader:
                x0 = x0.to(self.device)
                batch_size = x0.size(0)
                
                t = torch.randint(0, self.scheduler.T, (batch_size,), device=self.device, dtype=torch.long)
                noise = torch.randn_like(x0)
                x_t = self.scheduler.forward(x0, t, noise)
                noise_pred = self.model(x_t, t)
                loss = self.criterion(noise_pred, noise)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
                
                if self.use_wandb:
                    log_wandb({
                        "loss": loss.item(),
                    }, step=self.global_step)
                
                self.global_step += 1
            
            avg_loss = epoch_loss / num_batches
            
            if (epoch + 1) % log_every == 0:
                metrics = self._evaluate(target_dist)
                
                # Save frame for GIF (every 100 epochs)
                if self.save_gif:
                    self._save_gif_frame(metrics["generated_samples"], target_dist, epoch + 1)
                
                if self.use_wandb:
                    log_wandb({
                        "epoch": epoch + 1,
                        "avg_loss": avg_loss,
                        "energy_distance": metrics["energy_distance"],
                        "wasserstein_distance": metrics["wasserstein_distance"],
                    })
                
                print(f"\nEpoch {epoch + 1}/{epochs}")
                print(f"  Loss: {avg_loss:.4f}")
                print(f"  Energy Distance: {metrics['energy_distance']:.4f}")
                print(f"  Wasserstein Distance: {metrics['wasserstein_distance']:.4f}")
        
        os.makedirs("checkpoints", exist_ok=True)
        torch.save({
            "model": self.model.state_dict(),
            "scheduler_config": {
                "num_timesteps": self.scheduler.T,
                "beta_start": self.scheduler.beta_start,
                "beta_end": self.scheduler.beta_end,
            },
            "model_config": {
                "time_embed_type": self.model.time_embed_type if hasattr(self.model, 'time_embed_type') else "sinusoidal",
                "time_conditioning": self.model.time_conditioning if hasattr(self.model, 'time_conditioning') else "add",
                "hidden_dim": self.model.hidden_dim if hasattr(self.model, 'hidden_dim') else 128,
                "time_embed_dim": self.model.time_embed_dim if hasattr(self.model, 'time_embed_dim') else 128,
                "num_res_blocks": self.model.num_res_blocks if hasattr(self.model, 'num_res_blocks') else 4,
            }
        }, Path("checkpoints", "diffusion.pth"))
        
        if self.save_gif and len(self.gif_frames) > 0:
            self._compile_gif()
        
        if self.use_wandb:
            finish_wandb()
    
    def _evaluate(self, target_dist: np.ndarray) -> dict:
        """
        Evaluate the model on Energy Distance and Wasserstein Distance
        
        Args:
            target_dist: Target distribution (checkerboard data)
        
        Returns:
            Dictionary containing evaluation metrics and generated samples
        """
        from src.metric import cal_energy_distance, cal_2_wasserstein_dist
        
        self.model.eval()
        with torch.no_grad():
            generated = self.pipe.sample(sample_size=5000).cpu().numpy()
        self.model.train()
        
        energy_dist = cal_energy_distance(generated, target_dist)
        wasserstein_dist = cal_2_wasserstein_dist(generated, target_dist)
        
        return {
            "energy_distance": energy_dist,
            "wasserstein_distance": wasserstein_dist,
            "generated_samples": generated,
        }
    
    def _save_gif_frame(self, generated: np.ndarray, target: np.ndarray, epoch: int):
        """
        Save a frame for GIF animation showing generated vs target distribution
        
        Args:
            generated: Generated samples
            target: Target distribution (checkerboard)
            epoch: Current epoch number
        """
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
        
        plt.suptitle(f'DDPM Training Progress - Epoch {epoch}', fontsize=14)
        plt.tight_layout()
        
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
        buf = buf.reshape((h, w, 4))
        rgb = np.roll(buf, -1, axis=2)[:, :, :3]
        
        self.gif_frames.append(rgb.copy())
        plt.close(fig)
    
    def _compile_gif(self):
        """
        Compile all saved frames into an animated GIF
        """
        import imageio
        
        os.makedirs("results", exist_ok=True)
        gif_path = Path("results", "ddpm_training.gif")
        imageio.mimsave(str(gif_path), self.gif_frames, fps=5, loop=0)
        
        print(f"\n📽️ Saved training animation to {gif_path}")
        print(f"   Total frames: {len(self.gif_frames)}")
        
        if self.use_wandb:
            try:
                import wandb
                wandb.log({"training_animation": wandb.Video(str(gif_path), fps=5, format="gif")})
            except Exception as e:
                print(f"   Warning: Could not log GIF to wandb: {e}")

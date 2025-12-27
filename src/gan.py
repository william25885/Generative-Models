import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import trange

from src.utils import init_wandb, log_wandb, finish_wandb, WANDB_PROJECT


class GANGenerator(nn.Module):
    def __init__(self, z_dim: int = 8, hidden_dim: int = 128, output_dim: int = 2, output_scale: float = 5.0):
        """
        Generator network that maps random noise to 2D points.
        
        Args:
            z_dim: Dimension of input noise vector
            hidden_dim: Dimension of hidden layers
            output_dim: Dimension of output (2 for 2D points)
            output_scale: Scale factor for output (tanh range [-1,1] -> [-scale, scale])
        """
        super().__init__()
        self.z_dim = z_dim
        self.output_scale = output_scale
        
        self.net = nn.Sequential(
            # Input layer
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU(),
            
            # Hidden layers
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            
            # Output layer
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh()
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of generator.
        
        Args:
            z: Random noise tensor of shape (batch_size, z_dim)
        
        Returns:
            Generated 2D points of shape (batch_size, 2)
        """
        return self.net(z) * self.output_scale


class GANDiscriminator(nn.Module):
    def __init__(self, input_dim: int = 2, hidden_dim: int = 128, leaky_relu_slope: float = 0.2):
        """
        Discriminator network that classifies real vs fake 2D points.
        
        Args:
            input_dim: Dimension of input (2 for 2D points)
            hidden_dim: Dimension of hidden layers
            leaky_relu_slope: Negative slope for LeakyReLU activation
        """
        super().__init__()
        
        self.net = nn.Sequential(
            # Input layer
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(leaky_relu_slope),
            
            # Hidden layers
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(leaky_relu_slope),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(leaky_relu_slope),
            
            nn.Linear(hidden_dim, 1)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of discriminator.
        
        Args:
            x: Input 2D points of shape (batch_size, 2)
        
        Returns:
            Logits of shape (batch_size, 1) indicating real/fake probability
        """
        return self.net(x)


class GANPipeline:
    def __init__(self, generator: nn.Module, z_dim: int = 8, device: str = "cpu"):
        """
        Pipeline for generating samples from trained GAN generator.
        
        Args:
            generator: Trained generator model
            z_dim: Dimension of noise vector
            device: Device to run inference on
        """
        self.generator = generator.to(device)
        self.generator.eval()
        self.z_dim = z_dim
        self.device = device

    @torch.no_grad()
    def sample(self, sample_size: int = 5000) -> torch.Tensor:
        """Generate samples from the generator."""
        z = torch.randn(sample_size, self.z_dim, device=self.device)
        samples = self.generator(z)
        
        return samples


class GANTrainer:
    def __init__(
        self,
        generator: nn.Module,
        discriminator: nn.Module,
        z_dim: int = 8,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        beta1: float = 0.5,
        beta2: float = 0.999,
        device: str = "cpu",
        use_wandb: bool = True,
        save_gif: bool = True,
    ):
        """
        Trainer for GAN model.
        
        Args:
            generator: Generator network
            discriminator: Discriminator network
            z_dim: Dimension of noise vector
            lr: Learning rate
            weight_decay: Weight decay for AdamW optimizer
            beta1: AdamW optimizer beta1 parameter
            beta2: AdamW optimizer beta2 parameter
            device: Device to train on
            use_wandb: Whether to use wandb for logging
            save_gif: Whether to save training progression GIF
        """
        self.G = generator.to(device)
        self.D = discriminator.to(device)
        self.z_dim = z_dim
        self.opt_G = AdamW(self.G.parameters(), lr=lr, weight_decay=weight_decay, betas=(beta1, beta2))
        self.opt_D = AdamW(self.D.parameters(), lr=lr, weight_decay=weight_decay, betas=(beta1, beta2))
        self.criterion = nn.BCEWithLogitsLoss()
        self.device = device
        self.use_wandb = use_wandb
        self.save_gif = save_gif
        
        self.global_step = 0
        self.gif_frames = []

    def fit(self, dataloader: DataLoader, epochs: int, log_every: int) -> None:
        """Train the GAN model."""
        if self.use_wandb:
            init_wandb(
                method="gan",
                run_name="gan-checkerboard-adamw",
                tags=["gan", "checkerboard", "adamw"],
                config={
                    "optimizer": "AdamW",
                    "epochs": epochs,
                    "batch_size": dataloader.batch_size,
                    "lr": self.opt_G.param_groups[0]['lr'],
                    "weight_decay": self.opt_G.param_groups[0]['weight_decay'],
                    "beta1": self.opt_G.param_groups[0]['betas'][0],
                    "beta2": self.opt_G.param_groups[0]['betas'][1],
                    "z_dim": self.z_dim,
                    "generator_hidden_dim": self.G.net[0].out_features,
                    "discriminator_hidden_dim": self.D.net[0].out_features,
                    "output_scale": self.G.output_scale,
                    "architecture": "MLP",
                }
            )
        
        self.gif_frames = []
        target_dist = np.load(Path("data", "checkerboard.npy")).astype(np.float32)
        
        self.G.train()
        self.D.train()
        
        for epoch in trange(epochs, desc="Training GAN"):
            epoch_loss_G = 0.0
            epoch_loss_D = 0.0
            num_batches = 0
            
            for (real,) in dataloader:
                real = real.to(self.device)
                batch_size = real.size(0)
                
                real_labels = torch.ones(batch_size, 1, device=self.device)
                fake_labels = torch.zeros(batch_size, 1, device=self.device)
                
                # Train Discriminator
                self.opt_D.zero_grad()
                real_logits = self.D(real)
                loss_D_real = self.criterion(real_logits, real_labels)
                
                z = torch.randn(batch_size, self.z_dim, device=self.device)
                fake = self.G(z)
                fake_logits = self.D(fake.detach())
                loss_D_fake = self.criterion(fake_logits, fake_labels)
                
                loss_D = loss_D_real + loss_D_fake
                loss_D.backward()
                self.opt_D.step()
                
                # Train Generator
                self.opt_G.zero_grad()
                z = torch.randn(batch_size, self.z_dim, device=self.device)
                fake = self.G(z)
                fake_logits = self.D(fake)
                loss_G = self.criterion(fake_logits, real_labels)
                loss_G.backward()
                self.opt_G.step()
                
                epoch_loss_G += loss_G.item()
                epoch_loss_D += loss_D.item()
                num_batches += 1
                
                if self.use_wandb:
                    log_wandb({
                        "loss_G": loss_G.item(),
                        "loss_D": loss_D.item(),
                        "loss_D_real": loss_D_real.item(),
                        "loss_D_fake": loss_D_fake.item(),
                    }, step=self.global_step)
                
                self.global_step += 1
            
            avg_loss_G = epoch_loss_G / num_batches
            avg_loss_D = epoch_loss_D / num_batches
            
            if (epoch + 1) % log_every == 0:
                metrics = self._evaluate(target_dist)
                
                # Save frame for GIF
                if self.save_gif:
                    self._save_gif_frame(metrics["generated_samples"], target_dist, epoch + 1)
                
                if self.use_wandb:
                    log_wandb({
                        "epoch": epoch + 1,
                        "avg_loss_G": avg_loss_G,
                        "avg_loss_D": avg_loss_D,
                        "energy_distance": metrics["energy_distance"],
                        "wasserstein_distance": metrics["wasserstein_distance"],
                    })
                
                print(f"\nEpoch {epoch + 1}/{epochs}")
                print(f"  Loss G: {avg_loss_G:.4f}, Loss D: {avg_loss_D:.4f}")
                print(f"  Energy Distance: {metrics['energy_distance']:.4f}")
                print(f"  Wasserstein Distance: {metrics['wasserstein_distance']:.4f}")
        
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(
            {
                "generator": self.G.state_dict(),
                "discriminator": self.D.state_dict(),
                "z_dim": self.z_dim,
            },
            Path("checkpoints", "gan.pth")
        )
        
        # Compile and save GIF
        if self.save_gif and len(self.gif_frames) > 0:
            self._compile_gif()
        
        if self.use_wandb:
            finish_wandb()
    
    def _evaluate(self, target_dist: np.ndarray) -> dict:
        """Evaluate the generator on Energy Distance and Wasserstein Distance."""
        from src.metric import cal_energy_distance, cal_2_wasserstein_dist
        
        self.G.eval()
        with torch.no_grad():
            z = torch.randn(5000, self.z_dim, device=self.device)
            generated = self.G(z).cpu().numpy()
        self.G.train()
        
        energy_dist = cal_energy_distance(generated, target_dist)
        wasserstein_dist = cal_2_wasserstein_dist(generated, target_dist)
        
        return {
            "energy_distance": energy_dist,
            "wasserstein_distance": wasserstein_dist,
            "generated_samples": generated,
        }
    
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
        
        plt.suptitle(f'GAN Training Progress - Epoch {epoch}', fontsize=14)
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
        import imageio
        
        os.makedirs("results", exist_ok=True)
        gif_path = Path("results", "gan_training.gif")
        imageio.mimsave(str(gif_path), self.gif_frames, fps=5, loop=0)
        
        print(f"\n📽️ Saved training animation to {gif_path}")
        print(f"   Total frames: {len(self.gif_frames)}")
        
        if self.use_wandb:
            try:
                import wandb
                wandb.log({"training_animation": wandb.Video(str(gif_path), fps=5, format="gif")})
            except Exception as e:
                print(f"   Warning: Could not log GIF to wandb: {e}")

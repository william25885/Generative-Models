from argparse import ArgumentParser, Namespace
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset

from src.gan import GANGenerator, GANDiscriminator, GANPipeline, GANTrainer
from src.diffusion import DDPMScheduler, DDPMPipeline, Denoiser, DiffusionTrainer
from src.meanflow import MeanFlowNet, MeanFlowTrainer
from src.utils import set_random_seed, get_device


SAMPLE_SIZE = 5000


def parse_arguments() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "-m",
        "--method",
        type=str,
        default="ddpm",
        choices=["ddpm", "gan", "meanflow"],
    )
    return parser.parse_args()


if __name__ == "__main__":
    set_random_seed()
    args = parse_arguments()
    config = OmegaConf.load(Path("configs", f"{args.method}.yaml"))

    target_dist = np.load(Path("data", "checkerboard.npy")).astype(np.float32)
    dataset = TensorDataset(torch.from_numpy(target_dist))
    dataloader = DataLoader(dataset, **config.dataloader)
    device = get_device()

    if args.method == "ddpm":
        scheduler = DDPMScheduler(
            num_timesteps=config.scheduler.num_timesteps,
            beta_start=config.scheduler.beta_start,
            beta_end=config.scheduler.beta_end,
            device=device
        )
        
        model = Denoiser(
            input_dim=config.model.get("input_dim", 2),
            hidden_dim=config.model.hidden_dim,
            time_embed_dim=config.model.get("time_embed_dim", 128),
            num_timesteps=config.scheduler.num_timesteps,
            time_embed_type=config.model.get("time_embed_type", "sinusoidal"),
            time_conditioning=config.model.get("time_conditioning", "add"),
            num_res_blocks=config.model.get("num_res_blocks", 4),
        )
        
        pipe = DDPMPipeline(
            model=model,
            scheduler=scheduler,
            device=device
        )
        
        trainer = DiffusionTrainer(
            model=model,
            pipe=pipe,
            scheduler=scheduler,
            lr=config.trainer.lr,
            weight_decay=config.trainer.get("weight_decay", 1e-5),
            beta1=config.trainer.get("beta1", 0.9),
            beta2=config.trainer.get("beta2", 0.999),
            device=device,
            use_wandb=True,
            save_gif=config.trainer.get("save_gif", True)
        )
    elif args.method == "gan":
        z_dim = config.get("z_dim", 8)
        
        generator = GANGenerator(
            z_dim=z_dim,
            hidden_dim=config.generator.hidden_dim,
            output_dim=2,
            output_scale=config.generator.get("output_scale", 5.0)
        )
        
        discriminator = GANDiscriminator(
            input_dim=2,
            hidden_dim=config.discriminator.hidden_dim,
            leaky_relu_slope=config.discriminator.get("leaky_relu_slope", 0.2)
        )
        
        trainer = GANTrainer(
            generator=generator,
            discriminator=discriminator,
            z_dim=z_dim,
            lr=config.trainer.lr,
            weight_decay=config.trainer.get("weight_decay", 1e-5),
            beta1=config.trainer.get("beta1", 0.5),
            beta2=config.trainer.get("beta2", 0.999),
            device=device,
            use_wandb=True,
            save_gif=config.trainer.get("save_gif", True)
        )
    elif args.method == "meanflow":
        model = MeanFlowNet(
            input_dim=config.model.get("input_dim", 2),
            hidden_dim=config.model.hidden_dim,
            time_embed_dim=config.model.get("time_embed_dim", 128),
            num_res_blocks=config.model.get("num_res_blocks", 4),
        )
        
        t_beta = config.trainer.get("t_beta", None)
        if t_beta is not None:
            t_beta = tuple(t_beta)
        
        trainer = MeanFlowTrainer(
            model=model,
            lr=config.trainer.lr,
            weight_decay=config.trainer.get("weight_decay", 1e-5),
            beta1=config.trainer.get("beta1", 0.9),
            beta2=config.trainer.get("beta2", 0.999),
            loss_type=config.trainer.get("loss_type", "huber"),
            t_beta=t_beta,
            device=device,
            use_wandb=True,
            save_gif=config.trainer.get("save_gif", True),
        )
    else:
        raise ValueError(f"Unknown method: {args.method}")

    trainer.fit(dataloader, epochs=config.trainer.epochs, log_every=config.trainer.log_every)

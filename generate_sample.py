import os
from argparse import ArgumentParser, Namespace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset

from src.diffusion import DDPMScheduler, DDIMScheduler, DDPMPipeline, Denoiser, DDIMPipeline
from src.gan import GANGenerator, GANPipeline
from src.meanflow import MeanFlowNet, MeanFlowPipeline, load_meanflow_model
from src.metric import cal_2_wasserstein_dist, cal_energy_distance
from src.utils import set_random_seed, get_device


def parse_arguments() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "-m",
        "--method",
        type=str,
        default="ddpm",
        choices=["ddpm", "ddim", "gan", "meanflow"],
    )
    return parser.parse_args()


if __name__ == "__main__":
    set_random_seed()
    args = parse_arguments()
    config = OmegaConf.load(Path("configs", f"{args.method}.yaml"))

    target = np.load(Path("data", "checkerboard.npy"))
    target = target.astype(np.float32)
    dataset = TensorDataset(torch.from_numpy(target))
    dataloader = DataLoader(dataset, **config.dataloader)
    device = get_device()

    if args.method in ["ddpm", "ddim"]:
        checkpoint = torch.load(Path("checkpoints", "diffusion.pth"), map_location=device)
        model_config = checkpoint.get("model_config", {})
        time_embed_type = model_config.get("time_embed_type", config.model.get("time_embed_type", "sinusoidal"))
        time_conditioning = model_config.get("time_conditioning", config.model.get("time_conditioning", "add"))
        hidden_dim = model_config.get("hidden_dim", config.model.get("hidden_dim", 256))
        time_embed_dim = model_config.get("time_embed_dim", config.model.get("time_embed_dim", 128))
        num_res_blocks = model_config.get("num_res_blocks", config.model.get("num_res_blocks", 4))
        scheduler_config = checkpoint.get("scheduler_config", {})
        num_timesteps = scheduler_config.get("num_timesteps", config.scheduler.num_timesteps)
        beta_start = scheduler_config.get("beta_start", config.scheduler.beta_start)
        beta_end = scheduler_config.get("beta_end", config.scheduler.beta_end)
        
        model = Denoiser(
            input_dim=config.model.get("input_dim", 2),
            hidden_dim=hidden_dim,
            time_embed_dim=time_embed_dim,
            num_timesteps=num_timesteps,
            time_embed_type=time_embed_type,
            time_conditioning=time_conditioning,
            num_res_blocks=num_res_blocks,
        )
        model.load_state_dict(checkpoint["model"])

        if args.method == "ddpm":
            scheduler = DDPMScheduler(
                num_timesteps=num_timesteps,
                beta_start=beta_start,
                beta_end=beta_end,
                device=device
            )
            
            pipe = DDPMPipeline(
                model=model,
                scheduler=scheduler,
                device=device
            )

        elif args.method == "ddim":
            ddim_config = OmegaConf.load(Path("configs", "ddim.yaml"))
            eta = ddim_config.scheduler.get("eta", 0.0)
            num_inference_steps = ddim_config.sampling.get("num_inference_steps", 50)
            
            scheduler = DDIMScheduler(
                num_timesteps=num_timesteps,
                beta_start=beta_start,
                beta_end=beta_end,
                eta=eta,
                device=device
            )
            
            pipe = DDIMPipeline(
                model=model,
                scheduler=scheduler,
                num_inference_steps=num_inference_steps,
                device=device
            )
            
            print(f"Using DDIM with η={eta}, {num_inference_steps} inference steps (vs DDPM's 1000)")

    elif args.method == "gan":
        checkpoint = torch.load(Path("checkpoints", "gan.pth"), map_location=device)
        z_dim = checkpoint.get("z_dim", 8)
        
        generator = GANGenerator(
            z_dim=z_dim,
            hidden_dim=config.generator.hidden_dim,
            output_dim=2,
            output_scale=config.generator.get("output_scale", 5.0)
        )
        generator.load_state_dict(checkpoint["generator"])
        
        pipe = GANPipeline(
            generator=generator,
            z_dim=z_dim,
            device=device
        )

    elif args.method == "meanflow":
        model = load_meanflow_model(
            checkpoint_path=str(Path("checkpoints", "meanflow.pth")),
            device=device
        )
        
        pipe = MeanFlowPipeline(
            model=model,
            device=device
        )
        
        print("Using MeanFlow with 1-NFE")

    else:
        raise ValueError(f"Unknown method: {args.method}")

    samples = pipe.sample(sample_size=5000).cpu().numpy()
    wasserstein_dist = cal_2_wasserstein_dist(samples, target)
    energy_dist = cal_energy_distance(samples, target)
    print(f"2-Wasserstein Distance ({args.method.upper()}): {wasserstein_dist:.4f}")
    print(f"Energy Distance ({args.method.upper()}): {energy_dist:.4f}")


    os.makedirs("results", exist_ok=True)
    np.save(Path("results", f"{args.method}_generated_sample.npy"), samples)

    submission_path = Path("results", f"{args.method}_submission.csv")
    with open(submission_path, "w") as f:
        f.write("ID,x,y\n")
        for i, (x, y) in enumerate(samples, start=1):
            f.write(f"{i:04d},{x:.6f},{y:.6f}\n")
    print(f"Submission file saved to: {submission_path}")

    plt.figure(figsize=(6, 6), dpi=300)
    plt.scatter(target[:, 0], target[:, 1], s=10, alpha=0.75, label="Ground Truth", marker='.', edgecolors='none')
    plt.scatter(samples[:, 0], samples[:, 1], s=10, alpha=0.75, label=args.method.upper(), marker='.', edgecolors='none')
    plt.xlim(-8, 8)
    plt.ylim(-8, 8)
    plt.grid(alpha=0.3)
    plt.legend()    
    plt.savefig(Path("results", f"{args.method}.png"), dpi=300, bbox_inches='tight')
    plt.close()

"""DDIM 實驗腳本"""
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf

from src.diffusion import DDIMScheduler, DDIMPipeline, Denoiser
from src.metric import cal_2_wasserstein_dist, cal_energy_distance
from src.utils import set_random_seed, get_device


def load_ddpm_model(device):
    """載入訓練好的去噪模型"""
    checkpoint = torch.load(Path("checkpoints", "diffusion.pth"), map_location=device)
    model_config = checkpoint.get("model_config", {})
    scheduler_config = checkpoint.get("scheduler_config", {})
    ddpm_config = OmegaConf.load(Path("configs", "ddpm.yaml"))
    
    time_embed_type = model_config.get("time_embed_type", ddpm_config.model.get("time_embed_type", "sinusoidal"))
    time_conditioning = model_config.get("time_conditioning", ddpm_config.model.get("time_conditioning", "add"))
    hidden_dim = model_config.get("hidden_dim", ddpm_config.model.get("hidden_dim", 256))
    time_embed_dim = model_config.get("time_embed_dim", ddpm_config.model.get("time_embed_dim", 128))
    num_res_blocks = model_config.get("num_res_blocks", ddpm_config.model.get("num_res_blocks", 4))
    
    num_timesteps = scheduler_config.get("num_timesteps", ddpm_config.scheduler.num_timesteps)
    beta_start = scheduler_config.get("beta_start", ddpm_config.scheduler.beta_start)
    beta_end = scheduler_config.get("beta_end", ddpm_config.scheduler.beta_end)
    
    model = Denoiser(
        input_dim=2,
        hidden_dim=hidden_dim,
        time_embed_dim=time_embed_dim,
        num_timesteps=num_timesteps,
        time_embed_type=time_embed_type,
        time_conditioning=time_conditioning,
        num_res_blocks=num_res_blocks
    )
    model.load_state_dict(checkpoint["model"])
    
    return model, num_timesteps, beta_start, beta_end


def experiment_1_different_steps():
    """實驗 1: 測試不同的去噪步數"""
    print("實驗 1: 測試不同的去噪步數")
    
    set_random_seed()
    device = get_device()
    target = np.load(Path("data", "checkerboard.npy")).astype(np.float32)
    
    model, num_timesteps, beta_start, beta_end = load_ddpm_model(device)
    num_inference_steps_list = [1000, 500, 100, 10, 1]
    eta = 1.0
    
    energy_distances = []
    wasserstein_distances = []
    
    for num_steps in num_inference_steps_list:
        print(f"\n測試 {num_steps} 步...")
        
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
            num_inference_steps=num_steps,
            device=device
        )
        
        samples = pipe.sample(sample_size=5000).cpu().numpy()
        energy_dist = cal_energy_distance(samples, target)
        wasserstein_dist = cal_2_wasserstein_dist(samples, target)
        
        energy_distances.append(energy_dist)
        wasserstein_distances.append(wasserstein_dist)
        
        print(f"  Energy Distance: {energy_dist:.4f}")
        print(f"  Wasserstein Distance: {wasserstein_dist:.4f}")
    
    os.makedirs("results", exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(num_inference_steps_list, energy_distances, marker='o', linewidth=2, markersize=8)
    ax1.set_xlabel('Number of Denoising Steps', fontsize=12)
    ax1.set_ylabel('Energy Distance', fontsize=12)
    ax1.set_title(f'Energy Distance vs. Denoising Steps\n(η = {eta}, fixed)', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    
    ax2.plot(num_inference_steps_list, wasserstein_distances, marker='s', linewidth=2, markersize=8, color='orange')
    ax2.set_xlabel('Number of Denoising Steps', fontsize=12)
    ax2.set_ylabel('Wasserstein Distance', fontsize=12)
    ax2.set_title(f'Wasserstein Distance vs. Denoising Steps\n(η = {eta}, fixed)', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    
    plt.tight_layout()
    plt.savefig(Path("results", "ddim_steps_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    return energy_distances, wasserstein_distances


def experiment_2_different_eta():
    """實驗 2: 測試不同的 eta 值"""
    print("\n實驗 2: 測試不同的 eta 值")
    
    set_random_seed()
    device = get_device()
    target = np.load(Path("data", "checkerboard.npy")).astype(np.float32)
    
    model, num_timesteps, beta_start, beta_end = load_ddpm_model(device)
    eta_values = [0.0, 0.25, 0.5, 0.75, 1.0]
    num_inference_steps = 50
    
    energy_distances = []
    wasserstein_distances = []
    
    for eta in eta_values:
        print(f"\n測試 η = {eta}...")
        
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
        
        samples = pipe.sample(sample_size=5000).cpu().numpy()
        energy_dist = cal_energy_distance(samples, target)
        wasserstein_dist = cal_2_wasserstein_dist(samples, target)
        
        energy_distances.append(energy_dist)
        wasserstein_distances.append(wasserstein_dist)
        
        print(f"  Energy Distance: {energy_dist:.4f}")
        print(f"  Wasserstein Distance: {wasserstein_dist:.4f}")
    
    os.makedirs("results", exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(eta_values, energy_distances, marker='o', linewidth=2, markersize=8)
    ax1.set_xlabel('η (Eta)', fontsize=12)
    ax1.set_ylabel('Energy Distance', fontsize=12)
    ax1.set_title(f'Energy Distance vs. η (Stochasticity)\n({num_inference_steps} steps, fixed)', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(eta_values)
    
    ax2.plot(eta_values, wasserstein_distances, marker='s', linewidth=2, markersize=8, color='orange')
    ax2.set_xlabel('η (Eta)', fontsize=12)
    ax2.set_ylabel('Wasserstein Distance', fontsize=12)
    ax2.set_title(f'Wasserstein Distance vs. η (Stochasticity)\n({num_inference_steps} steps, fixed)', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(eta_values)
    
    plt.tight_layout()
    plt.savefig(Path("results", "ddim_eta_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    return energy_distances, wasserstein_distances


if __name__ == "__main__":
    if not Path("checkpoints", "diffusion.pth").exists():
        print("錯誤: 找不到訓練好的模型！")
        print("請先運行: uv run train.py -m ddpm")
        exit(1)
    
    experiment_1_different_steps()
    experiment_2_different_eta()
    print("\n實驗完成！圖表已保存至 results/")


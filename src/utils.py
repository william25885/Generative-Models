import random
from typing import Dict, Any, Optional

import numpy as np
import torch

try:
    import wandb
except ImportError:
    wandb = None


# Unified wandb project name for all models
WANDB_PROJECT = "ml-hw4-generative-models"


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.mps.is_available():
        return "mps"
    else:
        return "cpu"


def set_random_seed(random_seed: int = 0, deterministic: bool = True) -> None:
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic


def init_wandb(
    method: str,
    config: Dict[str, Any],
    run_name: Optional[str] = None,
    tags: Optional[list] = None,
) -> None:
    """
    Initialize wandb with unified project name for all generative models.
    
    All models (GAN, DDPM, DDIM, MeanFlow) will be logged to the same project
    for easy comparison and tracking.
    
    Args:
        method: Model method name ("gan", "ddpm", "ddim", "meanflow")
        config: Configuration dictionary to log
        run_name: Custom run name (default: "{method}-checkerboard")
        tags: List of tags for filtering runs
    """
    if wandb is None:
        return
    
    # Default run name if not provided
    if run_name is None:
        run_name = f"{method}-checkerboard"
    
    # Default tags
    if tags is None:
        tags = [method, "checkerboard"]
    
    # Add method to config
    config_with_method = {
        "method": method.upper(),
        **config
    }
    
    wandb.init(
        project=WANDB_PROJECT,
        name=run_name,
        tags=tags,
        config=config_with_method,
        reinit=True,  # Allow reinitialization for multiple runs
    )


def log_wandb(metrics: Dict[str, Any], step: Optional[int] = None) -> None:
    """
    Log metrics to wandb.
    
    Args:
        metrics: Dictionary of metrics to log
        step: Optional step number
    """
    if wandb is None:
        return
    
    if step is not None:
        wandb.log(metrics, step=step)
    else:
        wandb.log(metrics)


def finish_wandb() -> None:
    """Finish wandb run."""
    if wandb is not None:
        wandb.finish()

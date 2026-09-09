from __future__ import annotations

import os
from datetime import datetime
from typing import Tuple

import numpy as np
import torch

from .config import ExperimentConfig


def prepare_save_dir(path: str) -> str:
    if path == "./":
        path = datetime.now().strftime("%Y%m%d_%H%M%S") + "/"
        print(f"Save folder path not specified, using date string: {path}")
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def set_global_seeds(seed: int) -> None:
    np.random.Generator(np.random.PCG64(seed))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def set_deterministic(deterministic: bool) -> None:
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def configure_environment(cfg: ExperimentConfig) -> Tuple[torch.device, str]:
    set_global_seeds(cfg.seed)
    set_deterministic(cfg.deterministic)
    save_path = cfg.save_folder_path
    if cfg.save_uv_trajectories:
        save_path = prepare_save_dir(save_path)
    device = torch.device(cfg.device)
    return device, save_path

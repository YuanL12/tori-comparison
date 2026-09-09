from __future__ import annotations

import os
import tempfile
from contextlib import suppress
from typing import Any, Dict, Optional

import torch

from tori_comparison.config import ExperimentConfig


def _atomic_save_torch(obj: Any, path: str) -> None:
    """
    Atomically write a torch checkpoint to reduce corruption risk on ctrl+c.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=".ckpt_", suffix=".pt", dir=os.path.dirname(path)
    )
    os.close(fd)
    try:
        torch.save(obj, tmp_path)
        os.replace(tmp_path, path)
    finally:
        # Best-effort cleanup if os.replace fails.
        if os.path.exists(tmp_path):
            with suppress(OSError):
                os.remove(tmp_path)


def save_checkpoint(path: str, state: Dict[str, Any]) -> None:
    _atomic_save_torch(state, path)


def load_checkpoint(path: str, map_location: Optional[str] = None) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    # PyTorch 2.6 changed torch.load default `weights_only=True`, which rejects
    # checkpoints containing non-tensor objects (we store numpy arrays + metadata).
    # These checkpoints are locally created by this project, so full unpickling
    # is expected here.
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        # Compatibility for older PyTorch versions without `weights_only`.
        return torch.load(path, map_location=map_location)


def make_checkpoint_paths(cfg: ExperimentConfig) -> tuple[str, str]:
    """Return ``(run_ckpt_dir, checkpoint.pt path)`` for a run."""
    run_ckpt_dir = os.path.join(cfg.checkpoint_dir, cfg.run_name)
    ckpt_path = os.path.join(run_ckpt_dir, cfg.checkpoint_filename)
    return run_ckpt_dir, ckpt_path


def save_optimization_checkpoint(
    ckpt_path: str,
    *,
    img_f_uvs: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    resume_state: dict,
    wandb_run_id: str | None = None,
    extra: dict | None = None,
    lr_scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau | None = None,
) -> None:
    """Persist optimization state (UVs, optimizer, optional LR scheduler, wandb id)."""
    state: Dict[str, Any] = {
        "version": 1,
        "img_f_uvs": img_f_uvs.detach().cpu().numpy(),
        "optimizer_state_dict": optimizer.state_dict(),
        "resume_state": resume_state,
        "wandb_run_id": wandb_run_id,
    }
    if lr_scheduler is not None:
        state["lr_scheduler_state_dict"] = lr_scheduler.state_dict()
    if extra:
        state.update(extra)
    save_checkpoint(ckpt_path, state)

from typing import Iterable
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau


class ReduceLROnPlateauEMA:
    """
    ReduceLROnPlateau driven by an exponential moving average (EMA) of a metric.

    The EMA uses the same "weight" convention as W&B smoothing:
      ema <- weight * ema + (1 - weight) * x

    The underlying LR reduction logic is delegated to PyTorch ReduceLROnPlateau,
    but it sees the EMA value rather than the raw metric.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        ema_weight: float,
        factor: float = 0.5,
        patience: int = 2000,
        min_lr: float = 1e-7,
        tol: float = 1e-4,
        tol_mode: str = "rel",
    ) -> None:
        if not (0.0 <= float(ema_weight) < 1.0):
            raise ValueError(
                f"ema_weight must be in [0, 1), got {ema_weight!r}"
            )
        if tol_mode not in {"rel", "abs"}:
            raise ValueError(f"tol_mode must be 'rel' or 'abs', got {tol_mode!r}")

        self.ema_weight = float(ema_weight)
        self._ema: float | None = None
        self._scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=factor,
            patience=patience,
            min_lr=min_lr,
            threshold=tol,
            threshold_mode=tol_mode,
        )

    def step(self, metric: float) -> None:
        x = float(metric)
        if self._ema is None:
            self._ema = x
        else:
            w = self.ema_weight
            self._ema = w * self._ema + (1.0 - w) * x
        self._scheduler.step(self._ema)

    def state_dict(self) -> dict:
        return {
            "ema_weight": self.ema_weight,
            "ema": self._ema,
            "scheduler_state_dict": self._scheduler.state_dict(),
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.ema_weight = float(state_dict.get("ema_weight", self.ema_weight))
        ema = state_dict.get("ema")
        self._ema = None if ema is None else float(ema)
        sched_sd = state_dict.get("scheduler_state_dict")
        if sched_sd is not None:
            self._scheduler.load_state_dict(sched_sd)


def create_optimizer(
    name: str,
    params: Iterable[torch.nn.Parameter],
    lr: float,
) -> torch.optim.Optimizer:
    if name == "SGD":
        return torch.optim.SGD(params, lr=lr)
    if name == "Adam":
        return torch.optim.Adam(params, lr=lr)
    raise ValueError(f"Unknown optimizer: {name}")


def create_reduce_lr_on_plateau_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    factor: float = 0.5,
    patience: int = 2000,
    min_lr: float = 1e-7,
    tol: float = 1e-4,
    tol_mode: str = "rel",
) -> ReduceLROnPlateau:
    """Plateau-driven LR decay on a per-step metric (pass `energy` each step).

    `tol` / `tol_mode` map to PyTorch ReduceLROnPlateau `threshold` / `threshold_mode`.
    """
    if tol_mode not in {"rel", "abs"}:
        raise ValueError(f"tol_mode must be 'rel' or 'abs', got {tol_mode!r}")
    return ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=factor,
        patience=patience,
        min_lr=min_lr,
        threshold=tol,
        threshold_mode=tol_mode,
    )


class FlipRejectBackoffController:
    """Reject flipped steps and back off optimizer LR."""

    def __init__(self, *, factor: float = 0.5, min_lr: float = 1e-10) -> None:
        if not (0.0 < float(factor) < 1.0):
            raise ValueError(f"factor must be in (0, 1), got {factor!r}")
        if float(min_lr) <= 0.0:
            raise ValueError(f"min_lr must be > 0, got {min_lr!r}")
        self.factor = float(factor)
        self.min_lr = float(min_lr)
        self._accepted_uvs: torch.Tensor | None = None

    def initialize_accepted(self, img_f_uvs: torch.Tensor) -> None:
        self._accepted_uvs = img_f_uvs.detach().clone()

    def accept(self, img_f_uvs: torch.Tensor) -> None:
        self._accepted_uvs = img_f_uvs.detach().clone()

    def reject_and_backoff(
        self, optimizer: torch.optim.Optimizer, img_f_uvs: torch.Tensor
    ) -> float:
        if self._accepted_uvs is None:
            self._accepted_uvs = img_f_uvs.detach().clone()
        with torch.no_grad():
            img_f_uvs.copy_(self._accepted_uvs)
        return self.backoff_learning_rate(optimizer)

    def backoff_learning_rate(self, optimizer: torch.optim.Optimizer) -> float:
        new_lr_last = self.min_lr
        for group in optimizer.param_groups:
            current_lr = float(group.get("lr", 0.0))
            new_lr = max(self.min_lr, current_lr * self.factor)
            group["lr"] = new_lr
            new_lr_last = new_lr
        return new_lr_last


# Define a gradient hook to set the gradients of boundary vertices to be the same
class GradientHook:
    def __init__(self, shared_inds: list[list[int]]) -> None:
        self.shared_inds = (
            shared_inds  # each list is a group of vertices that are identified
        )

    def __call__(self, grad: torch.Tensor) -> torch.Tensor:
        """
        Set the gradients of boundary vertices to be the same.
        """
        grad = grad.clone()  # Avoid modifying the original gradient
        for inds in self.shared_inds:
            shared_grad = grad[inds].mean(dim=0)
            grad[inds] = shared_grad  # Set shared gradients
        return grad

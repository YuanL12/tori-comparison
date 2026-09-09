"""Early-stop detectors for optimization (energy plateau, too many flipped faces, etc.)."""

from __future__ import annotations

from collections import deque


class LossPlateauDetector:
    """Detect when a monitored scalar has plateaued via relative improvement."""

    def __init__(self, patience: int = 500, rel_tol: float = 1e-5) -> None:
        self.patience = patience
        self.rel_tol = rel_tol
        self.best_loss = float("inf")
        self.steps_without_improvement = 0

    def update(self, loss: float, flip_count: int) -> bool:
        """Return True if loss has plateaued."""
        if self.best_loss == float("inf"):
            self.best_loss = loss
            return False

        rel_improvement = (self.best_loss - loss) / max(abs(self.best_loss), 1e-12)

        # if improvement is significant and there was no flip, update the best value
        if rel_improvement > self.rel_tol and flip_count == 0:
            self.best_loss = loss
            self.steps_without_improvement = 0
        else:  # either there was a flip or the improvement is not significant
            self.steps_without_improvement += 1
        return self.steps_without_improvement >= self.patience


class TotalFlipStopDetector:
    """
    Rolling-window flipped-face count across optimization steps (same units as
    ``compute_flip_faces_count``). Flips indicate the map is no longer locally
    injective / a diffeomorphism onto its image.

    ``update`` returns True once the sum of the most recent ``window_size``
    flip counts strictly exceeds ``max_total_flips``.
    """

    def __init__(self, max_total_flips: int, window_size: int = 100) -> None:
        if int(window_size) <= 0:
            raise ValueError("window_size must be positive.")
        self.max_total_flips = int(max_total_flips)
        self.window_size = int(window_size)
        self._recent_flips: deque[int] = deque(maxlen=self.window_size)
        self._recent_flips_sum = 0

    def recent_flip_count(self) -> int:
        """Return the sum of flip counts in the current rolling window."""
        return int(self._recent_flips_sum)

    def update(self, flip_count: int) -> bool:
        flip_i = int(flip_count)

        if len(self._recent_flips) == self.window_size:
            self._recent_flips_sum -= self._recent_flips.popleft()
        self._recent_flips.append(flip_i)
        self._recent_flips_sum += flip_i

        return self._recent_flips_sum > self.max_total_flips

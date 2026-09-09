"""Simple iteration progress reporting (no tqdm) for long / resumable training runs."""

from __future__ import annotations

import math
import sys
import time
from typing import TextIO


def _format_duration(seconds: float) -> str:
    """Human-readable duration for ETA / elapsed."""
    if seconds < 0 or math.isnan(seconds):
        return "?"
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        m = int(seconds // 60)
        s = int(seconds % 60)
        return f"{m}m{s:02d}s"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    return f"{h}h{m:02d}m"


class IterationProgressReporter:
    """
    Print periodic progress for a fixed iteration budget.

    - Reports every ``interval`` completed iterations and on the last iteration.
    - Shows average rate (it/s), elapsed time, and ETA to finish this chunk.
    - Optionally includes a running global step counter for resumable jobs
      (``global_step_start`` is the step count *before* this chunk's first step).
    """

    def __init__(
        self,
        total: int,
        *,
        interval: int = 1000,
        desc: str = "Optimization",
        global_step_start: int | None = None,
        file: TextIO = sys.stdout,
    ) -> None:
        if total < 0:
            raise ValueError("total must be non-negative")
        if interval < 1:
            raise ValueError("interval must be >= 1")
        self._total = total
        self._interval = interval
        self._desc = desc
        self._global_step_start = global_step_start
        self._file = file

        self._completed = 0
        self._last_printed_at = 0
        self._t0 = time.perf_counter()
        self._closed = False

    @property
    def completed(self) -> int:
        return self._completed

    def step(self) -> None:
        """Call once after each finished optimization step in this chunk."""
        if self._closed:
            return
        self._completed += 1
        if self._completed > self._total:
            return
        if self._completed % self._interval == 0 or self._completed == self._total:
            self._print_line()

    def close(self) -> None:
        """Flush a final line if the last report did not land on ``completed``."""
        if self._closed:
            return
        self._closed = True
        if self._completed > 0 and self._last_printed_at != self._completed:
            self._print_line()

    def _print_line(self) -> None:
        self._last_printed_at = self._completed
        elapsed = time.perf_counter() - self._t0
        remaining = self._total - self._completed
        rate = self._completed / elapsed if elapsed > 0 and self._completed > 0 else 0.0
        eta_sec = remaining / rate if rate > 0 and remaining > 0 else float("nan")

        parts = [
            f"{self._desc}:",
            f"{self._completed}/{self._total}",
            f"elapsed {_format_duration(elapsed)}",
            f"{rate:.3g} it/s",
        ]
        if remaining > 0 and rate > 0:
            parts.append(f"ETA {_format_duration(eta_sec)}")
        elif remaining == 0:
            parts.append("done")

        if self._global_step_start is not None:
            g = self._global_step_start + self._completed
            parts.append(f"global_step={g}")

        print(" ".join(parts), file=self._file, flush=True)

    def __enter__(self) -> IterationProgressReporter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

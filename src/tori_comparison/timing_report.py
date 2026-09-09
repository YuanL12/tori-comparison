"""Periodic wall-clock timing summaries for optimization runs (stdout)."""

from __future__ import annotations

from typing import Dict

# Wall-clock breakdown printed every N steps (stdout).
TIMING_REPORT_INTERVAL = 100

# Sub-buckets from ``compute_geodesic_path_lengths`` (``par_geod`` ``timing_dict``).
# Order follows the geodesic pipeline; sums should match ``compute_geodesic_path_lengths`` wall time.
_GEO_SUB_KEYS = (
    "unitize_uv",
    "barycentric_coordinates_cpp",
    "edge_bookkeeping",
    "geodesic_paths_cpp",
    "uv_lift_to_3d",
    "geodesic_lengths_with_grad",
)

# Minimum averaged remainder (ms) to print under ``compute_geodesic_path_lengths``.
_REMAINDER_MS_EPS = 0.05


def _timing_key_label(key: str) -> str:
    if key.endswith("_s"):
        return key[:-2].replace("_", " ")
    return key.replace("_", " ")


def report_timing_window(
    window_sums: Dict[str, float], window_n: int, label: str, step_end: int
) -> None:
    """Print averaged bucket timings over the last ``window_n`` iterations (hierarchical)."""
    if window_n <= 0:
        return

    def avg_ms(key: str) -> float:
        return window_sums.get(key, 0.0) / window_n * 1000.0

    opt_s = window_sums.get("optimizer_step_s", 0.0)
    post_s = window_sums.get("post_iter_s", 0.0)
    wall_ms = (opt_s + post_s) / window_n * 1000.0

    print(f"[timing] {label} steps ending {step_end} (avg over {window_n} iters)")
    print(f"  wall≈{wall_ms:.1f}ms/iter")

    if opt_s > 0:
        print(f"  - optimizer step={avg_ms('optimizer_step_s'):.1f}ms")

        geo_total_key = "compute_geodesic_path_lengths"
        if window_sums.get(geo_total_key, 0.0) > 0:
            print(
                f"    - {_timing_key_label(geo_total_key)}="
                f"{avg_ms(geo_total_key):.1f}ms"
            )
            inner_sum_s = sum(window_sums.get(k, 0.0) for k in _GEO_SUB_KEYS)
            any_sub = any(window_sums.get(k, 0.0) > 0 for k in _GEO_SUB_KEYS)
            if any_sub:
                for ck in _GEO_SUB_KEYS:
                    if window_sums.get(ck, 0.0) > 0:
                        print(f"      - {_timing_key_label(ck)}={avg_ms(ck):.1f}ms")
                remainder_s = window_sums.get(geo_total_key, 0.0) - inner_sum_s
                remainder_ms = max(0.0, remainder_s) / window_n * 1000.0
                if remainder_ms >= _REMAINDER_MS_EPS:
                    print(f"      - others={remainder_ms:.1f}ms")
        elif window_sums.get("geom_primary_s", 0.0) > 0:
            print(
                f"    - {_timing_key_label('geom_primary_s')}="
                f"{avg_ms('geom_primary_s'):.1f}ms"
            )

        if window_sums.get("construct_PARs", 0.0) > 0:
            print(
                f"    - {_timing_key_label('construct_PARs')}="
                f"{avg_ms('construct_PARs'):.1f}ms"
            )
        elif window_sums.get("par_construct_s", 0.0) > 0:
            print(
                f"    - {_timing_key_label('par_construct_s')}="
                f"{avg_ms('par_construct_s'):.1f}ms"
            )

        if window_sums.get("loss_backward_s", 0.0) > 0:
            print(
                f"    - {_timing_key_label('loss_backward_s')}="
                f"{avg_ms('loss_backward_s'):.1f}ms"
            )

    if post_s > 0:
        print(f"  - {_timing_key_label('post_iter_s')}={avg_ms('post_iter_s'):.1f}ms")

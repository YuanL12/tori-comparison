from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
DEFAULT_MESH = REPO_ROOT / "test" / "input" / "torus" / "torus.obj"

sys.path.insert(0, str(REPO_ROOT / "build"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

import shapecomp as sc  # pyright: ignore[reportMissingImports]
from tori_comparison.cuda_locator import (
    build_uniform_grid_for_locator,
    find_bary_coords,
    load_locator,
    locate_uv_grid_tensors,
    locate_uv_tensors,
)


def make_points(tri_uv: np.ndarray, n_points: int) -> np.ndarray:
    centroids = tri_uv.mean(axis=1)
    idx = np.arange(n_points, dtype=np.int64) % centroids.shape[0]
    return centroids[idx].copy()


def median_ms(samples: list[float]) -> float:
    return statistics.median(samples) * 1000.0


def time_cpu(planar: sc.PlanarLocator, points: np.ndarray, repeats: int) -> float:
    times: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        planar.find_bary_coords_parallel(points)
        times.append(time.perf_counter() - t0)
    return median_ms(times)


def time_cuda_event_ms(fn: Callable[[], object], repeats: int) -> float:
    times: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) / 1000.0)
    return median_ms(times)


def time_cuda_sync_ms(fn: Callable[[], object], repeats: int) -> float:
    times: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return median_ms(times)


def time_cuda_api_ms(fn: Callable[[], object], repeats: int) -> float:
    times: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return median_ms(times)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark the CUDA UV locator.")
    parser.add_argument("--mesh", type=Path, default=DEFAULT_MESH)
    parser.add_argument("--sizes", type=int, nargs="+", default=[1_000, 10_000, 100_000])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--eps", type=float, default=1e-10)
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    parser.add_argument("--grid-res", type=int, default=64)
    parser.add_argument("--verbose-build", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available; benchmark requires a CUDA GPU.")

    mesh = sc.load_mesh(str(args.mesh))
    planar = sc.PlanarLocator(mesh)
    uv_np = np.asarray(planar.get_uv_positions(), dtype=np.float64)
    faces_np = np.asarray(planar.get_cutted_mesh_faces(), dtype=np.int64)
    tri_uv_np = uv_np[faces_np]

    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    uv_t = torch.as_tensor(uv_np, dtype=dtype, device="cuda")
    faces_t = torch.as_tensor(faces_np, dtype=torch.long, device="cuda")
    tri_uv_t = uv_t[faces_t].contiguous()
    cell_starts_t, cell_tris_t = build_uniform_grid_for_locator(tri_uv_t, args.grid_res)

    load_locator(verbose=args.verbose_build)
    print(f"mesh: {args.mesh}")
    print(
        f"faces: {tri_uv_np.shape[0]}  dtype: {args.dtype}  "
        f"grid: {args.grid_res}x{args.grid_res}  candidates: {cell_tris_t.numel()}  "
        f"repeats: {args.repeats}"
    )
    print(
        "N        CPU ms    brute sync ms    grid kernel ms    "
        "grid sync ms    grid API ms    grid speedup"
    )

    for n_points in args.sizes:
        points_np = make_points(tri_uv_np, n_points)
        points_t = torch.as_tensor(points_np, dtype=dtype, device="cuda")

        brute = lambda: locate_uv_tensors(points_t, tri_uv_t, args.eps)
        grid = lambda: locate_uv_grid_tensors(
            points_t, tri_uv_t, cell_starts_t, cell_tris_t, args.grid_res, args.eps
        )
        api = lambda: find_bary_coords(points_t, uv_t, faces_t, args.eps, args.grid_res)

        brute()
        grid()
        api()
        torch.cuda.synchronize()

        cpu_ms = time_cpu(planar, points_np, args.repeats)
        brute_sync_ms = time_cuda_sync_ms(brute, args.repeats)
        grid_kernel_ms = time_cuda_event_ms(grid, args.repeats)
        grid_sync_ms = time_cuda_sync_ms(grid, args.repeats)
        grid_api_ms = time_cuda_api_ms(api, args.repeats)
        speedup = cpu_ms / grid_sync_ms if grid_sync_ms > 0 else float("inf")
        print(
            f"{n_points:<8d} "
            f"{cpu_ms:>8.3f} "
            f"{brute_sync_ms:>16.3f} "
            f"{grid_kernel_ms:>17.3f} "
            f"{grid_sync_ms:>15.3f} "
            f"{grid_api_ms:>14.3f} "
            f"{speedup:>14.2f}x"
        )


if __name__ == "__main__":
    main()

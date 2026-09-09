from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.cpp_extension import load

THIS_DIR = Path(__file__).resolve().parent
CUDA_DIR = THIS_DIR.parents[1] / "experiments" / "cuda"
DEFAULT_GRID_RES = 64

_LOCATOR: Any | None = None
_GRID_CACHE: dict[tuple, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}


def load_locator(verbose: bool = False) -> Any:
    """Lazy-load the experimental CUDA UV locator extension."""
    global _LOCATOR
    if _LOCATOR is None:
        _LOCATOR = load(
            name="tori_cuda_locator",
            sources=[str(CUDA_DIR / "locator.cu")],
            extra_cflags=["-O2"],
            extra_cuda_cflags=["-O2"],
            verbose=verbose,
        )
    return _LOCATOR


def locate_uv_tensors(
    points: torch.Tensor,
    tri_uv: torch.Tensor,
    eps: float = 1e-10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the brute-force CUDA locator on already-built triangle UVs."""
    if not points.is_cuda or not tri_uv.is_cuda:
        raise ValueError("points and tri_uv must both be CUDA tensors.")
    if points.dtype not in (torch.float32, torch.float64):
        raise TypeError(f"points dtype must be float32 or float64, got {points.dtype}.")
    if tri_uv.dtype != points.dtype:
        raise TypeError("points and tri_uv must have the same dtype.")

    locator = load_locator()
    return locator.locate_uv(
        points.detach().contiguous(),
        tri_uv.detach().contiguous(),
        float(eps),
    )


def locate_uv_grid_tensors(
    points: torch.Tensor,
    tri_uv: torch.Tensor,
    cell_starts: torch.Tensor,
    cell_tris: torch.Tensor,
    grid_res: int = DEFAULT_GRID_RES,
    eps: float = 1e-10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the uniform-grid CUDA locator on already-built grid tensors."""
    if not points.is_cuda or not tri_uv.is_cuda:
        raise ValueError("points and tri_uv must both be CUDA tensors.")
    if not cell_starts.is_cuda or not cell_tris.is_cuda:
        raise ValueError("cell_starts and cell_tris must both be CUDA tensors.")
    if points.dtype not in (torch.float32, torch.float64):
        raise TypeError(f"points dtype must be float32 or float64, got {points.dtype}.")
    if tri_uv.dtype != points.dtype:
        raise TypeError("points and tri_uv must have the same dtype.")

    locator = load_locator()
    return locator.locate_uv_grid(
        points.detach().contiguous(),
        tri_uv.detach().contiguous(),
        cell_starts.detach().contiguous(),
        cell_tris.detach().contiguous(),
        int(grid_res),
        float(eps),
    )


def build_tri_uv_for_locator(
    points: torch.Tensor,
    codomain_uvs: torch.Tensor,
    cutted_codomain_faces: torch.Tensor,
) -> torch.Tensor:
    """Build (F, 3, 2) UV triangles on the same CUDA device/dtype as points."""
    if not points.is_cuda:
        raise ValueError("points must be a CUDA tensor.")
    faces = cutted_codomain_faces.detach().to(device=points.device, dtype=torch.long)
    uvs = codomain_uvs.detach().to(device=points.device, dtype=points.dtype)
    return uvs[faces].contiguous()


def build_uniform_grid_for_locator(
    tri_uv: torch.Tensor,
    grid_res: int = DEFAULT_GRID_RES,
    padding: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build CSR-style grid tensors: cell_starts and flattened cell_tris."""
    if not tri_uv.is_cuda:
        raise ValueError("tri_uv must be a CUDA tensor.")
    grid_res = int(grid_res)
    if grid_res <= 0:
        raise ValueError("grid_res must be positive.")

    bins: list[list[int]] = [[] for _ in range(grid_res * grid_res)]
    tri_np = tri_uv.detach().cpu().numpy()
    for face, tri in enumerate(tri_np):
        lo = np.floor(
            np.clip((tri.min(axis=0) - padding) * grid_res, 0, grid_res - 1)
        ).astype(np.int64)
        hi = np.floor(
            np.clip((tri.max(axis=0) + padding) * grid_res, 0, grid_res - 1)
        ).astype(np.int64)
        for gy in range(int(lo[1]), int(hi[1]) + 1):
            row = gy * grid_res
            for gx in range(int(lo[0]), int(hi[0]) + 1):
                bins[row + gx].append(int(face))

    starts = [0]
    flat: list[int] = []
    for bucket in bins:
        flat.extend(bucket)
        starts.append(len(flat))

    return (
        torch.as_tensor(starts, dtype=torch.long, device=tri_uv.device),
        torch.as_tensor(flat, dtype=torch.long, device=tri_uv.device),
    )


def _grid_cache_key(
    points: torch.Tensor,
    codomain_uvs: torch.Tensor,
    cutted_codomain_faces: torch.Tensor,
    grid_res: int,
) -> tuple:
    return (
        str(points.device),
        str(points.dtype),
        int(grid_res),
        tuple(codomain_uvs.shape),
        tuple(codomain_uvs.stride()),
        int(codomain_uvs.data_ptr()),
        tuple(cutted_codomain_faces.shape),
        tuple(cutted_codomain_faces.stride()),
        int(cutted_codomain_faces.data_ptr()),
    )


def grid_tensors_for_locator(
    points: torch.Tensor,
    codomain_uvs: torch.Tensor,
    cutted_codomain_faces: torch.Tensor,
    grid_res: int = DEFAULT_GRID_RES,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # ponytail: process-local cache assumes codomain UVs/faces are fixed during optimization.
    key = _grid_cache_key(points, codomain_uvs, cutted_codomain_faces, grid_res)
    cached = _GRID_CACHE.get(key)
    if cached is None:
        tri_uv = build_tri_uv_for_locator(points, codomain_uvs, cutted_codomain_faces)
        cell_starts, cell_tris = build_uniform_grid_for_locator(tri_uv, grid_res)
        cached = (tri_uv, cell_starts, cell_tris)
        _GRID_CACHE[key] = cached
    return cached


def find_bary_coords(
    points: torch.Tensor,
    codomain_uvs: torch.Tensor,
    cutted_codomain_faces: torch.Tensor,
    eps: float = 1e-10,
    grid_res: int = DEFAULT_GRID_RES,
) -> list[tuple[int, list[float]]]:
    """CUDA replacement for PlanarLocator.find_bary_coords_parallel."""
    tri_uv, cell_starts, cell_tris = grid_tensors_for_locator(
        points, codomain_uvs, cutted_codomain_faces, grid_res
    )
    out_face, out_bary = locate_uv_grid_tensors(
        points, tri_uv, cell_starts, cell_tris, grid_res, eps
    )

    faces = out_face.detach().cpu().tolist()
    bary = out_bary.detach().cpu().tolist()
    missing = sum(1 for face in faces if int(face) < 0)
    if missing:
        raise RuntimeError(
            f"CUDA locator failed to locate {missing} / {len(faces)} UV points."
        )
    return [(int(face), bary_i) for face, bary_i in zip(faces, bary)]

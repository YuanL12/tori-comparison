from __future__ import annotations

import os
from typing import Any, Dict, List

import numpy as np

from tori_comparison.checkpoint import save_checkpoint


def save_mesh_context(
    mesh_context_path: str,
    *,
    arrays: dict,
    domain_uvs: np.ndarray,
    codomain_uvs: np.ndarray,
    shared_inds: list[list[int]],
    distinct_direction: list[float] | None,
    tutte_embedding_type: str | None = None,
    extra: dict | None = None,
) -> None:
    """Write mesh arrays and UV glue data once per run (separate from optimization ckpt)."""
    state: Dict[str, Any] = {
        "version": 1,
        "arrays": arrays,
        "domain_uvs": domain_uvs,
        "codomain_uvs": codomain_uvs,
        "shared_inds": shared_inds,
        "distinct_direction": distinct_direction,
        "tutte_embedding_type": tutte_embedding_type,
    }
    if extra:
        state.update(extra)
    save_checkpoint(mesh_context_path, state)


def save_final_details(
    save_folder_path: str,
    saved_details: Dict[str, Any],
    *,
    run_name: str | None = None,
    delta_iterations: int = 10,
) -> None:
    uv_pos_res: List[np.ndarray] = saved_details["uv_pos_res"]

    # save the uv positions per delta_iterations steps
    uv_pos_res = uv_pos_res[::delta_iterations]
    os.makedirs(save_folder_path, exist_ok=True)
    filename = "uv_pos_res.npy" if not run_name else f"uv_pos_res__{run_name}.npy"
    np.save(os.path.join(save_folder_path, filename), np.array(uv_pos_res))

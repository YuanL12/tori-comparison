"""
Helpers for visualizing the three geodesic polylines of one cut-domain triangle on the codomain mesh.

Uses the same pipeline as ``core.par_geod.compute_geodesic_path_lengths`` (surface points → parallel geodesics).
"""

from __future__ import annotations

import dataclasses
import os
import sys
from typing import Any

import numpy as np
import torch

# shapecomp package lives in repo ``build/`` (same convention as ``tori_comparison.data``).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "build"))
import shapecomp as sc  # noqa: E402

from tori_comparison.checkpoint import load_checkpoint
from tori_comparison.config import ExperimentConfig, load_config
from tori_comparison.data import (
    get_uvs_and_shared,
    load_meshes_and_construct_planar_locators,
    numpy_to_torch_tensors,
)
from tori_comparison.par_geod import construct_surface_points_pairs_for_edges


def experiment_config_from_dict(d: dict) -> ExperimentConfig:
    kwargs: dict = {}
    for f in dataclasses.fields(ExperimentConfig):
        if not f.init:
            continue
        if f.name in d:
            kwargs[f.name] = d[f.name]
        elif f.default is not dataclasses.MISSING:
            kwargs[f.name] = f.default
        elif f.default_factory is not dataclasses.MISSING:
            kwargs[f.name] = f.default_factory()
        else:
            raise ValueError(f"config missing required key: {f.name!r}")
    return ExperimentConfig(**kwargs)


def _paths_from_config(cfg: ExperimentConfig) -> tuple[str, str]:
    run_dir = os.path.join(cfg.checkpoint_dir, cfg.run_name)
    return (
        os.path.join(run_dir, "mesh_context.pt"),
        os.path.join(run_dir, cfg.checkpoint_filename),
    )


def _load_mesh_and_planar(
    cfg: ExperimentConfig, mesh_ctx: dict
) -> tuple[Any, dict[str, torch.Tensor], np.ndarray]:
    distinct_direction = mesh_ctx.get("distinct_direction")
    if cfg.cut_on_shortest_generator and distinct_direction is None:
        raise RuntimeError("mesh_context missing distinct_direction for cut_on_shortest_generator.")

    codomain_swap_generators = bool(mesh_ctx.get("codomain_swap_generators", False))
    domain_tutte = mesh_ctx.get("domain_tutte_embedding_type", cfg.Tutte_embedding_type)
    codomain_tutte = mesh_ctx.get(
        "codomain_tutte_embedding_type", cfg.Tutte_embedding_type
    )

    planar1, planar2, arrays_rebuilt = load_meshes_and_construct_planar_locators(
        obj_path1=cfg.obj_path1,
        obj_path2=cfg.obj_path2,
        normalize_area_to_one=cfg.normalize_area_to_one,
        domain_Tutte_embedding_type=domain_tutte,
        codomain_Tutte_embedding_type=codomain_tutte,
        cut_on_shortest_generator=cfg.cut_on_shortest_generator,
        distinct_direction=distinct_direction,
        codomain_swap_generators=codomain_swap_generators,
    )
    domain_uvs_r, codomain_uvs_r, _ = get_uvs_and_shared(planar1, planar2)

    arrays = mesh_ctx["arrays"]
    if not np.array_equal(arrays_rebuilt["cutted_mesh_faces_1"], arrays["cutted_mesh_faces_1"]):
        raise RuntimeError("cutted_mesh_faces_1 mismatch: config/obj paths differ from mesh_context.")
    if not np.array_equal(arrays_rebuilt["cutted_mesh_faces_2"], arrays["cutted_mesh_faces_2"]):
        raise RuntimeError("cutted_mesh_faces_2 mismatch.")
    if not np.allclose(domain_uvs_r, mesh_ctx["domain_uvs"]):
        raise RuntimeError("domain_uvs mismatch (seed/direction/Tutte/config?).")
    if not np.allclose(codomain_uvs_r, mesh_ctx["codomain_uvs"]):
        raise RuntimeError("codomain_uvs mismatch.")

    torch_tensors = numpy_to_torch_tensors(arrays)
    return planar2, torch_tensors, mesh_ctx["codomain_uvs"]


def load_resumable_run_for_vis(
    config_path: str,
    *,
    mesh_context_path: str | None = None,
    checkpoint_path: str | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    """
    Load YAML config, ``mesh_context.pt``, ``checkpoint.pt``, and rebuild codomain ``PlanarLocator``.

    Returns a dict with keys: ``cfg``, ``planar2``, ``torch_tensors``, ``codomain_uvs_np``,
    ``img_f_uvs`` (tensor, requires_grad True), ``resume_state``, ``mesh_ctx``, ``ckpt``.
    """
    raw = load_config(config_path)
    cfg = experiment_config_from_dict(raw)
    mesh_path, ckpt_path = _paths_from_config(cfg)
    if mesh_context_path:
        mesh_path = mesh_context_path
    if checkpoint_path:
        ckpt_path = checkpoint_path

    dev = torch.device(device)
    mesh_ctx = load_checkpoint(mesh_path, map_location=str(dev))
    ckpt = load_checkpoint(ckpt_path, map_location=str(dev))

    planar2, torch_tensors, codomain_uvs_np = _load_mesh_and_planar(cfg, mesh_ctx)
    img_f_uvs = torch.tensor(ckpt["img_f_uvs"], dtype=torch.float64, device=dev, requires_grad=True)

    return {
        "cfg": cfg,
        "planar2": planar2,
        "torch_tensors": torch_tensors,
        "codomain_uvs_np": codomain_uvs_np,
        "img_f_uvs": img_f_uvs,
        "resume_state": ckpt.get("resume_state") or {},
        "mesh_ctx": mesh_ctx,
        "ckpt": ckpt,
    }


def canonical_edges_of_triangle(v0: int, v1: int, v2: int) -> list[tuple[int, int]]:
    """Undirected edges (min,max) in order (v0,v1), (v1,v2), (v2,v0)."""
    out: list[tuple[int, int]] = []
    for a, b in ((v0, v1), (v1, v2), (v2, v0)):
        lo, hi = (a, b) if a < b else (b, a)
        out.append((int(lo), int(hi)))
    return out


def geodesic_polylines_for_cutted_face(
    codomain_planar_locator: Any,
    codomain_vertices: torch.Tensor,
    codomain_faces: torch.Tensor,
    cutted_domain_faces: torch.Tensor,
    uv_images: torch.Tensor,
    face_idx: int,
    geodesic_type: str,
) -> tuple[list[np.ndarray], tuple[int, int, int], list[str]]:
    """
    Run the same geodesic pass as training and return the three polylines for one cut-domain face.

    Returns:
        polylines: three (K_i, 3) float64 arrays (polyline vertices in 3D).
        corners: (v0, v1, v2) vertex indices on the cut mesh.
        labels: short names for each edge, e.g. ``edge (24, 1433)``.
    """
    unitized_uvs = uv_images % 1
    uv_surface_points = codomain_planar_locator.find_bary_coords_parallel(
        unitized_uvs.detach().clone().numpy()
    )
    surface_points_pairs_collections, edge_to_its_index = construct_surface_points_pairs_for_edges(
        cutted_domain_faces,
        uv_surface_points,
    )

    V = codomain_vertices.detach().cpu().numpy()
    F = codomain_faces.detach().cpu().numpy()
    if geodesic_type == "geodesic":
        geo_paths = sc.geodesic_compute_parallel(
            surface_points_pairs_collections,
            V,
            F,
        )
    elif geodesic_type == "flip_geodesic":
        geo_paths = sc.flip_geodesic_compute_parallel(
            surface_points_pairs_collections,
            V,
            F,
        )
    else:
        raise ValueError(f"Unknown geodesic type: {geodesic_type}")

    faces_np = cutted_domain_faces.detach().cpu().numpy()
    tri = faces_np[int(face_idx)]
    v0, v1, v2 = int(tri[0]), int(tri[1]), int(tri[2])
    edges = canonical_edges_of_triangle(v0, v1, v2)

    polylines: list[np.ndarray] = []
    labels: list[str] = []
    for e in edges:
        if e not in edge_to_its_index:
            raise KeyError(f"Edge {e} not in edge_to_its_index (mesh bookkeeping mismatch).")
        idx = edge_to_its_index[e]
        polylines.append(np.asarray(geo_paths[idx], dtype=np.float64))
        labels.append(f"edge {e}")

    return polylines, (v0, v1, v2), labels

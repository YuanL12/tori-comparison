"""Initial Tutte embedding energy (area + shape-stretch) and minimal-type selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch

from tori_comparison.data import (
    load_mesh_and_construct_planar_locator,
    numpy_to_torch_tensors,
)
from tori_comparison.energy import compute_energy_torch
from tori_comparison.par_euc import consturct_euclidean_PARs_list_vectorized
from tori_comparison.par_geod import compute_geodesic_path_lengths, construct_geodesic_PARs_list
from tori_comparison.par_geod import lift_uv_coordinates_to_codomain_3d_batched

TutteType = Literal["Uniform", "coTan", "MeanValue", "Authalic"]

TUTTE_TYPES: list[TutteType] = ["Uniform", "coTan", "MeanValue", "Authalic"]


@dataclass(frozen=True)
class EnergyTerms:
    area_loss: float
    shape_stretch_loss: float
    energy: float
    flip_count: int


def sample_unit_direction(seed: int) -> list[float]:
    rng = np.random.default_rng(int(seed))
    vec = rng.normal(size=3)
    nrm = float(np.linalg.norm(vec))
    if nrm <= 1e-12:
        return [1.0, 0.0, 0.0]
    vec = vec / nrm
    return vec.tolist()


def _flip_faces_count(uv_image: torch.Tensor, domain_faces: torch.Tensor) -> int:
    # Same tolerance as core/energy.py
    f_uvs = uv_image[domain_faces]  # (n_faces, 3, 2)
    edge1 = f_uvs[:, 1] - f_uvs[:, 0]
    edge2 = f_uvs[:, 2] - f_uvs[:, 0]
    cross = edge1[:, 0] * edge2[:, 1] - edge1[:, 1] * edge2[:, 0]
    return int((cross < -1e-10).sum().item())


@dataclass(frozen=True)
class _PrebuiltDomain:
    tutte_type: TutteType
    planar: object
    arrays: dict[str, np.ndarray]
    domain_uvs_np: np.ndarray
    shared_inds: list[list[int]]


@dataclass(frozen=True)
class _PrebuiltCodomain:
    tutte_type: TutteType
    planar: object
    arrays: dict[str, np.ndarray]
    codomain_uvs_np: np.ndarray


def _compute_initial_energy_terms_from_prebuilt(
    *,
    domain: _PrebuiltDomain,
    codomain: _PrebuiltCodomain,
    edge_length_type: Literal["geodesic", "euclidean", "flip_geodesic"],
    device: str,
) -> EnergyTerms:
    # Merge arrays into the legacy structure expected by numpy_to_torch_tensors.
    arrays = {
        "domain_faces": domain.arrays["domain_faces"],
        "domain_vertices": domain.arrays["domain_vertices"],
        "cutted_mesh_vertices_1": domain.arrays["cutted_mesh_vertices_1"],
        "cutted_mesh_faces_1": domain.arrays["cutted_mesh_faces_1"],
        "codomain_faces": codomain.arrays["codomain_faces"],
        "codomain_vertices": codomain.arrays["codomain_vertices"],
        "cutted_mesh_vertices_2": codomain.arrays["cutted_mesh_vertices_2"],
        "cutted_mesh_faces_2": codomain.arrays["cutted_mesh_faces_2"],
    }

    torch_tensors = numpy_to_torch_tensors(arrays)
    codomain_faces = torch_tensors["codomain_faces"].to(device)
    codomain_vertices = torch_tensors["codomain_vertices"].to(device)
    cutted_domain_vertices = torch_tensors["cutted_domain_vertices"].to(device)
    cutted_domain_faces = torch_tensors["cutted_domain_faces"].to(device)
    cutted_codomain_faces = torch_tensors["cutted_codomain_faces"].to(device)

    codomain_uvs = torch.tensor(
        codomain.codomain_uvs_np, dtype=torch.float64, device=device
    )
    img_f_uvs = torch.tensor(domain.domain_uvs_np, dtype=torch.float64, device=device)

    planar2 = codomain.planar

    if edge_length_type in ("geodesic", "flip_geodesic"):
        edge_to_geo = compute_geodesic_path_lengths(
            cutted_domain_faces=cutted_domain_faces,
            uv_images=img_f_uvs,
            codomain_planar_locator=planar2,
            codomain_uvs=codomain_uvs,
            codomain_vertices=codomain_vertices,
            codomain_faces=codomain_faces,
            cutted_codomain_faces=cutted_codomain_faces,
            geodesic_type=edge_length_type,
        )
        Ps, As, Rs, Ds = construct_geodesic_PARs_list(
            cutted_domain_vertices=cutted_domain_vertices,
            cutted_domain_faces=cutted_domain_faces,
            edge_to_geodesic_edge_lengths=edge_to_geo,
        )
    elif edge_length_type == "euclidean":
        image_of_cutted_domain_vertices = lift_uv_coordinates_to_codomain_3d_batched(
            uv_coordinates=img_f_uvs,
            codomain_planar_locator=planar2,
            codomain_uvs=codomain_uvs,
            codomain_vertices=codomain_vertices,
            codomain_faces=codomain_faces,
            cutted_codomain_faces=cutted_codomain_faces,
        )
        Ps, As, Rs, Ds = consturct_euclidean_PARs_list_vectorized(
            cutted_domain_vertices=cutted_domain_vertices,
            cutted_domain_faces=cutted_domain_faces,
            image_of_cutted_domain_vertices=image_of_cutted_domain_vertices,
        )
    else:
        raise ValueError(f"Unknown edge_length_type: {edge_length_type}")

    area_loss, ss_loss = compute_energy_torch(Ps, As, Rs, D=Ds, smooth_max_type="none")
    flip_count = _flip_faces_count(img_f_uvs, cutted_domain_faces)

    a = float(area_loss.detach().cpu().item())
    s = float(ss_loss.detach().cpu().item())
    return EnergyTerms(
        area_loss=a,
        shape_stretch_loss=s,
        energy=a + s,
        flip_count=flip_count,
    )


def select_tutte_type_minimal_initial_energy(
    *,
    obj_path1: str,
    obj_path2: str,
    normalize_area_to_one: bool,
    edge_length_type: Literal["geodesic", "euclidean", "flip_geodesic"],
    cut_on_shortest_generator: bool,
    distinct_direction: list[float] | None,
    codomain_swap_generators: bool,
    device: str,
) -> tuple[str, str, dict[str, float]]:
    """
    Among the standard Tutte embedding types, choose the (domain_type, codomain_type)
    pair whose *initial* area+shape energy (unweighted sum) is smallest.

    Ties break deterministically by first occurrence in TUTTE_TYPES for domain,
    then codomain.

    Returns:
      domain_type: the best domain Tutte type
      codomain_type: the best codomain Tutte type
      energy_by_pair: a dictionary of energy by pair of Tutte types
    """
    # Prebuild each planar locator exactly once per type per side.
    prebuilt_domain: dict[TutteType, _PrebuiltDomain] = {}
    prebuilt_codomain: dict[TutteType, _PrebuiltCodomain] = {}

    for td in TUTTE_TYPES:
        planar_d, faces_d, verts_d, cut_v_d, cut_f_d = (
            load_mesh_and_construct_planar_locator(
                obj_path=obj_path1,
                normalize_area_to_one=normalize_area_to_one,
                Tutte_embedding_type=td,
                cut_on_shortest_generator=cut_on_shortest_generator,
                distinct_direction=distinct_direction,
                swap_generators=False,
            )
        )
        domain_uvs_np = planar_d.get_uv_positions()
        shared_inds: list[list[int]] = []
        for k, val in planar_d.get_identification_map().items():
            shared_inds.append([k, *val])

        prebuilt_domain[td] = _PrebuiltDomain(
            tutte_type=td,
            planar=planar_d,
            arrays={
                "domain_faces": faces_d,
                "domain_vertices": verts_d,
                "cutted_mesh_vertices_1": cut_v_d,
                "cutted_mesh_faces_1": cut_f_d,
            },
            domain_uvs_np=domain_uvs_np,
            shared_inds=shared_inds,
        )

    for tc in TUTTE_TYPES:
        planar_c, faces_c, verts_c, cut_v_c, cut_f_c = (
            load_mesh_and_construct_planar_locator(
                obj_path=obj_path2,
                normalize_area_to_one=normalize_area_to_one,
                Tutte_embedding_type=tc,
                cut_on_shortest_generator=cut_on_shortest_generator,
                distinct_direction=distinct_direction,
                swap_generators=codomain_swap_generators,
            )
        )
        codomain_uvs_np = planar_c.get_uv_positions()
        prebuilt_codomain[tc] = _PrebuiltCodomain(
            tutte_type=tc,
            planar=planar_c,
            arrays={
                "codomain_faces": faces_c,
                "codomain_vertices": verts_c,
                "cutted_mesh_vertices_2": cut_v_c,
                "cutted_mesh_faces_2": cut_f_c,
            },
            codomain_uvs_np=codomain_uvs_np,
        )

    # Compute initial energy for each pair of Tutte types and find the best pair.
    best: tuple[float, int, int, str, str] | None = None
    energies: dict[str, float] = {}
    for i, td in enumerate(TUTTE_TYPES):
        for j, tc in enumerate(TUTTE_TYPES):
            try:
                terms = _compute_initial_energy_terms_from_prebuilt(
                    domain=prebuilt_domain[td],
                    codomain=prebuilt_codomain[tc],
                    edge_length_type=edge_length_type,
                    device=device,
                )
            except ValueError as exc:
                print(
                    "WARNING: skipping Tutte pair "
                    f"(domain={td!r}, codomain={tc!r}) for minimal initial energy: {exc}",
                    flush=True,
                )
                continue
            key = f"{td}_{tc}"
            energies[key] = terms.energy
            cand = (terms.energy, i, j, td, tc)
            if best is None or cand < best:
                best = cand
    if best is None:
        raise ValueError(
            "No Tutte type pair produced a valid initial energy; "
            "every pair raised ValueError during computation."
        )
    return best[3], best[4], energies

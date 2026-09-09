"""Geodesic edge lengths and PAR metrics using surface geodesics (shapecomp).

Pipeline (see compute_geodesic_path_lengths): UVs → planar locator (surface
points) → unique edges + C++ geodesic polylines → lift UV coordinates to 3D codomain
positions (grad) → map each edge to polyline length w.r.t. endpoints. PAR
construction for training uses construct_geodesic_PARs_list on those lengths.
"""

from __future__ import annotations

import time

import numpy as np
import torch
import shapecomp as sc

PAR_GEOD_SANITY_CHECK = False
if PAR_GEOD_SANITY_CHECK:
    print("PAR_GEOD_SANITY_CHECK is enabled")
else:
    print("PAR_GEOD_SANITY_CHECK is disabled")


def round_bary_coordinates_numpy(arr: np.ndarray, decimals: int = 5) -> np.ndarray:
    """Round barycentric weights for stable C++ geodesic queries (sum 1, entries ≥ 0).

    The **smallest** weight (after clipping and normalization) is **not** rounded on
    its own. The other coordinates are rounded with ``np.round(..., decimals)``,
    then the smallest is set to ``1 - sum(those)``. That keeps tiny weights from
    being wiped to zero by independent rounding (or by floor-then-spread), which
    otherwise can move the point badly on the simplex and produce inconsistent
    geodesic edge lengths (e.g. triangle inequality failures on the three geodesics
    of one domain face).

    The old floor-and-remainder scheme sent the closure mass to vertices with the
    largest *fractional* parts of ``(arr - floor(arr))``, not necessarily to the
    coordinate that should absorb ``1 - sum(round(others))``; for near-boundary
    barycentrics that often zeroed the smallest component incorrectly.
    """
    arr = np.asarray(arr, dtype=np.float64).reshape(-1)
    arr = np.clip(arr, 0.0, None)
    total = float(np.sum(arr))
    if total <= 0.0:
        raise ValueError("round_bary_coordinates_numpy: weights are all non-positive.")
    arr = arr / total

    # Find the index of the smallest weight
    k = int(np.argmin(arr))
    # Initialize the output array
    out = np.empty_like(arr)
    # Round the weights
    for i in range(arr.shape[0]):
        if i != k:
            out[i] = np.round(arr[i], decimals)
    # Set the smallest weight to 1.0 - sum(the other weights)
    out[k] = 1.0 - sum(float(out[j]) for j in range(arr.shape[0]) if j != k)

    if out[k] < 0.0:
        # ``np.round`` on the larger weights can overshoot 1 (rare); fall back to
        # non-negative renormalize rather than leaving a negative weight.
        out = np.clip(out, 0.0, None)
        s = float(np.sum(out))
        if s > 0.0:
            out = out / s
    return out


def barycentric_coordinates_torch(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    P: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    (With Gradient) Compute the barycentric coordinates of a point P with respect to a triangle ABC.

    Args:
        A (torch.Tensor): Vertex A of the triangle, shape (2,).
        B (torch.Tensor): Vertex B of the triangle, shape (2,).
        C (torch.Tensor): Vertex C of the triangle, shape (2,).
        P (torch.Tensor): Point P, shape (2,).

    Returns:
        tuple: Barycentric coordinates (lambdaA, lambdaB, lambdaC).
    """
    # Compute reference determinant for triangle ABC
    denom = (B[0] - A[0]) * (C[1] - A[1]) - (C[0] - A[0]) * (B[1] - A[1])

    if denom == 0:
        raise ValueError("Degenerate triangle: The points A, B, and C are collinear.")

    # Compute sub-triangle determinants
    lambdaA = ((B[0] - P[0]) * (C[1] - P[1]) - (C[0] - P[0]) * (B[1] - P[1])) / denom
    lambdaB = ((C[0] - P[0]) * (A[1] - P[1]) - (A[0] - P[0]) * (C[1] - P[1])) / denom
    lambdaC = ((A[0] - P[0]) * (B[1] - P[1]) - (B[0] - P[0]) * (A[1] - P[1])) / denom

    # Sanity check if their sum is close to 1
    sum_coords = lambdaA + lambdaB + lambdaC
    if torch.abs(sum_coords - 1.0) > 1e-4:
        raise ValueError(
            f"Barycentric coordinates do not sum to 1: diff = {torch.abs(sum_coords - 1.0).item()}"
        )
    if lambdaA < 0 or lambdaB < 0 or lambdaC < 0:
        raise ValueError(
            f"Barycentric coordinates are negative: lambdaA = {lambdaA.item()}, lambdaB = {lambdaB.item()}, lambdaC = {lambdaC.item()}"
        )

    return lambdaA, lambdaB, lambdaC


def barycentric_coordinates_torch_batched(
    tri_uv: torch.Tensor,
    P: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Batched barycentric coordinates: one triangle ABC and query point P per row.

    Uses the same 2D determinant formula;
    requires P strictly inside the triangle (positive barycentrics) or raises.

    Args:
        tri_uv: (n, 3, 2) — per row, triangle corners A, B, C in 2D.
        P: (n, 2) — query points.

    Returns:
        Three tensors of shape (n,) — (lambda_A, lambda_B, lambda_C) per row.
    """
    A = tri_uv[:, 0, :]
    B = tri_uv[:, 1, :]
    C = tri_uv[:, 2, :]

    denom = (B[:, 0] - A[:, 0]) * (C[:, 1] - A[:, 1]) - (C[:, 0] - A[:, 0]) * (
        B[:, 1] - A[:, 1]
    )

    if torch.any(denom == 0):
        raise ValueError(
            "Degenerate triangle: The points A, B, and C are collinear (batched)."
        )

    lambdaA = (
        (B[:, 0] - P[:, 0]) * (C[:, 1] - P[:, 1])
        - (C[:, 0] - P[:, 0]) * (B[:, 1] - P[:, 1])
    ) / denom
    lambdaB = (
        (C[:, 0] - P[:, 0]) * (A[:, 1] - P[:, 1])
        - (A[:, 0] - P[:, 0]) * (C[:, 1] - P[:, 1])
    ) / denom
    lambdaC = (
        (A[:, 0] - P[:, 0]) * (B[:, 1] - P[:, 1])
        - (B[:, 0] - P[:, 0]) * (A[:, 1] - P[:, 1])
    ) / denom

    sum_coords = lambdaA + lambdaB + lambdaC
    if torch.any(torch.abs(sum_coords - 1.0) > 1e-4):
        raise ValueError(
            "Barycentric coordinates do not sum to 1 (batched): "
            f"max |sum-1| = {torch.max(torch.abs(sum_coords - 1.0)).item()}"
        )
    if torch.any((lambdaA < 0) | (lambdaB < 0) | (lambdaC < 0)):
        raise ValueError("Barycentric coordinates are negative (batched).")

    return lambdaA, lambdaB, lambdaC


def _geodesic_polyline_length_from_stacked_paths(
    starts: torch.Tensor,
    ends: torch.Tensor,
    paths: torch.Tensor,
) -> torch.Tensor:
    """
    Same length recipe as ``compute_geodesic_edge_length_with_grad``, vectorized.

    Args:
        starts: (B, 3), requires grad (typically).
        ends: (B, 3), requires grad (typically).
        paths: (B, n, 3), no grad (fixed polyline from C++).

    Returns:
        (B,) per-edge lengths.
    """
    n = paths.shape[1]
    if n < 2:
        raise ValueError("The path has less than 2 points (batched).")
    if n == 2:
        return torch.norm(starts - ends, dim=-1)
    if n == 3:
        mid = paths[:, 1, :]
        return torch.norm(starts - mid, dim=-1) + torch.norm(mid - ends, dim=-1)
    # n >= 4: interior segments are fixed (from C++); only first/last segments depend on starts/ends.
    with torch.no_grad():
        interior_deltas = paths[:, 2:-1, :] - paths[:, 1:-2, :]
        interior_len = torch.linalg.norm(interior_deltas, dim=-1).sum(dim=-1)
    return (
        torch.norm(starts - paths[:, 1, :], dim=-1)
        + interior_len
        + torch.norm(paths[:, -2, :] - ends, dim=-1)
    )


def map_geodesic_edge_lengths_with_grad(
    edge_to_its_index: dict[tuple[int, int], int],
    geo_paths: list,
    image_of_cutted_domain_vertices: torch.Tensor,
    surface_points_pairs_collections: list | None = None,
    sanity_check: bool = PAR_GEOD_SANITY_CHECK,
) -> dict[tuple[int, int], torch.Tensor]:
    """Map each undirected mesh edge to a scalar geodesic length that backprops through endpoints.

    geo_paths[i] is the polyline from C++ for edge list index i (same
    ordering as construct_surface_points_pairs_for_edges). Paths are
    grouped by vertex count n_pts so each batch can torch.stack without
    ragged tensors.

    Args:
        edge_to_its_index: dict[tuple[int, int], int]
            The edge to its index dictionary.
        geo_paths: list
            The geodesic paths.
        image_of_cutted_domain_vertices: torch.Tensor
            The image of the cutted domain vertices.
        surface_points_pairs_collections: list | None
            Surface point pair per edge index from construct_surface_points_pairs_for_edges.
        sanity_check: bool

    Returns:
        dict[tuple[int, int], torch.Tensor]
            The edge->geodesic_length_with_grad dictionary.
    """
    device = image_of_cutted_domain_vertices.device
    dt = image_of_cutted_domain_vertices.dtype
    out: dict[tuple[int, int], torch.Tensor] = {}

    # Group edges that share the same polyline length n_pts for vectorized length computation.
    by_n: dict[int, list[tuple[tuple[int, int], int]]] = {}
    for edge_tuple, edge_index in edge_to_its_index.items():
        raw = geo_paths[edge_index]
        n_pts = int(np.asarray(raw).shape[0])
        if n_pts == 0:
            if surface_points_pairs_collections is not None:
                surface_point_pair = surface_points_pairs_collections[edge_index]
            else:
                surface_point_pair = "unavailable"
            raise RuntimeError(
                f"Cannot find the geodesic path for edge {edge_tuple}\n"
                f"edge_index: {edge_index}\n"
                f"surface_point_pair: {surface_point_pair}\n"
                f"pts_on_geo_path: empty\n"
            )
        by_n.setdefault(n_pts, []).append((edge_tuple, edge_index))

    for n_pts in sorted(by_n.keys()):
        group = by_n[n_pts]  # all edges whose geodesic path has n_pts vertices
        edge_tuples = [g[0] for g in group]
        us = torch.tensor([t[0] for t in edge_tuples], device=device, dtype=torch.long)
        vs = torch.tensor([t[1] for t in edge_tuples], device=device, dtype=torch.long)
        starts = image_of_cutted_domain_vertices[us]
        ends = image_of_cutted_domain_vertices[vs]

        paths = torch.stack(
            [
                torch.as_tensor(
                    np.asarray(geo_paths[ei], dtype=np.float64),
                    device=device,
                    dtype=dt,
                )
                for _, ei in group
            ],
            dim=0,
        )
        if sanity_check:
            if not torch.all(torch.isclose(starts, paths[:, 0, :], atol=1e-5)):
                raise ValueError(
                    "Start vertex mismatch path[0] (batched): "
                    f"max err={torch.max(torch.abs(starts - paths[:, 0, :])).item()}"
                )
            if not torch.all(torch.isclose(ends, paths[:, -1, :], atol=1e-5)):
                raise ValueError(
                    "End vertex mismatch path[-1] (batched): "
                    f"max err={torch.max(torch.abs(ends - paths[:, -1, :])).item()}"
                )
            if not starts.requires_grad or not ends.requires_grad:
                raise ValueError(
                    "The start/end vertices must require grad (batched geodesic length)."
                )

        lengths = _geodesic_polyline_length_from_stacked_paths(starts, ends, paths)
        for i, et in enumerate(edge_tuples):
            out[et] = lengths[i]

    return out


def construct_geodesic_PARs_list(
    cutted_domain_vertices: torch.Tensor,
    cutted_domain_faces: torch.Tensor,
    edge_to_geodesic_edge_lengths: dict[tuple[int, int], torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Construct per-face
    - P (area ratio)
    - A (domain area)
    - R (shape quality)
    - D (cotangent-weighted PL Dirichlet density)
    using geodesic edge lengths with grad.

    Domain triangle edges use Euclidean lengths; image triangle edges use the
    scalar geodesic lengths.

    Args:
        cutted_domain_vertices: torch.Tensor, shape=(n, 3), grad is False
            The vertices of the cutted domain mesh.
        cutted_domain_faces: torch.Tensor, shape=(m, 3), grad is False
            The faces of the cutted domain mesh.
        edge_to_geodesic_edge_lengths: dict[tuple[int, int], torch.Tensor]
            The edge->geodesic_length dictionary.

    Returns:
        P: torch.Tensor, shape=(m,), grad is True
            The P values of the domain mesh.
        A: torch.Tensor, shape=(m,), grad is True
            The A values of the domain mesh.
        R: torch.Tensor, shape=(m,), grad is True
            The R values of the domain mesh.
        D: torch.Tensor, shape=(m,), grad is True
            Per-face Dirichlet term.
    """
    device = cutted_domain_vertices.device
    n_vertices = int(cutted_domain_vertices.shape[0])

    # --- Domain (Euclidean) edge lengths (vectorized, no grad) ---
    f_vertices = cutted_domain_vertices[cutted_domain_faces]  # (n_faces, 3, 3)
    edge_diffs_domain = torch.stack(
        [
            f_vertices[:, 1] - f_vertices[:, 0],  # v1 - v0
            f_vertices[:, 2] - f_vertices[:, 1],  # v2 - v1
            f_vertices[:, 0] - f_vertices[:, 2],  # v0 - v2
        ],
        dim=1,
    )  # (n_faces, 3, 3)
    euc_edges_domain = torch.norm(edge_diffs_domain, dim=2)  # (n_faces, 3)
    L1, L2, L3 = (
        euc_edges_domain[:, 0],
        euc_edges_domain[:, 1],
        euc_edges_domain[:, 2],
    )

    # --- Geodesic edge lengths: gather per-face edges in one shot ---
    # Build a sorted key table for edges present in `edge_to_geodesic_edge_lengths`.
    # Key encoding: key = u * n_vertices + v, where u < v.
    edge_items = list(edge_to_geodesic_edge_lengths.items())
    if len(edge_items) == 0:
        raise ValueError("edge_to_geodesic_edge_lengths is empty.")

    edges_uv = torch.tensor(
        [list(k) for (k, _) in edge_items],
        dtype=torch.long,
        device=device,
    )  # (n_edges, 2)
    geo_lens = torch.stack([v for (_, v) in edge_items], dim=0)  # (n_edges,)
    edge_keys = edges_uv[:, 0] * n_vertices + edges_uv[:, 1]  # (n_edges,)
    sort_idx = torch.argsort(edge_keys)
    edge_keys_sorted = edge_keys[sort_idx]
    geo_lens_sorted = geo_lens[sort_idx]

    # Per-face undirected edges (sorted pairs)
    f = cutted_domain_faces.to(dtype=torch.long, device=device)  # (n_faces, 3)
    e01 = torch.sort(f[:, [0, 1]], dim=1).values
    e12 = torch.sort(f[:, [1, 2]], dim=1).values
    e20 = torch.sort(f[:, [2, 0]], dim=1).values

    face_edge_keys = torch.stack(
        [
            e01[:, 0] * n_vertices + e01[:, 1],
            e12[:, 0] * n_vertices + e12[:, 1],
            e20[:, 0] * n_vertices + e20[:, 1],
        ],
        dim=1,
    )  # (n_faces, 3)

    keys_flat = face_edge_keys.reshape(-1)
    pos = torch.searchsorted(edge_keys_sorted, keys_flat)
    if pos.numel() == 0:
        raise ValueError("No face edges to lookup.")
    if torch.any(pos >= edge_keys_sorted.numel()):
        raise KeyError("Some face edges were not found in edge key table.")
    if not torch.all(edge_keys_sorted[pos] == keys_flat):
        raise KeyError("Some face edges were not found in edge key table.")

    geo_edges = geo_lens_sorted[pos].view(-1, 3)  # (n_faces, 3)
    l1, l2, l3 = geo_edges[:, 0], geo_edges[:, 1], geo_edges[:, 2]

    # --- Vectorized PAR: domain angles → cot weights; D = Dirichlet energy; P = sqrt(area ratio) ---
    alpha = torch.acos((L2**2 + L3**2 - L1**2) / (2 * L2 * L3))
    beta = torch.acos((L1**2 + L3**2 - L2**2) / (2 * L1 * L3))
    gamma = torch.acos((L1**2 + L2**2 - L3**2) / (2 * L1 * L2))

    cot_alpha = 1 / torch.tan(alpha)
    cot_beta = 1 / torch.tan(beta)
    cot_gamma = 1 / torch.tan(gamma)

    C = (L1 + L2 + L3) / 2
    A = torch.sqrt(C * (C - L1) * (C - L2) * (C - L3))
    A = torch.max(A, torch.tensor(1e-12, dtype=torch.float64, device=device))

    c = (l1 + l2 + l3) / 2
    D = (1 / (2 * A)) * (cot_alpha * l1**2 + cot_beta * l2**2 + cot_gamma * l3**2)
    P = torch.sqrt(
        (c * (c - l1) * (c - l2) * (c - l3)) / (C * (C - L1) * (C - L2) * (C - L3))
    )

    D_over_P = D / P
    R = torch.ones_like(D_over_P, dtype=torch.float64)
    # R from the quadratic in r + 1/r = D/P: r=1 when D/P=2; real roots when (D/P)²>4.
    close_to_2 = torch.isclose(
        D_over_P, torch.tensor(2.0, dtype=torch.float64, device=device), atol=1e-12
    )
    greater_than_4 = D_over_P**2 > 4
    degenerate = torch.isclose(
        P, torch.tensor(0.0, dtype=torch.float64, device=device), atol=1e-12
    )
    R[close_to_2] = 1.0
    R[greater_than_4] = (
        D_over_P[greater_than_4] + torch.sqrt(D_over_P[greater_than_4] ** 2 - 4)
    ) / 2
    R[degenerate] = 1.0
    unhandled = ~(close_to_2 | greater_than_4 | degenerate)
    if torch.any(unhandled):
        raise ValueError(
            f"{int(torch.sum(unhandled).item())} faces have unhandled D_over_P values"
        )

    return P, A, R, D


def construct_surface_points_pairs_for_edges(
    cutted_domain_faces: torch.Tensor,
    uv_surface_points: list,
) -> tuple[list, dict[tuple[int, int], int]]:
    """
    Turn “locator output per vertex” into “one surface-point pair per unique edge,” in a fixed order.

    uv_surface_points[i] is the locator output for vertex i: (face_idx, bary).
    We convert those to rounded (face_idx, bary list) pairs per vertex, then
    store them as a dict mapping each canonical edge (u, v)
    to its surface point pairs index in the surface_points_pairs_collections list.

    Args:
        cutted_domain_faces: torch.Tensor, shape=(m, 3), grad is False
            The faces of the cutted domain mesh.
        uv_surface_points: list
            The surface points for each vertex.

    Returns:
        surface_points_pairs_collections: list
            The surface point pairs for each edge.
        edge_to_its_index: dict[tuple[int, int], int]
            The edge to its index dictionary.
    """
    unique_edges = _unique_edges_canonical_first_occurrence(cutted_domain_faces)
    n_uv = len(uv_surface_points)
    needed_vertices = (
        np.unique(unique_edges.ravel())
        if unique_edges.size
        else np.empty(0, dtype=np.int64)
    )
    # Only vertices that appear on some unique edge need a surface point
    vertex_surface_cpp: list = [None] * n_uv
    for iv in needed_vertices:
        ii = int(iv)
        f0, b0 = uv_surface_points[ii]
        vertex_surface_cpp[ii] = construct_surface_point_from_cpp_results(f0, b0)

    # Store the surface point pairs for each edge in a list
    e = unique_edges.shape[0]
    surface_points_pairs_collections: list = []
    if e:
        edge_rows = unique_edges.tolist()
        surface_points_pairs_collections = [
            [vertex_surface_cpp[r[0]], vertex_surface_cpp[r[1]]] for r in edge_rows
        ]
    edge_to_its_index: dict[tuple[int, int], int] = {
        (int(unique_edges[i, 0]), int(unique_edges[i, 1])): i for i in range(e)
    }

    return surface_points_pairs_collections, edge_to_its_index


def construct_surface_point_from_cpp_results(
    face_idx: int, bary: np.ndarray | list | tuple
) -> tuple[int, list[float]]:
    """
    Construct a surface point (face index, barycentric coordinates)
        while rounding the barycentric coordinates such that
        they sum to 1 and all values are non-negative.

    Args:
        face_idx: int
            The index of the face.
        bary: np.ndarray | list | tuple
            The barycentric coordinates.

    Returns:
        tuple[int, list[float]]
            The surface point (face index, rounded barycentric coordinates).
    """
    arr = np.asarray(bary, dtype=np.float64).reshape(-1)
    rounded = round_bary_coordinates_numpy(arr)
    return (int(face_idx), rounded.tolist())


def _unique_edges_canonical_first_occurrence(
    cutted_domain_faces: torch.Tensor,
) -> np.ndarray:
    """
    Canonical (min, max) per undirected edge in first-seen order.

    Matches the legacy nested loop over faces and local edge indices 0, 1, 2.
    Returns (E, 2) int64; empty if there are no faces.
    """
    faces_np = cutted_domain_faces.detach().cpu().numpy()
    if faces_np.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    f = faces_np.astype(np.int64, copy=False)
    n = f.shape[0]
    # Get all possible edges from the faces
    u = np.empty(3 * n, dtype=np.int64)
    v = np.empty(3 * n, dtype=np.int64)
    u[0::3] = f[:, 0]
    u[1::3] = f[:, 1]
    u[2::3] = f[:, 2]
    v[0::3] = f[:, 1]
    v[1::3] = f[:, 2]
    v[2::3] = f[:, 0]
    lo = np.minimum(u, v)
    hi = np.maximum(u, v)
    all_edges = np.stack([lo, hi], axis=1)

    # Pack (lo, hi) into one int for O(1) duplicate detection while preserving first-seen order.
    max_v = int(all_edges.max()) + 1
    # assign each edge a unique integer
    packed = all_edges[:, 0] * max_v + all_edges[:, 1]
    seen: set[int] = set()
    order_idx: list[int] = []
    # loop over all edges and give the first seen edge its unique index
    for i in range(int(packed.shape[0])):
        p = int(packed[i])
        if p not in seen:
            seen.add(p)
            order_idx.append(i)
    return all_edges[order_idx]  # one row per unique undirected edge

    # # Same semantics: first-seen unique edges; keys are (lo, hi) instead of packed int.
    # # Compare speed vs packing + set[int] above (tuple hashing vs integer ops).
    # first_seen: dict[tuple[int, int], int] = {}
    # order_idx_tuple: list[int] = []
    # for i in range(int(all_edges.shape[0])):
    #     key = (int(all_edges[i, 0]), int(all_edges[i, 1]))
    #     if key not in first_seen:
    #         first_seen[key] = i
    #         order_idx_tuple.append(i)
    # return all_edges[order_idx_tuple]  # one row per unique undirected edge


def lift_unitized_uvs_to_codomain_image_vertices_batched(
    unitized_uvs: torch.Tensor,
    parallel_res: list,
    codomain_uvs: torch.Tensor,
    codomain_vertices: torch.Tensor,
    codomain_faces: torch.Tensor,
    cutted_codomain_faces: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Lift unitized UV coordinates to 3D positions using batched barycentric weights.

    parallel_res[i] is (face_idx, _) from find_bary_coords_parallel; only
    face_idx is used here: barycentrics are recomputed in Torch so gradients
    flow through unitized_uvs consistently with barycentric_coordinates_torch_batched.

    Args:
        unitized_uvs: (n, 2) in [0, 1)², typically uv_images % 1.
        parallel_res: Locator output, length n.
        codomain_uvs / codomain_vertices / codomain_faces: Fundamental domain mesh.
        cutted_codomain_faces: Face indices aligned with the locator (cut mesh).
        out: Optional preallocated (n, 3) buffer.

    Returns:
        3D positions (n, 3) (same tensor as out when provided).
    """
    n = unitized_uvs.shape[0]
    if len(parallel_res) != n:
        raise ValueError(
            f"parallel_res length {len(parallel_res)} != unitized_uvs rows {n}"
        )
    if out is None:
        out = torch.zeros((n, 3), dtype=torch.float64, device=unitized_uvs.device)
    elif out.shape != (n, 3):
        raise ValueError(f"out shape {out.shape} != ({n}, 3)")

    device = unitized_uvs.device
    face_idx = torch.as_tensor(
        [int(pr[0]) for pr in parallel_res],
        device=device,
        dtype=torch.long,
    )

    # locate the faces of the surface points on the cutted codomain mesh
    corners_on_cut = cutted_codomain_faces[face_idx]
    tri_uv = codomain_uvs[corners_on_cut]
    lam_a, lam_b, lam_c = barycentric_coordinates_torch_batched(tri_uv, unitized_uvs)

    # interpolate the 3D positions of the surface points on the codomain mesh
    corners_on_codomain = codomain_faces[face_idx]
    tri_v3 = codomain_vertices[corners_on_codomain]
    w = torch.stack((lam_a, lam_b, lam_c), dim=1).unsqueeze(-1)  # (n, 3, 1)
    out[:, :] = torch.sum(tri_v3 * w, dim=1)
    return out


def _find_bary_coords(
    codomain_planar_locator: sc.PlanarLocator,
    unitized_uvs: torch.Tensor,
    codomain_uvs: torch.Tensor,
    cutted_codomain_faces: torch.Tensor,
    use_cuda_locator: bool = False,
) -> list:
    if use_cuda_locator:
        if not unitized_uvs.is_cuda:
            raise ValueError("use_cuda_locator=True requires CUDA UV tensors.")
        from .cuda_locator import find_bary_coords

        return find_bary_coords(
            unitized_uvs,
            codomain_uvs,
            cutted_codomain_faces,
        )

    return codomain_planar_locator.find_bary_coords_parallel(
        unitized_uvs.detach().cpu().contiguous().numpy()
    )


def lift_uv_coordinates_to_codomain_3d_batched(
    uv_coordinates: torch.Tensor,
    codomain_planar_locator: sc.PlanarLocator,
    codomain_uvs: torch.Tensor,
    codomain_vertices: torch.Tensor,
    codomain_faces: torch.Tensor,
    cutted_codomain_faces: torch.Tensor,
    use_cuda_locator: bool = False,
) -> torch.Tensor:
    """
    Euclidean codomain lift:
    1. unitize UVs,
    2. find surface points on the codomain mesh,
    3. batched UV lift to codomain image vertices.
    """
    unitized_uv_coordinates = uv_coordinates % 1
    parallel_res = _find_bary_coords(
        codomain_planar_locator,
        unitized_uv_coordinates,
        codomain_uvs,
        cutted_codomain_faces,
        use_cuda_locator=use_cuda_locator,
    )
    return lift_unitized_uvs_to_codomain_image_vertices_batched(
        unitized_uv_coordinates,
        parallel_res,
        codomain_uvs,
        codomain_vertices,
        codomain_faces,
        cutted_codomain_faces,
    )


def compute_geodesic_path_lengths(
    cutted_domain_faces: torch.Tensor,
    uv_images: torch.Tensor,
    codomain_planar_locator: sc.PlanarLocator,
    codomain_uvs: torch.Tensor,
    codomain_vertices: torch.Tensor,
    codomain_faces: torch.Tensor,
    cutted_codomain_faces: torch.Tensor,
    geodesic_type: str,
    record_time: bool = False,
    timing_dict: dict[str, float] | None = None,
    use_cuda_locator: bool = False,
) -> dict[tuple[int, int], torch.Tensor]:
    """
    Compute the geodesic edge lengths with gradient.
    End-to-end:
    1. unitize UVs,
    2. find surface points on the codomain mesh,
    3. geodesic polylines,
    4. 3D lift,
    5. per-edge length tensors (grad).

    Args:
        cutted_domain_faces: torch.Tensor, shape=(n_faces, 3), grad is False
            The cutted domain triangles.
        uv_images: torch.Tensor, shape=(n_verts_cut, 2), grad is True
            The optimized UVs.
        codomain_planar_locator: sc.PlanarLocator
            The locator on the codomain UV layout.
        codomain_uvs: torch.Tensor, shape=(n_vertices_codomain, 2), grad is False
            The UVs of the codomain mesh.
        codomain_vertices: torch.Tensor, shape=(n_vertices_codomain, 3), grad is False
            The vertices of the codomain mesh.
        codomain_faces: torch.Tensor, shape=(n_faces_codomain, 3), grad is False
            The faces of the codomain mesh.
        cutted_codomain_faces: torch.Tensor, shape=(n_faces_cutted_codomain, 3), grad is False
            The faces indexing into UV/vertex arrays for the cut.
        geodesic_type: str
            The type of geodesic to compute.
        record_time: bool
            If True and timing_dict is set, fill stage wall times (seconds).
        timing_dict: dict[str, float] | None
            The dictionary to store the timing information.

    Returns:
        dict[tuple[int, int], torch.Tensor]
        The edge to geodesic edge lengths dictionary.
    """
    # initialize all time variables
    t0_unitize_uv, t1_unitize_uv = 0, 0
    t0_compute_barycentric_coordinates, t1_compute_barycentric_coordinates = 0, 0
    t0_edge_bookkeeping, t1_edge_bookkeeping = 0, 0
    t0_compute_geodesic_paths, t1_compute_geodesic_paths = 0, 0
    t0_uv_lift_to_3d, t1_uv_lift_to_3d = 0, 0
    t0_compute_geo_len_grad, t1_compute_geo_len_grad = 0, 0

    # Move all UVs to [0, 1]
    if record_time:
        t0_unitize_uv = time.perf_counter()
    unitized_uvs = uv_images % 1
    if record_time:
        t1_unitize_uv = time.perf_counter()

    # Find the barycentric coordinates (saved as surface points)
    if record_time:
        t0_compute_barycentric_coordinates = time.perf_counter()
    uv_surface_points = _find_bary_coords(
        codomain_planar_locator,
        unitized_uvs,
        codomain_uvs,
        cutted_codomain_faces,
        use_cuda_locator=use_cuda_locator,
    )
    if record_time:
        t1_compute_barycentric_coordinates = time.perf_counter()

    # Collect edges on the cutted domain and their surface points pairs
    if record_time:
        t0_edge_bookkeeping = time.perf_counter()
    surface_points_pairs_collections, edge_to_its_index = (
        construct_surface_points_pairs_for_edges(
            cutted_domain_faces,
            uv_surface_points,
        )
    )
    if record_time:
        t1_edge_bookkeeping = time.perf_counter()

    edge_to_geodesic_edge_lengths: dict[tuple[int, int], torch.Tensor] = {}

    # Compute the geodesic paths
    if record_time:
        t0_compute_geodesic_paths = time.perf_counter()
    codomain_vertices_np = codomain_vertices.detach().cpu().numpy()
    codomain_faces_np = codomain_faces.detach().cpu().numpy()
    if geodesic_type == "geodesic":
        geo_paths = sc.geodesic_compute_parallel(
            surface_points_pairs_collections,
            codomain_vertices_np,
            codomain_faces_np,
        )
    elif geodesic_type == "flip_geodesic":
        geo_paths = sc.flip_geodesic_compute_parallel(
            surface_points_pairs_collections,
            codomain_vertices_np,
            codomain_faces_np,
        )
    else:
        raise ValueError(f"Unknown geodesic type: {geodesic_type}")
    if record_time:
        t1_compute_geodesic_paths = time.perf_counter()

    if PAR_GEOD_SANITY_CHECK:
        # Print the exact problematic query inputs when geometry-central returns an empty polyline.
        # This is intended for debugging the C++ side; we still raise later (in map_*).
        for edge_tuple, edge_index in edge_to_its_index.items():
            raw = geo_paths[edge_index]
            if int(np.asarray(raw).shape[0]) == 0:
                spp = surface_points_pairs_collections[edge_index]
                print(
                    "EMPTY_GEODESIC_POLYLINE",
                    {
                        "edge": edge_tuple,
                        "edge_index": int(edge_index),
                        "surface_point_pair": spp,
                    },
                    flush=True,
                )

    # Compute the 3D positions of the cutted domain vertices on the codomain mesh
    if record_time:
        t0_uv_lift_to_3d = time.perf_counter()
    image_of_cutted_domain_vertices = torch.zeros(
        (uv_images.shape[0], 3), dtype=torch.float64, device=uv_images.device
    )
    lift_unitized_uvs_to_codomain_image_vertices_batched(
        unitized_uvs,
        uv_surface_points,
        codomain_uvs,
        codomain_vertices,
        codomain_faces,
        cutted_codomain_faces,
        out=image_of_cutted_domain_vertices,
    )
    if record_time:
        t1_uv_lift_to_3d = time.perf_counter()

    # Compute the geodesic edge lengths with gradient
    if record_time:
        t0_compute_geo_len_grad = time.perf_counter()
    edge_to_geodesic_edge_lengths = map_geodesic_edge_lengths_with_grad(
        edge_to_its_index,
        geo_paths,
        image_of_cutted_domain_vertices,
        surface_points_pairs_collections=surface_points_pairs_collections,
    )
    if record_time:
        t1_compute_geo_len_grad = time.perf_counter()

    if record_time and timing_dict is not None:
        timing_dict["unitize_uv"] = t1_unitize_uv - t0_unitize_uv
        timing_dict["barycentric_coordinates_cpp"] = (
            t1_compute_barycentric_coordinates - t0_compute_barycentric_coordinates
        )
        timing_dict["edge_bookkeeping"] = t1_edge_bookkeeping - t0_edge_bookkeeping
        timing_dict["geodesic_paths_cpp"] = (
            t1_compute_geodesic_paths - t0_compute_geodesic_paths
        )
        timing_dict["uv_lift_to_3d"] = t1_uv_lift_to_3d - t0_uv_lift_to_3d
        timing_dict["geodesic_lengths_with_grad"] = (
            t1_compute_geo_len_grad - t0_compute_geo_len_grad
        )
    return edge_to_geodesic_edge_lengths

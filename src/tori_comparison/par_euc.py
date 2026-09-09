"""Euclidean PAR (parameterization quality) metrics on 3D triangle meshes."""

from __future__ import annotations

import torch


def consturct_euclidean_PARs_list_vectorized(
    cutted_domain_vertices: torch.Tensor,
    cutted_domain_faces: torch.Tensor,
    image_of_cutted_domain_vertices: torch.Tensor,
    atol: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Vectorized version of consturct_euclidean_PARs_list for better performance.
    Construct the PARs for the domain mesh using tensor operations.
    Args:
        cutted_domain_vertices: torch.tensor, shape=(n, 3), grad is False
            The vertices of the cutted domain mesh.
        cutted_domain_faces: torch.tensor, shape=(m, 3), grad is False
            The faces of the cutted domain mesh.
        image_of_cutted_domain_vertices: torch.tensor, shape=(n, 3), grad is True
            The image of the cutted domain vertices.

    Returns:
        P: torch.tensor, shape=(m,), grad is True
            The P values of the domain mesh.
        A: torch.tensor, shape=(m,), grad is True
            The A values of the domain mesh.
        R: torch.tensor, shape=(m,), grad is True
            The R values of the domain mesh.
        D: torch.tensor, shape=(m,), grad is True
            Per-face Dirichlet term (cotangent weights on domain, squared image edges).
    """
    # Get all face vertices at once: shape (n_faces, 3, 3)
    f_vertices = cutted_domain_vertices[cutted_domain_faces]  # (n_faces, 3, 3)
    f_vertices_img = image_of_cutted_domain_vertices[
        cutted_domain_faces
    ]  # (n_faces, 3, 3)

    # Compute all edge differences at once
    # Edge 1: v1 - v0, Edge 2: v2 - v1, Edge 3: v0 - v2
    edge_diffs_domain = torch.stack(
        [
            f_vertices[:, 1] - f_vertices[:, 0],  # v1 - v0
            f_vertices[:, 2] - f_vertices[:, 1],  # v2 - v1
            f_vertices[:, 0] - f_vertices[:, 2],  # v0 - v2
        ],
        dim=1,
    )  # shape: (n_faces, 3, 3)

    edge_diffs_image = torch.stack(
        [
            f_vertices_img[:, 1] - f_vertices_img[:, 0],  # v1 - v0
            f_vertices_img[:, 2] - f_vertices_img[:, 1],  # v2 - v1
            f_vertices_img[:, 0] - f_vertices_img[:, 2],  # v0 - v2
        ],
        dim=1,
    )  # shape: (n_faces, 3, 3)

    # Compute all edge lengths at once using vectorized norm
    euc_edges_domain = torch.norm(edge_diffs_domain, dim=2)  # shape: (n_faces, 3)
    euc_edges_image = torch.norm(edge_diffs_image, dim=2)  # shape: (n_faces, 3)

    # Extract individual edge lengths for clarity
    L1, L2, L3 = euc_edges_domain[:, 0], euc_edges_domain[:, 1], euc_edges_domain[:, 2]
    l1, l2, l3 = euc_edges_image[:, 0], euc_edges_image[:, 1], euc_edges_image[:, 2]

    # Compute angles using vectorized cosine rule
    # alpha = acos((L2^2 + L3^2 - L1^2) / (2 * L2 * L3))
    alpha = torch.acos((L2**2 + L3**2 - L1**2) / (2 * L2 * L3))
    beta = torch.acos((L1**2 + L3**2 - L2**2) / (2 * L1 * L3))
    gamma = torch.acos((L1**2 + L2**2 - L3**2) / (2 * L1 * L2))

    # Compute cotangents
    cot_alpha = 1 / torch.tan(alpha)
    cot_beta = 1 / torch.tan(beta)
    cot_gamma = 1 / torch.tan(gamma)

    # Compute area using vectorized Heron's formula
    C = (L1 + L2 + L3) / 2  # Semi-perimeter
    A = torch.sqrt(C * (C - L1) * (C - L2) * (C - L3))
    A = torch.max(A, torch.tensor(1e-12))

    # Compute semi-perimeter of image triangles
    c = (l1 + l2 + l3) / 2

    # Compute D vectorized
    D = (1 / (2 * A)) * (cot_alpha * l1**2 + cot_beta * l2**2 + cot_gamma * l3**2)

    # Compute P vectorized
    P = torch.sqrt(
        (c * (c - l1) * (c - l2) * (c - l3)) / (C * (C - L1) * (C - L2) * (C - L3))
    )

    D_over_P = D / P

    # Handle special cases vectorized
    R = torch.ones_like(D_over_P, dtype=torch.float64)

    # Case 1: D_over_P close to 2.0
    close_to_2 = torch.isclose(
        D_over_P, torch.tensor(2.0, dtype=torch.float64), atol=atol
    )
    R[close_to_2] = 1.0

    # Case 2: D_over_P^2 > 4
    greater_than_4 = D_over_P**2 > 4
    R[greater_than_4] = (
        D_over_P[greater_than_4] + torch.sqrt(D_over_P[greater_than_4] ** 2 - 4)
    ) / 2

    # Case 3: P close to 0 (degenerate case)
    degenerate = torch.isclose(P, torch.tensor(0.0, dtype=torch.float64), atol=atol)
    R[degenerate] = 1.0

    # Check for any unhandled cases
    unhandled = ~(close_to_2 | greater_than_4 | degenerate)
    if torch.any(unhandled):
        print(f"Warning: {torch.sum(unhandled)} faces have unhandled D_over_P values")
        # For unhandled cases, set R to 1.0 as fallback
        R[unhandled] = 1.0

    return P, A, R, D

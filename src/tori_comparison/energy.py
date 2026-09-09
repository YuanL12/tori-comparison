"""Area / shape-stretch energy and UV flip penalties for parameterization optimization."""

from __future__ import annotations

import torch


def compute_energy_torch(
    P: torch.Tensor,
    A: torch.Tensor,
    R: torch.Tensor,
    *,
    D: torch.Tensor | None = None,
    D_over_P: torch.Tensor | None = None,
    smooth_max_type: str = "none",
    pnorm_p: float = 20.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the given expression:
    sqrt(sum((1 - sqrt(P))^2 * A)) + (1/2) * max(|log(R)|)

    Parameters:
    - P: 1D torch.Tensor of P values for each domain triangle.
    - A: 1D torch.Tensor of area values for each domain triangle.
    - R: 1D torch.Tensor of R values for each domain triangle.
    - D: optional per-face Dirichlet term; used with ``smooth_max_type == "Q_to_2"`` as D/P.
    - D_over_P: optional explicit D/P per face (overrides D-derived ratio when both given).
    - smooth_max_type: "none" (exact max), "logsumexp", or "pnorm", "Q_to_2"
    - pnorm_p: exponent for p-norm smooth max (only used when smooth_max_type == "pnorm")

    Returns:
    - two torch.tensor, the first is the area loss, the second is the shape stretch loss
    """
    term1 = (1 - torch.sqrt(P)) ** 2 * A
    area_loss = torch.sqrt(torch.sum(term1))
    abs_log_R = torch.abs(torch.log(R))
    return area_loss, shape_stretch_loss_torch(
        abs_log_R,
        D_over_P,
        D,
        P,
        smooth_max_type=smooth_max_type,
        pnorm_p=pnorm_p,
    )


def shape_stretch_loss_torch(
    abs_log_R: torch.Tensor,
    D_over_P: torch.Tensor | None = None,
    D: torch.Tensor | None = None,
    P: torch.Tensor | None = None,
    *,
    smooth_max_type: str = "none",
    pnorm_p: float = 20.0,
) -> torch.Tensor:
    if smooth_max_type == "logsumexp":
        shape_stretch_loss = 0.5 * torch.logsumexp(abs_log_R, dim=0)
    elif smooth_max_type == "pnorm":
        shape_stretch_loss = 0.5 * torch.norm(abs_log_R, p=pnorm_p)
    elif smooth_max_type == "Q_to_2":  # D/P -> 2 is equivalent R -> 1 but more stable
        if D_over_P is not None:
            q = D_over_P
        elif D is not None:
            q = D / torch.clamp(P, min=torch.finfo(P.dtype).eps * 1e3)
        else:
            raise ValueError("D_over_P or D is required for Q_to_2 smooth max")
        shape_stretch_loss = 0.5 * torch.norm(q - 2, p=pnorm_p)
    else:
        shape_stretch_loss = 0.5 * torch.max(abs_log_R)
    return shape_stretch_loss


def compute_log_area_shape_loss_torch(
    P: torch.Tensor,
    A: torch.Tensor,
    R: torch.Tensor,
    *,
    D: torch.Tensor | None = None,
    D_over_P: torch.Tensor | None = None,
    smooth_max_type: str = "none",
    pnorm_p: float = 20.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the given expression:
    sum(log(P)^2) + (1/2) * max(|log(R)|)
    The first term is desgined to avoid degenerate triangles with P = 0.

    Parameters:
    - P: 1D torch.Tensor of P values for each domain triangle.
    - A: 1D torch.Tensor of area values for each domain triangle.
    - R: 1D torch.Tensor of R values for each domain triangle.
    - D: optional per-face Dirichlet term; used with ``smooth_max_type == "Q_to_2"`` as D/P.
    - D_over_P: optional explicit D/P per face (overrides D-derived ratio when both given).
    - smooth_max_type: "none" (exact max), "logsumexp", or "pnorm", "Q_to_2"
    - pnorm_p: exponent for p-norm smooth max (only used when smooth_max_type == "pnorm")

    Returns:
    - two torch.tensor, the first is the area loss, the second is the shape stretch loss
    """
    area_loss = torch.sum(torch.log(P) ** 2)
    abs_log_R = torch.abs(torch.log(R))
    return area_loss, shape_stretch_loss_torch(
        abs_log_R,
        D_over_P,
        D,
        P,
        smooth_max_type=smooth_max_type,
        pnorm_p=pnorm_p,
    )


def compute_flip_loss(
    uv_image: torch.Tensor, domain_faces: torch.Tensor
) -> torch.Tensor:
    # Vectorized version (faster than Python loop; keeps autograd).
    f_uvs = uv_image[domain_faces]  # (n_faces, 3, 2)

    edge1 = f_uvs[:, 1] - f_uvs[:, 0]  # (n_faces, 2)
    edge2 = f_uvs[:, 2] - f_uvs[:, 0]  # (n_faces, 2)
    cross_products = edge1[:, 0] * edge2[:, 1] - edge1[:, 1] * edge2[:, 0]  # (n_faces,)

    mask = cross_products < -1e-10
    if not mask.any():
        return uv_image.new_zeros(())

    # Big penalty for inverted or tiny area (avoid vanishing gradient near 0).
    flipped = cross_products[mask]
    return 1e12 * (flipped.abs() + 1e-10).pow(2).sum()


def compute_flip_faces_count(uv_image: torch.Tensor, domain_faces: torch.Tensor) -> int:
    # Get all face UVs at once
    f_uvs = uv_image[domain_faces]  # Shape: (n_faces, 3, 2)

    # Compute edge differences for all faces at once
    edge1 = f_uvs[:, 1] - f_uvs[:, 0]  # Shape: (n_faces, 2)
    edge2 = f_uvs[:, 2] - f_uvs[:, 0]  # Shape: (n_faces, 2)

    # Compute cross products for all faces at once
    cross_products = (
        edge1[:, 0] * edge2[:, 1] - edge1[:, 1] * edge2[:, 0]
    )  # Shape: (n_faces,)

    # Count flipped faces with a small tolerance
    return (cross_products < -1e-10).sum().item()

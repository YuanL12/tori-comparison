"""Geometry objectives for UV optimization (area+stretch vs Dirichlet)."""

from __future__ import annotations

from typing import Any, Literal

import torch

from .energy import compute_energy_torch, compute_log_area_shape_loss_torch

EnergyObjectiveName = Literal["area_shape", "dirichlet", "log_area_shape"]


def geometry_objective_tensors(
    energy_objective: EnergyObjectiveName,
    Ps: torch.Tensor,
    As: torch.Tensor,
    Rs: torch.Tensor,
    Ds: torch.Tensor,
    *,
    smooth_max_type: str,
    smooth_max_pnorm_p: float,
    area_loss_weight: float = 1.0,
    shape_stretch_weight: float = 1.0,
) -> dict[str, Any]:
    """
    Compute the geometric loss (no flip penalty).
    Any future customized loss should define
    1. "loss_geom" used for optimization
    2. "energy_rec" used for record
    The above two can be the same or different.

    ``energy_rec`` matches the quantity used for plateau when flips are absent:
    area+shape record sum for ``area_shape``; discrete Dirichlet energy
    ``sum_i D_i * A_i`` for ``dirichlet`` (``A_i`` domain triangle area).
    """
    area_loss_rec, ss_loss_rec = compute_energy_torch(
        Ps, As, Rs, D=Ds, smooth_max_type="none"
    )

    if energy_objective == "dirichlet":
        # Discrete ∫ D dA over the domain triangulation: D piecewise constant per face.
        loss_geom = (Ds * As).sum()
        energy_rec = loss_geom
        return {
            "energy_objective": energy_objective,
            "loss_geom": loss_geom,
            "energy_rec": energy_rec,
            "area_loss_rec": area_loss_rec,
            "ss_loss_rec": ss_loss_rec,
        }

    if energy_objective == "area_shape":
        area_loss, shape_stretch_loss = compute_energy_torch(
            Ps,
            As,
            Rs,
            D=Ds,
            smooth_max_type=smooth_max_type,
            pnorm_p=smooth_max_pnorm_p,
        )
        weighted_area_loss = float(area_loss_weight) * area_loss
        weighted_shape_loss = float(shape_stretch_weight) * shape_stretch_loss
        loss_geom = weighted_area_loss + weighted_shape_loss
        energy_rec = area_loss_rec + ss_loss_rec
        return {
            "energy_objective": energy_objective,
            "loss_geom": loss_geom,
            "energy_rec": energy_rec,
            "area_loss_rec": area_loss_rec,
            "ss_loss_rec": ss_loss_rec,
            "area_loss_weight": float(area_loss_weight),
            "shape_stretch_weight": float(shape_stretch_weight),
        }

    if energy_objective == "log_area_shape":
        area_loss, shape_stretch_loss = compute_log_area_shape_loss_torch(
            Ps,
            As,
            Rs,
            D=Ds,
            smooth_max_type=smooth_max_type,
            pnorm_p=smooth_max_pnorm_p,
        )
        weighted_area_loss = float(area_loss_weight) * area_loss
        weighted_shape_loss = float(shape_stretch_weight) * shape_stretch_loss
        loss_geom = weighted_area_loss + weighted_shape_loss
        energy_rec = area_loss_rec + ss_loss_rec  # always record the original energy
        return {
            "energy_objective": energy_objective,
            "loss_geom": loss_geom,
            "energy_rec": energy_rec,
            "area_loss_rec": area_loss_rec,
            "ss_loss_rec": ss_loss_rec,
            "area_loss_weight": float(area_loss_weight),
            "shape_stretch_weight": float(shape_stretch_weight),
        }

    raise ValueError(
        f"Unknown energy_objective {energy_objective!r}; "
        "expected 'area_shape', 'dirichlet', or 'log_area_shape'."
    )

"""
Stop Reason:
1. Too many flipped faces: rolling-window flipped-face count exceeds a cap
   (flips ⇒ map is not locally injective / no longer a diffeomorphism onto its image).
   The returned dict from ``run_optimization_resumable`` uses ``stop_reason == "flips"``.
2. Energy plateau
3. Keyboard interrupt (ctrl+c)
4. Completed (ran all requested steps without early stop)
5. Uncaught exception during a step — checkpoint saved, then return with stop_reason="error"
"""

import sys
from typing import Any, Callable, Dict, List, Optional
import numpy as np
import torch

from .energy import compute_flip_faces_count, compute_flip_loss
from .objectives import EnergyObjectiveName, geometry_objective_tensors
from .iteration_progress import IterationProgressReporter
from .logging import Logger
from .optim import FlipRejectBackoffController
from .par_euc import consturct_euclidean_PARs_list_vectorized
from .par_geod import (
    compute_geodesic_path_lengths,
    construct_geodesic_PARs_list,
    lift_uv_coordinates_to_codomain_3d_batched,
)
from .stop_conditions import LossPlateauDetector, TotalFlipStopDetector


def _sync_resumable_checkpoint_counters(
    checkpoint_state: Dict[str, Any],
    *,
    global_step: int,
    plateau_detector: LossPlateauDetector,
    stop_reason: Optional[str],
) -> None:
    checkpoint_state.update(
        {
            "global_step": global_step,
            "plateau_best_loss": plateau_detector.best_loss,
            "plateau_steps_without_improvement": plateau_detector.steps_without_improvement,
            "stop_reason": stop_reason,
        }
    )


def per_face_area_and_shape_stretch(
    Ps: torch.Tensor, As: torch.Tensor, Rs: torch.Tensor
) -> tuple[np.ndarray, np.ndarray]:
    """Per-face area change and |log R| (same construction as former snapshot lists)."""
    shape_stretch_per_face = torch.abs(torch.log(Rs))
    area_change_per_face = (1 - torch.sqrt(Ps)) ** 2 * As
    return (
        area_change_per_face.detach().cpu().numpy(),
        shape_stretch_per_face.detach().cpu().numpy(),
    )


def snapshot_uv_iteration(
    uv_pos_res: List[np.ndarray], img_f_uvs: torch.Tensor
) -> None:
    """Append current UVs for optional trajectory export (save_uv_trajectories)."""
    uv_pos_res.append(img_f_uvs.detach().clone().numpy())


def _postprocess_step(
    *,
    iter_data: Dict[str, Any],
    img_f_uvs: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    logger: Logger,
    plateau_detector: LossPlateauDetector,
    flip_detector: TotalFlipStopDetector,
    checkpoint_state: Dict[str, Any],
    global_step: int,
    best_energy: float,
    step_norm: float,
    save_uv_trajectories: bool,
    uv_pos_res: List[np.ndarray],
) -> tuple[int, Optional[str], str, np.ndarray, np.ndarray]:
    """
    Post-step bookkeeping + logging + stop checks.

    Returns:
    - updated global_step
    - stop_message (None if not stopping)
    - stop_reason (\"completed\" if not stopping; otherwise \"flips\" or \"plateau\")
    - area_change_per_face (np)
    - shape_stretch_per_face (np)
    """
    energy_for_record = iter_data["energy_rec"]
    energy_val = energy_for_record.item()
    flip_count = iter_data["flip_count"]
    total_loss = iter_data["total_loss"]

    area_np, shape_np = per_face_area_and_shape_stretch(
        iter_data["Ps"], iter_data["As"], iter_data["Rs"]
    )
    checkpoint_state["area_change_per_face"] = area_np
    checkpoint_state["shape_stretch_per_face"] = shape_np
    if iter_data.get("energy_objective") == "dirichlet":
        checkpoint_state["dirichlet_per_face"] = iter_data["Ds"].detach().cpu().numpy()

    if save_uv_trajectories:
        snapshot_uv_iteration(uv_pos_res, img_f_uvs)

    # Flip-based early stop (highest priority)
    if flip_detector.update(flip_count):
        stop_message = (
            f"flip total stop at step {global_step} "
            f"(rolling flipped faces over last {flip_detector.window_size} steps = "
            f"{flip_detector.recent_flip_count()}, "
            f"max allowed = {flip_detector.max_total_flips}, "
            f"best energy = {best_energy:.6f})"
        )
        global_step += 1
        _sync_resumable_checkpoint_counters(
            checkpoint_state,
            global_step=global_step,
            plateau_detector=plateau_detector,
            stop_reason=stop_message,
        )
        return global_step, stop_message, "flips", area_np, shape_np

    # Loss plateau stop check (second priority)
    if plateau_detector.update(total_loss.item(), flip_count):
        stop_message = (
            f"loss plateau at step {global_step} (best energy = {best_energy:.6f})"
        )
        global_step += 1
        _sync_resumable_checkpoint_counters(
            checkpoint_state,
            global_step=global_step,
            plateau_detector=plateau_detector,
            stop_reason=stop_message,
        )
        return global_step, stop_message, "plateau", area_np, shape_np

    # log iteration data
    log_row: Dict[str, Any] = {
        "area_ratio_loss": iter_data["area_loss_rec"].item(),
        "shape_stretch_loss": iter_data["ss_loss_rec"].item(),
        "flip_count": flip_count,
        "total_loss": total_loss.item(),
        "energy": energy_val,
        "flip_loss": iter_data["flip_loss"].item(),  # used for flip penalty
        "learning_rate": optimizer.param_groups[0]["lr"],
        "effective_step_norm": step_norm,
    }
    logger.log(log_row, step=global_step)
    global_step += 1
    _sync_resumable_checkpoint_counters(
        checkpoint_state,
        global_step=global_step,
        plateau_detector=plateau_detector,
        stop_reason=None,
    )
    return global_step, None, "completed", area_np, shape_np


def _forward_backward_iter(
    edge_length_type: str,
    img_f_uvs: torch.Tensor,
    planar2: Any,
    codomain_uvs: torch.Tensor,
    codomain_vertices: torch.Tensor,
    codomain_faces: torch.Tensor,
    cutted_codomain_faces: torch.Tensor,
    cutted_domain_vertices: torch.Tensor,
    cutted_domain_faces: torch.Tensor,
    smooth_max_type: str,
    smooth_max_pnorm_p: float,
    iter_data: Dict[str, Any],
    energy_objective: EnergyObjectiveName = "area_shape",
    area_loss_weight: float = 1.0,
    shape_stretch_weight: float = 1.0,
) -> torch.Tensor:
    """Forward + backward for one optimizer eval; fills ``iter_data`` (no ``optimizer.zero_grad``)."""
    if edge_length_type in ["geodesic", "flip_geodesic"]:
        edge_to_geodesic_edge_lengths = compute_geodesic_path_lengths(
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
            edge_to_geodesic_edge_lengths=edge_to_geodesic_edge_lengths,
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
        raise ValueError(f"Unknown edge length type: {edge_length_type}")

    # compute the objective function values for optimization
    objective_tensors = geometry_objective_tensors(
        energy_objective,
        Ps,
        As,
        Rs,
        Ds,
        smooth_max_type=smooth_max_type,
        smooth_max_pnorm_p=smooth_max_pnorm_p,
        area_loss_weight=area_loss_weight,
        shape_stretch_weight=shape_stretch_weight,
    )

    # compute flip count and flip loss
    flip_count = compute_flip_faces_count(img_f_uvs, cutted_domain_faces)
    flip_loss = compute_flip_loss(img_f_uvs, cutted_domain_faces)

    # the total loss for optimization
    total_loss = objective_tensors["loss_geom"] + flip_loss

    # compute the gradient on uv coordinates
    total_loss.backward()

    iter_data.update(
        {
            "Ps": Ps,
            "As": As,
            "Rs": Rs,
            "Ds": Ds,
            "area_loss_rec": objective_tensors["area_loss_rec"],
            "ss_loss_rec": objective_tensors["ss_loss_rec"],
            "energy_rec": objective_tensors["energy_rec"],
            "energy_objective": objective_tensors["energy_objective"],
            "area_loss_weight": objective_tensors.get("area_loss_weight"),
            "shape_stretch_weight": objective_tensors.get("shape_stretch_weight"),
            "flip_count": flip_count,
            "flip_loss": flip_loss,
            "total_loss": total_loss,
        }
    )
    return total_loss


def run_optimization_resumable(
    n_iter: int,
    edge_length_type: str,
    optimizer: torch.optim.Optimizer,
    img_f_uvs: torch.Tensor,
    planar2: Any,
    codomain_uvs: torch.Tensor,
    codomain_vertices: torch.Tensor,
    codomain_faces: torch.Tensor,
    cutted_codomain_faces: torch.Tensor,
    cutted_domain_vertices: torch.Tensor,
    cutted_domain_faces: torch.Tensor,
    logger: Logger,
    save_uv_trajectories: bool,
    energy_objective: EnergyObjectiveName = "area_shape",
    area_loss_weight: float = 1.0,
    shape_stretch_weight: float = 1.0,
    smooth_max_type: str = "none",
    smooth_max_pnorm_p: float = 20.0,
    energy_plateau_patience: int = 500,
    energy_plateau_rel_tol: float = 1e-5,
    flip_stop_max_total_flips: int = 100,
    reset_plateau_steps_without_improvement: bool = False,
    iters_to_save_checkpoint: int = 1000,
    checkpoint_saver: Optional[Callable[[Dict[str, Any]], None]] = None,
    resume_state: Optional[Dict[str, Any]] = None,
    checkpoint_state: Optional[Dict[str, Any]] = None,
    lr_scheduler: Optional[Any] = None,
    flip_reject_backoff_enabled: bool = False,
    flip_reject_backoff_factor: float = 0.5,
    flip_reject_backoff_min_lr: float = 1e-10,
) -> Dict[str, Any]:
    """
    Resumable variant of `run_optimization()` that periodically saves an
    optimization-only checkpoint.

    - `n_iter` means: run up to this many optimization steps in this call.
    - `resume_state` provides the global counters + plateau state to resume from.
    - `checkpoint_state` (mutable dict) is updated in-place after each completed step.
    - If `checkpoint_saver` is provided, it is called:
      - periodically every `iters_to_save_checkpoint` steps
      - once at the end (normal exit or plateau)
      - on ctrl+c (KeyboardInterrupt)
      - on any other exception after a failed step (then the exception is re-raised)
    """

    uv_pos_res: List[np.ndarray] = []

    # resume status if available
    resume_state = resume_state or {}
    checkpoint_state = checkpoint_state if checkpoint_state is not None else {}
    global_step = int(resume_state.get("global_step", 0))
    plateau_best_loss = resume_state.get("plateau_best_loss", float("inf"))
    plateau_steps_without_improvement = int(
        resume_state.get("plateau_steps_without_improvement", 0)
    )
    best_energy = float("inf")
    last_energy_val: Optional[float] = None
    # reset the plateau steps without improvement
    if reset_plateau_steps_without_improvement:
        plateau_steps_without_improvement = 0

    # Initialize checkpoint state so the runner can save immediately on ctrl+c.
    checkpoint_state.update(
        {
            "global_step": global_step,
            "plateau_best_loss": plateau_best_loss,
            "plateau_steps_without_improvement": plateau_steps_without_improvement,
            "stop_reason": None,
        }
    )

    # Initialize the total loss for optimization
    total_loss = torch.tensor(0.0)
    stop_reason = "completed"
    stop_message: Optional[str] = None
    last_area_change_per_face: Optional[np.ndarray] = None
    last_shape_stretch_per_face: Optional[np.ndarray] = None

    try:
        plateau_detector = LossPlateauDetector(
            patience=energy_plateau_patience, rel_tol=energy_plateau_rel_tol
        )
        plateau_detector.best_loss = plateau_best_loss
        plateau_detector.steps_without_improvement = plateau_steps_without_improvement

        flip_detector = TotalFlipStopDetector(max_total_flips=flip_stop_max_total_flips)
        flip_reject_controller: FlipRejectBackoffController | None = None
        if bool(flip_reject_backoff_enabled):
            flip_reject_controller = FlipRejectBackoffController(
                factor=flip_reject_backoff_factor,
                min_lr=flip_reject_backoff_min_lr,
            )
            flip_reject_controller.initialize_accepted(img_f_uvs)

        with IterationProgressReporter(
            int(n_iter),
            desc="Optimization",
            global_step_start=global_step,
        ) as progress:
            next_save_global_step: Optional[int] = None
            if (
                checkpoint_saver is not None
                and iters_to_save_checkpoint is not None
                and int(iters_to_save_checkpoint) > 0
            ):
                next_save_global_step = global_step + int(iters_to_save_checkpoint)

            for _ in range(int(n_iter)):
                iter_data: Dict[str, Any] = {}

                saved_params = img_f_uvs.detach().clone()
                optimizer.zero_grad()
                total_loss = _forward_backward_iter(
                    edge_length_type,
                    img_f_uvs,
                    planar2,
                    codomain_uvs,
                    codomain_vertices,
                    codomain_faces,
                    cutted_codomain_faces,
                    cutted_domain_vertices,
                    cutted_domain_faces,
                    smooth_max_type,
                    smooth_max_pnorm_p,
                    iter_data,
                    energy_objective=energy_objective,
                    area_loss_weight=area_loss_weight,
                    shape_stretch_weight=shape_stretch_weight,
                )
                optimizer.step()

                with torch.no_grad():
                    last_energy_val = float(iter_data["energy_rec"].item())
                    best_energy = min(best_energy, last_energy_val)
                    delta = img_f_uvs - saved_params
                    step_norm = delta.abs().max().item()
                    post_flip_count = int(
                        compute_flip_faces_count(img_f_uvs, cutted_domain_faces)
                    )
                    iter_data["flip_count"] = post_flip_count
                    iter_data["flip_loss"] = compute_flip_loss(
                        img_f_uvs, cutted_domain_faces
                    )

                if (
                    flip_reject_controller is not None
                    and int(iter_data["flip_count"]) > 0
                ):
                    flip_reject_controller.reject_and_backoff(optimizer, img_f_uvs)
                    global_step += 1
                    _sync_resumable_checkpoint_counters(
                        checkpoint_state,
                        global_step=global_step,
                        plateau_detector=plateau_detector,
                        stop_reason=None,
                    )
                    progress.step()

                    if (
                        next_save_global_step is not None
                        and global_step >= next_save_global_step
                        and checkpoint_saver is not None
                    ):
                        checkpoint_saver(dict(checkpoint_state))
                        next_save_global_step = next_save_global_step + int(
                            iters_to_save_checkpoint
                        )
                    continue

                if flip_reject_controller is not None:
                    flip_reject_controller.accept(img_f_uvs)

                # drop learning rate if loss is plateaued
                if lr_scheduler is not None:
                    lr_scheduler.step(iter_data["total_loss"].item())

                # post-step bookkeeping + logging + stop checks
                (
                    global_step,
                    maybe_stop_message,
                    maybe_stop_reason,
                    last_area_change_per_face,
                    last_shape_stretch_per_face,
                ) = _postprocess_step(
                    iter_data=iter_data,
                    img_f_uvs=img_f_uvs,
                    optimizer=optimizer,
                    logger=logger,
                    plateau_detector=plateau_detector,
                    flip_detector=flip_detector,
                    checkpoint_state=checkpoint_state,
                    global_step=global_step,
                    best_energy=best_energy,
                    step_norm=step_norm,
                    save_uv_trajectories=save_uv_trajectories,
                    uv_pos_res=uv_pos_res,
                )
                if maybe_stop_message is not None:
                    stop_message = maybe_stop_message
                    stop_reason = maybe_stop_reason
                    progress.step()
                    break

                # update the progress bar
                progress.step()

                # save checkpoint if needed
                if (
                    next_save_global_step is not None
                    and global_step >= next_save_global_step
                    and checkpoint_saver is not None
                ):
                    checkpoint_saver(dict(checkpoint_state))
                    next_save_global_step = next_save_global_step + int(
                        iters_to_save_checkpoint
                    )

    except KeyboardInterrupt:  # ctrl+c
        stop_reason = "interrupted"
        checkpoint_state["stop_reason"] = "interrupted"
        if last_area_change_per_face is not None:
            checkpoint_state["area_change_per_face"] = last_area_change_per_face
        if last_shape_stretch_per_face is not None:
            checkpoint_state["shape_stretch_per_face"] = last_shape_stretch_per_face
        if checkpoint_saver is not None:
            checkpoint_saver(dict(checkpoint_state))

        if stop_message:
            print(f"Optimization stopped: {stop_message}")
        if save_uv_trajectories:
            logger.details_log({"uv_pos_res": uv_pos_res})

        return {
            "final_energy": best_energy,
            "final_loss": total_loss.detach(),
            "stop_reason": stop_reason,
            "checkpoint_state": dict(checkpoint_state),
        }

    except Exception as e:
        # Last successful step is reflected in global_step / per-face metrics; the failing
        # step did not complete (no increment). Save so debug can load state near the crash.
        msg = str(e)
        if len(msg) > 800:
            msg = msg[:800] + "..."
        checkpoint_state["stop_reason"] = f"error:{type(e).__name__}: {msg}"
        if last_area_change_per_face is not None:
            checkpoint_state["area_change_per_face"] = last_area_change_per_face
        if last_shape_stretch_per_face is not None:
            checkpoint_state["shape_stretch_per_face"] = last_shape_stretch_per_face
        if checkpoint_saver is not None:
            checkpoint_saver(dict(checkpoint_state))
            print(
                "Saved checkpoint after training failure "
                f"(global_step={checkpoint_state.get('global_step')}, "
                f"best_energy={best_energy}).",
                file=sys.stderr,
            )
        # Match the ctrl+c behavior: return a structured result so the caller can continue.
        stop_reason = "error"
        if save_uv_trajectories:
            logger.details_log({"uv_pos_res": uv_pos_res})
        return {
            "final_energy": best_energy,
            "final_loss": total_loss.detach(),
            "stop_reason": stop_reason,
            "checkpoint_state": dict(checkpoint_state),
            "error_type": type(e).__name__,
            "error_msg": msg,
        }

    if stop_message:
        print(f"Optimization stopped: {stop_message}")

    # Ensure we always leave the caller with an on-disk checkpoint at the
    # end of this optimization call.
    if checkpoint_state.get("stop_reason") is None:
        checkpoint_state["stop_reason"] = stop_reason
    # Per-face metrics are already updated in `checkpoint_state` as we go,
    # but keep this as a safety net in case the loop exits before updating.
    if last_area_change_per_face is not None:
        checkpoint_state["area_change_per_face"] = last_area_change_per_face
    if last_shape_stretch_per_face is not None:
        checkpoint_state["shape_stretch_per_face"] = last_shape_stretch_per_face
    if checkpoint_saver is not None:
        checkpoint_saver(dict(checkpoint_state))

    if save_uv_trajectories:
        logger.details_log({"uv_pos_res": uv_pos_res})

    return {
        "final_energy": best_energy,
        "final_loss": total_loss.detach(),
        "stop_reason": stop_reason,
        "checkpoint_state": dict(checkpoint_state),
    }

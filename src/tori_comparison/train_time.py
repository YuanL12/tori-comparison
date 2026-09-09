"""
Resumable optimization with wall-clock timing (stdout). Use from ``run_time.py``.

Only ``run_optimization_resumable`` is provided; core training logic is shared with
``core.train`` via imports of helpers and detectors.
"""

from __future__ import annotations

import sys
from typing import Any, Callable, Dict, List, Optional
import time

import numpy as np
import torch

from .energy import compute_flip_faces_count, compute_flip_loss
from .objectives import EnergyObjectiveName, geometry_objective_tensors
from .iteration_progress import IterationProgressReporter
from .logging import Logger
from .par_euc import consturct_euclidean_PARs_list_vectorized
from .par_geod import (
    compute_geodesic_path_lengths,
    construct_geodesic_PARs_list,
    lift_uv_coordinates_to_codomain_3d_batched,
)

from .stop_conditions import EnergyPlateauDetector, TotalFlipStopDetector
from .timing_report import TIMING_REPORT_INTERVAL, report_timing_window
from .train import (
    _sync_resumable_checkpoint_counters,
    per_face_area_and_shape_stretch,
    snapshot_uv_iteration,
)
from .optim import FlipRejectBackoffController


def _run_optimization_step(
    optimizer: torch.optim.Optimizer,
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
    energy_objective: EnergyObjectiveName,
    area_loss_weight: float,
    shape_stretch_weight: float,
    *,
    print_geodesic_time: bool,
) -> tuple[torch.Tensor, Dict[str, Any]]:
    """Forward + backward for one optimizer step; fills ``timing`` sub-dict (seconds)."""
    timing: Dict[str, float] = {}
    optimizer.zero_grad()

    t0 = time.perf_counter()
    if edge_length_type in ["geodesic", "flip_geodesic"]:
        geo_timing: Dict[str, float] = {}
        edge_to_geodesic_edge_lengths = compute_geodesic_path_lengths(
            cutted_domain_faces=cutted_domain_faces,
            uv_images=img_f_uvs,
            codomain_planar_locator=planar2,
            codomain_uvs=codomain_uvs,
            codomain_vertices=codomain_vertices,
            codomain_faces=codomain_faces,
            cutted_codomain_faces=cutted_codomain_faces,
            geodesic_type=edge_length_type,
            record_time=print_geodesic_time,
            timing_dict=geo_timing if print_geodesic_time else None,
        )

        t1 = time.perf_counter()
        if print_geodesic_time:
            timing.update(geo_timing)
        Ps, As, Rs, Ds = construct_geodesic_PARs_list(
            cutted_domain_vertices=cutted_domain_vertices,
            cutted_domain_faces=cutted_domain_faces,
            edge_to_geodesic_edge_lengths=edge_to_geodesic_edge_lengths,
        )
        t2 = time.perf_counter()
        timing["compute_geodesic_path_lengths"] = t1 - t0
        timing["construct_PARs"] = t2 - t1
    elif edge_length_type == "euclidean":
        image_of_cutted_domain_vertices = lift_uv_coordinates_to_codomain_3d_batched(
            uv_coordinates=img_f_uvs,
            codomain_planar_locator=planar2,
            codomain_uvs=codomain_uvs,
            codomain_vertices=codomain_vertices,
            codomain_faces=codomain_faces,
            cutted_codomain_faces=cutted_codomain_faces,
        )
        t1 = time.perf_counter()
        Ps, As, Rs, Ds = consturct_euclidean_PARs_list_vectorized(
            cutted_domain_vertices=cutted_domain_vertices,
            cutted_domain_faces=cutted_domain_faces,
            image_of_cutted_domain_vertices=image_of_cutted_domain_vertices,
        )
        t2 = time.perf_counter()
        timing["geom_primary_s"] = t1 - t0
        timing["par_construct_s"] = t2 - t1
    else:
        raise ValueError(f"Unknown edge length type: {edge_length_type}")

    t3 = time.perf_counter()
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
    flip_count = compute_flip_faces_count(img_f_uvs, cutted_domain_faces)
    flip_loss = compute_flip_loss(img_f_uvs, cutted_domain_faces)
    total_loss = objective_tensors["loss_geom"] + flip_loss
    total_loss.backward()
    t4 = time.perf_counter()
    timing["loss_backward_s"] = t4 - t3

    iter_payload: Dict[str, Any] = {
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
        "timing": timing,
    }
    return total_loss, iter_payload


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
    *,
    print_geodesic_time: bool = True,
) -> Dict[str, Any]:
    """
    Same contract as ``core.train.run_optimization_resumable``, plus periodic ``[timing]`` lines.
    """
    uv_pos_res: List[np.ndarray] = []
    resume_state = resume_state or {}
    checkpoint_state = checkpoint_state if checkpoint_state is not None else {}

    # resume status
    global_step = int(resume_state.get("global_step", 0))
    plateau_best_energy = resume_state.get("plateau_best_energy", float("inf"))
    plateau_steps_without_improvement = int(
        resume_state.get("plateau_steps_without_improvement", 0)
    )
    if reset_plateau_steps_without_improvement:
        plateau_steps_without_improvement = 0

    checkpoint_state.update(
        {
            "global_step": global_step,
            "plateau_best_energy": plateau_best_energy,
            "plateau_steps_without_improvement": plateau_steps_without_improvement,
            "stop_reason": None,
        }
    )

    energy_for_record = torch.tensor(0.0)
    total_loss = torch.tensor(0.0)
    stop_reason = "completed"
    stop_message: Optional[str] = None
    last_area_change_per_face: Optional[np.ndarray] = None
    last_shape_stretch_per_face: Optional[np.ndarray] = None

    try:
        plateau_detector = EnergyPlateauDetector(
            patience=energy_plateau_patience, rel_tol=energy_plateau_rel_tol
        )
        plateau_detector.best_energy = plateau_best_energy
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
            interval=100,
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

            timing_window_sums: Dict[str, float] = {}
            timing_window_n = 0

            def _flush_timing_window_resumable(step_label: int) -> None:
                nonlocal timing_window_sums, timing_window_n
                report_timing_window(
                    timing_window_sums, timing_window_n, "Optimization", step_label
                )
                timing_window_sums = {}
                timing_window_n = 0

            def _append_timing_row_resumable(step_label: int) -> None:
                nonlocal timing_window_sums, timing_window_n
                t_post1 = time.perf_counter()
                iter_data["timing"]["post_iter_s"] = t_post1 - t_post0
                for k, v in iter_data["timing"].items():
                    timing_window_sums[k] = timing_window_sums.get(k, 0.0) + v
                timing_window_n += 1
                if timing_window_n >= TIMING_REPORT_INTERVAL:
                    _flush_timing_window_resumable(step_label)

            for _ in range(int(n_iter)):
                iter_data: Dict[str, Any] = {}

                saved_params = img_f_uvs.detach().clone()
                t_opt0 = time.perf_counter()
                total_loss, step_payload = _run_optimization_step(
                    optimizer,
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
                    energy_objective,
                    area_loss_weight,
                    shape_stretch_weight,
                    print_geodesic_time=print_geodesic_time,
                )
                iter_data.update(step_payload)
                optimizer.step()
                t_opt1 = time.perf_counter()
                iter_data["timing"]["optimizer_step_s"] = t_opt1 - t_opt0

                t_post0 = time.perf_counter()

                with torch.no_grad():
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
                    _append_timing_row_resumable(global_step)
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

                energy_for_record = iter_data["energy_rec"]
                energy_val = energy_for_record.item()
                flip_count = iter_data["flip_count"]
                step_total_loss = iter_data["total_loss"]

                area_np, shape_np = per_face_area_and_shape_stretch(
                    iter_data["Ps"], iter_data["As"], iter_data["Rs"]
                )
                last_area_change_per_face = area_np
                last_shape_stretch_per_face = shape_np
                checkpoint_state["area_change_per_face"] = area_np
                checkpoint_state["shape_stretch_per_face"] = shape_np
                if iter_data.get("energy_objective") == "dirichlet":
                    checkpoint_state["dirichlet_per_face"] = (
                        iter_data["Ds"].detach().cpu().numpy()
                    )

                if save_uv_trajectories:
                    snapshot_uv_iteration(uv_pos_res, img_f_uvs)

                if lr_scheduler is not None:
                    lr_scheduler.step(energy_val)

                if flip_detector.update(flip_count):
                    stop_message = (
                        f"flip total stop at step {global_step} "
                        f"(rolling flipped faces over last {flip_detector.window_size} steps = "
                        f"{flip_detector.recent_flip_count()}, "
                        f"max allowed = {flip_detector.max_total_flips}, "
                        f"best energy = {plateau_detector.best_energy:.6f})"
                    )
                    stop_reason = "flips"
                    global_step += 1
                    _sync_resumable_checkpoint_counters(
                        checkpoint_state,
                        global_step=global_step,
                        plateau_detector=plateau_detector,
                        stop_reason=stop_message,
                    )
                    progress.step()
                    _append_timing_row_resumable(global_step)
                    break

                if plateau_detector.update(energy_val, flip_count):
                    stop_message = (
                        f"energy plateau at step {global_step} "
                        f"(best energy = {plateau_detector.best_energy:.6f})"
                    )
                    stop_reason = "plateau"
                    global_step += 1
                    _sync_resumable_checkpoint_counters(
                        checkpoint_state,
                        global_step=global_step,
                        plateau_detector=plateau_detector,
                        stop_reason=stop_message,
                    )
                    progress.step()
                    _append_timing_row_resumable(global_step)
                    break

                # Log the metrics
                log_row: Dict[str, Any] = {
                    "area_ratio_loss": iter_data["area_loss_rec"].item(),
                    "shape_stretch_loss": iter_data["ss_loss_rec"].item(),
                    "flip_count": flip_count,
                    "total_loss": step_total_loss.item(),
                    "energy": energy_val,
                    "flip_loss": iter_data["flip_loss"].item(),
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "effective_step_norm": step_norm,
                    "energy_objective": iter_data.get("energy_objective", "area_shape"),
                }
                logger.log(log_row, step=global_step)
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
                _append_timing_row_resumable(global_step)

            if timing_window_n > 0:
                _flush_timing_window_resumable(global_step)

    except KeyboardInterrupt:
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
            "final_energy": plateau_detector.best_energy,
            "final_loss": total_loss.detach(),
            "stop_reason": stop_reason,
            "checkpoint_state": dict(checkpoint_state),
        }

    except Exception as e:
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
                f"(global_step={checkpoint_state.get('global_step')}).",
                file=sys.stderr,
            )
        raise

    if stop_message:
        print(f"Optimization stopped: {stop_message}")

    if checkpoint_state.get("stop_reason") is None:
        checkpoint_state["stop_reason"] = stop_reason
    if last_area_change_per_face is not None:
        checkpoint_state["area_change_per_face"] = last_area_change_per_face
    if last_shape_stretch_per_face is not None:
        checkpoint_state["shape_stretch_per_face"] = last_shape_stretch_per_face
    if checkpoint_saver is not None:
        checkpoint_saver(dict(checkpoint_state))

    if save_uv_trajectories:
        logger.details_log({"uv_pos_res": uv_pos_res})

    return {
        "final_energy": plateau_detector.best_energy,
        "final_loss": total_loss.detach(),
        "stop_reason": stop_reason,
        "checkpoint_state": dict(checkpoint_state),
    }

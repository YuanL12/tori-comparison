"""
Timing-oriented resumable runner: same pipeline as ``run.py`` but uses
``core.train_time.run_optimization_resumable`` (stdout ``[timing]``) and ``NoOpLogger``
(no Weights & Biases unless enabled with ``--use_wandb``).

Usage (from ``experiments/``):
  python run_time.py --config ...

Optional:
  --use_wandb      Enable Weights & Biases logging.
"""

from __future__ import annotations

import os

import numpy as np
import torch

from tori_comparison.checkpoint import (
    load_checkpoint,
    make_checkpoint_paths,
    save_optimization_checkpoint,
)
from tori_comparison.config import ExperimentConfig
from tori_comparison.data import (
    get_uvs_and_shared,
    load_meshes_and_construct_planar_locators,
    numpy_to_torch_tensors,
)
from tori_comparison.logging import NoOpLogger, WandbLogger
from tori_comparison.optim import (
    GradientHook,
    create_optimizer,
    create_reduce_lr_on_plateau_scheduler,
    ReduceLROnPlateauEMA,
)
from tori_comparison.save import save_final_details, save_mesh_context
from tori_comparison.setup import configure_environment
from tori_comparison.train_time import run_optimization_resumable

PROJECT_NAME = "tori_comparison"


def _sample_unit_direction(seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    vec = rng.normal(size=3)
    nrm = float(np.linalg.norm(vec))
    if nrm <= 1e-12:
        return [1.0, 0.0, 0.0]
    vec = vec / nrm
    return vec.tolist()


if __name__ == "__main__":
    cfg = ExperimentConfig.from_command_line()

    run_name: str = cfg.run_name
    group_name: str = cfg.group_name
    use_wandb: bool = bool(cfg.use_wandb)

    normalize_area_to_one: bool = cfg.normalize_area_to_one
    lr: float = cfg.lr
    area_loss_weight = cfg.area_loss_weight
    shape_stretch_weight = cfg.shape_stretch_weight
    obj_path1: str = cfg.obj_path1
    obj_path2: str = cfg.obj_path2
    Tutte_embedding_type: str = cfg.Tutte_embedding_type
    edge_length_type_str = cfg.edge_length_type
    smooth_max_type = cfg.smooth_max_type
    cut_on_shortest_generator = cfg.cut_on_shortest_generator
    codomain_swap_generators = cfg.codomain_swap_generators

    if edge_length_type_str not in {"geodesic", "euclidean", "flip_geodesic"}:
        raise ValueError(f"Unknown edge length type: {edge_length_type_str}")

    iters_to_save_checkpoint = int(cfg.iters_to_save_checkpoint)

    device, save_folder_path = configure_environment(cfg)

    run_ckpt_dir, ckpt_path = make_checkpoint_paths(cfg)
    os.makedirs(run_ckpt_dir, exist_ok=True)

    logger = WandbLogger() if use_wandb else NoOpLogger()
    tags_for_wandb: list[str] = []
    if use_wandb:
        tags_for_wandb = [
            edge_length_type_str,
            cfg.optimizer,
            Tutte_embedding_type,
            f"smooth_max_{smooth_max_type}",
        ]
        if cfg.flip_reject_backoff_enabled:
            tags_for_wandb.append("flip_reject_backoff")
    logger.start(
        project=PROJECT_NAME if use_wandb else "",
        name=run_name,
        config=cfg.to_wandb_config(),
        tags=tags_for_wandb,
        group=group_name or None,
    )

    mesh_context_path = os.path.join(run_ckpt_dir, "mesh_context.pt")
    have_ckpt_files = os.path.exists(ckpt_path) and os.path.exists(mesh_context_path)
    do_resume = bool(cfg.resume_from_checkpoint) and have_ckpt_files

    if cfg.resume_from_checkpoint and not have_ckpt_files:
        raise RuntimeError(
            "resume_from_checkpoint=true but checkpoint files are missing. "
            "\n\tExpected both checkpoint.pt and mesh_context.pt to exist."
            "\n\tPlease set resume_from_checkpoint=false to start a new optimization."
        )

    ckpt = None
    if do_resume:
        ckpt = load_checkpoint(ckpt_path, map_location=str(device))

    mesh_ctx = None
    if do_resume:
        mesh_ctx = load_checkpoint(mesh_context_path, map_location=str(device))
    elif os.path.exists(mesh_context_path):
        print(
            "resume_from_checkpoint=false: ignoring existing mesh_context.pt; "
            "rebuilding from current config (file will be overwritten)."
        )

    arrays: dict
    domain_uvs_np: np.ndarray
    codomain_uvs_np: np.ndarray
    shared_inds: list[list[int]]
    distinct_direction: list[float] | None

    planar2: any

    if mesh_ctx is None:
        distinct_direction = None
        if cut_on_shortest_generator:
            distinct_direction = _sample_unit_direction(cfg.seed)

        planar1, planar2, arrays = load_meshes_and_construct_planar_locators(
            obj_path1=obj_path1,
            obj_path2=obj_path2,
            normalize_area_to_one=normalize_area_to_one,
            Tutte_embedding_type=Tutte_embedding_type,
            cut_on_shortest_generator=cut_on_shortest_generator,
            distinct_direction=distinct_direction,
            codomain_swap_generators=codomain_swap_generators,
        )
        domain_uvs_np, codomain_uvs_np, shared_inds = get_uvs_and_shared(
            planar1, planar2
        )

        save_mesh_context(
            mesh_context_path,
            arrays=arrays,
            domain_uvs=domain_uvs_np,
            codomain_uvs=codomain_uvs_np,
            shared_inds=shared_inds,
            distinct_direction=distinct_direction,
            extra={"codomain_swap_generators": codomain_swap_generators},
        )
        print(f"Saved mesh context to: {mesh_context_path}")
        print(f"Starting fresh optimization: {run_name}")
    else:
        arrays = mesh_ctx["arrays"]
        domain_uvs_np = mesh_ctx["domain_uvs"]
        codomain_uvs_np = mesh_ctx["codomain_uvs"]
        shared_inds = mesh_ctx["shared_inds"]
        distinct_direction = mesh_ctx.get("distinct_direction")

        if cut_on_shortest_generator and distinct_direction is None:
            raise RuntimeError(
                "mesh_context.pt is missing distinct_direction even though "
                "`cut_on_shortest_generator=true`. Can't safely resume."
            )

        if "codomain_swap_generators" in mesh_ctx:
            codomain_swap_generators = bool(mesh_ctx["codomain_swap_generators"])
        else:
            codomain_swap_generators = False

        planar1, planar2, arrays_rebuilt = load_meshes_and_construct_planar_locators(
            obj_path1=obj_path1,
            obj_path2=obj_path2,
            normalize_area_to_one=normalize_area_to_one,
            Tutte_embedding_type=Tutte_embedding_type,
            cut_on_shortest_generator=cut_on_shortest_generator,
            distinct_direction=distinct_direction,
            codomain_swap_generators=codomain_swap_generators,
        )
        domain_uvs_rebuilt, codomain_uvs_rebuilt, _shared_inds_rebuilt = (
            get_uvs_and_shared(planar1, planar2)
        )

        checks = [
            np.array_equal(
                arrays_rebuilt["cutted_mesh_faces_1"], arrays["cutted_mesh_faces_1"]
            ),
            np.array_equal(
                arrays_rebuilt["cutted_mesh_faces_2"], arrays["cutted_mesh_faces_2"]
            ),
            np.allclose(domain_uvs_rebuilt, domain_uvs_np),
            np.allclose(codomain_uvs_rebuilt, codomain_uvs_np),
        ]
        if not all(checks):
            raise RuntimeError(
                "Mesh context mismatch: rebuilt planar cuts/UVs do not match the saved "
                "mesh_context.pt. This usually means a different distinctDirection "
                "or input geometry."
            )

        print(f"Loaded mesh context from: {mesh_context_path}")

    torch_tensors = numpy_to_torch_tensors(arrays)
    codomain_faces = torch_tensors["codomain_faces"]
    codomain_vertices = torch_tensors["codomain_vertices"]
    cutted_domain_vertices = torch_tensors["cutted_domain_vertices"]
    cutted_domain_faces = torch_tensors["cutted_domain_faces"]
    cutted_codomain_faces = torch_tensors["cutted_codomain_faces"]

    codomain_uvs = torch.tensor(
        codomain_uvs_np, dtype=torch.float64, requires_grad=False
    )

    if ckpt is not None:
        img_f_uvs = torch.tensor(
            ckpt["img_f_uvs"], dtype=torch.float64, requires_grad=True
        )
        optimizer = create_optimizer(
            cfg.optimizer,
            [img_f_uvs],
            lr=lr,
        )
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        resume_state = ckpt.get("resume_state", {})
        print(f"Resumed optimization checkpoint: {ckpt_path}")
    else:
        img_f_uvs = torch.tensor(domain_uvs_np, dtype=torch.float64, requires_grad=True)
        optimizer = create_optimizer(
            cfg.optimizer,
            [img_f_uvs],
            lr=lr,
        )
        resume_state = {}
        print(f"Initialized optimization from identity UVs: {run_name}")

    lr_scheduler: object | None = None
    if cfg.reduce_lr_on_plateau:
        if cfg.lr_scheduler == "ReduceLROnPlateau":
            lr_scheduler = create_reduce_lr_on_plateau_scheduler(
                optimizer,
                factor=cfg.reduce_lr_factor,
                patience=cfg.reduce_lr_patience,
                min_lr=cfg.reduce_lr_min_lr,
                tol=cfg.reduce_lr_tol,
                tol_mode=cfg.reduce_lr_tol_mode,
            )
        elif cfg.lr_scheduler == "ReduceLROnPlateauEMA":
            lr_scheduler = ReduceLROnPlateauEMA(
                optimizer,
                ema_weight=cfg.reduce_lr_ema,
                factor=cfg.reduce_lr_factor,
                patience=cfg.reduce_lr_patience,
                min_lr=cfg.reduce_lr_min_lr,
                tol=cfg.reduce_lr_tol,
                tol_mode=cfg.reduce_lr_tol_mode,
            )
        else:
            raise RuntimeError(f"Unknown lr_scheduler: {cfg.lr_scheduler!r}")
        if (
            ckpt is not None
            and cfg.resume_lr_scheduler_state
            and "lr_scheduler_state_dict" in ckpt
        ):
            lr_scheduler.load_state_dict(ckpt["lr_scheduler_state_dict"])

    img_f_uvs.register_hook(GradientHook(shared_inds))

    save_uv_trajectories = bool(cfg.save_uv_trajectories)
    effective_lr = float(optimizer.param_groups[0]["lr"])
    cfg_snapshot = {
        "run_name": run_name,
        "edge_length_type": edge_length_type_str,
        "optimizer": cfg.optimizer,
        "learning_rate": effective_lr,
        "area_loss_weight": area_loss_weight,
        "shape_stretch_weight": shape_stretch_weight,
        "iters_to_save_checkpoint": iters_to_save_checkpoint,
    }

    def checkpoint_saver(resume_state_to_save: dict) -> None:
        save_optimization_checkpoint(
            ckpt_path,
            img_f_uvs=img_f_uvs,
            optimizer=optimizer,
            resume_state=resume_state_to_save,
            wandb_run_id=None,
            extra={"cfg_snapshot": cfg_snapshot},
            lr_scheduler=lr_scheduler,
        )

    try:
        result = run_optimization_resumable(
            n_iter=int(cfg.n_iter),
            edge_length_type=edge_length_type_str,
            optimizer=optimizer,
            img_f_uvs=img_f_uvs,
            planar2=planar2,
            codomain_uvs=codomain_uvs,
            codomain_vertices=codomain_vertices,
            codomain_faces=codomain_faces,
            cutted_codomain_faces=cutted_codomain_faces,
            cutted_domain_vertices=cutted_domain_vertices,
            cutted_domain_faces=cutted_domain_faces,
            logger=logger,
            save_uv_trajectories=save_uv_trajectories,
            energy_objective=cfg.energy_objective,
            area_loss_weight=area_loss_weight,
            shape_stretch_weight=shape_stretch_weight,
            smooth_max_type=smooth_max_type,
            smooth_max_pnorm_p=cfg.smooth_max_pnorm_p,
            energy_plateau_patience=cfg.energy_plateau_patience,
            energy_plateau_rel_tol=cfg.energy_plateau_rel_tol,
            flip_stop_max_total_flips=cfg.flip_stop_max_total_flips,
            reset_plateau_steps_without_improvement=cfg.reset_plateau_steps_without_improvement,
            iters_to_save_checkpoint=iters_to_save_checkpoint,
            checkpoint_saver=checkpoint_saver,
            resume_state=resume_state,
            lr_scheduler=lr_scheduler,
            flip_reject_backoff_enabled=cfg.flip_reject_backoff_enabled,
            flip_reject_backoff_factor=cfg.flip_reject_backoff_factor,
            flip_reject_backoff_min_lr=cfg.flip_reject_backoff_min_lr,
            print_geodesic_time=True,
        )
        print(
            f"Optimization finished with stop_reason={result['stop_reason']}, Best energy={result['final_energy']}"
        )
        if result.get("stop_reason") == "error":
            et = result.get("error_type", "?")
            em = result.get("error_msg", "")
            print(f"Optimization error: {et}: {em}")
    finally:
        logger.finish()

    if cfg.save_uv_trajectories:
        print(f"Saving UV trajectory to folder: {save_folder_path}")
        save_final_details(
            save_folder_path, saved_details=logger._details, run_name=run_name
        )

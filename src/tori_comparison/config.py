from dataclasses import dataclass, field
from typing import Literal, Optional
import argparse
import json
import math
import yaml

OptimizerName = Literal["SGD", "Adam"]
EdgeLengthType = Literal["geodesic", "euclidean"]
SmoothMaxType = Literal["none", "logsumexp", "pnorm"]
LRSchedulerName = Literal["ReduceLROnPlateau", "ReduceLROnPlateauEMA"]
EnergyObjectiveType = Literal["area_shape", "dirichlet", "log_area_shape"]


def load_config_from_json(config_path: str) -> dict:
    """Load configuration from JSON file."""
    with open(config_path) as f:
        return json.load(f)


def load_config_from_yaml(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_config(config_path: str) -> dict:
    """Load configuration from JSON or YAML file based on file extension."""
    if config_path.lower().endswith(".json"):
        return load_config_from_json(config_path)
    elif config_path.lower().endswith((".yaml", ".yml")):
        return load_config_from_yaml(config_path)
    else:
        raise ValueError(
            f"Unsupported config file format: {config_path}. Use .json, .yaml, or .yml"
        )


@dataclass
class ExperimentConfig:
    """Unified configuration class that handles both argument parsing and configuration management."""

    # Required parameters
    run_name: str
    obj_path1: str
    obj_path2: str

    # Optional parameters with defaults
    group_name: str = ""
    use_wandb: bool = False
    normalize_area_to_one: bool = False
    n_iter: int = 1000
    lr: float = 1e-6
    optimizer: OptimizerName = "Adam"
    area_loss_weight: float = 1.0
    shape_stretch_weight: float = 1.0
    Tutte_embedding_type: str = "Uniform"  # Uniform, coTan, MeanValue, Authalic
    Tutte_embedding_type_minimal: bool = (
        True  # if true on a fresh run, pick Tutte type with smallest initial energy
    )
    edge_length_type: EdgeLengthType = "euclidean"  # euclidean, geodesic, flip_geodesic
    energy_objective: EnergyObjectiveType = "area_shape"
    smooth_max_type: str = "none"  # none, logsumexp, pnorm
    smooth_max_pnorm_p: float = 20.0
    save_uv_trajectories: bool = False
    save_folder_path: str = "./"
    device: str = "cpu"
    seed: int = 42
    deterministic: bool = True
    cut_on_shortest_generator: bool = False
    codomain_swap_generators: bool = False
    energy_plateau_patience: int = 500
    energy_plateau_rel_tol: float = 1e-5
    flip_stop_max_total_flips: int = 100
    reset_plateau_steps_without_improvement: bool = False

    # ReduceLROnPlateau (per-step `energy`; separate patience from energy-plateau early stop)
    reduce_lr_on_plateau: bool = False
    lr_scheduler: LRSchedulerName = "ReduceLROnPlateau"
    reduce_lr_ema: float = 0.44
    reduce_lr_factor: float = 0.5
    reduce_lr_patience: int = 2000
    reduce_lr_min_lr: float = 1e-7
    reduce_lr_tol: float = 1e-4
    reduce_lr_tol_mode: Literal["rel", "abs"] = "rel"
    flip_reject_backoff_enabled: bool = False
    flip_reject_backoff_factor: float = 0.5
    flip_reject_backoff_min_lr: float = 1e-10

    # Resumable optimization / checkpointing (used by experiments/run_resumable.py)
    iters_to_save_checkpoint: int = 1000
    checkpoint_dir: str = "./checkpoints"
    checkpoint_filename: str = "checkpoint.pt"
    resume_from_checkpoint: bool = True
    resume_lr_scheduler_state: bool = True

    # Internal field for config file path
    _config_path: Optional[str] = field(default=None, init=False)

    @classmethod
    def from_command_line(cls) -> "ExperimentConfig":
        """Parse command line arguments and create configuration instance."""
        parser = argparse.ArgumentParser(
            description="Energy Geodesic Experiment Configuration"
        )

        # Config file argument
        parser.add_argument(
            "--config", type=str, help="Path to JSON or YAML configuration file"
        )

        # All configuration parameters
        parser.add_argument(
            "--n_iter", type=int, default=1000, help="Number of iterations"
        )
        parser.add_argument(
            "--lr",
            type=float,
            default=1e-6,
            help="Learning rate (e.g. 1e-6 to 1e-4 for Adam).",
        )
        parser.add_argument(
            "--optimizer", type=str, default="Adam", help="Optimizer: Adam or SGD"
        )
        parser.add_argument(
            "--area_loss_weight",
            type=float,
            default=1.0,
            help="Weight for area-loss term in area_shape objective.",
        )
        parser.add_argument(
            "--shape_stretch_weight",
            type=float,
            default=1.0,
            help="Shape stretch weight",
        )
        parser.add_argument("--obj_path1", type=str, help="domain mesh path")
        parser.add_argument("--obj_path2", type=str, help="codomain mesh path")
        parser.add_argument(
            "--save_folder_path", type=str, default="./", help="Save folder path"
        )
        parser.add_argument(
            "--save_uv_trajectories",
            type=bool,
            default=False,
            help="Save subsampled UV trajectory to npy (uv_pos_res.npy)",
        )
        parser.add_argument(
            "--smooth_max_type",
            type=str,
            default="none",
            help="Smooth max type: none (exact max), logsumexp, pnorm",
        )
        parser.add_argument(
            "--smooth_max_pnorm_p",
            type=float,
            default=20.0,
            help="Exponent p for p-norm smooth max approximation",
        )
        parser.add_argument(
            "--edge_length_type",
            type=str,
            default="euclidean",
            help="Edge length type: geodesic or euclidean",
        )
        parser.add_argument(
            "--energy_objective",
            type=str,
            default="area_shape",
            help=(
                "Geometry objective: area_shape (default), "
                "dirichlet (sum of per-face D), or log_area_shape."
            ),
        )
        parser.add_argument("--run_name", type=str, default="", help="Run name")
        parser.add_argument("--group_name", type=str, default="", help="Group name")
        parser.add_argument(
            "--use_wandb",
            action="store_true",
            help="Enable Weights & Biases logging.",
        )
        parser.add_argument(
            "--normalize_area_to_one",
            type=bool,
            default=False,
            help="Normalize area to one",
        )
        parser.add_argument("--device", type=str, default="cpu", help="Device to use")
        parser.add_argument("--seed", type=int, default=42, help="Random seed")
        parser.add_argument(
            "--deterministic",
            type=bool,
            default=True,
            help="Use deterministic behavior",
        )
        parser.add_argument(
            "--Tutte_embedding_type",
            type=str,
            default="Uniform",
            help="Tutte embedding type: Uniform, coTan, MeanValue, Authalic",
        )
        parser.add_argument(
            "--Tutte_embedding_type_minimal",
            type=bool,
            default=True,
            help="If true on a fresh run, select the Tutte type with smallest initial "
            "area+shape energy; if false, use Tutte_embedding_type.",
        )
        parser.add_argument(
            "--cut_on_shortest_generator",
            type=bool,
            default=False,
            help="Cut on shortest generator",
        )
        parser.add_argument(
            "--codomain_swap_generators",
            type=bool,
            default=False,
            help="Swap codomain meridian/longitude when building PlanarLocator.",
        )
        parser.add_argument(
            "--energy_plateau_patience",
            type=int,
            default=500,
            help="Steps without energy improvement before declaring plateau",
        )
        parser.add_argument(
            "--energy_plateau_rel_tol",
            type=float,
            default=1e-5,
            help="Relative improvement threshold for energy plateau detection",
        )
        parser.add_argument(
            "--reset_plateau_steps_without_improvement",
            type=bool,
            default=False,
            help="On resume, reset plateau steps counter to 0 so a new run won't immediately re-stop on the saved plateau state.",
        )
        parser.add_argument(
            "--flip_stop_max_total_flips",
            type=int,
            default=100,
            help="Stop when cumulative flipped-face count exceeds this value (strictly >). "
            "Must be positive.",
        )
        parser.add_argument(
            "--reduce_lr_on_plateau",
            type=bool,
            default=False,
            help="Enable torch.optim.lr_scheduler.ReduceLROnPlateau on reconstructed energy each step.",
        )
        parser.add_argument(
            "--lr_scheduler",
            type=str,
            default="ReduceLROnPlateau",
            help="LR scheduler type: ReduceLROnPlateau or ReduceLROnPlateauEMA.",
        )
        parser.add_argument(
            "--reduce_lr_ema",
            type=float,
            default=0.44,
            help="EMA weight (W&B convention) used only for ReduceLROnPlateauEMA.",
        )
        parser.add_argument(
            "--reduce_lr_factor",
            type=float,
            default=0.5,
            help="Factor to multiply LR when ReduceLROnPlateau triggers.",
        )
        parser.add_argument(
            "--reduce_lr_patience",
            type=int,
            default=2000,
            help="Scheduler patience: steps without sufficient energy improvement before LR reduction.",
        )
        parser.add_argument(
            "--reduce_lr_min_lr",
            type=float,
            default=1e-7,
            help="Minimum learning rate for ReduceLROnPlateau.",
        )
        parser.add_argument(
            "--reduce_lr_tol",
            type=float,
            default=1e-4,
            help="Relative/absolute tolerance for 'improvement' on energy (maps to ReduceLROnPlateau threshold).",
        )
        parser.add_argument(
            "--reduce_lr_tol_mode",
            type=str,
            default="rel",
            help="rel or abs (maps to ReduceLROnPlateau threshold_mode).",
        )
        parser.add_argument(
            "--flip_reject_backoff_enabled",
            type=bool,
            default=False,
            help="Enable strict post-step flip rejection with LR backoff.",
        )
        parser.add_argument(
            "--flip_reject_backoff_factor",
            type=float,
            default=0.5,
            help="LR multiply factor when a flipped post-step is rejected.",
        )
        parser.add_argument(
            "--flip_reject_backoff_min_lr",
            type=float,
            default=1e-10,
            help="Minimum LR floor for flip-reject backoff.",
        )

        # Checkpointing / resumable optimization
        parser.add_argument(
            "--iters_to_save_checkpoint",
            type=int,
            default=1000,
            help="How many optimization steps between checkpoint saves.",
        )
        parser.add_argument(
            "--checkpoint_dir",
            type=str,
            default="./checkpoints",
            help="Directory to save checkpoints (one subdir per run_name).",
        )
        parser.add_argument(
            "--checkpoint_filename",
            type=str,
            default="checkpoint.pt",
            help="Checkpoint filename inside checkpoint_dir/<run_name>/.",
        )
        parser.add_argument(
            "--resume_from_checkpoint",
            type=bool,
            default=True,
            help="If true and files exist, load checkpoint.pt and mesh_context.pt. "
            "If false, ignore both and rebuild mesh context from config (overwrites mesh_context.pt).",
        )
        parser.add_argument(
            "--resume_lr_scheduler_state",
            type=bool,
            default=True,
            help="If resuming and lr_scheduler_state_dict exists, restore the LR scheduler state. "
            "Set false to rebuild the scheduler from the current config (useful when changing reduce_lr_*).",
        )
        args = parser.parse_args()

        # If config file is provided, load it and override command line args
        if args.config:
            print(f"Loading configuration from: {args.config}")
            config = load_config(args.config)
            for key, raw_value in config.items():
                if hasattr(args, key):
                    # Get the expected type from the argument parser
                    action = parser._option_string_actions.get(f"--{key}")
                    if action is not None:
                        # Convert the value to the expected type
                        try:
                            value = raw_value
                            if action.type is int:
                                value = int(raw_value)
                            elif action.type is float:
                                value = float(raw_value)
                            elif action.type is bool:
                                if isinstance(raw_value, str):
                                    value = raw_value.lower() in (
                                        "true",
                                        "1",
                                        "yes",
                                        "on",
                                    )
                                else:
                                    value = bool(raw_value)
                            elif action.type is str:
                                value = str(raw_value)
                            else:
                                # e.g. argparse store_true actions where `action.type` is None
                                value = raw_value
                        except (ValueError, TypeError):
                            print(
                                f"Warning: Could not convert {key}={raw_value} to expected type {action.type}. Using original value."
                            )

                    setattr(args, key, value)
                    print(f"  {key}: {value} (type: {type(value).__name__})")

        # Validate required parameters
        if bool(getattr(args, "use_wandb", False)) and not args.run_name:
            raise RuntimeError(
                "Run name is not specified for wandb! Please specify a run name."
            )
        if not args.obj_path1:
            raise RuntimeError(
                "obj_path1 is required! Please specify the domain mesh path."
            )
        if not args.obj_path2:
            raise RuntimeError(
                "obj_path2 is required! Please specify the codomain mesh path."
            )

        # Create and return configuration instance
        config = cls(
            run_name=args.run_name,
            group_name=args.group_name,
            use_wandb=bool(args.use_wandb),
            normalize_area_to_one=bool(args.normalize_area_to_one),
            n_iter=int(args.n_iter),
            lr=float(args.lr),
            optimizer=args.optimizer,
            area_loss_weight=float(args.area_loss_weight),
            shape_stretch_weight=float(args.shape_stretch_weight),
            obj_path1=args.obj_path1,
            obj_path2=args.obj_path2,
            edge_length_type=args.edge_length_type,
            energy_objective=args.energy_objective,
            Tutte_embedding_type=args.Tutte_embedding_type,
            Tutte_embedding_type_minimal=bool(args.Tutte_embedding_type_minimal),
            smooth_max_type=args.smooth_max_type,
            smooth_max_pnorm_p=float(args.smooth_max_pnorm_p),
            save_uv_trajectories=bool(args.save_uv_trajectories),
            save_folder_path=args.save_folder_path,
            device=args.device,
            seed=int(args.seed),
            deterministic=bool(args.deterministic),
            cut_on_shortest_generator=bool(args.cut_on_shortest_generator),
            codomain_swap_generators=bool(args.codomain_swap_generators),
            energy_plateau_patience=int(args.energy_plateau_patience),
            energy_plateau_rel_tol=float(args.energy_plateau_rel_tol),
            reset_plateau_steps_without_improvement=bool(
                args.reset_plateau_steps_without_improvement
            ),
            flip_stop_max_total_flips=int(args.flip_stop_max_total_flips),
            reduce_lr_on_plateau=bool(args.reduce_lr_on_plateau),
            lr_scheduler=args.lr_scheduler,
            reduce_lr_ema=float(args.reduce_lr_ema),
            reduce_lr_factor=float(args.reduce_lr_factor),
            reduce_lr_patience=int(args.reduce_lr_patience),
            reduce_lr_min_lr=float(args.reduce_lr_min_lr),
            reduce_lr_tol=float(args.reduce_lr_tol),
            reduce_lr_tol_mode=args.reduce_lr_tol_mode,
            flip_reject_backoff_enabled=bool(args.flip_reject_backoff_enabled),
            flip_reject_backoff_factor=float(args.flip_reject_backoff_factor),
            flip_reject_backoff_min_lr=float(args.flip_reject_backoff_min_lr),
            iters_to_save_checkpoint=int(args.iters_to_save_checkpoint),
            checkpoint_dir=str(args.checkpoint_dir),
            checkpoint_filename=str(args.checkpoint_filename),
            resume_from_checkpoint=bool(args.resume_from_checkpoint),
            resume_lr_scheduler_state=bool(args.resume_lr_scheduler_state),
        )

        if config.reduce_lr_tol_mode not in ("rel", "abs"):
            raise RuntimeError(
                "reduce_lr_tol_mode must be 'rel' or 'abs', "
                f"got {config.reduce_lr_tol_mode!r}"
            )
        if config.lr_scheduler not in ("ReduceLROnPlateau", "ReduceLROnPlateauEMA"):
            raise RuntimeError(
                "lr_scheduler must be 'ReduceLROnPlateau' or 'ReduceLROnPlateauEMA', "
                f"got {config.lr_scheduler!r}"
            )
        if not (0.0 <= float(config.reduce_lr_ema) < 1.0):
            raise RuntimeError(
                "reduce_lr_ema must be in [0, 1) (W&B smoothing weight), "
                f"got {config.reduce_lr_ema!r}"
            )
        if config.flip_stop_max_total_flips <= 0:
            raise RuntimeError(
                "flip_stop_max_total_flips must be positive, "
                f"got {config.flip_stop_max_total_flips}"
            )
        if config.energy_objective not in ("area_shape", "dirichlet", "log_area_shape"):
            raise RuntimeError(
                "energy_objective must be 'area_shape', 'dirichlet', or "
                "'log_area_shape', "
                f"got {config.energy_objective!r}"
            )
        for name, value in (
            ("area_loss_weight", config.area_loss_weight),
            ("shape_stretch_weight", config.shape_stretch_weight),
        ):
            if not math.isfinite(float(value)):
                raise RuntimeError(f"{name} must be finite, got {value!r}")
            if float(value) < 0.0:
                raise RuntimeError(f"{name} must be non-negative, got {value!r}")
        if (
            float(config.area_loss_weight) == 0.0
            and float(config.shape_stretch_weight) == 0.0
        ):
            raise RuntimeError(
                "At least one of area_loss_weight or shape_stretch_weight must be > 0."
            )
        if not (0.0 < float(config.flip_reject_backoff_factor) < 1.0):
            raise RuntimeError(
                "flip_reject_backoff_factor must be in (0, 1), "
                f"got {config.flip_reject_backoff_factor!r}"
            )
        if float(config.flip_reject_backoff_min_lr) <= 0.0:
            raise RuntimeError(
                "flip_reject_backoff_min_lr must be > 0, "
                f"got {config.flip_reject_backoff_min_lr!r}"
            )

        # Store config path for reference
        config._config_path = getattr(args, "config", None)

        return config

    def to_wandb_config(self) -> dict:
        """Convert configuration to wandb-compatible format."""
        out = {
            "learning_rate": self.lr,
            "epochs": self.n_iter,
            "smooth_max_type": self.smooth_max_type,
            "smooth_max_pnorm_p": self.smooth_max_pnorm_p,
            "edge_length": self.edge_length_type,
            "energy_objective": self.energy_objective,
            "optimizer": self.optimizer,
            "Tutte_embedding_type_minimal": self.Tutte_embedding_type_minimal,
            "codomain_swap_generators": self.codomain_swap_generators,
            "area_loss_weight": self.area_loss_weight,
            "shape_stretch_weight": self.shape_stretch_weight,
            "reduce_lr_on_plateau": self.reduce_lr_on_plateau,
        }
        if self.reduce_lr_on_plateau:
            out["lr_scheduler"] = self.lr_scheduler
            out["reduce_lr_ema"] = self.reduce_lr_ema
            out["reduce_lr_factor"] = self.reduce_lr_factor
            out["reduce_lr_patience"] = self.reduce_lr_patience
            out["reduce_lr_min_lr"] = self.reduce_lr_min_lr
            out["reduce_lr_tol"] = self.reduce_lr_tol
            out["reduce_lr_tol_mode"] = self.reduce_lr_tol_mode
        out["flip_reject_backoff_enabled"] = self.flip_reject_backoff_enabled
        if self.flip_reject_backoff_enabled:
            out["flip_reject_backoff_factor"] = self.flip_reject_backoff_factor
            out["flip_reject_backoff_min_lr"] = self.flip_reject_backoff_min_lr
        return out

    def __str__(self) -> str:
        """String representation of the configuration."""
        lines = ["ExperimentConfig:"]
        for field_name, field_value in self.__dict__.items():
            if not field_name.startswith("_"):
                lines.append(f"  {field_name}: {field_value}")
        if self._config_path:
            lines.append(f"  config_file: {self._config_path}")
        return "\n".join(lines)


# Backward compatibility functions
def parse_arguments() -> ExperimentConfig:
    """Backward compatibility function that returns ExperimentConfig instead of argparse.Namespace."""
    return ExperimentConfig.from_command_line()

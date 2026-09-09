#!/usr/bin/env bash
# Run run_time.py with settings suited to wall-clock profiling ([timing] lines in stdout).
# Uses a dedicated checkpoint dir and disables resume by default so runs are reproducible.
#
# Usage (from anywhere):
#   ./experiments/run_timing.sh
#   ./experiments/run_timing.sh c2 a2
#   ./experiments/run_timing.sh c2 a2 b2 a2
#
# Recommended: run from repo root or cd experiments first; the script cds to this directory.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p logs

timestamp=$(date +"%Y%m%d_%H%M%S")
main_log="logs/timing_${timestamp}.log"

log_line() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$main_log"
}

# Same mesh keys as run_all_pairs.sh (subset is enough for timing)
declare -A meshes=(
    ["a"]="data/synthetic/torus_ratio_0.2_n_theta_10_n_phi_10.obj"
    ["a2"]="data/synthetic/torus_ratio_0.2_n_theta_20_n_phi_10.obj"
    ["b2"]="data/synthetic/torus_ratio_0.5_n_theta_20_n_phi_10.obj"
    ["c2"]="data/synthetic/torus_ratio_0.8_n_theta_20_n_phi_10.obj"
    ["Pop_f_145302"]="data/real/pelvis_subsampled/Pongo-pygmaeus-f-NMNH-145302-pelvis-L-000095230.obj"
    ["Pop_f_588109"]="data/real/pelvis_subsampled/Pongo-abelii-f-NMNH-588109-pelvis-L-000093958.obj"
)

# Profiling-oriented defaults (override below)
normalize_area_to_one=true
Tutte_embedding_type="Uniform"
group_name="timing"
smooth_max_type="none"
edge_length_type="flip_geodesic"
cut_on_shortest_generator=true
codomain_swap_generators=false
# >= 100 so at least one [timing] report (TIMING_REPORT_INTERVAL in core/train.py)
n_iter=100
optimizer="Adam"
lr=1e-4
save_uv_trajectories=false

energy_plateau_patience=100000
energy_plateau_rel_tol=1e-5
reset_plateau_steps_without_improvement=false

checkpoint_dir="./checkpoints_timing"
checkpoint_filename="checkpoint.pt"
resume_from_checkpoint=false
iters_to_save_checkpoint=100000

reduce_lr_on_plateau=false
reduce_lr_factor=0.5
reduce_lr_patience=2000
reduce_lr_min_lr=1e-7
reduce_lr_tol=1e-4
reduce_lr_tol_mode="rel"

pairs=(
    "a2 b2"
)

use_wandb=false
mesh_args=()
for arg in "$@"; do
    case "$arg" in
        --use_wandb|--wandb|--enable_wandb)
            use_wandb=true
            ;;
        *)
            mesh_args+=("$arg")
            ;;
    esac
done

if [ ${#mesh_args[@]} -gt 0 ]; then
    if [ $(( ${#mesh_args[@]} % 2 )) -ne 0 ]; then
        echo "Error: Provide mesh keys in pairs: mesh1 mesh2 [mesh3 mesh4 ...]" >&2
        exit 1
    fi
    pairs=()
    i=0
    while [ $i -lt ${#mesh_args[@]} ]; do
        pairs+=("${mesh_args[$i]} ${mesh_args[$((i + 1))]}")
        i=$((i + 2))
    done
fi

wandb_args=""
if [ "$use_wandb" = true ]; then
    wandb_args="--use_wandb"
fi

log_line "Starting timing run (${#pairs[@]} pair(s)); main log: $main_log"
log_line "n_iter=${n_iter} (need >=100 for one [timing] window), checkpoint_dir=${checkpoint_dir}, resume=${resume_from_checkpoint}"

for pair in "${pairs[@]}"; do
    read -r mesh1 mesh2 <<<"$pair"

    if [ -z "${meshes[$mesh1]+x}" ] || [ -z "${meshes[$mesh2]+x}" ]; then
        log_line "Unknown mesh key(s): $mesh1 $mesh2 — extend meshes[] in $0"
        exit 1
    fi

    run_name="timing_${mesh1}_vs_${mesh2}"
    exp_log="logs/timing_${timestamp}_${mesh1}_vs_${mesh2}.log"

    temp_config="temp_config_timing_${mesh1}_${mesh2}.yaml"
    cp argus.yaml "$temp_config"

    sed -i "s|run_name:.*|run_name: \"${run_name}\"|" "$temp_config"
    sed -i "s|obj_path1:.*|obj_path1: \"${meshes[$mesh1]}\"|" "$temp_config"
    sed -i "s|obj_path2:.*|obj_path2: \"${meshes[$mesh2]}\"|" "$temp_config"
    sed -i "s|group_name:.*|group_name: \"${group_name}\"|" "$temp_config"
    sed -i "s|n_iter:.*|n_iter: ${n_iter}|" "$temp_config"
    sed -i "s|normalize_area_to_one:.*|normalize_area_to_one: ${normalize_area_to_one}|" "$temp_config"
    sed -i "s|Tutte_embedding_type:.*|Tutte_embedding_type: \"${Tutte_embedding_type}\"|" "$temp_config"
    sed -i "s|edge_length_type:.*|edge_length_type: \"${edge_length_type}\"|" "$temp_config"
    sed -i "s|smooth_max_type:.*|smooth_max_type: \"${smooth_max_type}\"|" "$temp_config"
    sed -i "s|optimizer:.*|optimizer: \"${optimizer}\"|" "$temp_config"
    sed -i "s|lr:.*|lr: ${lr}|" "$temp_config"
    sed -i "s|cut_on_shortest_generator:.*|cut_on_shortest_generator: ${cut_on_shortest_generator}|" "$temp_config"
    sed -i "s|codomain_swap_generators:.*|codomain_swap_generators: ${codomain_swap_generators}|" "$temp_config"
    sed -i "s|save_uv_trajectories:.*|save_uv_trajectories: ${save_uv_trajectories}|" "$temp_config"
    sed -i "s|use_wandb:.*|use_wandb: ${use_wandb}|" "$temp_config"
    sed -i "s|save_folder_path:.*|save_folder_path: \"./\"|" "$temp_config"
    sed -i "s|checkpoint_dir:.*|checkpoint_dir: \"${checkpoint_dir}\"|" "$temp_config"
    sed -i "s|checkpoint_filename:.*|checkpoint_filename: \"${checkpoint_filename}\"|" "$temp_config"
    sed -i "s|resume_from_checkpoint:.*|resume_from_checkpoint: ${resume_from_checkpoint}|" "$temp_config"
    sed -i "s|iters_to_save_checkpoint:.*|iters_to_save_checkpoint: ${iters_to_save_checkpoint}|" "$temp_config"
    sed -i "s|energy_plateau_patience:.*|energy_plateau_patience: ${energy_plateau_patience}|" "$temp_config"
    sed -i "s|energy_plateau_rel_tol:.*|energy_plateau_rel_tol: ${energy_plateau_rel_tol}|" "$temp_config"
    sed -i "s|reset_plateau_steps_without_improvement:.*|reset_plateau_steps_without_improvement: ${reset_plateau_steps_without_improvement}|" "$temp_config"
    sed -i "s|reduce_lr_on_plateau:.*|reduce_lr_on_plateau: ${reduce_lr_on_plateau}|" "$temp_config"
    sed -i "s|reduce_lr_factor:.*|reduce_lr_factor: ${reduce_lr_factor}|" "$temp_config"
    sed -i "s|reduce_lr_patience:.*|reduce_lr_patience: ${reduce_lr_patience}|" "$temp_config"
    sed -i "s|reduce_lr_min_lr:.*|reduce_lr_min_lr: ${reduce_lr_min_lr}|" "$temp_config"
    sed -i "s|reduce_lr_tol:.*|reduce_lr_tol: ${reduce_lr_tol}|" "$temp_config"
    sed -i "s|reduce_lr_tol_mode:.*|reduce_lr_tol_mode: \"${reduce_lr_tol_mode}\"|" "$temp_config"

    log_line "Pair: $mesh1 vs $mesh2 → $exp_log"
    set +e
    python run_time.py --config "$temp_config" $wandb_args 2>&1 | tee -a "$exp_log" | tee -a "$main_log"
    status=$?
    set -e
    rm -f "$temp_config"

    if [ "$status" -eq 0 ]; then
        log_line "Finished pair $mesh1 vs $mesh2 (exit 0)"
    else
        log_line "Failed pair $mesh1 vs $mesh2 (exit $status)"
        exit "$status"
    fi
done

log_line "All timing runs completed."

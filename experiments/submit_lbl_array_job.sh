#!/bin/bash
#SBATCH --job-name=geomp_array
#
# === LBL Cluster Settings ===
#SBATCH --account=pc_mlgeometry
#SBATCH --qos=lr_normal
#SBATCH --time=34:00:00
#
# === Email Notifications ===
#SBATCH --mail-user=yluo3@lbl.gov
#SBATCH --mail-type=ALL
#
# === CPU Resources ===
#SBATCH --partition=lr5
#SBATCH --mincpus=28
#SBATCH --mem=62G
#SBATCH --nodes=1
#SBATCH --kill-on-invalid-dep=yes
#
# === Job Array ===
# Array range is set by ./submit_lbl_array.sh home via: sbatch --array=0-(N-1) ...

# ============================================================================
# SLURM Job Array Script for LBL Cluster (CPU Only)
# Based on: https://scienceit-docs.lbl.gov/hpc/running/script-examples/
#
# Do not submit this file directly
# use ./submit_lbl_array.sh home from the experiments/ directory (pairs live there).
# paritition settings:
# lr4
# --partition=lr4
# --mincpus=24
# --mem=62G
# 
# lr5
# --partition=lr5
# --mincpus=28
# --mem=62G
#
# lr6
# --partition=lr6
# --mincpus=32
# --mem=92G
# 
# Before submitting:
#   1. Find your account, partition, and qos names
#   2. Edit pairs in submit_lbl_array.sh
#
# Monitor: squeue --me
# Cancel all: scancel <job_array_id>
# Cancel one: scancel <job_array_id>_<task_id>
# ============================================================================

set -euo pipefail

# Slurm runs a copy under /var/spool/slurmd/...; BASH_SOURCE is not the repo path.
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    cd "$SLURM_SUBMIT_DIR" || exit 1
    SCRIPT_DIR="$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    cd "$SCRIPT_DIR" || exit 1
fi

# Per-submit run folder (set by submit_lbl_array.sh). Fallback: flat logs/ if submitted by hand.
RUN_LOG_DIR="${LBL_RUN_LOG_DIR:-logs}"
mkdir -p "$RUN_LOG_DIR"

# Print job info
echo "============================================"
echo "Job Array ID: ${SLURM_ARRAY_JOB_ID:-}"
echo "Task ID: ${SLURM_ARRAY_TASK_ID:-}"
echo "Node: ${SLURM_NODELIST:-}"
echo "Start time: $(date)"
echo "Working directory (after cd to script dir): $(pwd)"
echo "Run log directory: $RUN_LOG_DIR"
echo "Pairs file: ${LBL_PAIRS_FILE:-${SCRIPT_DIR}/lbl_array_pairs.txt}"
echo "============================================"

# Load required modules
module load gcc openmpi boost ninja miniforge3/25.9.1

# Home-prefix conda env (mamba: mamba activate /global/home/users/$USER/geom_home).
# Like the scratch job script, call this env's python directly instead of `mamba activate`
# so compute nodes do not depend on libmamba resolving the prefix.
ENV_PREFIX="/global/home/users/${USER}/geom_home"

# Pick a working interpreter: prefer versioned names; verify by exec (not only `[ -e ]`,
# which can disagree across login vs compute or for symlinks on some filesystems).
PYTHON_BIN=""
for _py in "${ENV_PREFIX}/bin/python3.12" "${ENV_PREFIX}/bin/python3" "${ENV_PREFIX}/bin/python"; do
    if "$_py" -c "import sys" >/dev/null 2>&1; then
        PYTHON_BIN="$_py"
        break
    fi
done

if [ -z "${PYTHON_BIN:-}" ]; then
    echo "ERROR: No usable Python under ${ENV_PREFIX}/bin (tried python3.12, python3, python)." >&2
    echo "On compute, confirm the env exists: ls -la ${ENV_PREFIX}/bin/python*" >&2
    echo "Create/repair env: module load miniforge3/25.9.1 && mamba create -y -p $ENV_PREFIX python=3.12" >&2
    exit 1
fi

# One-line sanity (fails fast on broken prefix or filesystem errors)
if ! "$PYTHON_BIN" -c "import sys; print(sys.executable)" >/dev/null 2>&1; then
    echo "ERROR: Cannot execute $PYTHON_BIN (I/O error or broken env). Try another node or recreate the env." >&2
    exit 1
fi

echo "Python path: $PYTHON_BIN"
echo "Python version: $($PYTHON_BIN --version 2>&1)"

# Mesh pairs: submit_lbl_array.sh writes an immutable per-run file under logs/<RUN_TS>/.
# Fallback keeps direct/manual submissions compatible with the old shared file.
PAIRS_FILE="${LBL_PAIRS_FILE:-${SCRIPT_DIR}/lbl_array_pairs.txt}"
if [[ ! -f "$PAIRS_FILE" ]]; then
    echo "ERROR: Missing pairs file ${PAIRS_FILE}. Submit with ./submit_lbl_array.sh from experiments/." >&2
    exit 1
fi
pairs=()
while IFS= read -r line || [[ -n "$line" ]]; do
    [[ -z "${line// }" ]] && continue
    [[ "$line" =~ ^[[:space:]]*# ]] && continue
    pairs+=("$line")
done < "$PAIRS_FILE"

if [ -z "${SLURM_ARRAY_TASK_ID+x}" ]; then
    echo "Error: SLURM_ARRAY_TASK_ID is unset. Submit with sbatch (job array), not run interactively without it." >&2
    exit 1
fi

if [ "$SLURM_ARRAY_TASK_ID" -ge "${#pairs[@]}" ]; then
    echo "Error: SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID >= number of pairs (${#pairs[@]}). Fix --array or pairs[]." >&2
    exit 1
fi

# Get the pair for this task
pair="${pairs[$SLURM_ARRAY_TASK_ID]}"
read -r mesh1 mesh2 <<< "$pair"

echo "Processing pair: $mesh1 vs $mesh2"

# ============================================================================
# Mesh path definitions (keep in sync with run_all_pairs_resumable.sh)
# ============================================================================
declare -A meshes=(
    ["r_1"]="data/synthetic/torus_ratio_0.1_n_theta_20_n_phi_10.obj"
    ["r_1_sub"]="data/synthetic/torus_ratio_0.1_n_theta_20_n_phi_10_subdivided.obj"
    ["r_1_d"]="data/synthetic/torus_ratio_0.1_n_theta_20_n_phi_30.obj"
    ["r_2"]="data/synthetic/torus_ratio_0.2_n_theta_20_n_phi_10.obj"
    ["r_2_sub"]="data/synthetic/torus_ratio_0.2_n_theta_20_n_phi_10_subdivided.obj"
    ["r_2_d"]="data/synthetic/torus_ratio_0.2_n_theta_20_n_phi_30.obj"
    ["r_5"]="data/synthetic/torus_ratio_0.5_n_theta_20_n_phi_10.obj"
    ["r_5_sub"]="data/synthetic/torus_ratio_0.5_n_theta_20_n_phi_10_subdivided.obj"
    ["r_5_d"]="data/synthetic/torus_ratio_0.5_n_theta_20_n_phi_30.obj"
    ["r_8"]="data/synthetic/torus_ratio_0.8_n_theta_20_n_phi_10.obj"
    ["r_8_sub"]="data/synthetic/torus_ratio_0.8_n_theta_20_n_phi_10_subdivided.obj"
    ["r_8_d"]="data/synthetic/torus_ratio_0.8_n_theta_20_n_phi_30.obj"
    ["r_9"]="data/synthetic/torus_ratio_0.9_n_theta_20_n_phi_10.obj"
    ["r_9_sub"]="data/synthetic/torus_ratio_0.9_n_theta_20_n_phi_10_subdivided.obj"
    ["r_9_d"]="data/synthetic/torus_ratio_0.9_n_theta_20_n_phi_30.obj"
    ["Gorilla_m_395635"]="data/real/pelvis_subsampled/Gorilla-beringei-beringei-m-NMNH-395636-pelvis-L-000091238.obj"
    ["Gorilla_m_397351"]="data/real/pelvis_subsampled/Gorilla-beringei-beringei-m-NMNH-397351-pelvis-L-000090881.obj"
    ["Hoolock_f_83416"]="data/real/pelvis_subsampled/Hoolock-hoolock-f-AMNH-83416-pelvis-L-000168063.obj"
    ["Hoolock_f_83423"]="data/real/pelvis_subsampled/Hoolock-hoolock-f-AMNH-83423-pelvis-L-000168067.obj"
    ["Pan_f_15296"]="data/real/pelvis_subsampled/Pan-paniscus-f-RMCA-15296-pelvis-L-000164217.obj"
    ["Pan_m_27696"]="data/real/pelvis_subsampled/Pan-paniscus-m-RMCA-27696-pelvis-L-000093793.obj"
    ["Papio_f_384235"]="data/real/pelvis_subsampled/Papio-anubis-neumanni-f-NMNH-384235-pelvis-L-000090811.obj"
    ["Papio_m_384229"]="data/real/pelvis_subsampled/Papio-anubis-neumanni-m-NMNH-384229-pelvis-L-000089800.obj"
    ["Pongo_f_588109"]="data/real/pelvis_subsampled/Pongo-abelii-f-NMNH-588109-pelvis-L-000093958.obj"
    ["Pongo_f_145302"]="data/real/pelvis_subsampled/Pongo-pygmaeus-f-NMNH-145302-pelvis-L-000095230.obj"
    ["Petauroides"]="data/real/pelvis_subsampled/SAMAM7326_Petauroides_volans_Pelvis_m_Polyga_GM.obj"
    ["ring_0"]="data/real/thingi10k_subsampled/ring_76715_nV_600.obj"           # 76715
    ["ring_1"]="data/real/thingi10k_subsampled/ring_76716_nV_600.obj"           # 76716
    ["nut_circle_0"]="data/real/thingi10k_subsampled/nut_circle_200966_nV_600.obj"    # 200966
    ["nut_circle_1"]="data/real/thingi10k_subsampled/nut_circle_200967_nV_600.obj"    # 200967
    ["roller_0"]="data/real/thingi10k_subsampled/roller_1207667_nV_600.obj"
    ["roller_1"]="data/real/thingi10k_subsampled/roller_1207669_nV_600.obj"
    ["sphere_0"]="data/real/thingi10k_subsampled/sphere_1396892_nV_600.obj"
    ["sphere_1"]="data/real/thingi10k_subsampled/sphere_1396893_nV_600.obj"
)

if [ -z "${meshes[$mesh1]+x}" ] || [ -z "${meshes[$mesh2]+x}" ]; then
    echo "Error: Unknown mesh key(s): $mesh1 $mesh2 (add to meshes[] or fix pairs[])" >&2
    exit 1
fi

# ============================================================================
# Experiment parameters (aligned with run_all_pairs_resumable.sh; tune for cluster)
# ============================================================================
# defaults
normalize_area_to_one=true
cut_on_shortest_generator=true
codomain_swap_generators=false
optimizer="Adam"
n_iter=300000
group_name=lbl_cluster_run
# checkpointing
checkpoint_dir="./checkpoints"
checkpoint_filename="checkpoint.pt"
iters_to_save_checkpoint=2000
# flip reject backoff
flip_reject_backoff_enabled=false
flip_reject_backoff_factor=0.5
flip_reject_backoff_min_lr=1e-10
# energy plateau detection
energy_plateau_patience=5000
energy_plateau_rel_tol=1e-5
reset_plateau_steps_without_improvement=true
# ReduceLROnPlateau
reduce_lr_on_plateau=true
lr_scheduler="ReduceLROnPlateauEMA"
reduce_lr_ema=0.5
reduce_lr_factor=0.5
reduce_lr_tol_mode=rel
reduce_lr_patience=2000
reduce_lr_min_lr=1e-7
reduce_lr_tol=1e-3
# energy objective
energy_objective="area_shape" # "area_shape" or "dirichlet"

# important settings
lr=1e-5
resume_from_checkpoint=true
smooth_max_type="none" # "none", "logsumexp", "pnorm"
Tutte_embedding_type_minimal=false
Tutte_embedding_type="Uniform" # "Uniform", "coTan", "MeanValue", "Authalic"
edge_length_type="geodesic" # "euclidean", "geodesic", "flip_geodesic"
save_uv_trajectories=false # big file
use_wandb=false # avoids large local experiments/wandb/run-* trees; does not affect checkpoint resume
area_loss_weight=1.0
shape_stretch_weight=1.0

# ============================================================================
# Run the experiment
# ============================================================================
timestamp=$(date +"%Y%m%d_%H%M%S")
exp_log="${RUN_LOG_DIR}/task_${SLURM_ARRAY_TASK_ID}_${mesh1}_vs_${mesh2}.log"
temp_config="temp_config_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.yaml"

# Save UV trajectories (if enabled) into one shared folder for the whole run.
UV_TRAJ_DIR="${RUN_LOG_DIR}/uv_trajectories/"
if [ "${save_uv_trajectories}" = true ]; then
    mkdir -p "$UV_TRAJ_DIR"
fi
cp argus.yaml "$temp_config"

sed -i "s|run_name:.*|run_name: \"${mesh1}_vs_${mesh2}_lrc\"|" "$temp_config"
sed -i "s|obj_path1:.*|obj_path1: \"${meshes[$mesh1]}\"|" "$temp_config"
sed -i "s|obj_path2:.*|obj_path2: \"${meshes[$mesh2]}\"|" "$temp_config"
sed -i "s|group_name:.*|group_name: \"${group_name}\"|" "$temp_config"
sed -i "s|n_iter:.*|n_iter: ${n_iter}|" "$temp_config"
sed -i "s|normalize_area_to_one:.*|normalize_area_to_one: ${normalize_area_to_one}|" "$temp_config"
sed -i "s|Tutte_embedding_type:.*|Tutte_embedding_type: \"${Tutte_embedding_type}\"|" "$temp_config"
sed -i "s|edge_length_type:.*|edge_length_type: \"${edge_length_type}\"|" "$temp_config"
sed -i "s|energy_objective:.*|energy_objective: \"${energy_objective}\"|" "$temp_config"
sed -i "s|smooth_max_type:.*|smooth_max_type: \"${smooth_max_type}\"|" "$temp_config"
sed -i "s|optimizer:.*|optimizer: \"${optimizer}\"|" "$temp_config"
sed -i "s|lr:.*|lr: ${lr}|" "$temp_config"
sed -i "s|area_loss_weight:.*|area_loss_weight: ${area_loss_weight}|" "$temp_config"
sed -i "s|shape_stretch_weight:.*|shape_stretch_weight: ${shape_stretch_weight}|" "$temp_config"
sed -i "s|cut_on_shortest_generator:.*|cut_on_shortest_generator: ${cut_on_shortest_generator}|" "$temp_config"
sed -i "s|codomain_swap_generators:.*|codomain_swap_generators: ${codomain_swap_generators}|" "$temp_config"
sed -i "s|save_uv_trajectories:.*|save_uv_trajectories: ${save_uv_trajectories}|" "$temp_config"
sed -i "s|use_wandb:.*|use_wandb: ${use_wandb}|" "$temp_config"
sed -i "s|save_folder_path:.*|save_folder_path: \"${UV_TRAJ_DIR}\"|" "$temp_config"
sed -i "s|checkpoint_dir:.*|checkpoint_dir: \"${checkpoint_dir}\"|" "$temp_config"
sed -i "s|checkpoint_filename:.*|checkpoint_filename: \"${checkpoint_filename}\"|" "$temp_config"
sed -i "s|resume_from_checkpoint:.*|resume_from_checkpoint: ${resume_from_checkpoint}|" "$temp_config"
sed -i "s|iters_to_save_checkpoint:.*|iters_to_save_checkpoint: ${iters_to_save_checkpoint}|" "$temp_config"
sed -i "s|energy_plateau_patience:.*|energy_plateau_patience: ${energy_plateau_patience}|" "$temp_config"
sed -i "s|energy_plateau_rel_tol:.*|energy_plateau_rel_tol: ${energy_plateau_rel_tol}|" "$temp_config"
sed -i "s|reset_plateau_steps_without_improvement:.*|reset_plateau_steps_without_improvement: ${reset_plateau_steps_without_improvement}|" "$temp_config"
sed -i "s|reduce_lr_on_plateau:.*|reduce_lr_on_plateau: ${reduce_lr_on_plateau}|" "$temp_config"
sed -i "s|lr_scheduler:.*|lr_scheduler: \"${lr_scheduler}\"|" "$temp_config"
sed -i "s|reduce_lr_ema:.*|reduce_lr_ema: ${reduce_lr_ema}|" "$temp_config"
sed -i "s|reduce_lr_factor:.*|reduce_lr_factor: ${reduce_lr_factor}|" "$temp_config"
sed -i "s|reduce_lr_patience:.*|reduce_lr_patience: ${reduce_lr_patience}|" "$temp_config"
sed -i "s|reduce_lr_min_lr:.*|reduce_lr_min_lr: ${reduce_lr_min_lr}|" "$temp_config"
sed -i "s|reduce_lr_tol:.*|reduce_lr_tol: ${reduce_lr_tol}|" "$temp_config"
sed -i "s|reduce_lr_tol_mode:.*|reduce_lr_tol_mode: \"${reduce_lr_tol_mode}\"|" "$temp_config"
sed -i "s|flip_reject_backoff_enabled:.*|flip_reject_backoff_enabled: ${flip_reject_backoff_enabled}|" "$temp_config"
sed -i "s|flip_reject_backoff_factor:.*|flip_reject_backoff_factor: ${flip_reject_backoff_factor}|" "$temp_config"
sed -i "s|flip_reject_backoff_min_lr:.*|flip_reject_backoff_min_lr: ${flip_reject_backoff_min_lr}|" "$temp_config"
sed -i "s|Tutte_embedding_type_minimal:.*|Tutte_embedding_type_minimal: ${Tutte_embedding_type_minimal}|" "$temp_config"

echo "Starting experiment: $mesh1 vs $mesh2"
echo "Config file: $temp_config"
echo "Log file: $exp_log"

if "$PYTHON_BIN" -u run.py --config "$temp_config" > "$exp_log" 2>&1; then
    echo "✅ Success: $mesh1 vs $mesh2"
    # Full training trace stays in $exp_log; repeat the final summary here for slurm_*.out
    if grep -q "Optimization finished" "$exp_log" 2>/dev/null; then
        grep -F "Optimization finished" "$exp_log" | tail -n 1
    fi
else
    echo "❌ Failed: $mesh1 vs $mesh2 (check $exp_log for details)"
    exit 1
fi

rm "$temp_config"

echo "============================================"
echo "Completed: $mesh1 vs $mesh2"
echo "End time: $(date)"
echo "============================================"

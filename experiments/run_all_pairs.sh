#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# -----------------------------------------------------------------------------
# Resumable batch runner
#
# Usage:
#   ./run_all_pairs.sh                 # run pairs from script
#   ./run_all_pairs.sh c2 a2           # run single pair (mesh keys)
#   ./run_all_pairs.sh c2 a2 b2 a2     # run multiple pairs
# -----------------------------------------------------------------------------

# Create logs directory if it doesn't exist
mkdir -p logs

# Get timestamp for unique log files
timestamp=$(date +"%Y%m%d_%H%M%S")
main_log="logs/batch_${timestamp}.log"

# Function to log with timestamp
log_with_timestamp() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$main_log"
}

# Define the mesh pairs (same keys as run_all_pairs.sh)
declare -A meshes=(
    ["simple_torus"]="data/synthetic/simple_torus.obj"
    ["simple_torus_flipped_edge"]="data/synthetic/simple_torus_flipped_edge.obj"
    ["test_torus"]="data/synthetic/torus_ratio_0.2_n_theta_3_n_phi_3.obj"
    ["a"]="data/synthetic/torus_ratio_0.2_n_theta_10_n_phi_10.obj"
    ["a_subdivided"]="data/synthetic/a_subdivided.obj"
    ["a2"]="data/synthetic/torus_ratio_0.2_n_theta_20_n_phi_10.obj"
    ["b"]="data/synthetic/torus_ratio_0.5_n_theta_10_n_phi_10.obj"
    ["b_subdivided"]="data/synthetic/b_subdivided.obj"
    ["ir_theta_2010"]="data/synthetic/irregular_gaussain_theta_0.5_20_10.obj"
    ["ir_phi_2010"]="data/synthetic/irregular_gaussain_phi_0.5_20_10.obj"
    ["b2"]="data/synthetic/torus_ratio_0.5_n_theta_20_n_phi_10.obj"
    ["b2_subdivide_single_face"]="data/synthetic/b2_subdivide_single_face.obj"
    ["b2_subdivide_each_face"]="data/synthetic/b2_subdivide_each_face.obj"
    ["b3030"]="data/synthetic/torus_ratio_0.5_n_theta_30_n_phi_30.obj"
    ["b4040"]="data/synthetic/torus_ratio_0.5_n_theta_40_n_phi_40.obj"
    ["c"]="data/synthetic/torus_ratio_0.8_n_theta_10_n_phi_10.obj"
    ["c2"]="data/synthetic/torus_ratio_0.8_n_theta_20_n_phi_10.obj"
    ["d2"]="data/synthetic/torus_ratio_0.1_n_theta_20_n_phi_10.obj"
    ["e2"]="data/synthetic/torus_ratio_0.9_n_theta_20_n_phi_10.obj"
    ["ACE_coffee_mug"]="data/real/ACE_Coffee_Mug_Model.obj"
    ["Cole_mug"]="data/real/Cole_Hardware_Mug_Classic_Blue.obj"
    ["room_essential"]="data/real/Room_Essentials_Mug_White_Yellow.obj"
    ["teapot"]="data/real/Threshold_Porcelain_Teapot_White.obj"
    ["L1_17"]="data/real/L1_17.obj"
    ["L3_15"]="data/real/L3_15.obj"
    ["L5_9"]="data/real/L5_9.obj"
    ["threshold_mug_remesh"]="data/real/remeshed/Threshold_Coffee_Mug_remesh.obj"
    ["ace_mug_remesh"]="data/real/remeshed/ACE_Coffee_Mug_Model_remesh.obj"
    ["cole_mug_remesh"]="data/real/remeshed/Cole_Hardware_Mug_Classic_Blue_remesh.obj"
    ["room_essential_remesh"]="data/real/remeshed/Room_Essentials_Mug_White_Yellow_remesh.obj"
    ["r_1"]="data/synthetic/torus_ratio_0.1_n_theta_20_n_phi_10.obj"
    ["r_1_5050"]="data/synthetic/torus_ratio_0.1_n_theta_50_n_phi_50.obj"
    ["r_1_sub"]="data/synthetic/torus_ratio_0.1_n_theta_20_n_phi_10_subdivided.obj"
    ["r_1_d"]="data/synthetic/torus_ratio_0.1_n_theta_20_n_phi_30.obj"
    ["r_2"]="data/synthetic/torus_ratio_0.2_n_theta_20_n_phi_10.obj"
    ["r_2_5050"]="data/synthetic/torus_ratio_0.2_n_theta_50_n_phi_50.obj"
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
    ["r_9_5050"]="data/synthetic/torus_ratio_0.9_n_theta_50_n_phi_50.obj"
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
    # ["ring_0"]="data/real/thingi10k/ring_754638_nV_606.obj"
    # ["ring_1"]="data/real/thingi10k/ring_370885_nV_389.obj"
    # ["nut_0"]="data/real/thingi10k_subsampled/nut_41275_nV_500.obj"
    # ["nut_1"]="data/real/thingi10k_subsampled/nut_370993_nV_500.obj"
    # ["gear_0"]="data/real/thingi10k/gear_100077_nV_264.obj"
    # ["gear_1"]="data/real/thingi10k_subsampled/gear_641140_nV_500.obj"
    # ["gear_2"]="data/real/thingi10k_subsampled/gear_1312974_nV_500.obj"
    # ["gear_3"]="data/real/thingi10k_subsampled/gear_1146170_nV_500.obj"
    # ["roller_0"]="data/real/thingi10k_subsampled/roller_1207667_nV_500.obj"
    # ["pendant_0"]="data/real/thingi10k_subsampled/pendant_428605_nV_500.obj"
    # ["pendant_1"]="data/real/thingi10k_subsampled/pendant_428607_nV_500.obj"
    ["sphere_0"]="data/real/thingi10k_subsampled/sphere_1396892_nV_600.obj"
    ["sphere_1"]="data/real/thingi10k_subsampled/sphere_1396893_nV_600.obj"
    ["ring_0"]="data/real/thingi10k_subsampled/ring_76715_nV_600.obj"           # 76715
    ["ring_1"]="data/real/thingi10k_subsampled/ring_76716_nV_600.obj"           # 76716
    ["nut_star_0"]="data/real/thingi10k_subsampled/nut_star_200962_nV_600.obj"  # 200962
    ["nut_star_1"]="data/real/thingi10k_subsampled/nut_star_200963_nV_600.obj"  # 200963
    ["nut_hexagon_0"]="data/real/thingi10k_subsampled/nut_hexagon_200961_nV_600.obj"  # 200961
    ["nut_hexagon_1"]="data/real/thingi10k_subsampled/nut_hexagon_200969_nV_600.obj"  # 200969
    ["nut_circle_0"]="data/real/thingi10k_subsampled/nut_circle_200966_nV_600.obj"    # 200966
    ["nut_circle_1"]="data/real/thingi10k_subsampled/nut_circle_200967_nV_600.obj"    # 200967
    ["gear_0"]="data/real/thingi10k_subsampled/gear_669969_nV_600.obj"
    ["gear_1"]="data/real/thingi10k_subsampled/gear_669970_nV_600.obj"
    ["trumpet_0"]="data/real/thingi10k_subsampled/trumpet_250397_nV_600.obj"
    ["trumpet_1"]="data/real/thingi10k_subsampled/trumpet_250398_nV_600.obj"
    ["toy_home_0"]="data/real/thingi10k/toy_home_100679_nV_248.obj"
    ["roller_0"]="data/real/thingi10k_subsampled/roller_1207667_nV_600.obj"
    ["roller_1"]="data/real/thingi10k_subsampled/roller_1207669_nV_600.obj"
    ["annulus_r1"]="data/synthetic/annulus_rin1_Rout6_iso.obj"
    ["annulus_r2"]="data/synthetic/annulus_rin2_Rout6_iso.obj"
    ["gear8_r1"]="data/synthetic/gear_rin1_Rtip6_8t_iso.obj"
    ["gear8_r2"]="data/synthetic/gear_rin2_Rtip6_8t_iso.obj"
    ["gear16_r1"]="data/synthetic/gear_rin1_Rtip6_16t_iso.obj"
    ["gear16_r2"]="data/synthetic/gear_rin2_Rtip6_16t_iso.obj"
)

# load mesh options
normalize_area_to_one=true
group_name=Apr_02

cut_on_shortest_generator=true
codomain_swap_generators=true
n_iter=150000
optimizer="Adam"

# energy plateau detection
energy_plateau_patience=5000
energy_plateau_rel_tol=1e-5
reset_plateau_steps_without_improvement=true

# resumable checkpoint controls (override argus.yaml)
# checkpoint_dir="./checkpoints_dirichlet"
# uv_trajectory_dir="logs/dirichlet_uv_trajectories"
checkpoint_filename="checkpoint.pt"
iters_to_save_checkpoint=2000

# ReduceLROnPlateau
resume_lr_scheduler_state=false # on resume, rebuild LR scheduler from current config (so reduce_lr_* changes take effect)
reduce_lr_on_plateau=true
lr_scheduler="ReduceLROnPlateauEMA"
reduce_lr_ema=0.5
reduce_lr_factor=0.5
reduce_lr_patience=2000
reduce_lr_tol=1e-3 # larger reduce_lr_tol -> only bigger improvements count -> plateau is detected sooner -> LR reduces more often.
reduce_lr_tol_mode=rel

# flip_reject_backoff_enabled=false # enable flip reject backoff
flip_reject_backoff_enabled=false # enable flip reject backoff
flip_reject_backoff_factor=0.5    # factor for flip reject backoff
flip_reject_backoff_min_lr=1e-10  # minimum step size for flip reject backoff

# Tutte embedding
# if Tutte_embedding_type_minimal is false, 
#    - each run uses Tutte_embedding_type from the loop below. 
# if Tutte_embedding_type_minimal is true, 
#    - fresh runs ignore Tutte_embedding_type from the loop below and 
#    - pick the minimal-initial-energy Tutte type (mesh_context.pt stores the choice).


# important settings
lr=1e-4                     # step size. If resume from checkpoint, the change here will not take effect.
edge_length_types=("flip_geodesic")
Tutte_embedding_type_minimal=true
Tutte_embedding_type_options=("Uniform") # "Uniform", "coTan", "MeanValue", "Authalic"
smooth_max_type_options=("none") # "none", "logsumexp", "pnorm"
area_loss_weight=1.0      # weight for area-loss term in area_shape objective
shape_stretch_weight=1.0    # weight for shape-stretch term in area_shape objective
reduce_lr_min_lr=1e-8      # minimum step size
save_uv_trajectories=false   # save subsampled UV trajectory to npy (uv_pos_res.npy)
use_wandb=true                   # use wandb for logging
resume_from_checkpoint=false     # resume from checkpoint
checkpoint_dir="./checkpoints"
uv_trajectory_dir="logs/uv_trajectories"
energy_objective="area_shape"    # "area_shape", "dirichlet", or "log_area_shape"


# RNG seed (torch/numpy; used for Reeb direction when cut_on_shortest_generator)
seed=10

# Pairs to run (used when no CLI args are provided)
pairs=(
    # "nut_circle_0 nut_circle_1" 
    "r_1 r_5" "r_5 r_1"
)

# CLI pair override: pass an even number of mesh keys
if [ "$#" -gt 0 ]; then
    if [ $(( $# % 2 )) -ne 0 ]; then
        echo "Error: Provide mesh keys in pairs: mesh1 mesh2 [mesh3 mesh4 ...]" >&2
        exit 1
    fi
    pairs=()
    while [ "$#" -gt 0 ]; do
        mesh1="$1"
        mesh2="$2"
        pairs+=("$mesh1 $mesh2")
        shift 2
    done
fi

log_with_timestamp "Starting batch with ${#pairs[@]} pairs"
log_with_timestamp "Main log file: $main_log"

for edge_length_type in "${edge_length_types[@]}"; do
    for Tutte_embedding_type in "${Tutte_embedding_type_options[@]}"; do
        for smooth_max_type in "${smooth_max_type_options[@]}"; do
            log_with_timestamp "Running experiments with Tutte=$Tutte_embedding_type, smooth_max=$smooth_max_type"
            for pair in "${pairs[@]}"; do
                read -r mesh1 mesh2 <<< "$pair"

                if [ -z "${meshes[$mesh1]+x}" ] || [ -z "${meshes[$mesh2]+x}" ]; then
                    log_with_timestamp "❌ Unknown mesh key(s): $mesh1 $mesh2"
                    continue
                fi

                exp_log="logs/${timestamp}_${mesh1}_vs_${mesh2}.log"
                log_with_timestamp "Running pair: $mesh1 vs $mesh2"
                log_with_timestamp "Individual log: $exp_log"

                temp_config="temp_config_${mesh1}_${mesh2}.yaml"
                cp argus.yaml "$temp_config"
                if [ "${save_uv_trajectories}" = true ]; then
                    mkdir -p "$uv_trajectory_dir"
                fi

                sed -i "s|run_name:.*|run_name: \"${mesh1}_vs_${mesh2}\"|" "$temp_config"
                sed -i "s|obj_path1:.*|obj_path1: \"${meshes[$mesh1]}\"|" "$temp_config"
                sed -i "s|obj_path2:.*|obj_path2: \"${meshes[$mesh2]}\"|" "$temp_config"
                sed -i "s|group_name:.*|group_name: \"${group_name}\"|" "$temp_config"
                sed -i "s|n_iter:.*|n_iter: ${n_iter}|" "$temp_config"
                sed -i "s|normalize_area_to_one:.*|normalize_area_to_one: ${normalize_area_to_one}|" "$temp_config"
                sed -i "s|Tutte_embedding_type:.*|Tutte_embedding_type: \"${Tutte_embedding_type}\"|" "$temp_config"
                sed -i "s|Tutte_embedding_type_minimal:.*|Tutte_embedding_type_minimal: ${Tutte_embedding_type_minimal}|" "$temp_config"
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
                sed -i "s|save_folder_path:.*|save_folder_path: \"${uv_trajectory_dir}\"|" "$temp_config"
                sed -i "s|checkpoint_dir:.*|checkpoint_dir: \"${checkpoint_dir}\"|" "$temp_config"
                sed -i "s|checkpoint_filename:.*|checkpoint_filename: \"${checkpoint_filename}\"|" "$temp_config"
                sed -i "s|resume_from_checkpoint:.*|resume_from_checkpoint: ${resume_from_checkpoint}|" "$temp_config"
                sed -i "s|resume_lr_scheduler_state:.*|resume_lr_scheduler_state: ${resume_lr_scheduler_state}|" "$temp_config"
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
                sed -i "s|seed:.*|seed: ${seed}|" "$temp_config"

                log_with_timestamp "Starting experiment: $mesh1 vs $mesh2"
                if python run.py --config "$temp_config" > "$exp_log" 2>&1; then
                    log_with_timestamp "✅ Success: $mesh1 vs $mesh2"
                else
                    log_with_timestamp "❌ Failed: $mesh1 vs $mesh2 (check $exp_log for details)"
                fi

                if [ -f "$exp_log" ]; then
                    if grep -Eq "stopped:|ctrl\\+c received|budget_exhausted" "$exp_log"; then
                        while IFS= read -r line; do
                            log_with_timestamp "  $line"
                        done < <(grep -E "stopped:|ctrl\\+c received|budget_exhausted" "$exp_log")
                    fi
                fi

                rm "$temp_config"
                log_with_timestamp "Completed: $mesh1 vs $mesh2"
                log_with_timestamp "----------------------------------------"
            done
        done
    done
done

log_with_timestamp "All experiments completed!"


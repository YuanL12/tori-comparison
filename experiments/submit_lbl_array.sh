#!/bin/bash
#
# From experiments/:
#   chmod +x submit_lbl_array.sh   # once
#   ./submit_lbl_array.sh
#
# Edit the pairs[] block below; this script writes logs/<RUN_TS>/lbl_array_pairs.txt and submits with
#   sbatch --array=0-(N-1)  where N is the number of pairs.
#
# This creates logs/<YYYYMMDD_HHMMSS>/ with slurm_%A_%a.out|err and per-task *.log files.
# Checkpoints still use ./checkpoints. UV trajectories (when enabled) go to logs/<RUN_TS>/uv_trajectories/.

set -euo pipefail

# get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# cd to the script directory
cd "$SCRIPT_DIR" || exit 1

# ============================================================================
# Mesh pairs for lbl array jobs (one element per pair: "mesh1 mesh2")
# ============================================================================
pairs=(
    # "ring_0 ring_1" "ring_0 roller_0" "ring_0 roller_1" "ring_0 sphere_0" "ring_0 sphere_1" "ring_0 nut_circle_0" "ring_0 nut_circle_1"
    # "ring_1 ring_0" "ring_1 roller_0" "ring_1 roller_1" "ring_1 sphere_0" "ring_1 sphere_1" "ring_1 nut_circle_0" "ring_1 nut_circle_1"
    # "roller_0 ring_0" "roller_0 ring_1" "roller_0 roller_1" "roller_0 sphere_0" "roller_0 sphere_1" "roller_0 nut_circle_0" "roller_0 nut_circle_1"
    # "roller_1 ring_0" "roller_1 ring_1" "roller_1 roller_0" "roller_1 sphere_0" "roller_1 sphere_1" "roller_1 nut_circle_0" "roller_1 nut_circle_1"
    # "sphere_0 ring_0" "sphere_0 ring_1" "sphere_0 roller_0" "sphere_0 roller_1" "sphere_0 sphere_1" "sphere_0 nut_circle_0" "sphere_0 nut_circle_1"
    # "sphere_1 ring_0" "sphere_1 ring_1" "sphere_1 roller_0" "sphere_1 roller_1" "sphere_1 sphere_0" "sphere_1 nut_circle_0" "sphere_1 nut_circle_1"

    # Re-run pairs that previously ended with stop_reason=completed (see records_thingi10k.md Status `C`)
    "nut_circle_0 nut_circle_1"
    "nut_circle_0 ring_0"
    "nut_circle_0 roller_0"
    "nut_circle_1 nut_circle_0"
    "roller_0 roller_1"
    "sphere_0 sphere_1"
    "sphere_1 ring_1"
    "sphere_1 roller_1"
)

N=${#pairs[@]}
if (( N < 1 )); then
    echo "pairs[] is empty; add pairs in submit_lbl_array.sh" >&2
    exit 1
fi
ARRAY_LAST=$((N - 1))

if (( $# > 0 )); then
    echo "Usage: $0" >&2
    echo "This launcher is home-env only; no mode argument is needed." >&2
    exit 1
fi

JOB_SCRIPT="${SCRIPT_DIR}/submit_lbl_array_job.sh"
MODE_LABEL="home env (${JOB_SCRIPT##*/})"

RUN_TS=$(date +"%Y%m%d_%H%M%S")
LOG_ROOT="logs/${RUN_TS}"
mkdir -p "$LOG_ROOT"

# Absolute path avoids ambiguity if SLURM_SUBMIT_DIR ever differs from cwd.
LOG_ROOT_ABS="$(cd "$LOG_ROOT" && pwd)"
PAIRS_FILE="${LOG_ROOT}/lbl_array_pairs.txt"
PAIRS_FILE_ABS="${LOG_ROOT_ABS}/lbl_array_pairs.txt"
printf '%s\n' "${pairs[@]}" > "$PAIRS_FILE"

echo "Submitting array job ($MODE_LABEL); run log directory: $LOG_ROOT_ABS"
echo "Pairs: ${N} tasks (Slurm array indices 0-${ARRAY_LAST}); list: ${PAIRS_FILE_ABS}"

exec sbatch \
    --array="0-${ARRAY_LAST}" \
    --output="${LOG_ROOT}/slurm_%A_%a.out" \
    --error="${LOG_ROOT}/slurm_%A_%a.err" \
    --export=ALL,LBL_RUN_LOG_DIR="${LOG_ROOT_ABS}",LBL_PAIRS_FILE="${PAIRS_FILE_ABS}" \
    "$JOB_SCRIPT"

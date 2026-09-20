#!/usr/bin/env bash
# Reproduce the original 48-episode prefix of the composable-slot pilot.
#
# The schedule is the first 12 ordered pairs from pilot_96, with four seeds
# per pair (3000--3047).  Keeping those pair/run names makes regenerated v4
# data directly comparable to the earlier pilot artifacts.
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <gpu-index>" >&2
    exit 2
fi

gpu="$1"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

python_bin="${PYTHON_BIN:-python}"
config="tactile_memory_match_slot_calibration_96"
seed_base=3000
repetitions=4

# This is intentionally the historical 48-trajectory prefix, not a new
# balanced split.  Use run_composable_slot_pilot_96.sh for the full pilot.
pairs=(
    "light_smooth_rigid heavy_smooth_rigid h1_weight"
    "light_smooth_rigid heavy_rough_rigid h2_weight_roughness"
    "light_smooth_rigid heavy_rough_soft h3_all"
    "light_smooth_soft light_rough_soft h1_roughness"
    "light_smooth_soft heavy_smooth_rigid h2_weight_hardness"
    "light_smooth_soft heavy_rough_rigid h3_all"
    "light_rough_rigid light_rough_soft h1_hardness"
    "light_rough_rigid heavy_smooth_rigid h2_weight_roughness"
    "light_rough_rigid heavy_smooth_soft h3_all"
    "light_rough_soft heavy_rough_soft h1_weight"
    "light_rough_soft light_smooth_rigid h2_roughness_hardness"
    "light_rough_soft heavy_smooth_rigid h3_all"
)

for index in "${!pairs[@]}"; do
    read -r reference distractor difficulty <<< "${pairs[$index]}"
    start_seed=$((seed_base + index * repetitions))
    end_seed=$((start_seed + repetitions - 1))
    run_name="pilot48_$(printf '%02d' "$index")_${difficulty}"

    echo "[$((index + 1))/${#pairs[@]}] ${reference} vs ${distractor}; seeds ${start_seed}-${end_seed}"
    "$python_bin" scripts/collect_data.py tactile_memory_match "$config" \
        --episode_num "$repetitions" \
        --start_seed "$start_seed" \
        --max_seed "$end_seed" \
        --gpu "$gpu" \
        --run_name "$run_name" \
        --task_cfg_override "compact_fixed_pair_scene=true" \
        --task_cfg_override "reference_class_key=${reference}" \
        --task_cfg_override "distractor_class_key=${distractor}"
done

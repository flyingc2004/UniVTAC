#!/usr/bin/env bash
# Three two-slot contrasts x four expert seeds.  No roughness label changes.
set -euo pipefail

gpu="${1:-0}"
task="tactile_memory_match"
config="tactile_memory_match_weight_hardness_smoke"

declare -a cases=(
  "weight_only light_smooth_rigid heavy_smooth_rigid 5000"
  "hardness_only light_smooth_rigid light_smooth_soft 5004"
  "weight_hardness light_smooth_rigid heavy_smooth_soft 5008"
)

for entry in "${cases[@]}"; do
  read -r name reference distractor start_seed <<< "${entry}"
  end_seed=$((start_seed + 3))
  python scripts/collect_data.py "${task}" "${config}" \
    --episode_num 4 \
    --start_seed "${start_seed}" \
    --max_seed "${end_seed}" \
    --gpu "${gpu}" \
    --run_name "weight_hardness_${name}" \
    --task_cfg_override compact_fixed_pair_scene=true \
    --task_cfg_override reference_class_key="${reference}" \
    --task_cfg_override distractor_class_key="${distractor}"
done

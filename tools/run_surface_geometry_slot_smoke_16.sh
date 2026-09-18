#!/usr/bin/env bash
# Four roughness-containing contrasts x four official-expert seeds.
set -euo pipefail

gpu="${1:-0}"
task="tactile_memory_match"
config="tactile_memory_match_surface_roughness_smoke"

declare -a cases=(
  "roughness_only light_smooth_rigid light_rough_rigid 5100"
  "weight_roughness light_smooth_rigid heavy_rough_rigid 5104"
  "roughness_hardness light_smooth_rigid light_rough_soft 5108"
  "weight_roughness_hardness light_smooth_rigid heavy_rough_soft 5112"
)

for entry in "${cases[@]}"; do
  read -r name reference distractor start_seed <<< "${entry}"
  end_seed=$((start_seed + 3))
  python scripts/collect_data.py "${task}" "${config}" \
    --episode_num 4 \
    --start_seed "${start_seed}" \
    --max_seed "${end_seed}" \
    --gpu "${gpu}" \
    --run_name "surface_roughness_${name}" \
    --task_cfg_override compact_fixed_pair_scene=true \
    --task_cfg_override reference_class_key="${reference}" \
    --task_cfg_override distractor_class_key="${distractor}"
done

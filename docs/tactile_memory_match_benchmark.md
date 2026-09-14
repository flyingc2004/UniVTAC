# Composable Tactile Memory Match Benchmark

`tactile_memory_match` is a model-agnostic reference-to-candidate benchmark.
It has eight hidden physical combinations:

```text
weight {light, heavy} x roughness {smooth, rough} x hardness {rigid, soft}
```

All objects use `Can_d4cm.usd`.  The soft variant is the same mesh with
`StableNeoHookean(youngs_modulus=0.1, poisson_rate=0.45)`; there is no cuboid
fallback.  A reference and its matching candidate have identical hidden
properties.  The distractor differs in one, two, or three slots.

## Public Measurement Contract

The benchmark provides a controlled measurement primitive:

```python
spec = task.get_public_probe_spec()
response = task.run_public_probe("reference_object")
```

The task performs the same sequence for reference, left candidate, and right
candidate:

```text
side grasp -> adaptive close -> bilateral-contact gate -> preload
-> short lift -> hold -> lower -> release -> clearance
```

`response` contains raw depth/marker frames, gripper qpos/control frames, and
a `tactile_probe.v3` summary.  Its three canonical windows are `preload`,
`lift_motion`, and `hold`; `lift_motion` is captured during the actual lift,
not after the motion completes.  The task never returns a live actor pose,
weight, friction, Young's modulus, match label, reward, success, a similarity
score, or a chosen candidate.

## Expert Data Split

The official expert calls the same public probe for all three objects.  It uses
the private match label only afterwards to place the correct candidate and
validate physical feasibility.

The calibration schedule has 56 ordered reference/distractor pairs and five
repetitions each:

```bash
python scripts/collect_data.py tactile_memory_match tactile_memory_match_slot_calibration \
  --episode_num 280 --start_seed 3000 --max_seed 3279 --gpu <GPU>
```

Before a full collection, verify that the same `Can_d4cm.usd` mesh is stable
under the soft-body constitution.  This smoke forces soft reference/match and
a rigid distractor, so it does not hide a soft-physics failure behind a random
draw:

```bash
python scripts/collect_data.py tactile_memory_match tactile_memory_match_composable_smoke \
  --episode_num 1 --start_seed 4000 --max_seed 4000 --gpu <GPU>
```

The first three repetitions of each ordered pair are the development set
(168 episodes).  The final two are the frozen independent set (112 episodes).

Normal output is intentionally compact:

```text
data/tactile_memory_match/tactile_memory_match_slot_calibration/
  metadata.json           # public v3 probe summaries only
  public_probe/<seed>.npz # public raw probe segments only
  private_metadata.json   # labels, poses, expert outcome; offline scoring only
```

## Frozen Slot Expression

Use expert development trajectories to choose one small, interpretable public
vector per slot, fit robust median/IQR scalers, and freeze the expression:

```bash
python tools/analyze_tactile_slot_expression.py \
  --data-root data/tactile_memory_match/tactile_memory_match_slot_calibration \
  --output-dir analysis/raw_distance_v3
```

The tool writes `tactile_slot_expression.v1.json`, `manifest.jsonl`,
`predictions.csv`, and `summary.json`.  It reports per-slot Hamming-distance-1
accuracy, fused identity accuracy, coverage, selective accuracy, margins, and
quality failures.  The frozen expression contains feature names, robust
scalers, and aggregate reliability only; it contains no individual trajectory,
hidden label, pose, or physical parameter.

The benchmark gates are: official expert success at least 80%, three-probe
coverage at least 90%, and each slot plus fused identity macro accuracy at
least 80% on the frozen set.  A failure from low probe coverage calls for a
probe/physics repair.  A failure with good coverage calls for a learned tactile
encoder experiment instead of further hand-vector tuning.

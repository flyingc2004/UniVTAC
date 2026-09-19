#!/usr/bin/env python3
"""Audit provisional composable-slot matching from saved public probes.

The selector uses only public ``tactile_probe.v3`` values.  Private metadata
is opened only after each prediction to score the saved hidden left/right
match assignment.  Raw NPZ arrays are intentionally not loaded.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


OBJECTS = ("reference", "candidate_left", "candidate_right")
SLOTS = {
    "weight": (
        ("lift_minus_preload", "left", "marker_displacement_px"),
        ("lift_minus_preload", "right", "marker_displacement_px"),
        ("lift_minus_preload", "left", "marker_coherence"),
        ("lift_minus_preload", "right", "marker_coherence"),
    ),
    "roughness": (
        ("lift_minus_preload", "left", "marker_displacement_px"),
        ("lift_minus_preload", "right", "marker_displacement_px"),
        ("lift_minus_preload", "left", "marker_coherence"),
        ("lift_minus_preload", "right", "marker_coherence"),
    ),
    "hardness": (
        ("lift_minus_preload", "left", "depth_mm"),
        ("lift_minus_preload", "right", "depth_mm"),
        ("lift_minus_preload", "left", "marker_displacement_px"),
        ("lift_minus_preload", "right", "marker_displacement_px"),
        ("lift_minus_preload", "left", "marker_coherence"),
        ("lift_minus_preload", "right", "marker_coherence"),
    ),
}
FAMILY = {
    ("lift_minus_preload", "left", "depth_mm"): ("lift_depth_delta_2d", 0),
    ("lift_minus_preload", "right", "depth_mm"): ("lift_depth_delta_2d", 1),
    ("lift_minus_preload", "left", "marker_displacement_px"): (
        "lift_marker_displacement_delta_2d",
        0,
    ),
    ("lift_minus_preload", "right", "marker_displacement_px"): (
        "lift_marker_displacement_delta_2d",
        1,
    ),
    ("lift_minus_preload", "left", "marker_coherence"): (
        "lift_marker_coherence_delta_2d",
        0,
    ),
    ("lift_minus_preload", "right", "marker_coherence"): (
        "lift_marker_coherence_delta_2d",
        1,
    ),
}


def _valid(probe: Any) -> bool:
    if not isinstance(probe, dict) or probe.get("schema_version") != "tactile_probe.v3":
        return False
    quality = probe.get("quality", {})
    return bool(quality.get("valid")) and int(quality.get("lift_motion_frame_count", 0)) > 0


def _value(probe: dict[str, Any], key: tuple[str, str, str]) -> float:
    section, hand, field = key
    value = float(probe[section][hand][field])
    if not np.isfinite(value):
        raise ValueError(f"non-finite public value at {section}.{hand}.{field}")
    return value


def _accuracy(rows: list[dict[str, Any]]) -> dict[str, float | int | None]:
    if not rows:
        return {"n": 0, "accuracy": None, "mean_margin": None}
    return {
        "n": len(rows),
        "accuracy": sum(bool(row["correct"]) for row in rows) / len(rows),
        "mean_margin": float(np.mean([row["margin"] for row in rows])),
    }


def _pair_report(rows: list[dict[str, Any]]) -> dict[str, dict[str, float | int | None]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[f'{row["reference_class"]}__vs__{row["distractor_class"]}'].append(row)
    return {key: _accuracy(value) for key, value in sorted(grouped.items())}


def _predict(episode: dict[str, Any], scales: dict[tuple[str, str, str], float]) -> dict[str, Any]:
    reference = episode["probes"]["reference"]
    slot_distances: dict[str, dict[str, float]] = {}
    fused: dict[str, float] = {}
    for candidate_name in ("candidate_left", "candidate_right"):
        candidate = episode["probes"][candidate_name]
        distances = {}
        for slot_name, keys in SLOTS.items():
            deltas = [(_value(reference, key) - _value(candidate, key)) / scales[key] for key in keys]
            distances[slot_name] = float(np.sqrt(np.mean(np.square(deltas))))
        slot_distances[candidate_name] = distances
        fused[candidate_name] = float(np.mean(list(distances.values())))

    selected = min(fused, key=fused.get)
    return {
        "slot_distances": slot_distances,
        "score_left": fused["candidate_left"],
        "score_right": fused["candidate_right"],
        "margin": abs(fused["candidate_left"] - fused["candidate_right"]),
        "selected": selected,
        "correct": selected == episode["match"],
    }


def _fold_scales(episodes: list[dict[str, Any]]) -> dict[tuple[str, str, str], float]:
    scales = {}
    for key in FAMILY:
        values = [_value(episode["probes"][name], key) for episode in episodes for name in OBJECTS]
        q1, q3 = np.percentile(values, [25, 75])
        scales[key] = max(float(q3 - q1), 1e-6)
    return scales


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--scaler", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    scaler = json.loads(args.scaler.read_text(encoding="utf-8"))
    if scaler.get("schema_version") != "tactile_probe_scaler.v0":
        raise ValueError("expected tactile_probe_scaler.v0")
    runtime_scales = {
        key: max(float(scaler["families"][family]["scaler"]["iqr"][index]), 1e-6)
        for key, (family, index) in FAMILY.items()
    }

    root = args.data_root.resolve()
    rejected: dict[str, int] = defaultdict(int)
    episodes = []
    for public_path in sorted(root.rglob("metadata.json")):
        private_path = public_path.with_name("private_metadata.json")
        if not private_path.exists():
            rejected["missing_private_metadata"] += 1
            continue
        try:
            public_records = json.loads(public_path.read_text(encoding="utf-8"))
            private_records = json.loads(private_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            rejected["invalid_metadata"] += 1
            continue
        for seed, public in public_records.items():
            probes = public.get("probes", {}) if isinstance(public, dict) else {}
            if not all(_valid(probes.get(name)) for name in OBJECTS):
                rejected["invalid_three_object_probe"] += 1
                continue
            audit = private_records.get(str(seed), {})
            match = audit.get("match_candidate_public_name")
            if match not in {"candidate_left", "candidate_right"}:
                rejected["missing_oracle_audit"] += 1
                continue
            episodes.append(
                {
                    "seed": int(seed),
                    "run": str(public_path.parent.relative_to(root)),
                    "reference_class": audit.get("reference_class_key"),
                    "distractor_class": audit.get("distractor_class_key"),
                    "match": match,
                    "probes": probes,
                }
            )

    if not episodes:
        raise RuntimeError("no episodes with three valid public probes and offline oracle audit")

    runtime_rows = []
    for episode in episodes:
        row = {key: value for key, value in episode.items() if key != "probes"}
        row.update(_predict(episode, runtime_scales))
        runtime_rows.append(row)

    oof_rows = []
    for fold in range(max(2, args.folds)):
        train = [episode for episode in episodes if episode["seed"] % args.folds != fold]
        test = [episode for episode in episodes if episode["seed"] % args.folds == fold]
        if not train or not test:
            continue
        scales = _fold_scales(train)
        for episode in test:
            row = {key: value for key, value in episode.items() if key != "probes"}
            row["fold"] = fold
            row.update(_predict(episode, scales))
            oof_rows.append(row)

    report = {
        "schema_version": "provisional_slot_v0_validation.v1",
        "valid_episodes": len(episodes),
        "rejected": dict(sorted(rejected.items())),
        "runtime_scaler_diagnostic": {**_accuracy(runtime_rows), "per_pair": _pair_report(runtime_rows), "predictions": runtime_rows},
        "five_fold_out_of_fold": {**_accuracy(oof_rows), "per_pair": _pair_report(oof_rows), "predictions": oof_rows},
        "caveat": (
            "Only public tactile_probe.v3 fields are used for feature values and scaling. "
            "Private match assignment is read after prediction for offline audit. "
            "Weight and roughness intentionally share marker-delta features in provisional_slot_expression.v0, "
            "so this fusion double-counts that family."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "valid_episodes": len(episodes),
        "runtime_scaler": _accuracy(runtime_rows),
        "five_fold_oof": _accuracy(oof_rows),
        "rejected": dict(sorted(rejected.items())),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

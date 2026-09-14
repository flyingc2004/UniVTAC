#!/usr/bin/env python3
"""Audit the historical expert comparator without using raw frames or GT labels as features.

The task metadata contains the expert's already-recorded static and dynamic
signatures.  This script compares three modes: the original expert score,
the static public signature only, and a dynamic-public ablation that removes
all actor-pose-derived terms from the historical weight response.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PUBLIC_DYNAMIC_FEATURES = (
    ("depth_delta_drop", 6.0, 0.75),
    ("contact_area_drop", 0.50, 0.75),
    ("after_depth_delta_mm", 6.0, 0.75),
    ("after_contact_area", 0.50, 0.75),
    ("gripper_qpos_after", 0.02, 0.50),
)


def _best(scores: dict[str, Any]) -> str | None:
    if set(scores) != {"candidate_left", "candidate_right"}:
        return None
    if not all(math.isfinite(float(value)) for value in scores.values()):
        return None
    return min(scores, key=lambda key: float(scores[key]))


def _public_dynamic_distance(reference: dict[str, Any], candidate: dict[str, Any]) -> float:
    if not reference.get("valid", False) or not candidate.get("valid", False):
        return float("inf")

    penalty = 0.0
    if bool(reference.get("contact_lost", False)) != bool(candidate.get("contact_lost", False)):
        penalty += 1.0
    if bool(reference.get("after_both_contact", False)) != bool(candidate.get("after_both_contact", False)):
        penalty += 0.5

    weighted_sum = 0.0
    weight_sum = 0.0
    for key, scale, weight in PUBLIC_DYNAMIC_FEATURES:
        ref_value = reference.get(key)
        candidate_value = candidate.get(key)
        if ref_value is None or candidate_value is None:
            continue
        ref_value = float(ref_value)
        candidate_value = float(candidate_value)
        if not math.isfinite(ref_value) or not math.isfinite(candidate_value):
            continue
        normalized = (candidate_value - ref_value) / scale
        weighted_sum += weight * normalized * normalized
        weight_sum += weight
    if weight_sum == 0.0:
        return float("inf")
    return math.sqrt(weighted_sum / weight_sum) + penalty


def _accuracy(rows: list[dict[str, Any]], prediction_key: str) -> dict[str, Any]:
    valid = [row for row in rows if row[prediction_key] is not None]
    correct = [row for row in valid if row[prediction_key] == row["match_candidate"]]
    by_class: dict[str, list[bool]] = defaultdict(list)
    for row in valid:
        by_class[str(row["reference_class"])].append(row[prediction_key] == row["match_candidate"])
    per_class = {name: sum(values) / len(values) for name, values in sorted(by_class.items())}
    return {
        "valid_episodes": len(valid),
        "correct": len(correct),
        "overall_accuracy": len(correct) / len(valid) if valid else None,
        "per_reference_class_accuracy": per_class,
        "macro_accuracy": sum(per_class.values()) / len(per_class) if per_class else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    metadata_path = args.dataset_root / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for seed, item in sorted(metadata.items(), key=lambda pair: int(pair[0])):
        static_scores = item.get("tactile_static_similarity_scores") or {}
        full_scores = item.get("tactile_similarity_scores") or {}
        reference_dynamic = item.get("tactile_weight_signature_reference") or {}
        dynamic_scores = {
            name: _public_dynamic_distance(
                reference_dynamic,
                item.get(f"tactile_weight_signature_{name}") or {},
            )
            for name in ("candidate_left", "candidate_right")
        }
        rows.append(
            {
                "seed": int(seed),
                "reference_class": item.get("reference_class"),
                "distractor_class": item.get("distractor_class"),
                "match_candidate": item.get("match_candidate_public_name"),
                "recorded_selection": item.get("selected_candidate"),
                "static_only": _best(static_scores),
                "expert_full": _best(full_scores),
                "public_dynamic": _best(dynamic_scores),
                "static_left": static_scores.get("candidate_left"),
                "static_right": static_scores.get("candidate_right"),
                "full_left": full_scores.get("candidate_left"),
                "full_right": full_scores.get("candidate_right"),
                "public_dynamic_left": dynamic_scores["candidate_left"],
                "public_dynamic_right": dynamic_scores["candidate_right"],
                "task_result": item.get("result"),
                "failure_stage": item.get("failure_stage"),
            }
        )

    output_dir = args.output_dir or args.dataset_root / "analysis" / "expert_memory_audit_v1"
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "schema_version": "expert_memory_audit.v1",
        "dataset_root": str(args.dataset_root.resolve()),
        "episode_count": len(rows),
        "task_results": dict(Counter(str(row["task_result"]) for row in rows)),
        "methods": {
            method: _accuracy(rows, method)
            for method in ("recorded_selection", "static_only", "public_dynamic", "expert_full")
        },
        "note": (
            "expert_full includes actor-pose-derived terms through the historical weight response; "
            "public_dynamic excludes object_lift_delta, lift_follow_ratio, inhand_z_error, and gripper_lift_delta."
        ),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

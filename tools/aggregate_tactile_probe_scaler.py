#!/usr/bin/env python3
"""Aggregate public tactile_probe.v3 summaries into a provisional scaler.

This utility intentionally reads only ``metadata.json`` files.  The raw NPZ
arrays remain untouched, so it is safe to run while a multi-GPU collection is
still producing large probe archives.  The result is a task/protocol-local
Median/IQR normalizer, not a classifier and not a frozen benchmark expression.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


OBJECT_NAMES = ("reference", "candidate_left", "candidate_right")
FEATURE_FAMILIES: dict[str, tuple[str, str]] = {
    "preload_depth_2d": ("preload", "depth_mm"),
    "preload_marker_displacement_2d": ("preload", "marker_displacement_px"),
    "preload_marker_coherence_2d": ("preload", "marker_coherence"),
    "lift_depth_delta_2d": ("lift_minus_preload", "depth_mm"),
    "lift_marker_displacement_delta_2d": ("lift_minus_preload", "marker_displacement_px"),
    "lift_marker_coherence_delta_2d": ("lift_minus_preload", "marker_coherence"),
}


def _valid_probe(probe: Any) -> bool:
    if not isinstance(probe, dict) or probe.get("schema_version") != "tactile_probe.v3":
        return False
    quality = probe.get("quality", {})
    return bool(quality.get("valid")) and int(quality.get("lift_motion_frame_count", 0)) > 0


def _vector(probe: dict[str, Any], section: str, field: str) -> list[float]:
    values = [float(probe[section][hand][field]) for hand in ("left", "right")]
    if not np.isfinite(values).all():
        raise ValueError("non-finite public tactile value")
    return values


def _scaler(vectors: list[list[float]], min_iqr: float) -> dict[str, list[float]]:
    matrix = np.asarray(vectors, dtype=np.float64)
    median = np.median(matrix, axis=0)
    q1, q3 = np.percentile(matrix, [25, 75], axis=0)
    return {
        "median": median.tolist(),
        "iqr": np.maximum(q3 - q1, min_iqr).tolist(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Pilot root containing one or more run subdirectories with metadata.json.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-iqr", type=float, default=1e-6)
    args = parser.parse_args()

    root = args.data_root.expanduser().resolve()
    metadata_files = sorted(root.rglob("metadata.json"))
    if not metadata_files:
        raise FileNotFoundError(f"No metadata.json found below {root}")

    vectors = {name: [] for name in FEATURE_FAMILIES}
    diagnostics: Counter[str] = Counter()
    accepted: list[dict[str, Any]] = []
    seen = set()

    for metadata_path in metadata_files:
        try:
            records = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            diagnostics[f"invalid_metadata:{type(error).__name__}"] += 1
            continue
        if not isinstance(records, dict):
            diagnostics["metadata_not_mapping"] += 1
            continue
        for seed, episode in records.items():
            if not isinstance(episode, dict):
                diagnostics["episode_not_mapping"] += 1
                continue
            key = (str(metadata_path.resolve()), str(seed))
            if key in seen:
                continue
            seen.add(key)
            probes = episode.get("probes", {})
            if not isinstance(probes, dict) or not all(_valid_probe(probes.get(name)) for name in OBJECT_NAMES):
                diagnostics["episode_missing_valid_three_probe"] += 1
                continue
            try:
                episode_vectors = {name: [] for name in FEATURE_FAMILIES}
                for family, (section, field) in FEATURE_FAMILIES.items():
                    for object_name in OBJECT_NAMES:
                        episode_vectors[family].append(_vector(probes[object_name], section, field))
            except (KeyError, TypeError, ValueError) as error:
                diagnostics[f"invalid_public_values:{type(error).__name__}"] += 1
                continue
            for family, values in episode_vectors.items():
                vectors[family].extend(values)
            accepted.append({"metadata": str(metadata_path.relative_to(root)), "seed": str(seed)})

    if not accepted:
        raise RuntimeError("No episodes with valid reference/left/right tactile_probe.v3 records")

    output = {
        "schema_version": "tactile_probe_scaler.v0",
        "scope": "provisional_task_protocol_local",
        "data_root": str(root),
        "protocol": "tactile_probe.v3",
        "episode_count": len(accepted),
        "object_vector_count_per_family": len(accepted) * len(OBJECT_NAMES),
        "families": {
            family: {
                "section": section,
                "field": field,
                "hands": ["left", "right"],
                "scaler": _scaler(vectors[family], args.min_iqr),
            }
            for family, (section, field) in FEATURE_FAMILIES.items()
        },
        "diagnostics": dict(sorted(diagnostics.items())),
        "accepted_episodes": accepted,
        "notes": [
            "Computed from public tactile_probe.v3 summaries only; raw NPZ archives are not read.",
            "Median cancels in direct reference-candidate differences; per-feature IQR sets relative distance scale.",
            "This is not a classifier, centroid, task-success signal, or final cross-task expression.",
            "An unbalanced pilot is acceptable for an engineering smoke, not for frozen generalization claims.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "episodes": len(accepted),
                "vectors_per_family": len(accepted) * len(OBJECT_NAMES),
                "diagnostics": dict(sorted(diagnostics.items())),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

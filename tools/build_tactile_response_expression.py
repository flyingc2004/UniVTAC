#!/usr/bin/env python3
"""Freeze a model-agnostic tactile response expression from public v4 probes.

The tool consumes only ``tactile_memory_match_public_episode.v4`` metadata.
It never opens private metadata, actor poses, physical class labels, rewards,
or task-success fields.  Its output is a read-only Median/IQR normalizer for
the standard public probe, intended for downstream agents to own memory and
reference-to-candidate comparison.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


EXPRESSION_SCHEMA = "tactile_response_expression.v1"
PUBLIC_PROBE_SCHEMA = "tactile_probe.v4"
PUBLIC_EPISODE_SCHEMA = "tactile_memory_match_public_episode.v4"
OBJECTS = ("reference", "candidate_left", "candidate_right")


def _field(path: str, *, transform: str = "identity", snr_paths: tuple[str, ...]) -> dict[str, Any]:
    return {"path": path, "transform": transform, "snr_paths": list(snr_paths)}


# Response blocks deliberately describe measured contact modes rather than
# semantic properties.  This avoids double-counting one marker response as
# both a "weight" and a "roughness" slot during runtime comparison.
RESPONSE_BLOCKS: tuple[dict[str, Any], ...] = (
    {
        "name": "normal_static",
        "description": "Bilateral preload indentation.",
        "quality_sections": ("preload",),
        "fields": (
            _field("preload.left.depth_mm", snr_paths=("preload.noise.left.depth_mm.snr",)),
            _field("preload.right.depth_mm", snr_paths=("preload.noise.right.depth_mm.snr",)),
        ),
    },
    {
        "name": "normal_dynamic",
        "description": "Bilateral lift-minus-preload indentation response.",
        "quality_sections": ("preload", "lift_motion"),
        "fields": (
            _field(
                "lift_minus_preload.left.depth_mm",
                snr_paths=("preload.noise.left.depth_mm.snr", "lift_motion.noise.left.depth_mm.snr"),
            ),
            _field(
                "lift_minus_preload.right.depth_mm",
                snr_paths=("preload.noise.right.depth_mm.snr", "lift_motion.noise.right.depth_mm.snr"),
            ),
        ),
    },
    {
        "name": "surface_spatial_static",
        "description": "Bilateral preload marker-grid spatial deformation.",
        "quality_sections": ("preload",),
        "fields": (
            _field(
                "preload.left.marker_row_gradient_px",
                snr_paths=("preload.noise.left.marker_row_gradient_px.snr",),
            ),
            _field(
                "preload.right.marker_row_gradient_px",
                snr_paths=("preload.noise.right.marker_row_gradient_px.snr",),
            ),
            _field(
                "preload.left.marker_col_gradient_px",
                snr_paths=("preload.noise.left.marker_col_gradient_px.snr",),
            ),
            _field(
                "preload.right.marker_col_gradient_px",
                snr_paths=("preload.noise.right.marker_col_gradient_px.snr",),
            ),
            _field(
                "preload.left.marker_anisotropy_ratio",
                transform="log",
                snr_paths=("preload.noise.left.marker_anisotropy_ratio.snr",),
            ),
            _field(
                "preload.right.marker_anisotropy_ratio",
                transform="log",
                snr_paths=("preload.noise.right.marker_anisotropy_ratio.snr",),
            ),
        ),
    },
)


def _path_value(source: dict[str, Any], path: str) -> Any:
    value: Any = source
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            raise KeyError(path)
        value = value[part]
    return value


def _transform(value: Any, transform: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("non-finite public tactile value")
    if transform == "identity":
        return number
    if transform == "log":
        if number <= 0.0:
            raise ValueError("log transform requires a positive public tactile value")
        return float(math.log(number))
    raise ValueError(f"unknown public response transform {transform!r}")


def _probe_is_valid(probe: Any) -> bool:
    if not isinstance(probe, dict) or probe.get("schema_version") != PUBLIC_PROBE_SCHEMA:
        return False
    quality = probe.get("quality", {})
    return bool(quality.get("valid")) and int(quality.get("lift_motion_frame_count", 0)) > 0


def _block_vector(probe: dict[str, Any], block: dict[str, Any]) -> list[float]:
    return [_transform(_path_value(probe, field["path"]), field["transform"]) for field in block["fields"]]


def _robust_scaler(vectors: list[list[float]], minimum_iqr: float) -> dict[str, list[float]]:
    matrix = np.asarray(vectors, dtype=np.float64)
    median = np.median(matrix, axis=0)
    q1, q3 = np.percentile(matrix, [25, 75], axis=0)
    return {
        "median": median.tolist(),
        "iqr": np.maximum(q3 - q1, float(minimum_iqr)).tolist(),
    }


def _load_public_episodes(root: Path) -> tuple[list[dict[str, Any]], Counter[str]]:
    accepted: list[dict[str, Any]] = []
    rejected: Counter[str] = Counter()
    for metadata_path in sorted(root.rglob("metadata.json")):
        try:
            records = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            rejected[f"invalid_metadata:{type(error).__name__}"] += 1
            continue
        if not isinstance(records, dict):
            rejected["metadata_not_mapping"] += 1
            continue
        for seed, episode in records.items():
            if not isinstance(episode, dict) or episode.get("schema_version") != PUBLIC_EPISODE_SCHEMA:
                rejected["non_v4_public_episode"] += 1
                continue
            probes = episode.get("probes", {})
            if not isinstance(probes, dict) or not all(_probe_is_valid(probes.get(name)) for name in OBJECTS):
                rejected["invalid_three_object_probe"] += 1
                continue
            try:
                protocol_id = str(probes["reference"]["protocol_id"])
                if any(str(probes[name]["protocol_id"]) != protocol_id for name in OBJECTS):
                    raise ValueError("mixed protocol ids")
                for probe in (probes[name] for name in OBJECTS):
                    for block in RESPONSE_BLOCKS:
                        _block_vector(probe, block)
            except (KeyError, TypeError, ValueError) as error:
                rejected[f"invalid_public_response:{type(error).__name__}"] += 1
                continue
            accepted.append(
                {
                    "metadata": str(metadata_path.relative_to(root)),
                    "seed": str(seed),
                    "protocol_id": protocol_id,
                    "probes": probes,
                }
            )
    return accepted, rejected


def build_expression(root: Path, minimum_iqr: float) -> dict[str, Any]:
    episodes, rejected = _load_public_episodes(root)
    if not episodes:
        raise RuntimeError("No valid tactile_probe.v4 reference/left/right episodes were found")
    protocol_ids = sorted({episode["protocol_id"] for episode in episodes})
    if len(protocol_ids) != 1:
        raise RuntimeError(f"Expected one public probe protocol, found {protocol_ids}")

    blocks = []
    for source_block in RESPONSE_BLOCKS:
        vectors = [
            _block_vector(episode["probes"][object_name], source_block)
            for episode in episodes
            for object_name in OBJECTS
        ]
        blocks.append(
            {
                "name": source_block["name"],
                "description": source_block["description"],
                "fields": [dict(field) for field in source_block["fields"]],
                "quality_sections": list(source_block["quality_sections"]),
                "scaler": _robust_scaler(vectors, minimum_iqr),
            }
        )

    return {
        "schema_version": EXPRESSION_SCHEMA,
        "scope": "development_calibrated_public_response",
        "public_episode_schema_version": PUBLIC_EPISODE_SCHEMA,
        "public_probe_schema_version": PUBLIC_PROBE_SCHEMA,
        "protocol_id": protocol_ids[0],
        "episode_count": len(episodes),
        "object_vector_count_per_block": len(episodes) * len(OBJECTS),
        "response_blocks": blocks,
        "quality_rule": {
            "requires_probe_valid": True,
            "candidate_confidence": "minimum_reference_candidate_block_confidence",
            "block_confidence": "minimum_bilateral_ratio_times_median_declared_snr",
            "snr_cap": 10.0,
        },
        "comparison_rule": {
            "version": "quality_weighted_block_rms.v1",
            "block_distance": "rms((reference-candidate)/iqr)",
            "fused_distance": "sum(candidate_confidence*block_distance)/sum(candidate_confidence)",
            "tie_break": "candidate_left",
        },
        "accepted_episodes": [
            {key: episode[key] for key in ("metadata", "seed")} for episode in episodes
        ],
        "diagnostics": dict(sorted(rejected.items())),
        "notes": [
            "Only public tactile_probe.v4 values were read; no labels or private metadata were opened.",
            "This expression is protocol/sensor-local development calibration, not a cross-task generalization claim.",
            "Response blocks describe normal and spatial contact response; they are not mutually independent semantic attributes.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-iqr", type=float, default=1e-6)
    args = parser.parse_args()
    if args.minimum_iqr <= 0:
        parser.error("--minimum-iqr must be positive")

    expression = build_expression(args.data_root.expanduser().resolve(), args.minimum_iqr)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(expression, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "episodes": expression["episode_count"],
                "protocol_id": expression["protocol_id"],
                "blocks": [block["name"] for block in expression["response_blocks"]],
                "diagnostics": expression["diagnostics"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

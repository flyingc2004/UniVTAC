#!/usr/bin/env python3
"""Re-extract directional marker features from public tactile_probe.v3 raw NPZ.

This is an offline diagnostic for ``tactile_memory_match``.  It reads only
public raw tactile probe recordings to form descriptors, then reads the
private match assignment *after prediction* to score reference-to-candidate
matching.  It does not use pose, density, friction, hardness, reward, or task
success as features.

The tool compares the current magnitude/coherence delta descriptor with
direction-preserving mean marker ``dx/dy``, cross-finger shear, depth, and a
diagnostic grid-gradient anisotropy.  The anisotropy assumes TacEx's 64 marker
indices form the same row-major 8x8 grid already used by the existing raw
analysis tools; it is deliberately labelled ``v0`` rather than treated as an
exact reproduction of another implementation.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


OBJECTS = ("reference", "candidate_left", "candidate_right")
HANDS = ("left", "right")
SEGMENTS = ("preload", "lift_motion")
SCHEMA_VERSION = "marker_feature_reextraction.v1"


def _finite_median(values: np.ndarray, label: str) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError(f"{label} contains no finite values")
    return float(np.median(values))


def _depth_frame_metrics(depth: np.ndarray, far_plane_mm: float) -> dict[str, float]:
    values = np.asarray(depth, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("depth frame contains no finite values")
    indentation = np.clip(float(far_plane_mm) - values, 0.0, None)
    return {
        # Existing v3 summary: robust indentation, equivalent to far_plane - p5(depth).
        "depth_p5_mm": float(np.percentile(indentation, 95.0)),
        # Candidate for the colleague's min(depth) formulation; intentionally less robust.
        "depth_min_mm": float(np.max(indentation)),
    }


def _marker_frame_metrics(marker: np.ndarray) -> dict[str, float]:
    marker = np.asarray(marker, dtype=np.float64)
    if marker.shape != (2, 64, 2):
        raise ValueError(f"expected marker shape (2, 64, 2), received {marker.shape}")
    flow = marker[1] - marker[0]
    valid = flow[np.isfinite(flow).all(axis=1)]
    if valid.size == 0:
        raise ValueError("marker frame contains no finite flow vectors")

    magnitudes = np.linalg.norm(valid, axis=1)
    mean_magnitude = float(np.mean(magnitudes))
    mean_flow = np.mean(valid, axis=0)
    coherence = 0.0 if mean_magnitude <= 1e-12 else float(np.linalg.norm(mean_flow) / mean_magnitude)

    # This is a spatial deformation statistic, not a marker-flow direction.
    # TacEx marker ordering is assumed to be row-major 8x8, as in the existing
    # raw descriptor extractor.  The signed form avoids an unbounded ratio.
    grid = flow.reshape(8, 8, 2)
    row_gradient = float(np.sqrt(np.nanmean(np.square(np.diff(grid, axis=0)))))
    col_gradient = float(np.sqrt(np.nanmean(np.square(np.diff(grid, axis=1)))))
    denominator = row_gradient + col_gradient
    signed_anisotropy = 0.0 if denominator <= 1e-12 else float((row_gradient - col_gradient) / denominator)

    return {
        "marker_dx_px": float(mean_flow[0]),
        "marker_dy_px": float(mean_flow[1]),
        "marker_displacement_px": mean_magnitude,
        "marker_coherence": float(np.clip(coherence, 0.0, 1.0)),
        "marker_row_gradient_px": row_gradient,
        "marker_col_gradient_px": col_gradient,
        "marker_anisotropy_signed_v0": float(np.clip(signed_anisotropy, -1.0, 1.0)),
        "marker_anisotropy_ratio_v0": float(row_gradient / max(col_gradient, 1e-12)),
    }


def _segment_metrics(npz: Any, object_name: str, segment: str, hand: str, far_plane_mm: float) -> dict[str, float]:
    prefix = f"{object_name}__{segment}__{hand}"
    depth_key = f"{prefix}_depth"
    marker_key = f"{prefix}_marker"
    if depth_key not in npz or marker_key not in npz:
        raise ValueError(f"missing raw arrays {depth_key!r} or {marker_key!r}")
    depth_frames = np.asarray(npz[depth_key])
    marker_frames = np.asarray(npz[marker_key])
    if len(depth_frames) == 0 or len(marker_frames) == 0:
        raise ValueError(f"empty raw segment {object_name}/{segment}/{hand}")
    if len(depth_frames) != len(marker_frames):
        raise ValueError(f"frame count mismatch in {object_name}/{segment}/{hand}")

    per_frame = []
    for depth, marker in zip(depth_frames, marker_frames, strict=True):
        row = _depth_frame_metrics(depth, far_plane_mm)
        row.update(_marker_frame_metrics(marker))
        per_frame.append(row)
    return {name: _finite_median(np.asarray([row[name] for row in per_frame]), name) for name in per_frame[0]}


def _object_features(npz: Any, object_name: str, far_plane_mm: float) -> dict[str, float]:
    values: dict[str, float] = {}
    for segment in SEGMENTS:
        for hand in HANDS:
            for name, value in _segment_metrics(npz, object_name, segment, hand, far_plane_mm).items():
                values[f"{segment}.{hand}.{name}"] = value
    for hand in HANDS:
        for name in (
            "depth_p5_mm",
            "depth_min_mm",
            "marker_dx_px",
            "marker_dy_px",
            "marker_displacement_px",
            "marker_coherence",
            "marker_row_gradient_px",
            "marker_col_gradient_px",
            "marker_anisotropy_signed_v0",
            "marker_anisotropy_ratio_v0",
        ):
            values[f"lift_minus_preload.{hand}.{name}"] = (
                values[f"lift_motion.{hand}.{name}"] - values[f"preload.{hand}.{name}"]
            )
    return values


def _hand_fields(segment: str, metric: str) -> tuple[str, str]:
    return tuple(f"{segment}.{hand}.{metric}" for hand in HANDS)


def _delta_fields(metric: str) -> tuple[str, str]:
    return tuple(f"lift_minus_preload.{hand}.{metric}" for hand in HANDS)


DESCRIPTORS: dict[str, tuple[str, ...]] = {
    # Exact family currently used by the provisional v0 expression.
    "current_marker_delta_4d": _delta_fields("marker_displacement_px") + _delta_fields("marker_coherence"),
    # Preserves the mean marker flow vector which the current public summary discards.
    "directional_flow_8d": (
        _hand_fields("preload", "marker_dx_px")
        + _hand_fields("preload", "marker_dy_px")
        + _delta_fields("marker_dx_px")
        + _delta_fields("marker_dy_px")
    ),
    # Colleague-compatible cross-finger x shear: static and lift-minus-preload.
    "cross_finger_x_shear_2d": (),
    "cross_finger_xy_shear_4d": (),
    # Candidate row/column gradient interpretation of the screenshot's anisotropy term.
    "grid_gradient_anisotropy_4d": (
        _hand_fields("preload", "marker_anisotropy_signed_v0")
        + _delta_fields("marker_anisotropy_signed_v0")
    ),
    "depth_p5_4d": _hand_fields("preload", "depth_p5_mm") + _delta_fields("depth_p5_mm"),
    "depth_min_4d": _hand_fields("preload", "depth_min_mm") + _delta_fields("depth_min_mm"),
    # Deliberately exposed only as a diagnostic combination, never as a frozen expression.
    "combined_direction_depth_anisotropy_16d": (
        _hand_fields("preload", "marker_dx_px")
        + _hand_fields("preload", "marker_dy_px")
        + _delta_fields("marker_dx_px")
        + _delta_fields("marker_dy_px")
        + _hand_fields("preload", "depth_p5_mm")
        + _delta_fields("depth_p5_mm")
        + _hand_fields("preload", "marker_anisotropy_signed_v0")
        + _delta_fields("marker_anisotropy_signed_v0")
    ),
}

# Candidate image-to-common-contact-frame transforms for the *right* sensor.
# They span the 2D mirror/axis-swap possibilities caused by opposite sensor
# mounting.  This is a diagnostic sweep only: a production protocol must fix
# the transform from saved sensor poses and a label-free calibration motion.
RIGHT_FLOW_TRANSFORMS: dict[str, np.ndarray] = {
    "identity": np.array(((1.0, 0.0), (0.0, 1.0))),
    "flip_x": np.array(((-1.0, 0.0), (0.0, 1.0))),
    "flip_y": np.array(((1.0, 0.0), (0.0, -1.0))),
    "flip_xy": np.array(((-1.0, 0.0), (0.0, -1.0))),
    "swap_xy": np.array(((0.0, 1.0), (1.0, 0.0))),
    "swap_flip_x": np.array(((0.0, -1.0), (1.0, 0.0))),
    "swap_flip_y": np.array(((0.0, 1.0), (-1.0, 0.0))),
    "swap_flip_xy": np.array(((0.0, -1.0), (-1.0, 0.0))),
}
for _transform_name in RIGHT_FLOW_TRANSFORMS:
    DESCRIPTORS[f"cross_finger_xy_right_{_transform_name}_4d"] = ()

# Each entry is (base descriptor, dynamic outlier gate in training-fold IQRs).
# We report a small sensitivity sweep rather than silently treating one
# hand-picked threshold as a final measurement rule.
GATED_DESCRIPTORS = {
    "grid_gradient_anisotropy_gate_4iqr": ("grid_gradient_anisotropy_4d", 4.0),
    "grid_gradient_anisotropy_gate_8iqr": ("grid_gradient_anisotropy_4d", 8.0),
    "grid_gradient_anisotropy_gate_12iqr": ("grid_gradient_anisotropy_4d", 12.0),
}


def _base_descriptor_name(descriptor_name: str) -> str:
    return GATED_DESCRIPTORS.get(descriptor_name, (descriptor_name, None))[0]


def _is_right_transform_descriptor(descriptor_name: str) -> bool:
    return descriptor_name.startswith("cross_finger_xy_right_") and descriptor_name.endswith("_4d")


def _right_transform_name(descriptor_name: str) -> str:
    prefix = "cross_finger_xy_right_"
    return descriptor_name[len(prefix) : -len("_4d")]


def _vector(features: dict[str, float], descriptor_name: str) -> np.ndarray:
    descriptor_name = _base_descriptor_name(descriptor_name)
    if _is_right_transform_descriptor(descriptor_name):
        transform_name = _right_transform_name(descriptor_name)
        transform = RIGHT_FLOW_TRANSFORMS[transform_name]

        def cross_finger(phase: str) -> np.ndarray:
            left = np.asarray(
                [
                    features[f"{phase}.left.marker_dx_px"],
                    features[f"{phase}.left.marker_dy_px"],
                ],
                dtype=np.float64,
            )
            right = np.asarray(
                [
                    features[f"{phase}.right.marker_dx_px"],
                    features[f"{phase}.right.marker_dy_px"],
                ],
                dtype=np.float64,
            )
            return transform @ right - left

        return np.concatenate((cross_finger("preload"), cross_finger("lift_minus_preload")))
    if descriptor_name == "cross_finger_x_shear_2d":
        return np.asarray(
            [
                features["preload.right.marker_dx_px"] - features["preload.left.marker_dx_px"],
                features["lift_minus_preload.right.marker_dx_px"]
                - features["lift_minus_preload.left.marker_dx_px"],
            ],
            dtype=np.float64,
        )
    if descriptor_name == "cross_finger_xy_shear_4d":
        return np.asarray(
            [
                features["preload.right.marker_dx_px"] - features["preload.left.marker_dx_px"],
                features["preload.right.marker_dy_px"] - features["preload.left.marker_dy_px"],
                features["lift_minus_preload.right.marker_dx_px"]
                - features["lift_minus_preload.left.marker_dx_px"],
                features["lift_minus_preload.right.marker_dy_px"]
                - features["lift_minus_preload.left.marker_dy_px"],
            ],
            dtype=np.float64,
        )
    fields = DESCRIPTORS[descriptor_name]
    return np.asarray([features[field] for field in fields], dtype=np.float64)


def _descriptor_fields(descriptor_name: str) -> tuple[str, ...] | None:
    """Return direct public feature fields, or None for derived shear vectors."""
    base_name = _base_descriptor_name(descriptor_name)
    if base_name in {"cross_finger_x_shear_2d", "cross_finger_xy_shear_4d"} or _is_right_transform_descriptor(base_name):
        return None
    return DESCRIPTORS[base_name]


def _fit_scaler(episodes: list[dict[str, Any]], descriptor_name: str) -> dict[str, list[float]]:
    matrix = np.stack(
        [_vector(episode["features"][object_name], descriptor_name) for episode in episodes for object_name in OBJECTS]
    )
    median = np.median(matrix, axis=0)
    q1, q3 = np.percentile(matrix, [25, 75], axis=0)
    return {"median": median.tolist(), "iqr": np.maximum(q3 - q1, 1e-6).tolist()}


def _predict(episode: dict[str, Any], descriptor_name: str, scaler: dict[str, list[float]]) -> dict[str, Any]:
    scale = np.asarray(scaler["iqr"], dtype=np.float64)
    reference = _vector(episode["features"]["reference"], descriptor_name)
    fields = _descriptor_fields(descriptor_name)
    threshold = GATED_DESCRIPTORS.get(descriptor_name, (None, None))[1]
    active = np.ones(len(reference), dtype=bool)
    omitted_fields: list[str] = []
    if threshold is not None:
        if fields is None:
            raise RuntimeError(f"Dynamic gate is unsupported for derived descriptor {descriptor_name}")
        median = np.asarray(scaler["median"], dtype=np.float64)
        # A dynamic field is accepted only if reference, left, and right are all
        # within the same training-fold robustness envelope.  This prevents one
        # candidate from receiving a shorter/easier distance than the other.
        for index, field in enumerate(fields):
            if not field.startswith("lift_minus_preload."):
                continue
            z_scores = [
                abs((_vector(episode["features"][name], descriptor_name)[index] - median[index]) / scale[index])
                for name in OBJECTS
            ]
            if max(z_scores) > threshold:
                active[index] = False
                omitted_fields.append(field)
        if not np.any(active):
            raise ValueError(f"No public feature dimensions remain after {descriptor_name} quality gate")

    scores = {}
    for candidate in ("candidate_left", "candidate_right"):
        vector = _vector(episode["features"][candidate], descriptor_name)
        normalized_delta = ((reference - vector) / scale)[active]
        # Mean-squared normalization keeps the two candidate scores comparable
        # when a shared low-quality dynamic field is omitted.
        scores[candidate] = float(np.sqrt(np.mean(np.square(normalized_delta))))
    selected = min(scores, key=scores.get)
    return {
        "score_left": scores["candidate_left"],
        "score_right": scores["candidate_right"],
        "margin": abs(scores["candidate_left"] - scores["candidate_right"]),
        "selected": selected,
        "correct": bool(selected == episode["match_candidate"]),
        "used_dimensions": int(np.count_nonzero(active)),
        "omitted_dynamic_fields": omitted_fields,
    }


def _summary(rows: list[dict[str, Any]]) -> dict[str, float | int | None]:
    if not rows:
        return {"n": 0, "accuracy": None, "mean_margin": None}
    return {
        "n": len(rows),
        "accuracy": float(np.mean([row["correct"] for row in rows])),
        "mean_margin": float(np.mean([row["margin"] for row in rows])),
    }


def _slot_changes(reference_class: str | None, distractor_class: str | None) -> str:
    if not reference_class or not distractor_class:
        return "unknown"
    reference = str(reference_class).split("_")
    distractor = str(distractor_class).split("_")
    if len(reference) != 3 or len(distractor) != 3:
        return "unknown"
    slots = ("weight", "roughness", "hardness")
    return "+".join(slot for slot, left, right in zip(slots, reference, distractor, strict=True) if left != right)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_episodes(data_root: Path, far_plane_mm: float) -> tuple[list[dict[str, Any]], Counter[str], list[dict[str, Any]]]:
    episodes: list[dict[str, Any]] = []
    rejected: Counter[str] = Counter()
    manifest: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for metadata_path in sorted(data_root.rglob("metadata.json")):
        private_path = metadata_path.with_name("private_metadata.json")
        if not private_path.exists():
            rejected["missing_private_metadata"] += 1
            continue
        try:
            public_records = json.loads(metadata_path.read_text(encoding="utf-8"))
            private_records = json.loads(private_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            rejected["invalid_metadata_json"] += 1
            manifest.append({"path": str(metadata_path), "accepted": False, "reason": repr(exc)})
            continue
        if not isinstance(public_records, dict) or not isinstance(private_records, dict):
            rejected["metadata_not_mapping"] += 1
            continue
        for seed_text, public in public_records.items():
            run = str(metadata_path.parent.relative_to(data_root))
            unique_key = (run, str(seed_text))
            if unique_key in seen:
                continue
            seen.add(unique_key)
            base = {"run": run, "seed": str(seed_text), "metadata_path": str(metadata_path)}
            if not isinstance(public, dict) or public.get("schema_version") not in {
                "tactile_memory_match_public_episode.v3",
                "tactile_memory_match_public_episode.v4",
            }:
                rejected["not_public_probe_v3"] += 1
                manifest.append({**base, "accepted": False, "reason": "not_public_probe_v3"})
                continue
            probes = public.get("probes", {})
            if not all(isinstance(probes.get(name), dict) and probes[name].get("quality", {}).get("valid") for name in OBJECTS):
                rejected["invalid_three_object_probe"] += 1
                manifest.append({**base, "accepted": False, "reason": "invalid_three_object_probe"})
                continue
            private = private_records.get(str(seed_text), {})
            match = private.get("match_candidate_public_name") if isinstance(private, dict) else None
            if match not in {"candidate_left", "candidate_right"}:
                rejected["missing_offline_match_audit"] += 1
                manifest.append({**base, "accepted": False, "reason": "missing_offline_match_audit"})
                continue
            raw_relative = public.get("raw_probe_path")
            raw_path = metadata_path.parent / str(raw_relative) if raw_relative else None
            if raw_path is None or not raw_path.exists():
                rejected["missing_raw_npz"] += 1
                manifest.append({**base, "accepted": False, "reason": "missing_raw_npz", "raw_path": str(raw_path)})
                continue
            try:
                with np.load(raw_path, allow_pickle=False) as npz:
                    features = {name: _object_features(npz, name, far_plane_mm) for name in OBJECTS}
            except (OSError, ValueError, KeyError) as exc:
                rejected["invalid_raw_probe"] += 1
                manifest.append({**base, "accepted": False, "reason": f"invalid_raw_probe:{exc}", "raw_path": str(raw_path)})
                continue
            episode = {
                **base,
                "seed": int(seed_text),
                "raw_path": str(raw_path),
                "match_candidate": str(match),
                # Labels below are audit-only.  They do not enter extraction, scaling, or prediction.
                "reference_class": private.get("reference_class_key"),
                "distractor_class": private.get("distractor_class_key"),
                "changed_slots": _slot_changes(private.get("reference_class_key"), private.get("distractor_class_key")),
                "features": features,
            }
            episodes.append(episode)
            manifest.append({**base, "accepted": True, "raw_path": str(raw_path)})
    return episodes, rejected, manifest


def _run_oof(episodes: list[dict[str, Any]], descriptor_name: str, folds: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fold in range(folds):
        train = [episode for episode in episodes if episode["seed"] % folds != fold]
        test = [episode for episode in episodes if episode["seed"] % folds == fold]
        if not train or not test:
            continue
        scaler = _fit_scaler(train, descriptor_name)
        for episode in test:
            row = {
                key: episode[key]
                for key in ("run", "seed", "reference_class", "distractor_class", "changed_slots", "match_candidate")
            }
            row.update({"descriptor": descriptor_name, "fold": fold, **_predict(episode, descriptor_name, scaler)})
            rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True, help="Task data directory containing metadata.json files")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--far-plane-mm", type=float, default=34.0, help="TacEx sensor far plane used to convert depth to indentation")
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    if args.folds < 2:
        raise ValueError("--folds must be at least 2")

    episodes, rejected, manifest = _load_episodes(args.data_root.resolve(), args.far_plane_mm)
    if not episodes:
        raise RuntimeError("No valid three-object public raw probe episodes found")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "manifest.jsonl", manifest)
    _write_jsonl(
        output_dir / "feature_records.jsonl",
        [
            {
                key: episode[key]
                for key in ("run", "seed", "reference_class", "distractor_class", "changed_slots", "match_candidate", "features")
            }
            for episode in episodes
        ],
    )

    all_rows: list[dict[str, Any]] = []
    reports: dict[str, Any] = {}
    for descriptor_name in (*DESCRIPTORS, *GATED_DESCRIPTORS):
        rows = _run_oof(episodes, descriptor_name, args.folds)
        all_rows.extend(rows)
        by_change: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_change[str(row["changed_slots"])].append(row)
        reports[descriptor_name] = {
            "description": {
                "current_marker_delta_4d": "Current v0 public marker magnitude/coherence lift-minus-preload.",
                "directional_flow_8d": "Per-hand static mean dx/dy plus lift-minus-preload dx/dy.",
                "cross_finger_x_shear_2d": "Cross-finger x shear at preload and lift-minus-preload.",
                "cross_finger_xy_shear_4d": "Cross-finger x/y shear at preload and lift-minus-preload.",
                "grid_gradient_anisotropy_4d": "Signed row-vs-column marker-grid gradient statistic; candidate v0 interpretation.",
                "depth_p5_4d": "Robust p5 indentation at preload plus lift-minus-preload.",
                "depth_min_4d": "Max indentation (far-plane minus min depth) at preload plus lift-minus-preload.",
                "combined_direction_depth_anisotropy_16d": "Diagnostic concatenation; not a frozen memory expression.",
                "grid_gradient_anisotropy_gate_4iqr": "Same anisotropy vector; sensitivity gate excluding dynamic fields beyond 4 training-fold IQRs.",
                "grid_gradient_anisotropy_gate_8iqr": "Same anisotropy vector; sensitivity gate excluding dynamic fields beyond 8 training-fold IQRs.",
                "grid_gradient_anisotropy_gate_12iqr": "Same anisotropy vector; sensitivity gate excluding dynamic fields beyond 12 training-fold IQRs.",
                **{
                    f"cross_finger_xy_right_{name}_4d": (
                        "Diagnostic right-hand image-frame transform "
                        f"{name}; must not be frozen without a physical frame calibration."
                    )
                    for name in RIGHT_FLOW_TRANSFORMS
                },
            }[descriptor_name],
            "five_fold_oof": _summary(rows),
            "by_changed_slots": {name: _summary(group) for name, group in sorted(by_change.items())},
        }

    prediction_path = output_dir / "predictions.csv"
    columns = [
        "descriptor", "run", "seed", "fold", "reference_class", "distractor_class", "changed_slots",
        "match_candidate", "selected", "correct", "score_left", "score_right", "margin",
        "used_dimensions", "omitted_dynamic_fields",
    ]
    with prediction_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(all_rows)

    report = {
        "schema_version": SCHEMA_VERSION,
        "data_root": str(args.data_root.resolve()),
        "far_plane_mm": float(args.far_plane_mm),
        "valid_episodes": len(episodes),
        "rejected": dict(sorted(rejected.items())),
        "important_limitations": [
            "Private match labels are used only after reference-to-candidate predictions for offline scoring.",
            "The saved v3 raw probe has one preload and one lift-motion segment, not two controlled squeeze levels.",
            "grid_gradient_anisotropy_v0 is a transparent candidate definition, not an exact implementation of any external metric.",
            "Right-hand flow-transform variants are diagnostic only.  Future raw archives include tactile attachment poses; use them with a label-free calibration motion before freezing a common contact frame.",
            "OOF scaling is unsupervised median/IQR scaling on training-fold public values; no classifier is trained.",
        ],
        "descriptors": reports,
        "artifacts": {
            "manifest": "manifest.jsonl",
            "features": "feature_records.jsonl",
            "predictions": "predictions.csv",
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    concise = {
        "output_dir": str(output_dir),
        "valid_episodes": len(episodes),
        "rejected": dict(sorted(rejected.items())),
        "five_fold_oof_accuracy": {
            name: value["five_fold_oof"]["accuracy"] for name, value in reports.items()
        },
    }
    print(json.dumps(concise, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

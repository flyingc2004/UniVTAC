#!/usr/bin/env python3
"""Freeze source-task tactile slots and apply them to one public probe record.

This is an explicit diagnostic bridge, not a benchmark scorer.  It freezes
Median/IQR scaling and class centroids from validated VitaForge single-slot
datasets, then performs label-free reference-to-candidate matching on a
``tactile_probe.v3`` record.  Private metadata is read only after selection
for the optional audit field.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


SLOT_SPECS = {
    "weight": {
        "label_key": "label",
        "source_labels": ("light", "heavy"),
        "columns": tuple(f"expert_lift_marker_v2_{index}" for index in range(4)),
        "target_fields": (
            ("marker_displacement_px", "left"),
            ("marker_displacement_px", "right"),
            ("marker_coherence", "left"),
            ("marker_coherence", "right"),
        ),
        "validation_macro_accuracy": 0.8984867591424968,
    },
    "hardness": {
        "label_key": "label",
        "source_labels": ("soft", "hard"),
        "columns": tuple(f"delta_all_{index}" for index in range(6)),
        "target_fields": (
            ("depth_mm", "left"),
            ("depth_mm", "right"),
            ("marker_displacement_px", "left"),
            ("marker_displacement_px", "right"),
            ("marker_coherence", "left"),
            ("marker_coherence", "right"),
        ),
        "validation_macro_accuracy": 1.0,
    },
}


def read_csv_vectors(path: Path, columns: tuple[str, ...], labels: tuple[str, ...]) -> tuple[np.ndarray, list[str]]:
    vectors: list[list[float]] = []
    row_labels: list[str] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            label = str(row.get("label", ""))
            if label not in labels:
                continue
            vectors.append([float(row[column]) for column in columns])
            row_labels.append(label)
    if not vectors:
        raise ValueError(f"no usable rows in {path}")
    return np.asarray(vectors, dtype=np.float64), row_labels


def fit_encoder(vectors: np.ndarray, labels: list[str], class_labels: tuple[str, ...]) -> dict[str, Any]:
    median = np.median(vectors, axis=0)
    q25, q75 = np.percentile(vectors, [25, 75], axis=0)
    iqr = q75 - q25
    iqr[np.abs(iqr) < 1e-9] = 1.0
    encoded = (vectors - median) / iqr
    centers = {
        label: np.mean(encoded[np.asarray(labels) == label], axis=0).tolist()
        for label in class_labels
    }
    return {"median": median.tolist(), "iqr": iqr.tolist(), "centers": centers}


def target_vector(probe: dict[str, Any], fields: tuple[tuple[str, str], ...]) -> np.ndarray:
    delta = probe["lift_minus_preload"]
    values = [float(delta[hand][field]) for field, hand in fields]
    vector = np.asarray(values, dtype=np.float64)
    if not np.isfinite(vector).all():
        raise ValueError("target probe contains non-finite tactile values")
    return vector


def probe_confidence(probe: dict[str, Any]) -> float:
    quality = probe.get("quality", {})
    if not quality.get("valid", False):
        return 0.0
    return float(
        min(
            quality.get("preload_bilateral_contact_ratio", 0.0),
            quality.get("lift_motion_bilateral_contact_ratio", 0.0),
        )
    )


def apply_encoder(vector: np.ndarray, encoder: dict[str, Any]) -> np.ndarray:
    return (vector - np.asarray(encoder["median"], dtype=np.float64)) / np.asarray(
        encoder["iqr"], dtype=np.float64
    )


def class_distances(encoded: np.ndarray, centers: dict[str, Any]) -> dict[str, float]:
    return {
        label: float(np.linalg.norm(encoded - np.asarray(center, dtype=np.float64)))
        for label, center in centers.items()
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weight-features", type=Path, required=True)
    parser.add_argument("--hardness-features", type=Path, required=True)
    parser.add_argument("--roughness-encoder", type=Path, required=True)
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    encoders: dict[str, dict[str, Any]] = {}
    for slot, spec in SLOT_SPECS.items():
        source_path = args.weight_features if slot == "weight" else args.hardness_features
        vectors, labels = read_csv_vectors(source_path, spec["columns"], spec["source_labels"])
        encoders[slot] = {
            "slot": slot,
            "source_task": "weight_classify" if slot == "weight" else "hardness_classify",
            "feature_order": list(spec["target_fields"]),
            "encoder": fit_encoder(vectors, labels, spec["source_labels"]),
            "validation_macro_accuracy": spec["validation_macro_accuracy"],
        }

    roughness = json.loads(args.roughness_encoder.read_text(encoding="utf-8"))
    if roughness.get("descriptor") != "dynamic_marker":
        raise ValueError("this bridge expects roughness dynamic_marker descriptor")
    encoders["roughness"] = {
        "slot": "roughness",
        "source_task": str(roughness["source_task"]),
        "feature_order": [
            ("marker_displacement_px", "left"),
            ("marker_displacement_px", "right"),
            ("marker_coherence", "left"),
            ("marker_coherence", "right"),
        ],
        "encoder": {
            "median": roughness["scaler"]["median"],
            "iqr": roughness["scaler"]["iqr"],
            "centers": roughness["centers"],
        },
        "validation_macro_accuracy": float(roughness["validation"]["macro_accuracy"]),
    }

    public_records = json.loads((args.target_root / "metadata.json").read_text(encoding="utf-8"))
    public = public_records[str(args.seed)]
    probes = public["probes"]
    names = ("reference", "candidate_left", "candidate_right")
    output: dict[str, Any] = {
        "schema_version": "external_slot_reference_transfer.v1",
        "status": "pending",
        "transfer_status": "exploratory_block_to_can",
        "seed": args.seed,
        "source_encoders": encoders,
        "objects": {},
        "selection": {},
    }

    for name in names:
        probe = probes[name]
        confidence = probe_confidence(probe)
        slot_records: dict[str, Any] = {}
        for slot, slot_record in encoders.items():
            fields = tuple(tuple(field) for field in slot_record["feature_order"])
            vector = target_vector(probe, fields)
            encoded = apply_encoder(vector, slot_record["encoder"])
            distances = class_distances(encoded, slot_record["encoder"]["centers"])
            slot_records[slot] = {
                "raw_vector": vector.tolist(),
                "encoded_vector": encoded.tolist(),
                "class_distances": distances,
                "class_prediction": min(distances, key=distances.get),
            }
        output["objects"][name] = {"probe_confidence": confidence, "slots": slot_records}

    reference = output["objects"]["reference"]
    fused: dict[str, float] = {}
    slot_scores: dict[str, dict[str, Any]] = {}
    for candidate in ("candidate_left", "candidate_right"):
        weighted_sum = 0.0
        weight_sum = 0.0
        candidate_scores: dict[str, Any] = {}
        for slot, slot_record in encoders.items():
            ref_vector = np.asarray(reference["slots"][slot]["encoded_vector"], dtype=np.float64)
            candidate_vector = np.asarray(output["objects"][candidate]["slots"][slot]["encoded_vector"], dtype=np.float64)
            distance = float(np.sqrt(np.mean((ref_vector - candidate_vector) ** 2)))
            confidence = min(reference["probe_confidence"], output["objects"][candidate]["probe_confidence"])
            weight = float(slot_record["validation_macro_accuracy"]) * confidence
            candidate_scores[slot] = {"distance": distance, "confidence": confidence, "weight": weight}
            weighted_sum += weight * distance
            weight_sum += weight
        fused[candidate] = weighted_sum / weight_sum if weight_sum else float("inf")
        slot_scores[candidate] = candidate_scores

    selected = min(fused, key=fused.get)
    other = "candidate_right" if selected == "candidate_left" else "candidate_left"
    output["selection"] = {
        "selected_candidate": selected,
        "fused_scores": fused,
        "margin": float(fused[other] - fused[selected]),
        "per_slot": slot_scores,
        "rule": "validated_source_scaling + reference_distance + macro_accuracy_weighted_fusion",
    }
    private_path = args.target_root / "private_metadata.json"
    if private_path.exists():
        private = json.loads(private_path.read_text(encoding="utf-8")).get(str(args.seed), {})
        match_name = private.get("match_candidate_public_name")
        output["oracle_audit"] = {
            "match_candidate": match_name,
            "correct": selected == match_name,
        }
    output["status"] = "complete"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"selection": output["selection"], "oracle_audit": output.get("oracle_audit")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

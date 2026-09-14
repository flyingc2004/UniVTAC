#!/usr/bin/env python3
"""Freeze and evaluate an interpretable composable tactile expression.

The input is the public ``tactile_probe.v3`` record written by
``tactile_memory_match``.  Private metadata is read only after all public
feature distances have been computed, to split repeated expert trajectories and
score predictions.  The exported expression contains selected public fields,
robust scalers, and aggregate reliability only; it contains no labels, poses,
or individual trajectories.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


SLOTS = ("weight", "roughness", "hardness")
OBJECTS = ("reference", "candidate_left", "candidate_right")
EXPRESSION_VERSION = "tactile_slot_expression.v1"

# A small, auditable menu.  Selection is development-only and each slot freezes
# one member of this menu along with its robust scaler.
FEATURE_FAMILIES: dict[str, tuple[tuple[str, str], ...]] = {
    "preload_depth_2d": (("preload", "depth_mm"),),
    "preload_marker_displacement_2d": (("preload", "marker_displacement_px"),),
    "preload_marker_coherence_2d": (("preload", "marker_coherence"),),
    "lift_depth_delta_2d": (("lift_minus_preload", "depth_mm"),),
    "lift_marker_displacement_delta_2d": (("lift_minus_preload", "marker_displacement_px"),),
    "lift_marker_coherence_delta_2d": (("lift_minus_preload", "marker_coherence"),),
}


@dataclass(frozen=True)
class Episode:
    seed: int
    public: dict[str, Any]
    private: dict[str, Any]
    pair: tuple[str, str]
    repetition: int

    @property
    def reference_key(self) -> str:
        return str(self.private["reference_class_key"])

    @property
    def distractor_key(self) -> str:
        return str(self.private["distractor_class_key"])

    @property
    def match_name(self) -> str:
        return str(self.private["match_candidate_public_name"])


def parse_class_key(class_key: str) -> dict[str, str]:
    parts = str(class_key).split("_")
    if len(parts) != 3:
        raise ValueError(f"Expected weight_roughness_hardness class key, got {class_key!r}")
    return dict(zip(SLOTS, parts, strict=True))


def load_episodes(data_root: Path) -> tuple[list[Episode], list[dict[str, Any]]]:
    public_path = data_root / "metadata.json"
    private_path = data_root / "private_metadata.json"
    with public_path.open("r", encoding="utf-8") as handle:
        public_records = json.load(handle)
    with private_path.open("r", encoding="utf-8") as handle:
        private_records = json.load(handle)

    episodes: list[Episode] = []
    rejects: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str], list[tuple[int, dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for seed_text, public in public_records.items():
        private = private_records.get(str(seed_text))
        if not isinstance(private, dict):
            rejects.append({"seed": seed_text, "reason": "missing_private_label_record"})
            continue
        if public.get("schema_version") != "tactile_memory_match_public_episode.v3":
            rejects.append({"seed": seed_text, "reason": "not_tactile_probe_v3"})
            continue
        try:
            seed = int(seed_text)
            pair = (str(private["reference_class_key"]), str(private["distractor_class_key"]))
            parse_class_key(pair[0])
            parse_class_key(pair[1])
        except (KeyError, TypeError, ValueError) as exc:
            rejects.append({"seed": seed_text, "reason": f"invalid_private_label_record:{exc}"})
            continue
        grouped[pair].append((seed, public, private))

    for pair, records in grouped.items():
        for repetition, (seed, public, private) in enumerate(sorted(records, key=lambda row: row[0])):
            episodes.append(Episode(seed=seed, public=public, private=private, pair=pair, repetition=repetition))
    return sorted(episodes, key=lambda episode: episode.seed), rejects


def split_development_and_frozen(episodes: Iterable[Episode]) -> tuple[list[Episode], list[Episode], dict[str, Any]]:
    development: list[Episode] = []
    frozen: list[Episode] = []
    pair_counts: Counter[tuple[str, str]] = Counter()
    for episode in episodes:
        pair_counts[episode.pair] += 1
        if episode.repetition < 3:
            development.append(episode)
        elif episode.repetition < 5:
            frozen.append(episode)
    expected_pairs = 56
    complete_pairs = sum(count >= 5 for count in pair_counts.values())
    schedule = {
        "expected_ordered_pairs": expected_pairs,
        "observed_ordered_pairs": len(pair_counts),
        "complete_ordered_pairs": complete_pairs,
        "expected_development_episodes": expected_pairs * 3,
        "observed_development_episodes": len(development),
        "expected_frozen_episodes": expected_pairs * 2,
        "observed_frozen_episodes": len(frozen),
        "complete": complete_pairs == expected_pairs and len(development) == 168 and len(frozen) == 112,
    }
    return development, frozen, schedule


def probe_for(episode: Episode, object_name: str) -> dict[str, Any]:
    probes = episode.public.get("probes", {})
    if object_name not in probes or not isinstance(probes[object_name], dict):
        raise ValueError(f"missing public probe {object_name}")
    probe = probes[object_name]
    if probe.get("schema_version") != "tactile_probe.v3":
        raise ValueError(f"{object_name} does not have tactile_probe.v3")
    return probe


def probe_is_valid(probe: dict[str, Any]) -> bool:
    quality = probe.get("quality", {})
    return bool(quality.get("valid")) and int(quality.get("lift_motion_frame_count", 0)) > 0


def vector_from_probe(probe: dict[str, Any], family: str) -> np.ndarray:
    if family not in FEATURE_FAMILIES:
        raise KeyError(f"Unknown feature family {family}")
    values: list[float] = []
    for section, field in FEATURE_FAMILIES[family]:
        source = probe.get(section, {})
        for hand in ("left", "right"):
            values.append(float(source[hand][field]))
    vector = np.asarray(values, dtype=np.float64)
    if not np.isfinite(vector).all():
        raise ValueError(f"non-finite public feature in {family}")
    return vector


def robust_scaler(vectors: Iterable[np.ndarray]) -> dict[str, list[float]]:
    matrix = np.stack(list(vectors))
    median = np.median(matrix, axis=0)
    q1, q3 = np.percentile(matrix, [25, 75], axis=0)
    scale = np.maximum(q3 - q1, 1e-6)
    return {"median": median.tolist(), "iqr": scale.tolist()}


def transform(vector: np.ndarray, scaler: dict[str, Any]) -> np.ndarray:
    return (vector - np.asarray(scaler["median"], dtype=np.float64)) / np.asarray(scaler["iqr"], dtype=np.float64)


def distance(reference: np.ndarray, candidate: np.ndarray, scaler: dict[str, Any]) -> float:
    return float(np.linalg.norm(transform(reference, scaler) - transform(candidate, scaler)))


def slot_confidence(probe: dict[str, Any], family: str) -> float:
    """A label-free quality value based on contact coverage and field SNR."""
    quality = probe.get("quality", {})
    if not probe_is_valid(probe):
        return 0.0
    contact = min(
        float(quality.get("preload_bilateral_contact_ratio", 0.0)),
        float(quality.get("lift_motion_bilateral_contact_ratio", 0.0)),
    )
    section, field = FEATURE_FAMILIES[family][0]
    if section == "lift_minus_preload":
        # Delta noise is not stored separately; use the two source windows as
        # a conservative diagnostic and cap it to avoid huge ratios dominating.
        noise_sections = ("preload", "lift_motion")
    else:
        noise_sections = (section,)
    snrs = []
    for noise_section in noise_sections:
        noise = probe.get(noise_section, {}).get("noise", {})
        for hand in ("left", "right"):
            snrs.append(float(noise.get(hand, {}).get(field, {}).get("snr", 0.0)))
    return float(np.clip(contact * min(1.0, float(np.median(snrs)) / 10.0), 0.0, 1.0))


def episode_has_valid_probes(episode: Episode) -> bool:
    try:
        return all(probe_is_valid(probe_for(episode, name)) for name in OBJECTS)
    except ValueError:
        return False


def pair_prediction(episode: Episode, family: str, scaler: dict[str, Any]) -> dict[str, Any] | None:
    if not episode_has_valid_probes(episode):
        return None
    reference = vector_from_probe(probe_for(episode, "reference"), family)
    left = vector_from_probe(probe_for(episode, "candidate_left"), family)
    right = vector_from_probe(probe_for(episode, "candidate_right"), family)
    left_distance = distance(reference, left, scaler)
    right_distance = distance(reference, right, scaler)
    predicted = "candidate_left" if left_distance <= right_distance else "candidate_right"
    return {
        "predicted": predicted,
        "correct": predicted == episode.match_name,
        "left_distance": left_distance,
        "right_distance": right_distance,
        "margin": abs(left_distance - right_distance),
    }


def macro_accuracy(rows: Iterable[dict[str, Any]], label_key: str) -> float | None:
    grouped: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        grouped[str(row[label_key])].append(bool(row["correct"]))
    if not grouped:
        return None
    return float(np.mean([np.mean(values) for values in grouped.values()]))


def slot_contrast_rows(episodes: Iterable[Episode], slot: str) -> list[Episode]:
    rows = []
    for episode in episodes:
        ref = parse_class_key(episode.reference_key)
        distractor = parse_class_key(episode.distractor_key)
        differences = [name for name in SLOTS if ref[name] != distractor[name]]
        if differences == [slot]:
            rows.append(episode)
    return rows


def choose_slot_expression(development: list[Episode]) -> dict[str, Any]:
    expression_slots: dict[str, Any] = {}
    for slot in SLOTS:
        contrast = slot_contrast_rows(development, slot)
        candidates: list[dict[str, Any]] = []
        for family in FEATURE_FAMILIES:
            vectors = []
            for episode in development:
                for object_name in OBJECTS:
                    try:
                        vectors.append(vector_from_probe(probe_for(episode, object_name), family))
                    except (KeyError, TypeError, ValueError):
                        pass
            if not vectors:
                continue
            scaler = robust_scaler(vectors)
            scores = []
            for episode in contrast:
                prediction = pair_prediction(episode, family, scaler)
                if prediction is not None:
                    scores.append({**prediction, "reference_value": parse_class_key(episode.reference_key)[slot]})
            accuracy = float(np.mean([row["correct"] for row in scores])) if scores else 0.0
            macro = macro_accuracy(scores, "reference_value") or 0.0
            margin = float(np.mean([row["margin"] for row in scores])) if scores else 0.0
            candidates.append(
                {
                    "family": family,
                    "scaler": scaler,
                    "development_accuracy": accuracy,
                    "development_macro_accuracy": macro,
                    "development_margin": margin,
                    "development_coverage": len(scores) / max(len(contrast), 1),
                }
            )
        if not candidates:
            raise RuntimeError(f"No public features available to derive slot {slot}")
        candidates.sort(
            key=lambda row: (
                row["development_macro_accuracy"],
                row["development_accuracy"],
                row["development_margin"],
                -len(FEATURE_FAMILIES[row["family"]]),
                row["family"],
            ),
            reverse=True,
        )
        selected = candidates[0]
        expression_slots[slot] = {
            "feature_family": selected["family"],
            "fields": [f"{section}.{hand}.{field}" for section, field in FEATURE_FAMILIES[selected["family"]] for hand in ("left", "right")],
            "scaler": selected["scaler"],
            "quality_rule": {"requires_probe_valid": True, "snr_cap": 10.0},
            "frozen_reliability": selected["development_macro_accuracy"],
            "development": {key: value for key, value in selected.items() if key != "scaler"},
        }
    return {
        "schema_version": EXPRESSION_VERSION,
        "public_probe_schema_version": "tactile_probe.v3",
        "slots": expression_slots,
        "fusion": "normalized_reliability_x_slot_confidence_weighted_distance",
    }


def predict_identity(episode: Episode, expression: dict[str, Any]) -> dict[str, Any]:
    scores = {"candidate_left": 0.0, "candidate_right": 0.0}
    score_weights = {"candidate_left": 0.0, "candidate_right": 0.0}
    slot_details: dict[str, Any] = {}
    for slot, slot_expression in expression["slots"].items():
        family = slot_expression["feature_family"]
        try:
            ref_probe = probe_for(episode, "reference")
            ref = vector_from_probe(ref_probe, family)
            ref_confidence = slot_confidence(ref_probe, family)
        except (KeyError, TypeError, ValueError):
            slot_details[slot] = {"valid": False, "reason": "missing_reference_probe"}
            continue
        slot_details[slot] = {"valid": ref_confidence > 0.0, "candidates": {}}
        for candidate in ("candidate_left", "candidate_right"):
            try:
                candidate_probe = probe_for(episode, candidate)
                candidate_vector = vector_from_probe(candidate_probe, family)
                confidence = min(ref_confidence, slot_confidence(candidate_probe, family))
            except (KeyError, TypeError, ValueError):
                confidence = 0.0
                candidate_vector = None
            if confidence <= 0.0 or candidate_vector is None:
                slot_details[slot]["candidates"][candidate] = {"valid": False}
                continue
            value = distance(ref, candidate_vector, slot_expression["scaler"])
            weight = float(slot_expression["frozen_reliability"]) * confidence
            scores[candidate] += weight * value
            score_weights[candidate] += weight
            slot_details[slot]["candidates"][candidate] = {
                "valid": True,
                "distance": value,
                "confidence": confidence,
                "weight": weight,
            }
    normalized = {
        candidate: (scores[candidate] / score_weights[candidate] if score_weights[candidate] > 0 else math.nan)
        for candidate in scores
    }
    valid_candidates = {candidate: value for candidate, value in normalized.items() if np.isfinite(value)}
    predicted = min(valid_candidates, key=valid_candidates.get) if valid_candidates else None
    margin = abs(normalized["candidate_left"] - normalized["candidate_right"]) if len(valid_candidates) == 2 else math.nan
    return {
        "predicted": predicted,
        "correct": predicted == episode.match_name if predicted is not None else False,
        "scores": normalized,
        "margin": margin,
        "slot_details": slot_details,
        "coverage": len(valid_candidates) == 2,
    }


def summarise(rows: list[dict[str, Any]], schedule: dict[str, Any], slot_reports: dict[str, Any]) -> dict[str, Any]:
    forced = [row for row in rows if row["predicted"] is not None]
    covered = [row for row in rows if row["coverage"]]
    all_accuracy = float(np.mean([row["correct"] for row in rows])) if rows else None
    forced_accuracy = float(np.mean([row["correct"] for row in forced])) if forced else None
    selective_accuracy = float(np.mean([row["correct"] for row in covered])) if covered else None
    classes = defaultdict(list)
    for row in forced:
        classes[row["reference_class"]].append(row["correct"])
    macro = float(np.mean([np.mean(values) for values in classes.values()])) if classes else None
    hamming = defaultdict(list)
    for row in forced:
        hamming[str(row["hamming_distance"])].append(row["correct"])
    return {
        "schema_version": "tactile_slot_expression_report.v1",
        "schedule": schedule,
        "identity": {
            "episodes": len(rows),
            "forced_choice_accuracy": forced_accuracy,
            "selective_accuracy": selective_accuracy,
            "macro_accuracy": macro,
            "coverage": len(covered) / max(len(rows), 1),
            "all_episode_accuracy": all_accuracy,
            "accuracy_by_hamming_distance": {key: float(np.mean(values)) for key, values in hamming.items()},
        },
        "slots": slot_reports,
    }


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def run(data_root: Path, output_dir: Path) -> dict[str, Any]:
    episodes, rejects = load_episodes(data_root)
    development, frozen, schedule = split_development_and_frozen(episodes)
    expression = choose_slot_expression(development)

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "tactile_slot_expression.v1.json").open("w", encoding="utf-8") as handle:
        json.dump(expression, handle, ensure_ascii=False, indent=2)

    slot_reports: dict[str, Any] = {}
    for slot, slot_expression in expression["slots"].items():
        family = slot_expression["feature_family"]
        rows = []
        for episode in slot_contrast_rows(frozen, slot):
            prediction = pair_prediction(episode, family, slot_expression["scaler"])
            if prediction is not None:
                rows.append({**prediction, "reference_value": parse_class_key(episode.reference_key)[slot]})
        slot_reports[slot] = {
            "feature_family": family,
            "contrast_episodes": len(slot_contrast_rows(frozen, slot)),
            "valid_episodes": len(rows),
            "coverage": len(rows) / max(len(slot_contrast_rows(frozen, slot)), 1),
            "accuracy": float(np.mean([row["correct"] for row in rows])) if rows else None,
            "macro_accuracy": macro_accuracy(rows, "reference_value"),
        }

    prediction_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = list(rejects)
    for episode in frozen:
        result = predict_identity(episode, expression)
        ref_slots = parse_class_key(episode.reference_key)
        distractor_slots = parse_class_key(episode.distractor_key)
        hamming_distance = sum(ref_slots[slot] != distractor_slots[slot] for slot in SLOTS)
        prediction_rows.append(
            {
                "seed": episode.seed,
                "reference_class": episode.reference_key,
                "distractor_class": episode.distractor_key,
                "hamming_distance": hamming_distance,
                "predicted": result["predicted"],
                "match_candidate": episode.match_name,
                "correct": result["correct"],
                "coverage": result["coverage"],
                "score_left": result["scores"]["candidate_left"],
                "score_right": result["scores"]["candidate_right"],
                "margin": result["margin"],
                "slot_details": result["slot_details"],
            }
        )
    for episode in episodes:
        manifest_rows.append(
            {
                "seed": episode.seed,
                "split": "development" if episode in development else "frozen" if episode in frozen else "unused",
                "pair": list(episode.pair),
                "repetition": episode.repetition,
                "all_probes_valid": episode_has_valid_probes(episode),
            }
        )

    write_jsonl(output_dir / "manifest.jsonl", manifest_rows)
    with (output_dir / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "seed", "reference_class", "distractor_class", "hamming_distance", "predicted",
                "match_candidate", "correct", "coverage", "score_left", "score_right", "margin", "slot_details",
            ],
        )
        writer.writeheader()
        for row in prediction_rows:
            writer.writerow({**row, "slot_details": json.dumps(row["slot_details"], sort_keys=True)})
    report = summarise(prediction_rows, schedule, slot_reports)
    report["quality"] = {
        "frozen_all_probe_valid_coverage": sum(episode_has_valid_probes(episode) for episode in frozen) / max(len(frozen), 1),
        "physics_success_rate": float(np.mean([episode.private.get("result") == "success" for episode in episodes])) if episodes else None,
    }
    report["gates"] = {
        "physics_success": bool((report["quality"]["physics_success_rate"] or 0.0) >= 0.80),
        "probe_coverage": bool(report["quality"]["frozen_all_probe_valid_coverage"] >= 0.90),
        "slot_macro_accuracy": {slot: bool((row["macro_accuracy"] or 0.0) >= 0.80) for slot, row in slot_reports.items()},
        "identity_macro_accuracy": bool((report["identity"]["macro_accuracy"] or 0.0) >= 0.80),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.data_root, args.output_dir)
    identity = report["identity"]
    print(
        "[tactile-slot-expression] "
        f"coverage={identity['coverage']:.3f} forced={identity['forced_choice_accuracy']} "
        f"macro={identity['macro_accuracy']} schedule_complete={report['schedule']['complete']}"
    )


if __name__ == "__main__":
    main()

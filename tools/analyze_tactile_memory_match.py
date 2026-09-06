#!/usr/bin/env python3
"""Offline raw tactile distance calibration for ``tactile_memory_match``.

This tool deliberately separates raw feature extraction from offline scoring:
the extractor reads only tactile depth/marker observations and ``atom.tag``;
``metadata.json`` is consulted only after a prediction has been made.  It does
not read actor poses, physical parameters, task success, or the expert's
private weight-response fields as input features.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


OBJECTS = ("reference", "candidate_left", "candidate_right")
KNOWN_CLASSES = ("light-smooth", "light-rough", "heavy-smooth", "heavy-rough")
GRID_SIZE = 4
STATIC_FRAME_COUNT = 8
DYNAMIC_FRAME_COUNT = 6
DESCRIPTOR_VERSION = "raw_tactile_distance.v1"


@dataclass(frozen=True)
class RawFrame:
    """One cached observation containing only public raw tactile inputs."""

    step: int
    tag: str
    depth: dict[str, np.ndarray]
    marker: dict[str, np.ndarray]


@dataclass(frozen=True)
class ProbeDescriptor:
    """Raw tactile descriptor and audit metadata for one probe window."""

    static_depth: np.ndarray
    static_marker: np.ndarray
    dynamic: np.ndarray
    close_steps: tuple[int, ...]
    dynamic_steps: tuple[int, ...]

    def vectors(self) -> dict[str, np.ndarray]:
        static = np.concatenate((self.static_depth, self.static_marker))
        return {
            "depth_only": np.concatenate((self.static_depth, self.dynamic_depth_contact())),
            "marker_only": np.concatenate((self.static_marker, self.dynamic_marker())),
            "static_only": static,
            "dynamic_only": self.dynamic,
            "combined": np.concatenate((static, self.dynamic)),
        }

    def dynamic_depth_contact(self) -> np.ndarray:
        # Per frame and hand: mean indentation and contact-area fraction.
        return self.dynamic.reshape(DYNAMIC_FRAME_COUNT, 2, 5)[:, :, :2].reshape(-1)

    def dynamic_marker(self) -> np.ndarray:
        # Per frame and hand: mean marker dx, dy, and displacement magnitude.
        return self.dynamic.reshape(DYNAMIC_FRAME_COUNT, 2, 5)[:, :, 2:].reshape(-1)


@dataclass(frozen=True)
class PreparedEpisode:
    seed: str
    descriptors: dict[str, ProbeDescriptor]


@dataclass(frozen=True)
class OfflineLabel:
    """Offline-only label used after distance prediction, never for features."""

    reference_class: str
    distractor_class: str
    match_candidate: str


@dataclass(frozen=True)
class RobustScaler:
    median: np.ndarray
    scale: np.ndarray

    def transform(self, vector: np.ndarray) -> np.ndarray:
        return (vector - self.median) / self.scale


def _close_tag(name: str) -> str:
    if name == "reference":
        return "reference_close"
    return f"{name}_probe_close"


def _weight_prefix(name: str) -> str:
    return "reference_weight" if name == "reference" else f"{name}_weight"


def _numeric_key(path: Path) -> int:
    try:
        return int(path.stem)
    except ValueError:
        return sys.maxsize


def _load_raw_frames(cache_dir: Path) -> list[RawFrame]:
    """Load raw tactile fields and atom tags from one episode cache directory."""

    frames: list[RawFrame] = []
    for path in sorted(cache_dir.glob("*.pkl"), key=_numeric_key):
        try:
            with path.open("rb") as handle:
                payload = pickle.load(handle)
            tactile = payload["tactile"]
            atom = payload.get("atom") or {}
            tag = str(atom.get("tag") or "")
            step = int(payload.get("step", path.stem))
            depth: dict[str, np.ndarray] = {}
            marker: dict[str, np.ndarray] = {}
            for hand, source in (("left", "left_tactile"), ("right", "right_tactile")):
                hand_obs = tactile[source]
                depth[hand] = np.asarray(hand_obs["depth"], dtype=np.float64)
                marker[hand] = np.asarray(hand_obs["marker"], dtype=np.float64)
        except (KeyError, OSError, TypeError, ValueError, pickle.UnpicklingError) as exc:
            raise ValueError(f"Could not read raw tactile frame {path}: {exc}") from exc
        frames.append(RawFrame(step=step, tag=tag, depth=depth, marker=marker))
    return frames


def _mean_pool_2d(array: np.ndarray, grid_size: int = GRID_SIZE) -> np.ndarray:
    if array.ndim != 2 or min(array.shape) < grid_size:
        raise ValueError(f"Expected 2D tactile depth at least {grid_size}x{grid_size}, got {array.shape}")
    row_edges = np.linspace(0, array.shape[0], grid_size + 1, dtype=int)
    col_edges = np.linspace(0, array.shape[1], grid_size + 1, dtype=int)
    return np.asarray(
        [
            [
                float(array[row_edges[row] : row_edges[row + 1], col_edges[col] : col_edges[col + 1]].mean())
                for col in range(grid_size)
            ]
            for row in range(grid_size)
        ],
        dtype=np.float64,
    )


def _marker_delta(marker: np.ndarray) -> np.ndarray:
    if marker.shape != (2, 64, 2):
        raise ValueError(f"Expected marker flow with shape (2, 64, 2), got {marker.shape}")
    return marker[1] - marker[0]


def _pool_marker_delta(marker: np.ndarray) -> np.ndarray:
    # TacEx documents marker flow as [initial/current, 64 markers, xy].
    delta = _marker_delta(marker).reshape(8, 8, 2)
    pooled = delta.reshape(GRID_SIZE, 2, GRID_SIZE, 2, 2).mean(axis=(1, 3))
    return pooled


def _select_uniform(frames: list[RawFrame], count: int) -> list[RawFrame]:
    if len(frames) < count:
        raise ValueError(f"Need at least {count} frames, found {len(frames)}")
    indices = np.linspace(0, len(frames) - 1, count).round().astype(int)
    return [frames[index] for index in indices]


def _dynamic_frames(name: str, frames: list[RawFrame]) -> list[RawFrame]:
    prefix = _weight_prefix(name)
    lift_tag = f"{prefix}_lift"
    lower_tag = f"{prefix}_lower"
    lift_frames = [frame for frame in frames if frame.tag == lift_tag]
    lower_frames = [frame for frame in frames if frame.tag == lower_tag]
    if not lift_frames or not lower_frames:
        raise ValueError(f"Missing {lift_tag} or {lower_tag} frames")
    start_step = min(frame.step for frame in lift_frames)
    end_step = max(frame.step for frame in lower_frames)
    return [frame for frame in frames if start_step <= frame.step <= end_step]


def _static_frames(name: str, frames: list[RawFrame]) -> list[RawFrame]:
    close_frames = [frame for frame in frames if frame.tag == _close_tag(name)]
    if len(close_frames) < STATIC_FRAME_COUNT:
        raise ValueError(f"{name} has only {len(close_frames)} close frames")
    return close_frames


def extract_probe_descriptor(name: str, frames: list[RawFrame]) -> ProbeDescriptor:
    """Build a descriptor using only raw depth, marker flow, and atom tags."""

    close_frames = _static_frames(name, frames)
    baseline_frames = close_frames[:STATIC_FRAME_COUNT]
    final_frames = close_frames[-STATIC_FRAME_COUNT:]
    dynamic_frames = _select_uniform(_dynamic_frames(name, frames), DYNAMIC_FRAME_COUNT)

    depth_parts: list[np.ndarray] = []
    marker_parts: list[np.ndarray] = []
    dynamic_parts: list[np.ndarray] = []
    baselines: dict[str, np.ndarray] = {}
    for hand in ("left", "right"):
        baselines[hand] = np.median(np.stack([frame.depth[hand] for frame in baseline_frames]), axis=0)
        final_depth = np.median(np.stack([frame.depth[hand] for frame in final_frames]), axis=0)
        indentation = np.maximum(baselines[hand] - final_depth, 0.0)
        depth_parts.append(_mean_pool_2d(indentation).reshape(-1))
        final_marker = np.median(np.stack([frame.marker[hand] for frame in final_frames]), axis=0)
        marker_parts.append(_pool_marker_delta(final_marker).reshape(-1))

    for frame in dynamic_frames:
        frame_parts: list[np.ndarray] = []
        for hand in ("left", "right"):
            indentation = np.maximum(baselines[hand] - frame.depth[hand], 0.0)
            marker_delta = _marker_delta(frame.marker[hand])
            marker_magnitude = np.linalg.norm(marker_delta, axis=1)
            frame_parts.append(
                np.asarray(
                    [
                        float(indentation.mean()),
                        float((indentation > 0.0).mean()),
                        float(marker_delta[:, 0].mean()),
                        float(marker_delta[:, 1].mean()),
                        float(marker_magnitude.mean()),
                    ],
                    dtype=np.float64,
                )
            )
        dynamic_parts.append(np.concatenate(frame_parts))

    return ProbeDescriptor(
        static_depth=np.concatenate(depth_parts),
        static_marker=np.concatenate(marker_parts),
        dynamic=np.concatenate(dynamic_parts),
        close_steps=tuple(frame.step for frame in final_frames),
        dynamic_steps=tuple(frame.step for frame in dynamic_frames),
    )


def _load_labels(metadata: dict[str, Any]) -> dict[str, OfflineLabel]:
    """Read labels only for post-hoc evaluation, never descriptor extraction."""

    labels: dict[str, OfflineLabel] = {}
    for seed, record in metadata.items():
        match_candidate = record.get("match_candidate_public_name")
        reference_class = record.get("reference_class")
        distractor_class = record.get("distractor_class")
        if (
            match_candidate in {"candidate_left", "candidate_right"}
            and isinstance(reference_class, str)
            and isinstance(distractor_class, str)
        ):
            labels[str(seed)] = OfflineLabel(
                reference_class=reference_class,
                distractor_class=distractor_class,
                match_candidate=match_candidate,
            )
    return labels


def _stable_fold(seed: str, folds: int) -> int:
    digest = hashlib.sha256(seed.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big") % folds


def _fit_scaler(vectors: Iterable[np.ndarray]) -> RobustScaler:
    stacked = np.stack(list(vectors))
    median = np.median(stacked, axis=0)
    q75, q25 = np.percentile(stacked, [75, 25], axis=0)
    scale = q75 - q25
    scale[np.abs(scale) < 1e-9] = 1.0
    return RobustScaler(median=median, scale=scale)


def _distance(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.linalg.norm(left - right))


def _confidence_interval(values: np.ndarray, samples: int, seed: int = 0) -> tuple[float, float]:
    if values.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, values.size, size=(samples, values.size))
    estimates = values[draws].mean(axis=1)
    return (float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975)))


def _evaluate_variant(
    episodes: list[PreparedEpisode],
    labels: dict[str, OfflineLabel],
    variant: str,
    folds: int,
    bootstrap_samples: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], RobustScaler | None]:
    predictions: list[dict[str, Any]] = []
    scalers: list[RobustScaler] = []
    for fold in range(folds):
        test_episodes = [episode for episode in episodes if _stable_fold(episode.seed, folds) == fold]
        train_episodes = [episode for episode in episodes if _stable_fold(episode.seed, folds) != fold]
        if not test_episodes or not train_episodes:
            continue
        train_vectors = [
            episode.descriptors[name].vectors()[variant]
            for episode in train_episodes
            for name in OBJECTS
        ]
        scaler = _fit_scaler(train_vectors)
        scalers.append(scaler)
        for episode in test_episodes:
            label = labels[episode.seed]
            reference = scaler.transform(episode.descriptors["reference"].vectors()[variant])
            left = scaler.transform(episode.descriptors["candidate_left"].vectors()[variant])
            right = scaler.transform(episode.descriptors["candidate_right"].vectors()[variant])
            left_distance = _distance(reference, left)
            right_distance = _distance(reference, right)
            selected = "candidate_left" if left_distance <= right_distance else "candidate_right"
            distractor = "candidate_right" if label.match_candidate == "candidate_left" else "candidate_left"
            match_distance = left_distance if label.match_candidate == "candidate_left" else right_distance
            distractor_distance = right_distance if label.match_candidate == "candidate_left" else left_distance
            correct = selected == label.match_candidate
            predicted_class = label.reference_class if correct else label.distractor_class
            predictions.append(
                {
                    "seed": episode.seed,
                    "fold": fold,
                    "variant": variant,
                    "reference_class": label.reference_class,
                    "distractor_class": label.distractor_class,
                    "match_candidate": label.match_candidate,
                    "selected_candidate": selected,
                    "correct": correct,
                    "left_distance": left_distance,
                    "right_distance": right_distance,
                    "match_distance": match_distance,
                    "distractor_distance": distractor_distance,
                    "margin": distractor_distance - match_distance,
                    "predicted_class": predicted_class,
                    "static_reference_steps": ";".join(map(str, episode.descriptors["reference"].close_steps)),
                    "dynamic_reference_steps": ";".join(map(str, episode.descriptors["reference"].dynamic_steps)),
                    "static_left_steps": ";".join(map(str, episode.descriptors["candidate_left"].close_steps)),
                    "dynamic_left_steps": ";".join(map(str, episode.descriptors["candidate_left"].dynamic_steps)),
                    "static_right_steps": ";".join(map(str, episode.descriptors["candidate_right"].close_steps)),
                    "dynamic_right_steps": ";".join(map(str, episode.descriptors["candidate_right"].dynamic_steps)),
                }
            )

    if not predictions:
        return predictions, {"valid_episodes": 0}, None

    correct_values = np.asarray([float(row["correct"]) for row in predictions])
    per_class: dict[str, float] = {}
    for class_name in KNOWN_CLASSES:
        class_values = [float(row["correct"]) for row in predictions if row["reference_class"] == class_name]
        if class_values:
            per_class[class_name] = float(np.mean(class_values))
    macro_accuracy = float(np.mean(list(per_class.values()))) if per_class else float("nan")
    ci_low, ci_high = _confidence_interval(correct_values, bootstrap_samples)
    metrics = {
        "valid_episodes": len(predictions),
        "overall_accuracy": float(correct_values.mean()),
        "macro_accuracy": macro_accuracy,
        "per_class_accuracy": per_class,
        "bootstrap_95_ci": [ci_low, ci_high],
        "mean_match_distance": float(np.mean([row["match_distance"] for row in predictions])),
        "mean_distractor_distance": float(np.mean([row["distractor_distance"] for row in predictions])),
        "mean_margin": float(np.mean([row["margin"] for row in predictions])),
        "positive_margin_rate": float(np.mean([row["margin"] > 0.0 for row in predictions])),
    }
    return predictions, metrics, scalers[0] if scalers else None


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)


def _write_manifest(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def _write_predictions(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_plots(output_dir: Path, predictions: list[dict[str, Any]]) -> None:
    if not predictions:
        return
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    combined = [row for row in predictions if row["variant"] == "combined"]
    if not combined:
        return
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    match = np.asarray([row["match_distance"] for row in combined])
    distractor = np.asarray([row["distractor_distance"] for row in combined])
    fig, axis = plt.subplots(figsize=(7, 4))
    axis.hist(match, bins="auto", alpha=0.7, label="match")
    axis.hist(distractor, bins="auto", alpha=0.7, label="distractor")
    axis.set_xlabel("Scaled Euclidean distance")
    axis.set_ylabel("Episodes")
    axis.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "distance_distribution.png", dpi=160)
    plt.close(fig)

    margin = np.asarray([row["margin"] for row in combined])
    fig, axis = plt.subplots(figsize=(7, 4))
    axis.hist(margin, bins="auto", color="#3478bf")
    axis.axvline(0.0, color="#222222", linewidth=1)
    axis.set_xlabel("Distractor distance - match distance")
    axis.set_ylabel("Episodes")
    fig.tight_layout()
    fig.savefig(figures_dir / "margin_distribution.png", dpi=160)
    plt.close(fig)

    index = {class_name: position for position, class_name in enumerate(KNOWN_CLASSES)}
    matrix = np.zeros((len(KNOWN_CLASSES), len(KNOWN_CLASSES)), dtype=int)
    for row in combined:
        matrix[index[row["reference_class"]], index[row["predicted_class"]]] += 1
    fig, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_xticks(range(len(KNOWN_CLASSES)), KNOWN_CLASSES, rotation=30, ha="right")
    axis.set_yticks(range(len(KNOWN_CLASSES)), KNOWN_CLASSES)
    axis.set_xlabel("Physical class of selected candidate")
    axis.set_ylabel("Reference physical class")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            axis.text(col, row, str(matrix[row, col]), ha="center", va="center")
    fig.colorbar(image, ax=axis)
    fig.tight_layout()
    fig.savefig(figures_dir / "class_confusion_matrix.png", dpi=160)
    plt.close(fig)


def run_calibration(
    dataset_root: Path,
    output_dir: Path,
    *,
    folds: int = 5,
    bootstrap_samples: int = 5000,
    min_valid_per_class: int = 20,
) -> dict[str, Any]:
    """Run the label-isolated calibration and write all offline artifacts."""

    if folds < 2:
        raise ValueError("folds must be at least 2")
    metadata_path = dataset_root / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if not isinstance(metadata, dict):
        raise ValueError("metadata.json must contain an episode mapping")

    labels = _load_labels(metadata)
    episodes: list[PreparedEpisode] = []
    manifest: list[dict[str, Any]] = []
    rejection_counts: Counter[str] = Counter()
    for seed in sorted(metadata, key=lambda value: int(value) if str(value).isdigit() else sys.maxsize):
        seed_text = str(seed)
        cache_dir = dataset_root / ".cache" / seed_text
        manifest_record: dict[str, Any] = {"seed": seed_text, "cache_dir": str(cache_dir)}
        if seed_text not in labels:
            manifest_record.update({"status": "rejected", "reason": "missing_offline_label"})
            rejection_counts["missing_offline_label"] += 1
            manifest.append(manifest_record)
            continue
        if not cache_dir.is_dir():
            manifest_record.update({"status": "rejected", "reason": "missing_cache"})
            rejection_counts["missing_cache"] += 1
            manifest.append(manifest_record)
            continue
        try:
            frames = _load_raw_frames(cache_dir)
            if not frames:
                raise ValueError("empty_cache")
            descriptors = {name: extract_probe_descriptor(name, frames) for name in OBJECTS}
        except ValueError as exc:
            reason = str(exc)
            manifest_record.update({"status": "rejected", "reason": reason})
            rejection_counts[reason] += 1
            manifest.append(manifest_record)
            continue
        episodes.append(PreparedEpisode(seed=seed_text, descriptors=descriptors))
        manifest_record.update(
            {
                "status": "valid",
                "frame_count": len(frames),
                "static_frame_count": STATIC_FRAME_COUNT,
                "dynamic_frame_count": DYNAMIC_FRAME_COUNT,
                "probe_steps": {
                    name: {
                        "close": list(descriptor.close_steps),
                        "dynamic": list(descriptor.dynamic_steps),
                    }
                    for name, descriptor in descriptors.items()
                },
            }
        )
        manifest.append(manifest_record)

    all_predictions: list[dict[str, Any]] = []
    variant_metrics: dict[str, Any] = {}
    scaler_payload: dict[str, Any] = {}
    for variant in ("depth_only", "marker_only", "static_only", "dynamic_only", "combined"):
        rows, metrics, scaler = _evaluate_variant(episodes, labels, variant, folds, bootstrap_samples)
        all_predictions.extend(rows)
        variant_metrics[variant] = metrics
        if scaler is not None:
            scaler_payload[variant] = {
                "median": scaler.median.tolist(),
                "iqr_scale": scaler.scale.tolist(),
            }

    class_counts = Counter(labels[episode.seed].reference_class for episode in episodes)
    combined_metrics = variant_metrics.get("combined", {})
    enough_data = (
        len(episodes) >= min_valid_per_class * len(KNOWN_CLASSES)
        and all(class_counts[class_name] >= min_valid_per_class for class_name in KNOWN_CLASSES)
    )
    accuracy_pass = bool(
        combined_metrics.get("overall_accuracy", 0.0) >= 0.80
        and combined_metrics.get("macro_accuracy", 0.0) >= 0.80
    )
    summary = {
        "schema_version": DESCRIPTOR_VERSION,
        "dataset_root": str(dataset_root),
        "feature_policy": {
            "allowed_inputs": ["tactile.depth", "tactile.marker", "atom.tag"],
            "forbidden_feature_inputs": [
                "density",
                "friction",
                "actor_pose",
                "object_lift_delta",
                "lift_follow_ratio",
                "reward",
                "success",
                "task_completed",
            ],
            "labels_used_only_for_scoring": True,
        },
        "descriptor": {
            "static": "per hand: 4x4 depth indentation + 4x4 marker dx/dy",
            "dynamic": "six uniform lift/hold/lower frames; per hand depth/contact/marker summary",
            "dimensions": {
                "depth_only": 56,
                "marker_only": 100,
                "static_only": 96,
                "dynamic_only": 60,
                "combined": 156,
            },
            "scaling": "five-fold seed-hash split; train-fold median/IQR; no labels, PCA, or classifier fitting",
            "classifier": "argmin scaled Euclidean distance from reference to left/right candidate",
        },
        "quality": {
            "metadata_episodes": len(metadata),
            "valid_episodes": len(episodes),
            "reference_class_counts": dict(sorted(class_counts.items())),
            "rejections": dict(sorted(rejection_counts.items())),
        },
        "metrics": variant_metrics,
        "gate": {
            "required_valid_per_class": min_valid_per_class,
            "required_total_valid": min_valid_per_class * len(KNOWN_CLASSES),
            "required_overall_accuracy": 0.80,
            "required_macro_accuracy": 0.80,
            "enough_balanced_data": enough_data,
            "combined_accuracy_pass": accuracy_pass,
            "calibration_pass": bool(enough_data and accuracy_pass),
        },
    }
    _write_manifest(output_dir / "manifest.jsonl", manifest)
    _write_predictions(output_dir / "predictions.csv", all_predictions)
    _write_json(output_dir / "descriptor_stats.json", scaler_payload)
    _write_json(output_dir / "summary.json", summary)
    _write_plots(output_dir, all_predictions)
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/tactile_memory_match/tactile_memory_match_smoke"),
        help="Official collect output containing metadata.json and .cache/<seed>/*.pkl.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Offline report directory (default: <dataset-root>/analysis/raw_distance_v1).",
    )
    parser.add_argument("--folds", type=int, default=5, help="Seed-hash robust-scaling folds.")
    parser.add_argument("--bootstrap-samples", type=int, default=5000, help="Accuracy bootstrap draws.")
    parser.add_argument(
        "--min-valid-per-class",
        type=int,
        default=20,
        help="Gate requirement per physical reference class.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    dataset_root = args.dataset_root.resolve()
    output_dir = (args.output_dir or dataset_root / "analysis" / "raw_distance_v1").resolve()
    summary = run_calibration(
        dataset_root,
        output_dir,
        folds=args.folds,
        bootstrap_samples=args.bootstrap_samples,
        min_valid_per_class=args.min_valid_per_class,
    )
    gate = summary["gate"]
    print(f"[raw-distance] report={output_dir}")
    print(
        "[raw-distance] valid={valid} enough_data={enough} accuracy_pass={accuracy} calibration_pass={passed}".format(
            valid=summary["quality"]["valid_episodes"],
            enough=gate["enough_balanced_data"],
            accuracy=gate["combined_accuracy_pass"],
            passed=gate["calibration_pass"],
        )
    )
    if summary["quality"]["valid_episodes"] == 0:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

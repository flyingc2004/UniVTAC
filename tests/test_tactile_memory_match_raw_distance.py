"""Small, Isaac-free tests for the raw tactile distance calibration tool."""

from __future__ import annotations

import importlib.util
import json
import pickle
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "analyze_tactile_memory_match.py"
SPEC = importlib.util.spec_from_file_location("raw_distance", MODULE_PATH)
assert SPEC and SPEC.loader
raw_distance = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = raw_distance
SPEC.loader.exec_module(raw_distance)


def _marker(offset_x: float, offset_y: float) -> np.ndarray:
    rows, cols = np.meshgrid(np.arange(8), np.arange(8), indexing="ij")
    initial = np.stack((cols.reshape(-1), rows.reshape(-1)), axis=1).astype(np.float64)
    current = initial + np.asarray([offset_x, offset_y])
    return np.stack((initial, current))


def _frame(step: int, tag: str, indentation: float, marker_x: float, marker_y: float):
    baseline = np.full((8, 8), 34.0, dtype=np.float64)
    depth = baseline - indentation
    return raw_distance.RawFrame(
        step=step,
        tag=tag,
        depth={"left": depth, "right": depth + 0.1},
        marker={"left": _marker(marker_x, marker_y), "right": _marker(marker_x, -marker_y)},
    )


def _probe_frames(name: str, start: int, indentation: float, marker_x: float):
    close_tag = raw_distance._close_tag(name)
    prefix = raw_distance._weight_prefix(name)
    frames = [_frame(start + index, close_tag, 0.0 if index < 2 else indentation, marker_x, 0.0) for index in range(8)]
    frames.extend(_frame(start + 10 + index, f"{prefix}_lift", indentation, marker_x, 0.2 * index) for index in range(4))
    frames.extend(_frame(start + 14 + index, "delay", indentation, marker_x, 0.8) for index in range(2))
    frames.extend(_frame(start + 16 + index, f"{prefix}_lower", indentation, marker_x, 0.8 - 0.2 * index) for index in range(4))
    return frames


class RawDistanceDescriptorTests(unittest.TestCase):
    def test_descriptor_dimensions_and_windows(self):
        descriptor = raw_distance.extract_probe_descriptor("reference", _probe_frames("reference", 0, 3.0, 1.0))
        vectors = descriptor.vectors()
        self.assertEqual(vectors["depth_only"].shape, (56,))
        self.assertEqual(vectors["marker_only"].shape, (100,))
        self.assertEqual(vectors["static_only"].shape, (96,))
        self.assertEqual(vectors["dynamic_only"].shape, (60,))
        self.assertEqual(vectors["combined"].shape, (156,))
        self.assertEqual(vectors["semantic_signature_8d"].shape, (8,))
        self.assertEqual(len(descriptor.close_steps), raw_distance.STATIC_FRAME_COUNT)
        self.assertEqual(len(descriptor.dynamic_steps), raw_distance.DYNAMIC_FRAME_COUNT)
        self.assertEqual(len(descriptor.semantic_static_steps), raw_distance.STATIC_FRAME_COUNT)
        self.assertEqual(len(descriptor.semantic_dynamic_steps), raw_distance.DYNAMIC_FRAME_COUNT)
        # Dynamic semantic evidence ends before the lower motion begins.
        self.assertLess(max(descriptor.semantic_dynamic_steps), 16)

    def test_semantic_signature_keeps_raw_marker_pixels_and_coherence(self):
        descriptor = raw_distance.extract_probe_descriptor(
            "reference", _probe_frames("reference", 0, 3.0, 2.0)
        )
        signature = descriptor.vectors()["semantic_signature_8d"]

        self.assertEqual(signature[0], 3.0)
        self.assertAlmostEqual(signature[1], 2.9)
        self.assertEqual(signature[2], 2.0)
        self.assertEqual(signature[3], 2.0)
        self.assertEqual(signature[6], 1.0)
        self.assertEqual(signature[7], 1.0)
        self.assertGreater(signature[4], 0.0)
        self.assertGreater(signature[5], 0.0)

    def test_semantic_signature_rejects_shallow_bilateral_contact(self):
        with self.assertRaisesRegex(ValueError, "static left shallow_contact"):
            raw_distance.extract_probe_descriptor(
                "reference", _probe_frames("reference", 0, 0.0, 1.0)
            )

    def test_marker_coherence_is_low_for_symmetric_flow(self):
        marker = _marker(0.0, 0.0)
        marker[1, :32, 0] = marker[0, :32, 0] - 2.0
        marker[1, 32:, 0] = marker[0, 32:, 0] + 2.0
        self.assertAlmostEqual(raw_distance._marker_coherence(marker), 0.0)

    def test_same_probe_is_nearer_than_changed_probe(self):
        reference = raw_distance.extract_probe_descriptor("reference", _probe_frames("reference", 0, 3.0, 1.0))
        left = raw_distance.extract_probe_descriptor("candidate_left", _probe_frames("candidate_left", 30, 3.0, 1.0))
        right = raw_distance.extract_probe_descriptor("candidate_right", _probe_frames("candidate_right", 60, 1.0, 4.0))
        self.assertLess(
            raw_distance._distance(reference.vectors()["combined"], left.vectors()["combined"]),
            raw_distance._distance(reference.vectors()["combined"], right.vectors()["combined"]),
        )

    def test_marker_shape_is_checked(self):
        bad = _frame(0, "reference_close", 0.0, 0.0, 0.0)
        bad_marker = dict(bad.marker)
        bad_marker["left"] = np.zeros((2, 63, 2), dtype=np.float64)
        malformed = raw_distance.RawFrame(step=bad.step, tag=bad.tag, depth=bad.depth, marker=bad_marker)
        with self.assertRaisesRegex(ValueError, "marker flow"):
            raw_distance._pool_marker_delta(malformed.marker["left"])

    def test_calibration_scores_only_after_raw_feature_extraction(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir) / "dataset"
            cache_root = root / ".cache"
            metadata = {}
            for seed in range(8):
                reference_class = raw_distance.KNOWN_CLASSES[seed % len(raw_distance.KNOWN_CLASSES)]
                match_left = bool(seed % 2 == 0)
                match_name = "candidate_left" if match_left else "candidate_right"
                distractor_name = "candidate_right" if match_left else "candidate_left"
                metadata[str(seed)] = {
                    "reference_class": reference_class,
                    "distractor_class": "heavy-rough" if reference_class != "heavy-rough" else "light-smooth",
                    "match_candidate_public_name": match_name,
                    # These intentionally forbidden fields prove the fixture may carry them
                    # without the extractor needing them.
                    "reference_variant": {"density": 650.0, "friction_ratio": 2.8},
                    "tactile_weight_signature_reference": {"object_lift_delta": 99.0},
                }
                frames = _probe_frames("reference", 0, 3.0, 1.0)
                left_spec = (3.0, 1.0) if match_left else (1.0, 4.0)
                right_spec = (1.0, 4.0) if match_left else (3.0, 1.0)
                frames += _probe_frames("candidate_left", 30, *left_spec)
                frames += _probe_frames("candidate_right", 60, *right_spec)
                cache_dir = cache_root / str(seed)
                cache_dir.mkdir(parents=True)
                for frame in frames:
                    payload = {
                        "step": frame.step,
                        "atom": {"tag": frame.tag},
                        "tactile": {
                            "left_tactile": {"depth": frame.depth["left"], "marker": frame.marker["left"]},
                            "right_tactile": {"depth": frame.depth["right"], "marker": frame.marker["right"]},
                        },
                    }
                    with (cache_dir / f"{frame.step}.pkl").open("wb") as handle:
                        pickle.dump(payload, handle)
            root.mkdir(parents=True, exist_ok=True)
            (root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            summary = raw_distance.run_calibration(
                root,
                root / "analysis",
                folds=2,
                bootstrap_samples=8,
                min_valid_per_class=1,
            )
            self.assertEqual(summary["quality"]["valid_episodes"], 8)
            self.assertEqual(summary["metrics"]["combined"]["overall_accuracy"], 1.0)
            self.assertEqual(summary["metrics"]["semantic_signature_8d"]["overall_accuracy"], 1.0)
            self.assertTrue((root / "analysis" / "predictions.csv").is_file())
            self.assertTrue((root / "analysis" / "semantic_8d_features.csv").is_file())
            self.assertTrue((root / "analysis" / "figures" / "class_confusion_matrix.png").is_file())


if __name__ == "__main__":
    unittest.main()

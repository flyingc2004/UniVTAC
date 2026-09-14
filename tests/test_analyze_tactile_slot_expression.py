from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "analyze_tactile_slot_expression.py"
SPEC = importlib.util.spec_from_file_location("analyze_tactile_slot_expression", MODULE_PATH)
assert SPEC and SPEC.loader
ANALYZER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ANALYZER
SPEC.loader.exec_module(ANALYZER)


def _window(depth: float, displacement: float, coherence: float) -> dict:
    def hand(scale: float) -> dict:
        return {
            "depth_mm": depth * scale,
            "marker_displacement_px": displacement * scale,
            "marker_coherence": coherence,
            "contact_area": 0.5,
        }

    return {
        "frame_count": 8,
        "start_step": 1,
        "end_step": 8,
        "bilateral_contact_ratio": 1.0,
        "left": hand(1.0),
        "right": hand(1.01),
        "noise": {
            side: {field: {"value": 1.0, "mad": 0.01, "snr": 100.0} for field in (
                "depth_mm", "marker_displacement_px", "marker_coherence", "contact_area"
            )}
            for side in ("left", "right")
        },
        "gripper_qpos": 0.4,
    }


def _probe(class_key: str, *, valid: bool = True) -> dict:
    weight, roughness, hardness = class_key.split("_")
    values = {
        "weight": 1.0 if weight == "light" else 10.0,
        "roughness": 2.0 if roughness == "smooth" else 20.0,
        "hardness": 0.2 if hardness == "rigid" else 0.9,
    }
    preload = _window(values["weight"], 0.1, 0.1)
    motion = _window(values["weight"], values["roughness"] + 0.1, values["hardness"] + 0.1)
    delta = {
        side: {
            "depth_mm": motion[side]["depth_mm"] - preload[side]["depth_mm"],
            "marker_displacement_px": motion[side]["marker_displacement_px"] - preload[side]["marker_displacement_px"],
            "marker_coherence": motion[side]["marker_coherence"] - preload[side]["marker_coherence"],
        }
        for side in ("left", "right")
    }
    return {
        "schema_version": "tactile_probe.v3",
        "preload": preload,
        "lift_motion": motion,
        "hold": motion,
        "lift_minus_preload": delta,
        "quality": {
            "valid": valid,
            "preload_bilateral_contact_ratio": 1.0 if valid else 0.0,
            "lift_motion_bilateral_contact_ratio": 1.0 if valid else 0.0,
            "lift_motion_frame_count": 8 if valid else 0,
        },
    }


def _write_balanced_dataset(root: Path) -> None:
    public_records: dict[str, dict] = {}
    private_records: dict[str, dict] = {}
    classes = [
        f"{weight}_{roughness}_{hardness}"
        for weight in ("light", "heavy")
        for roughness in ("smooth", "rough")
        for hardness in ("rigid", "soft")
    ]
    seed = 3000
    for reference in classes:
        for distractor in classes:
            if distractor == reference:
                continue
            for repetition in range(5):
                match_name = "candidate_left" if (seed + repetition) % 2 == 0 else "candidate_right"
                candidates = {
                    "candidate_left": reference if match_name == "candidate_left" else distractor,
                    "candidate_right": reference if match_name == "candidate_right" else distractor,
                }
                public_records[str(seed)] = {
                    "schema_version": "tactile_memory_match_public_episode.v3",
                    "probes": {
                        "reference": _probe(reference),
                        "candidate_left": _probe(candidates["candidate_left"]),
                        "candidate_right": _probe(candidates["candidate_right"]),
                    },
                }
                private_records[str(seed)] = {
                    "seed": seed,
                    "result": "success",
                    "reference_class_key": reference,
                    "distractor_class_key": distractor,
                    "match_candidate_public_name": match_name,
                    "density": 999.0,
                }
                seed += 1
    (root / "metadata.json").write_text(json.dumps(public_records), encoding="utf-8")
    (root / "private_metadata.json").write_text(json.dumps(private_records), encoding="utf-8")


def test_frozen_slot_expression_uses_three_repetitions_then_two(tmp_path: Path):
    _write_balanced_dataset(tmp_path)
    report = ANALYZER.run(tmp_path, tmp_path / "analysis")

    assert report["schedule"]["complete"] is True
    assert report["schedule"]["observed_development_episodes"] == 168
    assert report["schedule"]["observed_frozen_episodes"] == 112
    assert report["identity"]["macro_accuracy"] == 1.0
    assert report["identity"]["coverage"] == 1.0
    assert all(report["slots"][slot]["macro_accuracy"] == 1.0 for slot in ANALYZER.SLOTS)

    expression = json.loads((tmp_path / "analysis" / "tactile_slot_expression.v1.json").read_text())
    serialized = json.dumps(expression)
    assert expression["schema_version"] == "tactile_slot_expression.v1"
    assert "density" not in serialized
    assert "reference_class_key" not in serialized
    assert set(expression["slots"]) == set(ANALYZER.SLOTS)


def test_invalid_probe_reduces_coverage_without_becoming_a_label(tmp_path: Path):
    _write_balanced_dataset(tmp_path)
    metadata_path = tmp_path / "metadata.json"
    public_records = json.loads(metadata_path.read_text())
    # This is a frozen repetition (the fourth record of the first ordered pair).
    public_records["3003"]["probes"]["candidate_left"] = _probe("light_smooth_rigid", valid=False)
    metadata_path.write_text(json.dumps(public_records), encoding="utf-8")

    report = ANALYZER.run(tmp_path, tmp_path / "analysis")
    assert report["identity"]["coverage"] < 1.0
    assert report["identity"]["coverage"] > 0.0

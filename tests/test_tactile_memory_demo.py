from __future__ import annotations

from tools.tactile_memory_demo import build_demo_memory


def _probe(static: float, dynamic: float, *, valid: bool = True) -> dict:
    def bilateral(value: float) -> dict:
        return {
            "left": {
                "depth_mm": value,
                "marker_displacement_px": value * 10.0,
                "marker_coherence": value * 0.1,
            },
            "right": {
                "depth_mm": value * 1.1,
                "marker_displacement_px": value * 11.0,
                "marker_coherence": value * 0.11,
            },
        }

    return {
        "preload": bilateral(static),
        "lift_minus_preload": bilateral(dynamic),
        "quality": {
            "valid": valid,
            "preload_bilateral_contact_ratio": 1.0 if valid else 0.0,
            "lift_motion_bilateral_contact_ratio": 1.0 if valid else 0.0,
            "lift_motion_frame_count": 8 if valid else 0,
        },
    }


def test_public_memory_selector_picks_nearest_candidate():
    memory = build_demo_memory(
        {
            "reference": _probe(2.0, 0.3),
            "candidate_left": _probe(6.0, 1.4),
            "candidate_right": _probe(2.1, 0.32),
        },
        "COMPARE MEMORY",
    )

    selection = memory["selection"]
    assert selection["status"] == "selected"
    assert selection["selected_candidate"] == "candidate_right"
    assert selection["fused_scores"]["candidate_right"] < selection["fused_scores"]["candidate_left"]
    assert selection["margin"] > 0.0
    vector = memory["evidence"]["reference"]["memory_vector"]
    assert vector["schema_version"] == "tactile_memory_vector.v1"
    assert vector["static_preload"]["left_depth_mm"] == 2.0
    assert vector["dynamic_lift_minus_preload"]["right_lift_marker_displacement_delta_px"] == 3.3


def test_public_memory_selector_refuses_invalid_probe():
    memory = build_demo_memory(
        {
            "reference": _probe(2.0, 0.3),
            "candidate_left": _probe(6.0, 1.4, valid=False),
            "candidate_right": _probe(2.1, 0.32),
        },
        "COMPARE MEMORY",
    )

    assert memory["selection"]["status"] == "pending"
    assert memory["selection"]["selected_candidate"] is None
    assert memory["evidence"]["candidate_left"]["quality"]["valid"] is False

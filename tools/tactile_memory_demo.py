"""Public-feature tactile memory selector used by the teaching demo.

The selector deliberately has no access to task metadata.  It receives the
three public ``tactile_probe.v3`` records, stores their human-readable static
and dynamic signatures, and selects the candidate with the smallest symmetric
per-trial normalized distance to the reference.
"""

from __future__ import annotations

from typing import Any

import numpy as np


SCHEMA_VERSION = "memory_demo.v1"
OBJECTS = ("reference", "candidate_left", "candidate_right")
FEATURES = (
    ("preload", "left", "depth_mm", "left_depth_mm"),
    ("preload", "right", "depth_mm", "right_depth_mm"),
    ("preload", "left", "marker_displacement_px", "left_marker_displacement_px"),
    ("preload", "right", "marker_displacement_px", "right_marker_displacement_px"),
    ("preload", "left", "marker_coherence", "left_marker_coherence"),
    ("preload", "right", "marker_coherence", "right_marker_coherence"),
    ("lift_minus_preload", "left", "depth_mm", "left_lift_depth_delta_mm"),
    ("lift_minus_preload", "right", "depth_mm", "right_lift_depth_delta_mm"),
    (
        "lift_minus_preload",
        "left",
        "marker_displacement_px",
        "left_lift_marker_displacement_delta_px",
    ),
    (
        "lift_minus_preload",
        "right",
        "marker_displacement_px",
        "right_lift_marker_displacement_delta_px",
    ),
    ("lift_minus_preload", "left", "marker_coherence", "left_lift_marker_coherence_delta"),
    ("lift_minus_preload", "right", "marker_coherence", "right_lift_marker_coherence_delta"),
)
STATIC_FEATURE_NAMES = tuple(feature[3] for feature in FEATURES[:6])
DYNAMIC_FEATURE_NAMES = tuple(feature[3] for feature in FEATURES[6:])
VECTOR_SCHEMA_VERSION = "tactile_memory_vector.v1"


def _quality(probe: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(probe, dict):
        return {"valid": False, "reason": "missing_probe"}
    quality = probe.get("quality")
    if not isinstance(quality, dict):
        return {"valid": False, "reason": "missing_quality"}
    valid = bool(quality.get("valid")) and int(quality.get("lift_motion_frame_count", 0)) > 0
    return {
        "valid": valid,
        "preload_bilateral_contact_ratio": float(quality.get("preload_bilateral_contact_ratio", 0.0)),
        "lift_motion_bilateral_contact_ratio": float(quality.get("lift_motion_bilateral_contact_ratio", 0.0)),
        "lift_motion_frame_count": int(quality.get("lift_motion_frame_count", 0)),
        "reason": None if valid else "probe_quality_invalid",
    }


def _feature_map(probe: dict[str, Any]) -> dict[str, float]:
    values: dict[str, float] = {}
    for section, hand, field, name in FEATURES:
        value = float(probe[section][hand][field])
        if not np.isfinite(value):
            raise ValueError(f"non-finite tactile feature: {name}")
        values[name] = value
    return values


def _memory_vector(features: dict[str, float]) -> dict[str, Any]:
    """Expose the exact public values used as the demo working memory."""
    return {
        "schema_version": VECTOR_SCHEMA_VERSION,
        "static_preload": {name: float(features[name]) for name in STATIC_FEATURE_NAMES},
        "dynamic_lift_minus_preload": {name: float(features[name]) for name in DYNAMIC_FEATURE_NAMES},
    }


def _symmetric_distance(
    reference: dict[str, float],
    left: dict[str, float],
    right: dict[str, float],
    feature_names: tuple[str, ...],
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """Score each candidate after normalizing each public field within trial."""
    details: dict[str, dict[str, float]] = {}
    squared: dict[str, list[float]] = {"candidate_left": [], "candidate_right": []}
    for name in feature_names:
        ref = reference[name]
        left_delta = left[name] - ref
        right_delta = right[name] - ref
        scale = max(abs(left_delta), abs(right_delta), 1e-9)
        details[name] = {
            "reference": ref,
            "candidate_left": left[name],
            "candidate_right": right[name],
            "normalization_scale": scale,
        }
        squared["candidate_left"].append((left_delta / scale) ** 2)
        squared["candidate_right"].append((right_delta / scale) ** 2)
    return (
        {name: float(np.sqrt(np.mean(values))) for name, values in squared.items()},
        details,
    )


def build_demo_memory(probes: dict[str, dict[str, Any]], phase: str) -> dict[str, Any]:
    """Build a serializable memory card and optional public-feature decision."""
    quality = {name: _quality(probes.get(name)) for name in OBJECTS}
    evidence: dict[str, dict[str, Any]] = {}
    for name in OBJECTS:
        probe = probes.get(name)
        if quality[name]["valid"]:
            try:
                features = _feature_map(probe)
                evidence[name] = {
                    "quality": quality[name],
                    "features": features,
                    "memory_vector": _memory_vector(features),
                }
            except (KeyError, TypeError, ValueError) as error:
                quality[name] = {**quality[name], "valid": False, "reason": str(error)}
                evidence[name] = {"quality": quality[name], "features": {}, "memory_vector": {}}
        else:
            evidence[name] = {"quality": quality[name], "features": {}, "memory_vector": {}}

    memory: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "phase": phase,
        "selector": "symmetric_public_feature_distance.v1",
        "evidence": evidence,
        "selection": {
            "status": "pending",
            "selected_candidate": None,
            "static_scores": {},
            "dynamic_scores": {},
            "fused_scores": {},
            "margin": None,
            "reason": "waiting_for_all_valid_probes",
        },
    }
    if not all(quality[name]["valid"] for name in OBJECTS):
        return memory

    reference = evidence["reference"]["features"]
    left = evidence["candidate_left"]["features"]
    right = evidence["candidate_right"]["features"]
    static_scores, static_features = _symmetric_distance(reference, left, right, STATIC_FEATURE_NAMES)
    dynamic_scores, dynamic_features = _symmetric_distance(reference, left, right, DYNAMIC_FEATURE_NAMES)
    fused_scores = {
        name: float(0.5 * static_scores[name] + 0.5 * dynamic_scores[name])
        for name in ("candidate_left", "candidate_right")
    }
    selected = min(fused_scores, key=fused_scores.get)
    other = "candidate_right" if selected == "candidate_left" else "candidate_left"
    memory["feature_details"] = {"static": static_features, "dynamic": dynamic_features}
    memory["selection"] = {
        "status": "selected",
        "selected_candidate": selected,
        "static_scores": static_scores,
        "dynamic_scores": dynamic_scores,
        "fused_scores": fused_scores,
        "margin": float(fused_scores[other] - fused_scores[selected]),
        "reason": "lowest_symmetric_public_feature_distance",
    }
    return memory

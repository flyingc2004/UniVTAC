#!/usr/bin/env python3
"""Summarize composable tactile-probe collection outcomes.

This is an offline diagnostic for official-expert collection.  It never
computes a tactile match or uses hidden physical labels as an input.  It only
reports the result already written by the collector plus public probe-quality
flags, so a failed seed can be assigned to reset/physics, probe, grasp, or
placement rather than being silently counted as an ordinary mismatch.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


OBJECTS = ("reference", "candidate_left", "candidate_right")
PROBE_SCHEMA_VERSION = "tactile_memory_match_public_episode.v3"
EXECUTION_FLAGS = (
    "approach_ok",
    "close_ok",
    "bilateral_gate",
    "lift_command_ok",
    "lower_ok",
    "release_ok",
    "clearance_ok",
)


def _load_mapping(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise RuntimeError(f"Invalid JSON in {path}: {error}") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected a JSON object in {path}")
    return {str(seed): value for seed, value in payload.items() if isinstance(value, dict)}


def _sort_seed(seed: str) -> tuple[int, str]:
    try:
        return int(seed), seed
    except ValueError:
        return 2**63 - 1, seed


def _load_suc_map(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    entries = path.read_text(encoding="utf-8").split()
    return {str(seed): value for seed, value in enumerate(entries) if value in {"0", "1"}}


def _probe_failure_reasons(probe: dict[str, Any]) -> list[str]:
    if not probe:
        return ["missing_probe"]
    quality = probe.get("quality")
    if not isinstance(quality, dict):
        return ["missing_quality"]
    if quality.get("valid") is True:
        return []

    reasons = [flag for flag in EXECUTION_FLAGS if quality.get(flag) is False]
    if float(quality.get("preload_bilateral_contact_ratio", 0.0)) <= 0.0:
        reasons.append("no_preload_bilateral_contact")
    if float(quality.get("lift_motion_bilateral_contact_ratio", 0.0)) <= 0.0:
        reasons.append("no_lift_bilateral_contact")
    return reasons or ["quality_invalid"]


def _probe_status(public_record: dict[str, Any]) -> tuple[dict[str, bool], list[str]]:
    probes = public_record.get("probes", {})
    if not isinstance(probes, dict):
        probes = {}
    valid: dict[str, bool] = {}
    reasons: list[str] = []
    for object_name in OBJECTS:
        probe = probes.get(object_name, {})
        quality = probe.get("quality", {}) if isinstance(probe, dict) else {}
        valid[object_name] = bool(isinstance(quality, dict) and quality.get("valid") is True)
        if not valid[object_name]:
            reasons.extend(f"{object_name}:{reason}" for reason in _probe_failure_reasons(probe))
    return valid, reasons


def _failure_reason(public_record: dict[str, Any], private_record: dict[str, Any]) -> str:
    failure_stage = private_record.get("failure_stage")
    if failure_stage:
        return str(failure_stage)

    _, probe_reasons = _probe_status(public_record)
    if probe_reasons:
        return ";".join(probe_reasons)

    result = private_record.get("result")
    if result:
        return f"collector_{result}"
    return "missing_terminal_metadata"


def _compact(value: Any) -> str:
    return "-" if value is None else str(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Collection directory containing metadata.json and private_metadata.json.",
    )
    parser.add_argument(
        "--max-failures",
        type=int,
        default=30,
        help="Maximum failed seed rows to print (default: 30).",
    )
    args = parser.parse_args()

    root = args.dataset_root.expanduser().resolve()
    public_records = _load_mapping(root / "metadata.json")
    private_records = _load_mapping(root / "private_metadata.json")
    suc_map = _load_suc_map(root / "suc_map.txt")
    seeds = sorted(set(public_records) | set(private_records) | set(suc_map), key=_sort_seed)

    terminal_counts: Counter[str] = Counter()
    failure_counts: Counter[str] = Counter()
    probe_valid_counts: Counter[str] = Counter()
    probe_invalid_reasons: Counter[str] = Counter()
    probe_v3_records = 0
    probe_non_v3_records = 0
    failed_rows: list[dict[str, str]] = []

    for seed in seeds:
        public_record = public_records.get(seed, {})
        private_record = private_records.get(seed, {})
        map_status = suc_map.get(seed)
        result = private_record.get("result")
        terminal = str(result or ("success" if map_status == "1" else "failed" if map_status == "0" else "unknown"))
        terminal_counts[terminal] += 1

        is_v3_probe_record = public_record.get("schema_version") == PROBE_SCHEMA_VERSION
        probe_valid, probe_reasons = _probe_status(public_record)
        if is_v3_probe_record:
            probe_v3_records += 1
            for object_name, valid in probe_valid.items():
                probe_valid_counts[f"{object_name}:{'valid' if valid else 'invalid'}"] += 1
            for reason in probe_reasons:
                probe_invalid_reasons[reason] += 1
        elif public_record:
            probe_non_v3_records += 1

        succeeded = map_status == "1" or result == "success"
        if succeeded:
            continue

        reason = _failure_reason(public_record, private_record)
        failure_counts[reason] += 1
        failed_rows.append(
            {
                "seed": seed,
                "result": terminal,
                "reason": reason,
                "probes": (
                    "/".join(f"{name[0].upper()}={'ok' if probe_valid[name] else 'bad'}" for name in OBJECTS)
                    if is_v3_probe_record
                    else "n/a"
                ),
                "selected": _compact(private_record.get("selected_candidate")),
                "match_placed": _compact(private_record.get("match_candidate_placed")),
                "distractor_placed": _compact(private_record.get("distractor_placed")),
            }
        )

    known_success = sum(1 for value in suc_map.values() if value == "1")
    known_attempted = len(suc_map)
    success_rate = 100.0 * known_success / known_attempted if known_attempted else 0.0
    print(f"Dataset root: {root}")
    print(
        "Seeds: "
        f"metadata={len(public_records)}, private={len(private_records)}, "
        f"collector_attempted={known_attempted}, collector_success={known_success}, "
        f"collector_success_rate={success_rate:.2f}%"
    )
    print("Terminal results:")
    for name, count in sorted(terminal_counts.items()):
        print(f"  {name}: {count}")
    print(f"Probe records: tactile_probe.v3={probe_v3_records}, non_v3_or_legacy={probe_non_v3_records}")
    print("Probe quality (tactile_probe.v3 only):")
    for object_name in OBJECTS:
        valid = probe_valid_counts[f"{object_name}:valid"]
        invalid = probe_valid_counts[f"{object_name}:invalid"]
        print(f"  {object_name}: valid={valid}, invalid={invalid}")
    if probe_invalid_reasons:
        print("Invalid probe causes:")
        for reason, count in probe_invalid_reasons.most_common():
            print(f"  {reason}: {count}")
    if failure_counts:
        print("Failed-seed causes:")
        for reason, count in failure_counts.most_common():
            print(f"  {reason}: {count}")

    if failed_rows:
        print("Failed seed details:")
        print("  seed | result | reason | probes | selected | match_placed | distractor_placed")
        for row in failed_rows[: max(args.max_failures, 0)]:
            print(
                "  {seed} | {result} | {reason} | {probes} | {selected} | "
                "{match_placed} | {distractor_placed}".format(**row)
            )
        remaining = len(failed_rows) - max(args.max_failures, 0)
        if remaining > 0:
            print(f"  ... {remaining} more failed seeds omitted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Video-oriented, transparent tactile-memory demonstration.

This task is intentionally separate from ``tactile_memory_match``.  It reuses
the benchmark's public probe and transport primitives but makes the candidate
choice with a visible public-feature memory card.  The private match label is
revealed only after that decision and is never passed into the selector.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from .tactile_memory_match import Task as BenchmarkTask
from .tactile_memory_match import TaskCfg as BenchmarkTaskCfg
from tools.tactile_memory_demo import OBJECTS, build_demo_memory
from isaaclab.utils import configclass


@configclass
class TaskCfg(BenchmarkTaskCfg):
    video_size = (1920, 480)


class Task(BenchmarkTask):
    """Benchmark wrapper that shows a demo-only working-memory card."""

    _panel_width = 780
    _panel_background = (20, 27, 35)
    _panel_text = (238, 242, 245)
    _panel_ok = (89, 201, 132)
    _panel_bad = (235, 103, 93)
    _panel_accent = (89, 177, 255)

    def _reset_episode_state(self):
        super()._reset_episode_state()
        self.demo_display_phase = "RESET: waiting for reference probe"
        self.demo_memory = build_demo_memory(self.tactile_probes, self.demo_display_phase)
        self.demo_oracle_audit: dict[str, Any] | None = None

    def _play_once(self):
        self.demo_display_phase = "PROBE REFERENCE: collect tactile evidence"
        self._probe_reference()
        self._refresh_demo_memory("REFERENCE STORED")
        if not self.plan_success:
            return

        self.demo_display_phase = "PROBE LEFT: same measurement protocol"
        self._probe_candidate("candidate_left")
        self._refresh_demo_memory("LEFT EVIDENCE STORED")
        if not self.plan_success:
            return

        self.demo_display_phase = "PROBE RIGHT: same measurement protocol"
        self._probe_candidate("candidate_right")
        self._refresh_demo_memory("COMPARE MEMORY")
        if not self.plan_success:
            return

        selection = self.demo_memory["selection"]
        selected = selection.get("selected_candidate")
        if selected not in {"candidate_left", "candidate_right"}:
            self.demo_display_phase = "STOP: insufficient tactile evidence"
            self._mark_failure("demo_memory_insufficient_evidence")
            self._refresh_demo_memory("STOP: insufficient tactile evidence")
            self.delay(10, is_save=True)
            return

        self.demo_display_phase = f"MEMORY SELECTED: {selected}"
        self._sync_metadata()
        self._select_and_place_candidate(str(selected))
        self._finalize_demo_audit()
        # The final audit must be visible in the recorded video, not only in
        # its JSON artifact.
        self.delay(18, is_save=True)

    def _refresh_demo_memory(self, phase: str):
        self.demo_display_phase = phase
        self.demo_memory = build_demo_memory(self.tactile_probes, phase)
        selection = self.demo_memory["selection"]
        self.metadata["demo_memory_selector"] = self.demo_memory["selector"]
        self.metadata["demo_memory_phase"] = phase
        self.metadata["demo_memory_selected_candidate"] = selection["selected_candidate"]
        self.metadata["demo_memory_margin"] = selection["margin"]

    def _finalize_demo_audit(self):
        selected = self.demo_memory["selection"].get("selected_candidate")
        self._update_task_state()
        self.demo_oracle_audit = {
            "revealed_after_selection": True,
            "selected_candidate": selected,
            "oracle_match_candidate": self.match_candidate_public_name,
            "selection_correct": bool(selected == self.match_candidate_public_name),
            "task_success": bool(self.check_success()),
            "match_candidate_placed": bool(self.match_candidate_placed),
            "distractor_placed": bool(self.distractor_placed),
        }
        self.demo_memory["oracle_audit"] = self.demo_oracle_audit
        self.demo_display_phase = "AUDIT REVEALED AFTER TRANSPORT"
        self.metadata["demo_memory_oracle_audit"] = self.demo_oracle_audit

    def _save_metadata(self):
        super()._save_metadata()
        if not hasattr(self, "demo_memory"):
            return
        path = Path(self.save_root) / "demo_memory_trace.json"
        payload = {
            "schema_version": "memory_demo_trace.v1",
            "seed": int(self.cfg.seed),
            "memory": self.demo_memory,
        }
        existing: dict[str, Any] = {}
        if path.exists():
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                existing = {}
        existing[str(self.cfg.seed)] = payload
        path.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")

    def get_frame_shot(self, obs):
        base = super().get_frame_shot(obs)
        if isinstance(base, torch.Tensor):
            frame = base.detach().cpu().numpy()
        else:
            frame = np.asarray(base)
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0.0, 1.0)
            frame = (frame * 255.0).astype(np.uint8)

        target_height = int(self.cfg.video_size[1])
        panel_width = self._panel_width
        visual_width = int(self.cfg.video_size[0]) - panel_width
        visual = cv2.resize(frame, (visual_width, target_height), interpolation=cv2.INTER_AREA)
        panel = self._render_memory_panel(target_height, panel_width)
        return np.concatenate((visual, panel), axis=1)

    def _render_memory_panel(self, height: int, width: int) -> np.ndarray:
        panel = np.full((height, width, 3), self._panel_background, dtype=np.uint8)
        # BaseTask renders a preview while reset is still settling.  The
        # episode-level memory fields are created slightly later by
        # _reset_episode_state, so the first frame is intentionally an empty
        # initialization card rather than an AttributeError.
        phase = getattr(self, "demo_display_phase", "INITIALIZING: reset settling")
        memory = getattr(self, "demo_memory", {})
        oracle_audit = getattr(self, "demo_oracle_audit", None)
        y = 24
        y = self._panel_line(panel, "TACTILE MEMORY DEMO", y, 0.62, self._panel_accent, 2)
        y = self._panel_line(panel, phase, y, 0.40, self._panel_text)
        y += 3

        evidence = memory.get("evidence", {})
        for name, short in (("reference", "A reference"), ("candidate_left", "B left"), ("candidate_right", "C right")):
            quality = evidence.get(name, {}).get("quality", {})
            valid = bool(quality.get("valid"))
            status = "stored" if valid else "waiting"
            color = self._panel_ok if valid else self._panel_text
            y = self._panel_line(panel, f"{short}: {status}", y, 0.47, color)

        y += 2
        y = self._panel_line(panel, "PUBLIC TACTILE MEMORY VECTOR", y, 0.45, self._panel_accent, 1)
        y = self._panel_line(panel, "field                         A ref       B left      C right", y, 0.34, self._panel_text)
        for label, feature, unit, accent in self._memory_vector_rows():
            values = [self._memory_value(evidence, name, feature) for name in ("reference", "candidate_left", "candidate_right")]
            formatted = "  ".join(self._format_memory_value(value, unit) for value in values)
            y = self._panel_line(panel, f"{label:<27} {formatted}", y, 0.33, accent)

        selection = memory.get("selection", {})
        if selection.get("status") == "selected":
            y += 4
            static_scores = selection["static_scores"]
            dynamic_scores = selection["dynamic_scores"]
            fused_scores = selection["fused_scores"]
            y = self._panel_line(
                panel,
                f"static  L {static_scores['candidate_left']:.3f}  R {static_scores['candidate_right']:.3f}",
                y,
                0.38,
                self._panel_text,
            )
            y = self._panel_line(
                panel,
                f"dynamic L {dynamic_scores['candidate_left']:.3f}  R {dynamic_scores['candidate_right']:.3f}",
                y,
                0.38,
                self._panel_text,
            )
            y = self._panel_line(
                panel,
                f"fused   L {fused_scores['candidate_left']:.3f}  R {fused_scores['candidate_right']:.3f}",
                y,
                0.38,
                self._panel_text,
            )
            y = self._panel_line(panel, f"margin: {selection['margin']:.3f}", y, 0.38, self._panel_text)
            y = self._panel_line(
                panel,
                f"SELECTED: {selection['selected_candidate']}",
                y + 4,
                0.52,
                self._panel_ok,
                2,
            )

        if oracle_audit is not None:
            audit = oracle_audit
            y += 4
            y = self._panel_line(panel, "ORACLE AUDIT (revealed now)", y, 0.42, self._panel_accent, 1)
            y = self._panel_line(panel, f"match: {audit['oracle_match_candidate']}", y, 0.38, self._panel_text)
            audit_color = self._panel_ok if audit["selection_correct"] else self._panel_bad
            status = "CORRECT" if audit["selection_correct"] else "INCORRECT"
            self._panel_line(panel, status, y, 0.52, audit_color, 2)
        return panel

    @staticmethod
    def _memory_vector_rows():
        return (
            ("S depth L [mm]", "left_depth_mm", "mm", Task._panel_text),
            ("S depth R [mm]", "right_depth_mm", "mm", Task._panel_text),
            ("S marker disp L [px]", "left_marker_displacement_px", "px", Task._panel_text),
            ("S marker disp R [px]", "right_marker_displacement_px", "px", Task._panel_text),
            ("S marker coherence L", "left_marker_coherence", "unit", Task._panel_text),
            ("S marker coherence R", "right_marker_coherence", "unit", Task._panel_text),
            ("D depth L [mm]", "left_lift_depth_delta_mm", "mm", Task._panel_ok),
            ("D depth R [mm]", "right_lift_depth_delta_mm", "mm", Task._panel_ok),
            ("D marker disp L [px]", "left_lift_marker_displacement_delta_px", "px", Task._panel_ok),
            ("D marker disp R [px]", "right_lift_marker_displacement_delta_px", "px", Task._panel_ok),
            ("D marker coherence L", "left_lift_marker_coherence_delta", "unit", Task._panel_ok),
            ("D marker coherence R", "right_lift_marker_coherence_delta", "unit", Task._panel_ok),
        )

    @staticmethod
    def _memory_value(evidence: dict[str, Any], object_name: str, feature: str) -> float | None:
        value = evidence.get(object_name, {}).get("features", {}).get(feature)
        return float(value) if isinstance(value, (int, float)) else None

    @staticmethod
    def _format_memory_value(value: float | None, unit: str) -> str:
        if value is None:
            return "-          "
        precision = 4 if unit == "unit" else 3
        return f"{value:>9.{precision}f}"

    @staticmethod
    def _panel_line(
        panel: np.ndarray,
        text: str,
        y: int,
        scale: float,
        color: tuple[int, int, int],
        thickness: int = 1,
    ) -> int:
        max_width = panel.shape[1] - 16
        while scale > 0.28 and cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0][0] > max_width:
            scale -= 0.02
        cv2.putText(panel, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)
        return y + max(18, int(20 * scale + 10))

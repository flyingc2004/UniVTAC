from ._base_task import *
import csv
import json
import numpy as np


TACTILE_CLASSES = {
    "light_smooth": {
        "label": "light-smooth",
        "asset": "Can_d4cm.usd",
        "diameter": 4,
        "density": 300.0,
        "friction_ratio": 1.5,
    },
    "light_rough": {
        "label": "light-rough",
        "asset": "Can_d4cm.usd",
        "diameter": 4,
        "density": 300.0,
        "friction_ratio": 2.8,
    },
    "heavy_smooth": {
        "label": "heavy-smooth",
        "asset": "Can_d4cm.usd",
        "diameter": 4,
        "density": 650.0,
        "friction_ratio": 1.5,
    },
    "heavy_rough": {
        "label": "heavy-rough",
        "asset": "Can_d4cm.usd",
        "diameter": 4,
        "density": 650.0,
        "friction_ratio": 2.8,
    },
}


@configclass
class TaskCfg(BaseTaskCfg):
    step_lim = 500
    use_adaptive_grasp = True
    adaptive_grasp_depth_threshold = 27.75


class Task(BaseTask):
    reference_xy = np.array([0.68, 0.24], dtype=np.float64)
    candidate_xy = {
        "candidate_left": np.array([0.60, -0.345], dtype=np.float64),
        "candidate_right": np.array([0.60, -0.175], dtype=np.float64),
    }
    match_slot_xy = np.array([0.42, 0.00], dtype=np.float64)
    stash_xy = np.array([-2.0, -2.0], dtype=np.float64)
    can_rot = [1.0, 0.0, 0.0, 0.0]
    xy_jitter = 0.006
    candidate_jitter = 0.004
    safe_gripper_z = 0.160
    release_retreat_z = 0.130
    release_z_clearance = 0.018
    transport_xy_step = 0.025
    descend_z_step = 0.008
    placement_mode = "pad_overlap"
    placement_supported_z_threshold = 0.012
    placement_stable_steps_required = 10
    pad_half_extents = np.array([0.045, 0.045], dtype=np.float64)
    can_length = 0.120
    placement_overlap_margin = 0.004
    timeline_frequency = 5
    probe_delay_steps = 18
    weight_probe_lift_height = 0.018
    weight_probe_hold_steps = 10
    weight_probe_return_steps = 6
    post_release_wait_steps = 35
    occlusion_enabled = True
    occlusion_opacity = 1.0
    occlusion_center_xy = np.array([0.60, -0.26], dtype=np.float64)
    occlusion_inner_width = 0.42
    occlusion_inner_depth = 0.36
    occlusion_wall_base_z = 0.001
    occlusion_wall_height = 0.32
    occlusion_wall_thickness = 0.025
    occlusion_lid_height = 0.180
    occlusion_lid_thickness = 0.030
    occlusion_color = np.array([0.01, 0.01, 0.012], dtype=np.float32)
    capx_premove_role = "reference_object"
    capx_premove_time_dilation_factor = 0.5

    def __init__(
        self,
        cfg: BaseTaskCfg,
        mode: Literal["collect", "eval"] = "collect",
        render_mode: str | None = None,
        **kwargs,
    ):
        cfg.sim.physics_material.dynamic_friction = 1.5
        cfg.sim.physics_material.static_friction = 1.5
        cfg.uipc_sim.contact.default_friction_ratio = 2.5
        super().__init__(cfg, mode, render_mode, **kwargs)

    def create_actors(self):
        self.match_slot = self._actor_manager.add_from_usd_file(
            name="match_slot",
            asset_path="GreenPad.usd",
            pose=Pose([self.match_slot_xy[0], self.match_slot_xy[1], 0.01], [1, 0, 0, 0]),
            density=1e5,
        )
        self._create_occlusion_box()

        self.reference_actors: dict[str, Actor] = {}
        self.candidate_actors: dict[str, Actor] = {}
        for idx, (class_key, class_cfg) in enumerate(TACTILE_CLASSES.items()):
            pose = self._stash_pose(idx)
            self.reference_actors[class_key] = self._actor_manager.add_from_usd_file(
                name=f"reference_variant_{idx}",
                asset_path=str(class_cfg["asset"]),
                pose=pose,
                density=float(class_cfg["density"]),
                friction_ratio=float(class_cfg["friction_ratio"]),
            )
            self.candidate_actors[class_key] = self._actor_manager.add_from_usd_file(
                name=f"candidate_variant_{idx}",
                asset_path=str(class_cfg["asset"]),
                pose=self._stash_pose(idx + len(TACTILE_CLASSES)),
                density=float(class_cfg["density"]),
                friction_ratio=float(class_cfg["friction_ratio"]),
            )

    def _reset_actors(self):
        for idx, actor in enumerate(self.reference_actors.values()):
            actor.set_pose(self._stash_pose(idx))
        for idx, actor in enumerate(self.candidate_actors.values()):
            actor.set_pose(self._stash_pose(idx + len(TACTILE_CLASSES)))

        class_keys = list(TACTILE_CLASSES.keys())
        self.reference_class_key = str(self.rng.choice(class_keys))
        distractor_choices = [key for key in class_keys if key != self.reference_class_key]
        self.distractor_class_key = str(self.rng.choice(distractor_choices))
        self.reference_class = TACTILE_CLASSES[self.reference_class_key]
        self.distractor_class = TACTILE_CLASSES[self.distractor_class_key]
        self.match_on_left = bool(self.rng.random() < 0.5)
        self.match_candidate_public_name = "candidate_left" if self.match_on_left else "candidate_right"
        self.distractor_candidate_public_name = "candidate_right" if self.match_on_left else "candidate_left"

        self.reference_object = self.reference_actors[self.reference_class_key]
        self.match_actor = self.candidate_actors[self.reference_class_key]
        self.distractor_actor = self.candidate_actors[self.distractor_class_key]
        self.public_candidate_actors = {
            self.match_candidate_public_name: self.match_actor,
            self.distractor_candidate_public_name: self.distractor_actor,
        }
        self.public_actor_map = {
            "reference_object": self.reference_object,
            "candidate_left": self.public_candidate_actors["candidate_left"],
            "candidate_right": self.public_candidate_actors["candidate_right"],
        }

        self.reference_start_pose = Pose(
            [*self._jitter_xy(self.reference_xy, self.xy_jitter), self._resting_z(self.reference_class)],
            self.can_rot,
        )
        self.reference_object.set_pose(self.reference_start_pose)

        self.candidate_start_poses = {}
        for public_name, actor in self.public_candidate_actors.items():
            variant = self._variant_for_public_candidate(public_name)
            pose = Pose(
                [
                    *self._jitter_xy(self.candidate_xy[public_name], self.candidate_jitter),
                    self._resting_z(variant),
                ],
                self.can_rot,
            )
            self.candidate_start_poses[public_name] = pose
            actor.set_pose(pose)

        self.match_slot_pose = Pose(
            [self.match_slot_xy[0], self.match_slot_xy[1], self._resting_z(self.reference_class)],
            self.can_rot,
        )
        self._reset_episode_state()

    def _reset_episode_state(self):
        self.task_phase = "reset"
        self.active_public_name = None
        self.selected_candidate = None
        self.selection_correct = False
        self.reference_touched = False
        self.reference_lifted = False
        self.reference_tactile_signature_valid = False
        self.candidate_touched = {"candidate_left": False, "candidate_right": False}
        self.candidate_placed = {"candidate_left": False, "candidate_right": False}
        self.candidate_place_stable_count = {"candidate_left": 0, "candidate_right": 0}
        self.match_candidate_placed = False
        self.distractor_placed = False
        self.failure_stage = None
        self.tactile_timeline = []
        self.reference_initial_z = float(self.reference_object.get_pose().p[2])
        self.candidate_initial_z = {
            name: float(actor.get_pose().p[2])
            for name, actor in self.public_candidate_actors.items()
        }
        self.tactile_signatures = {
            "reference": {},
            "candidate_left": {},
            "candidate_right": {},
        }
        self.tactile_weight_signatures = {
            "reference": {},
            "candidate_left": {},
            "candidate_right": {},
        }
        self.metadata.update(
            {
                "reference_class": str(self.reference_class["label"]),
                "reference_variant": self._public_variant(self.reference_class),
                "distractor_class": str(self.distractor_class["label"]),
                "distractor_variant": self._public_variant(self.distractor_class),
                "match_candidate_public_name": self.match_candidate_public_name,
                "distractor_candidate_public_name": self.distractor_candidate_public_name,
                "candidate_left_internal_role": (
                    "match" if self.match_candidate_public_name == "candidate_left" else "distractor"
                ),
                "candidate_right_internal_role": (
                    "match" if self.match_candidate_public_name == "candidate_right" else "distractor"
                ),
                "reference_start_pose": self.reference_start_pose.tolist(),
                "candidate_left_start_pose": self.candidate_start_poses["candidate_left"].tolist(),
                "candidate_right_start_pose": self.candidate_start_poses["candidate_right"].tolist(),
                "match_slot_pose": self.match_slot_pose.tolist(),
                "selected_candidate": None,
                "selection_correct": False,
                "candidate_left_touched": False,
                "candidate_right_touched": False,
                "candidate_left_placed": False,
                "candidate_right_placed": False,
                "match_candidate_placed": False,
                "distractor_placed": False,
                "reference_touched": False,
                "reference_lifted": False,
                "reference_tactile_signature_valid": False,
                "tactile_signature_reference": {},
                "tactile_signature_candidate_left": {},
                "tactile_signature_candidate_right": {},
                "tactile_weight_signature_reference": {},
                "tactile_weight_signature_candidate_left": {},
                "tactile_weight_signature_candidate_right": {},
                "occlusion_enabled": bool(self.occlusion_enabled),
                "occlusion_opacity": float(self.occlusion_opacity),
                "occlusion_wall_base_z": float(self.occlusion_wall_base_z),
                "occlusion_wall_height": float(self.occlusion_wall_height),
                "occlusion_inner_width": float(self.occlusion_inner_width),
                "occlusion_inner_depth": float(self.occlusion_inner_depth),
                "wrist_view_darkened": bool(self.occlusion_enabled),
                "expert": "tactile_similarity_match",
                "expert_policy": "probe_reference_probe_candidates_select_by_tactile_similarity",
                "selection_method": None,
                "tactile_similarity_scores": {},
                "tactile_static_similarity_scores": {},
                "tactile_weight_similarity_scores": {},
                "tactile_similarity_margin": None,
                "tactile_weight_similarity_margin": None,
                "failure_stage": None,
            }
        )

    def _create_occlusion_box(self):
        if not self.occlusion_enabled:
            self.occlusion_walls = []
            return

        center = np.asarray(self.occlusion_center_xy, dtype=np.float64)
        height = float(self.occlusion_wall_height)
        thickness = float(self.occlusion_wall_thickness)
        inner_width = float(self.occlusion_inner_width)
        inner_depth = float(self.occlusion_inner_depth)
        z_center = float(self.occlusion_wall_base_z) + 0.5 * height
        lid_z = float(self.occlusion_lid_height)
        color = np.asarray(self.occlusion_color, dtype=np.float32)

        wall_specs = [
            (
                "memory_match_occlusion_back",
                [center[0] + 0.5 * inner_depth + 0.5 * thickness, center[1], z_center],
                [thickness, inner_width + 2.0 * thickness, height],
            ),
            (
                "memory_match_occlusion_front",
                [center[0] - 0.5 * inner_depth - 0.5 * thickness, center[1], z_center],
                [thickness, inner_width + 2.0 * thickness, height],
            ),
            (
                "memory_match_occlusion_left",
                [center[0], center[1] - 0.5 * inner_width - 0.5 * thickness, z_center],
                [inner_depth + 2.0 * thickness, thickness, height],
            ),
            (
                "memory_match_occlusion_right",
                [center[0], center[1] + 0.5 * inner_width + 0.5 * thickness, z_center],
                [inner_depth + 2.0 * thickness, thickness, height],
            ),
            (
                "memory_match_occlusion_camera_shield",
                [center[0] - 0.025, center[1] + 0.5 * inner_width + 2.0 * thickness, z_center],
                [inner_depth + 0.08, thickness, height],
            ),
            (
                "memory_match_occlusion_lid",
                [center[0], center[1], lid_z],
                [
                    inner_depth + 2.0 * thickness,
                    inner_width + 2.0 * thickness,
                    float(self.occlusion_lid_thickness),
                ],
            ),
        ]

        self.occlusion_walls = []
        for name, position, scale in wall_specs:
            wall = VisualCuboid(
                prim_path=f"/World/envs/env_0/{name}",
                name=name,
                position=np.asarray(position, dtype=np.float32),
                orientation=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                scale=np.asarray(scale, dtype=np.float32),
                size=1.0,
                color=color,
            )
            self.occlusion_walls.append(wall)

    def pre_move(self):
        self.task_phase = "premove_reference"
        self.delay(10, is_save=False)
        self.move(self.atom.open_gripper(1.0), tag="reference_premove_open", is_save=False)

        grasp_pose = self._lift_can_style_grasp_pose(self.reference_object)
        grasp_idx = self.reference_object.register_point(pose=grasp_pose, type="contact")
        ok = self.move(
            self.atom.grasp_actor(self.reference_object, contact_point_id=grasp_idx, is_close=False),
            tag="reference_premove_anchor",
            time_dilation_factor=self.capx_premove_time_dilation_factor,
            is_save=False,
        )
        if not ok:
            self._mark_failure("reference_premove_anchor_failed")
        self.delay(5, is_save=False)

    def _play_once(self):
        self._probe_reference()
        if not self.plan_success:
            return
        self._probe_candidate("candidate_left")
        if not self.plan_success:
            return
        self._probe_candidate("candidate_right")
        if not self.plan_success:
            return
        selected = self._select_candidate_by_tactile_similarity()
        self._select_and_place_candidate(selected)
        self.delay(20, is_save=False)

    def _probe_reference(self) -> bool:
        self.active_public_name = "reference_object"
        self.task_phase = "reference_grasp"
        if not self._close_current_grasp("reference_close"):
            self._mark_failure("reference_close_failed")
            return False
        self.delay(self.probe_delay_steps, is_save=True)
        self.reference_touched = self._has_tactile_contact()
        signature = self._capture_tactile_signature("reference")
        self.reference_tactile_signature_valid = bool(signature.get("contact", False))
        weight_signature = self._probe_weight_response(
            "reference",
            self.reference_object,
            signature,
            phase="reference_lift",
        )
        self.reference_lifted = bool(weight_signature.get("object_lift_delta", 0.0) > 0.010)
        self.move(self.atom.open_gripper(1.0), tag="reference_release_open", is_save=True)
        self.delay(10, is_save=True)
        self._sync_metadata()
        return True

    def _probe_candidate(self, public_name: str) -> bool:
        self.active_public_name = public_name
        self.task_phase = f"{public_name}_probe"
        actor = self.public_candidate_actors[public_name]
        if not self._move_to_actor_grasp(public_name, actor, tag=f"{public_name}_probe_grasp_actor"):
            self._mark_failure(f"{public_name}_probe_approach_failed")
            return False
        if not self._close_current_grasp(f"{public_name}_probe_close"):
            self._mark_failure(f"{public_name}_probe_close_failed")
            return False
        self.delay(self.probe_delay_steps, is_save=True)
        self.candidate_touched[public_name] = self._has_tactile_contact()
        signature = self._capture_tactile_signature(public_name)
        self._probe_weight_response(
            public_name,
            actor,
            signature,
            phase=f"{public_name}_weight_probe",
        )
        self.move(self.atom.open_gripper(1.0), tag=f"{public_name}_probe_open", is_save=True)
        self.delay(10, is_save=True)
        self._sync_metadata()
        return True

    def _select_candidate_by_tactile_similarity(self) -> str:
        reference = self.tactile_signatures.get("reference", {})
        reference_weight = self.tactile_weight_signatures.get("reference", {})
        static_scores = {
            public_name: self._tactile_signature_distance(
                reference,
                self.tactile_signatures.get(public_name, {}),
            )
            for public_name in ("candidate_left", "candidate_right")
        }
        weight_scores = {
            public_name: self._weight_signature_distance(
                reference_weight,
                self.tactile_weight_signatures.get(public_name, {}),
            )
            for public_name in ("candidate_left", "candidate_right")
        }
        scores = {}
        for public_name in ("candidate_left", "candidate_right"):
            weight_score = weight_scores[public_name]
            scores[public_name] = static_scores[public_name]
            if np.isfinite(weight_score):
                scores[public_name] += weight_score

        selected = min(scores, key=scores.get)
        other = "candidate_right" if selected == "candidate_left" else "candidate_left"
        self.metadata["selection_method"] = "tactile_static_weight_similarity"
        self.metadata["tactile_similarity_scores"] = {
            name: float(score) for name, score in scores.items()
        }
        self.metadata["tactile_static_similarity_scores"] = {
            name: float(score) for name, score in static_scores.items()
        }
        self.metadata["tactile_weight_similarity_scores"] = {
            name: float(score) for name, score in weight_scores.items()
        }
        self.metadata["tactile_similarity_margin"] = float(scores[other] - scores[selected])
        self.metadata["tactile_weight_similarity_margin"] = float(
            weight_scores[other] - weight_scores[selected]
            if np.isfinite(weight_scores[other]) and np.isfinite(weight_scores[selected])
            else 0.0
        )
        print(
            "[tactile-memory-match] selection_method=tactile_static_weight_similarity "
            f"selected={selected} "
            f"candidate_left={scores['candidate_left']:.4f} "
            f"candidate_right={scores['candidate_right']:.4f} "
            f"static_left={static_scores['candidate_left']:.4f} "
            f"static_right={static_scores['candidate_right']:.4f} "
            f"weight_left={weight_scores['candidate_left']:.4f} "
            f"weight_right={weight_scores['candidate_right']:.4f} "
            f"margin={self.metadata['tactile_similarity_margin']:.4f}"
        )
        return selected

    def _probe_weight_response(self, key: str, actor: Actor, before_signature: dict, phase: str) -> dict:
        previous_phase = self.task_phase
        self.task_phase = phase
        start_actor_z = float(actor.get_pose().p[2])
        start_gripper_z = float(self._robot_manager.get_gripper_center_pose().p[2])
        commanded_lift = float(self.weight_probe_lift_height)

        ok = self._role_move(
            key,
            self.atom.move_by_displacement(z=commanded_lift, xyz_coord="world"),
            tag=f"{key}_weight_lift",
            time_dilation_factor=0.5,
            is_save=True,
        )
        self.delay(self.weight_probe_hold_steps, is_save=True)

        after_signature = self._read_tactile_signature()
        end_actor_z = float(actor.get_pose().p[2])
        end_gripper_z = float(self._robot_manager.get_gripper_center_pose().p[2])
        object_lift_delta = float(end_actor_z - start_actor_z)
        gripper_lift_delta = float(end_gripper_z - start_gripper_z)
        lift_follow_ratio = float(object_lift_delta / max(abs(gripper_lift_delta), 1e-6))
        depth_delta_drop = float(
            before_signature.get("mean_depth_delta_mm", 0.0)
            - after_signature.get("mean_depth_delta_mm", 0.0)
        )
        contact_area_drop = float(
            before_signature.get("mean_contact_area", 0.0)
            - after_signature.get("mean_contact_area", 0.0)
        )
        contact_lost = bool(before_signature.get("contact", False) and not after_signature.get("contact", False))
        response = {
            "valid": bool(ok),
            "commanded_lift": commanded_lift,
            "object_lift_delta": object_lift_delta,
            "gripper_lift_delta": gripper_lift_delta,
            "lift_follow_ratio": lift_follow_ratio,
            "inhand_z_error": float(abs(end_gripper_z - end_actor_z)),
            "before_depth_delta_mm": float(before_signature.get("mean_depth_delta_mm", 0.0)),
            "after_depth_delta_mm": float(after_signature.get("mean_depth_delta_mm", 0.0)),
            "depth_delta_drop": depth_delta_drop,
            "before_contact_area": float(before_signature.get("mean_contact_area", 0.0)),
            "after_contact_area": float(after_signature.get("mean_contact_area", 0.0)),
            "contact_area_drop": contact_area_drop,
            "contact_lost": contact_lost,
            "after_contact": bool(after_signature.get("contact", False)),
            "after_both_contact": bool(after_signature.get("both_contact", False)),
            "gripper_qpos_after": float(self._robot_manager.get_gripper_qpos()),
            "step": int(self.step_count),
            "phase": str(phase),
        }
        self._store_weight_signature(key, response)

        if ok:
            self._role_move(
                key,
                self.atom.move_by_displacement(z=-commanded_lift, xyz_coord="world"),
                tag=f"{key}_weight_lower",
                time_dilation_factor=0.5,
                is_save=True,
            )
            self.delay(self.weight_probe_return_steps, is_save=True)
        self.task_phase = previous_phase
        self._sync_metadata()
        return response

    def _store_weight_signature(self, key: str, signature: dict):
        if key in self.tactile_weight_signatures:
            self.tactile_weight_signatures[key] = signature
        if key == "reference":
            self.metadata["tactile_weight_signature_reference"] = signature
        elif key in {"candidate_left", "candidate_right"}:
            self.metadata[f"tactile_weight_signature_{key}"] = signature

    @classmethod
    def _weight_signature_distance(cls, reference: dict, candidate: dict) -> float:
        if not reference.get("valid", False) or not candidate.get("valid", False):
            return float("inf")
        features = [
            ("object_lift_delta", 0.025, 1.0),
            ("gripper_lift_delta", 0.025, 0.5),
            ("lift_follow_ratio", 1.0, 1.0),
            ("depth_delta_drop", 6.0, 0.75),
            ("contact_area_drop", 0.50, 0.75),
            ("after_depth_delta_mm", 6.0, 0.75),
            ("after_contact_area", 0.50, 0.75),
            ("inhand_z_error", 0.20, 0.5),
            ("gripper_qpos_after", 0.02, 0.5),
        ]
        penalty = 0.0
        if bool(reference.get("contact_lost", False)) != bool(candidate.get("contact_lost", False)):
            penalty += 1.0
        if bool(reference.get("after_both_contact", False)) != bool(candidate.get("after_both_contact", False)):
            penalty += 0.5

        weighted_sum = 0.0
        weight_sum = 0.0
        for key, scale, weight in features:
            ref_value = reference.get(key)
            cand_value = candidate.get(key)
            if ref_value is None or cand_value is None:
                continue
            ref_value = float(ref_value)
            cand_value = float(cand_value)
            if not np.isfinite(ref_value) or not np.isfinite(cand_value):
                continue
            normalized = (cand_value - ref_value) / max(float(scale), 1e-6)
            weighted_sum += float(weight) * normalized * normalized
            weight_sum += float(weight)

        if weight_sum <= 0.0:
            return float("inf")
        return float(np.sqrt(weighted_sum / weight_sum) + penalty)

    @classmethod
    def _tactile_signature_distance(cls, reference: dict, candidate: dict) -> float:
        features = [
            (("mean_depth_delta_mm",), 6.0, 1.0),
            (("mean_contact_area",), 0.50, 1.0),
            (("gripper_qpos",), 0.02, 0.5),
            (("left", "depth_delta_mm"), 6.0, 0.75),
            (("right", "depth_delta_mm"), 6.0, 0.75),
            (("left", "contact_area"), 0.50, 0.75),
            (("right", "contact_area"), 0.50, 0.75),
            (("left", "marker_centroid_x"), 320.0, 0.15),
            (("right", "marker_centroid_x"), 320.0, 0.15),
            (("left", "marker_centroid_y"), 240.0, 0.15),
            (("right", "marker_centroid_y"), 240.0, 0.15),
        ]

        def get_nested(data: dict, path: tuple[str, ...]):
            value = data
            for key in path:
                if not isinstance(value, dict) or key not in value:
                    return None
                value = value[key]
            return value

        penalty = 0.0
        if not reference.get("contact", False):
            penalty += 10.0
        if not candidate.get("contact", False):
            penalty += 10.0
        if bool(reference.get("both_contact", False)) != bool(candidate.get("both_contact", False)):
            penalty += 1.0

        weighted_sum = 0.0
        weight_sum = 0.0
        for path, scale, weight in features:
            ref_value = get_nested(reference, path)
            cand_value = get_nested(candidate, path)
            if ref_value is None or cand_value is None:
                continue
            ref_value = float(ref_value)
            cand_value = float(cand_value)
            if not np.isfinite(ref_value) or not np.isfinite(cand_value):
                continue
            normalized = (cand_value - ref_value) / max(float(scale), 1e-6)
            weighted_sum += float(weight) * normalized * normalized
            weight_sum += float(weight)

        if weight_sum <= 0.0:
            return float("inf")
        return float(np.sqrt(weighted_sum / weight_sum) + penalty)

    def _select_and_place_candidate(self, public_name: str) -> bool:
        self.selected_candidate = public_name
        self.selection_correct = public_name == self.match_candidate_public_name
        self.active_public_name = public_name
        self.task_phase = "candidate_selected"
        self._sync_metadata()

        actor = self.public_candidate_actors[public_name]
        if not self._move_to_actor_grasp(public_name, actor, tag=f"{public_name}_final_grasp_actor"):
            self._mark_failure(f"{public_name}_final_approach_failed")
            return False
        if not self._close_current_grasp(f"{public_name}_final_close"):
            self._mark_failure(f"{public_name}_final_close_failed")
            return False

        self.task_phase = "candidate_lift"
        start_z = float(actor.get_pose().p[2])
        if not self._role_move(
            public_name,
            self.atom.move_by_displacement(z=0.030),
            tag=f"{public_name}_final_lift",
            time_dilation_factor=0.5,
            is_save=True,
        ):
            self._mark_failure(f"{public_name}_final_lift_failed")
            return False
        self.delay(12, is_save=True)
        self.metadata[f"{public_name}_final_lift_delta"] = float(actor.get_pose().p[2] - start_z)

        self.task_phase = "candidate_place"
        if not self._transport_held_actor_xy(public_name, actor, self.match_slot_pose.p[:2]):
            self._mark_failure(f"{public_name}_transport_failed")
            return False
        release_z = float(self.match_slot_pose.p[2] + self.release_z_clearance)
        if not self._descend_held_actor_to_z(public_name, actor, release_z):
            self._mark_failure(f"{public_name}_descend_failed")
            return False
        if not self._role_move(public_name, self.atom.open_gripper(1.0), tag=f"{public_name}_release_open", is_save=True):
            self._mark_failure(f"{public_name}_release_failed")
            return False
        self.delay(self.post_release_wait_steps, is_save=True)
        self._retreat_after_release(public_name)
        self.delay(10, is_save=True)
        self._update_task_state()
        return True

    def _move_to_actor_grasp(self, public_name: str, actor: Actor, tag: str) -> bool:
        if not self._move_gripper_center_to_z(self.safe_gripper_z, tag=f"{public_name}_safe_z"):
            return False
        self._role_move(public_name, self.atom.open_gripper(1.0), tag=f"{public_name}_open", is_save=True)
        grasp_pose = self._lift_can_style_grasp_pose(actor)
        grasp_idx = actor.register_point(pose=grasp_pose, type="contact")
        return self._role_move(
            public_name,
            self.atom.grasp_actor(actor, contact_point_id=grasp_idx, is_close=False),
            tag=tag,
            time_dilation_factor=0.5,
            is_save=True,
        )

    def _close_current_grasp(self, tag: str) -> bool:
        ok = self._role_move(
            self.active_public_name or "unknown",
            self.atom.close_gripper(0.0, depth_threshold="auto"),
            tag=tag,
            is_save=True,
        )
        return bool(ok)

    def _transport_held_actor_xy(self, public_name: str, actor: Actor, target_xy: np.ndarray) -> bool:
        target_xy = np.asarray(target_xy, dtype=np.float64).reshape(2)
        for idx in range(32):
            actor_xy = np.asarray(actor.get_pose().p[:2], dtype=np.float64)
            delta_xy = target_xy - actor_xy
            xy_error = float(np.linalg.norm(delta_xy))
            if xy_error < 0.012:
                return True
            step_xy = np.zeros(2, dtype=np.float64)
            axis = int(np.argmax(np.abs(delta_xy)))
            step_xy[axis] = float(np.clip(delta_xy[axis], -self.transport_xy_step, self.transport_xy_step))
            if not self._role_move(
                public_name,
                self.atom.move_by_displacement(
                    x=float(step_xy[0]),
                    y=float(step_xy[1]),
                    z=0.0,
                    xyz_coord="world",
                ),
                tag=f"{public_name}_transport_xy_{idx}",
                time_dilation_factor=0.5,
                is_save=True,
            ):
                return False
            self.delay(8, is_save=True)

        actor_xy = np.asarray(actor.get_pose().p[:2], dtype=np.float64)
        self.metadata[f"{public_name}_transport_xy_error"] = float(np.linalg.norm(target_xy - actor_xy))
        self._sync_metadata()
        return float(np.linalg.norm(target_xy - actor_xy)) < 0.04

    def _descend_held_actor_to_z(self, public_name: str, actor: Actor, release_z: float) -> bool:
        for idx in range(10):
            actor_z = float(actor.get_pose().p[2])
            delta_z = float(release_z - actor_z)
            if abs(delta_z) < 0.006:
                return True
            step_z = float(np.clip(delta_z, -self.descend_z_step, self.descend_z_step))
            if not self._role_move(
                public_name,
                self.atom.move_by_displacement(z=step_z, xyz_coord="world"),
                tag=f"{public_name}_descend_{idx}",
                time_dilation_factor=0.5,
                is_save=True,
            ):
                return False
            self.delay(8, is_save=True)

        actor_z = float(actor.get_pose().p[2])
        self.metadata[f"{public_name}_release_z_error"] = float(abs(release_z - actor_z))
        self._sync_metadata()
        return abs(release_z - actor_z) < 0.03

    def _retreat_after_release(self, public_name: str) -> bool:
        return self._move_gripper_center_to_z(self.release_retreat_z, tag=f"{public_name}_release_retreat")

    def _move_gripper_center_to_z(self, target_z: float, tag: str) -> bool:
        gripper_center = self._robot_manager.get_gripper_center_pose()
        current_z = float(gripper_center.p[2])
        target_z = max(float(target_z), current_z)
        if abs(target_z - current_z) < 0.005:
            return True
        target_gripper = Pose(
            [gripper_center.p[0], gripper_center.p[1], target_z],
            gripper_center.q,
        )
        target_ee = self._robot_manager.gripper_center_to_ee(target_gripper)
        return self._role_move(
            self.active_public_name or "unknown",
            self.atom.move_to_pose(target_ee),
            tag=tag,
            time_dilation_factor=0.5,
            is_save=True,
        )

    def _role_move(self, public_name: str, actions: list[Action], **kwargs) -> bool:
        if actions is None:
            return False
        return self.move(actions, **kwargs)

    def _mark_failure(self, stage: str):
        self.failure_stage = stage
        self.metadata["failure_stage"] = stage
        self._sync_metadata()

    def _step(self, is_save: bool = True):
        ret = super()._step(is_save=is_save)
        self._update_task_state()
        self._record_tactile_timeline()
        return ret

    def _update_task_state(self):
        if not hasattr(self, "public_candidate_actors"):
            return

        self.reference_lifted = bool(
            float(self.reference_object.get_pose().p[2]) - self.reference_initial_z > 0.010
        )
        for public_name in ("candidate_left", "candidate_right"):
            if self._is_candidate_placed(public_name):
                self.candidate_place_stable_count[public_name] += 1
            else:
                self.candidate_place_stable_count[public_name] = 0
            if self.candidate_place_stable_count[public_name] >= self.placement_stable_steps_required:
                self.candidate_placed[public_name] = True

        self.match_candidate_placed = bool(self.candidate_placed[self.match_candidate_public_name])
        self.distractor_placed = bool(self.candidate_placed[self.distractor_candidate_public_name])
        self.selection_correct = bool(self.selected_candidate == self.match_candidate_public_name)
        self._sync_metadata()

    def _is_candidate_placed(self, public_name: str) -> bool:
        actor = self.public_candidate_actors[public_name]
        actor_pose = actor.get_pose()
        xy_error = float(np.linalg.norm(actor_pose.p[:2] - self.match_slot_pose.p[:2]))
        z_error = float(abs(actor_pose.p[2] - self.match_slot_pose.p[2]))
        supported = bool(z_error < self.placement_supported_z_threshold)
        gripper_open = bool(self._robot_manager.get_gripper_qpos() > 0.020)
        if str(self.placement_mode) == "pad_overlap":
            on_pad = self._object_footprint_overlaps_pad(actor_pose, self.match_slot_pose, self._variant_for_public_candidate(public_name))
        else:
            on_pad = xy_error < 0.035
        self.metadata[f"{public_name}_xy_error"] = xy_error
        self.metadata[f"{public_name}_z_error"] = z_error
        self.metadata[f"{public_name}_placement_supported"] = supported
        self.metadata[f"{public_name}_placement_gripper_open"] = gripper_open
        self.metadata[f"{public_name}_on_match_pad"] = bool(on_pad)
        return bool(on_pad)

    def _object_footprint_overlaps_pad(self, actor_pose: Pose, target_pose: Pose, variant: dict) -> bool:
        center_xy = np.asarray(actor_pose.p[:2], dtype=np.float64)
        target_xy = np.asarray(target_pose.p[:2], dtype=np.float64)
        radius = 0.005 * float(variant.get("diameter", 4))
        half_length = 0.5 * float(variant.get("length", self.can_length))
        expanded_half_extents = np.asarray(self.pad_half_extents, dtype=np.float64) + radius + float(
            self.placement_overlap_margin
        )

        transform = actor_pose.to_transformation_matrix()
        axis_xy = np.asarray(transform[:2, 0], dtype=np.float64)
        axis_norm = float(np.linalg.norm(axis_xy))
        if axis_norm < 1e-6:
            delta = np.abs(center_xy - target_xy)
            return bool(np.all(delta <= expanded_half_extents))

        axis_xy = axis_xy / axis_norm
        p0 = center_xy - axis_xy * half_length
        p1 = center_xy + axis_xy * half_length
        return self._segment_intersects_aabb(p0, p1, target_xy, expanded_half_extents)

    @staticmethod
    def _segment_intersects_aabb(
        p0: np.ndarray,
        p1: np.ndarray,
        box_center: np.ndarray,
        box_half_extents: np.ndarray,
    ) -> bool:
        p0 = np.asarray(p0, dtype=np.float64) - np.asarray(box_center, dtype=np.float64)
        p1 = np.asarray(p1, dtype=np.float64) - np.asarray(box_center, dtype=np.float64)
        half = np.asarray(box_half_extents, dtype=np.float64)
        delta = p1 - p0
        t_min = 0.0
        t_max = 1.0
        for axis in range(2):
            if abs(delta[axis]) < 1e-12:
                if p0[axis] < -half[axis] or p0[axis] > half[axis]:
                    return False
                continue
            inv_delta = 1.0 / delta[axis]
            t1 = (-half[axis] - p0[axis]) * inv_delta
            t2 = (half[axis] - p0[axis]) * inv_delta
            if t1 > t2:
                t1, t2 = t2, t1
            t_min = max(t_min, t1)
            t_max = min(t_max, t2)
            if t_min > t_max:
                return False
        return True

    def get_public_pose_map(self) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        return {
            "reference_object": (
                np.asarray(self.reference_object.get_pose().p, dtype=np.float32).reshape(3),
                quat.copy(),
                np.array([0.04, 0.04, 0.12], dtype=np.float32),
            ),
            "candidate_left": (
                np.asarray(self.public_candidate_actors["candidate_left"].get_pose().p, dtype=np.float32).reshape(3),
                quat.copy(),
                np.array([0.04, 0.04, 0.12], dtype=np.float32),
            ),
            "candidate_right": (
                np.asarray(self.public_candidate_actors["candidate_right"].get_pose().p, dtype=np.float32).reshape(3),
                quat.copy(),
                np.array([0.04, 0.04, 0.12], dtype=np.float32),
            ),
            "match_slot": (
                np.asarray(self.match_slot_pose.p, dtype=np.float32).reshape(3),
                quat.copy(),
                np.array([0.10, 0.10, 0.03], dtype=np.float32),
            ),
        }

    def get_public_regions(self) -> dict:
        return {
            "reference_region": self._region_record(self.reference_xy, "reference"),
            "candidate_left_region": self._region_record(self.candidate_xy["candidate_left"], "candidate"),
            "candidate_right_region": self._region_record(self.candidate_xy["candidate_right"], "candidate"),
            "match_slot_region": {
                "kind": "place",
                "center_xy": [float(self.match_slot_xy[0]), float(self.match_slot_xy[1])],
                "half_extents": self.pad_half_extents.tolist(),
                "hover_z": float(self.safe_gripper_z),
                "release_z": float(self.match_slot_pose.p[2] + self.release_z_clearance),
                "description": "public placement region for the selected matching candidate",
            },
        }

    def get_public_grasp_actor(self, object_name: str):
        key = self._resolve_public_object_name(object_name)
        if key is None:
            return None
        return self.public_actor_map.get(key)

    def get_public_grasp_pose(self, object_name: str, *, grasp_height: float = 0.04):
        key = self._resolve_public_object_name(object_name)
        if key is None:
            raise KeyError(f"public grasp object {object_name!r} is not available")
        actor = self.public_actor_map[key]
        grasp_pose = self._lift_can_style_grasp_pose(actor)
        return (
            np.asarray(grasp_pose.p, dtype=np.float32),
            np.asarray(grasp_pose.q, dtype=np.float32),
        )

    def make_public_grasp_pose(self, object_name: str, actor: Actor | None = None, *, grasp_height: float = 0.04):
        key = self._resolve_public_object_name(object_name)
        if key is None:
            raise KeyError(f"public grasp object {object_name!r} is not available")
        return self._lift_can_style_grasp_pose(actor or self.public_actor_map[key])

    def _resolve_public_object_name(self, object_name: str) -> str | None:
        key = str(object_name).strip().lower().replace(" ", "_")
        aliases = {
            "reference": "reference_object",
            "reference_object": "reference_object",
            "object_a": "reference_object",
            "a": "reference_object",
            "candidate_left": "candidate_left",
            "left_candidate": "candidate_left",
            "left": "candidate_left",
            "candidate_1": "candidate_left",
            "candidate_right": "candidate_right",
            "right_candidate": "candidate_right",
            "right": "candidate_right",
            "candidate_2": "candidate_right",
        }
        return aliases.get(key)

    def _region_record(self, xy: np.ndarray, kind: str) -> dict:
        return {
            "kind": kind,
            "center_xy": [float(xy[0]), float(xy[1])],
            "half_extents": [float(self.xy_jitter + 0.035), float(self.xy_jitter + 0.035)],
            "hover_z": float(self.safe_gripper_z),
            "search_z_range": [0.018, float(self.safe_gripper_z)],
        }

    def _read_tactile_signature(self) -> dict:
        try:
            tactile_obs = self._tactile_manager.get_observations(["depth", "marker"])
        except Exception:
            tactile_obs = {}
        left = self._tactile_stats(tactile_obs.get("left_tactile", {}))
        right = self._tactile_stats(tactile_obs.get("right_tactile", {}))
        signature = {
            "contact": bool(left["contact"] or right["contact"]),
            "both_contact": bool(left["contact"] and right["contact"]),
            "left": left,
            "right": right,
            "mean_depth_delta_mm": float((left["depth_delta_mm"] + right["depth_delta_mm"]) / 2.0),
            "mean_contact_area": float((left["contact_area"] + right["contact_area"]) / 2.0),
            "gripper_qpos": float(self._robot_manager.get_gripper_qpos()),
            "step": int(self.step_count),
            "phase": str(self.task_phase),
        }
        return signature

    def _capture_tactile_signature(self, key: str) -> dict:
        signature = self._read_tactile_signature()
        if key in self.tactile_signatures:
            self.tactile_signatures[key] = signature
        if key == "reference":
            self.metadata["tactile_signature_reference"] = signature
            self.metadata["reference_tactile_signature_valid"] = bool(signature["contact"])
        elif key in {"candidate_left", "candidate_right"}:
            self.metadata[f"tactile_signature_{key}"] = signature
        self._sync_metadata()
        return signature

    def _has_tactile_contact(self) -> bool:
        signature = self._capture_tactile_signature("_contact_check")
        return bool(signature.get("contact", False))

    def _record_tactile_timeline(self):
        if not hasattr(self, "tactile_timeline") or self.step_count % self.timeline_frequency != 0:
            return
        try:
            tactile_obs = self._tactile_manager.get_observations(["depth", "marker"])
        except Exception:
            tactile_obs = {}

        row = {
            "step": int(self.step_count),
            "phase": self.task_phase,
            "active_public_name": self.active_public_name,
            "selected_candidate": self.selected_candidate,
            "gripper_qpos": float(self._robot_manager.get_gripper_qpos()),
        }
        actor_map = {
            "reference_object": self.reference_object,
            "candidate_left": self.public_candidate_actors["candidate_left"],
            "candidate_right": self.public_candidate_actors["candidate_right"],
        }
        for public_name, actor in actor_map.items():
            pose = actor.get_pose()
            row[f"{public_name}_x"] = float(pose.p[0])
            row[f"{public_name}_y"] = float(pose.p[1])
            row[f"{public_name}_z"] = float(pose.p[2])
        for hand_name in ("left_tactile", "right_tactile"):
            stats = self._tactile_stats(tactile_obs.get(hand_name, {}))
            prefix = "left" if hand_name.startswith("left") else "right"
            for key, value in stats.items():
                row[f"{prefix}_{key}"] = value
        self.tactile_timeline.append(row)

    def _tactile_stats(self, hand_obs: dict) -> dict:
        depth = hand_obs.get("depth")
        marker = hand_obs.get("marker")
        stats = {
            "contact": False,
            "depth_min": None,
            "depth_delta_mm": 0.0,
            "contact_area": 0.0,
            "marker_centroid_x": None,
            "marker_centroid_y": None,
        }
        if isinstance(depth, torch.Tensor):
            depth_np = depth.detach().cpu().numpy()
        elif depth is not None:
            depth_np = np.asarray(depth)
        else:
            depth_np = None
        if depth_np is not None and depth_np.size > 0:
            depth_np = np.asarray(depth_np, dtype=np.float64)
            finite = np.isfinite(depth_np)
            if finite.any():
                finite_depth = depth_np[finite]
                depth_min = float(finite_depth.min())
                far_plane = float(self.cfg.robot.tactile_far_plane)
                depth_delta = max(0.0, far_plane - depth_min)
                stats["depth_min"] = depth_min
                stats["depth_delta_mm"] = float(depth_delta)
                stats["contact_area"] = float(np.mean(finite_depth < far_plane - 0.1))
                stats["contact"] = bool(depth_delta > 0.5 or stats["contact_area"] > 0.001)

        if isinstance(marker, torch.Tensor):
            marker_np = marker.detach().cpu().numpy()
        elif marker is not None:
            marker_np = np.asarray(marker)
        else:
            marker_np = None
        if marker_np is not None and marker_np.size > 0:
            marker_np = np.asarray(marker_np, dtype=np.float64).reshape(-1, marker_np.shape[-1])
            valid = np.isfinite(marker_np).all(axis=1)
            if valid.any() and marker_np.shape[1] >= 2:
                stats["marker_centroid_x"] = float(marker_np[valid, 0].mean())
                stats["marker_centroid_y"] = float(marker_np[valid, 1].mean())
        return stats

    def _sync_metadata(self):
        if not hasattr(self, "public_candidate_actors"):
            return
        self.metadata.update(
            {
                "selected_candidate": self.selected_candidate,
                "selection_correct": bool(self.selection_correct),
                "candidate_left_touched": bool(self.candidate_touched["candidate_left"]),
                "candidate_right_touched": bool(self.candidate_touched["candidate_right"]),
                "candidate_left_placed": bool(self.candidate_placed["candidate_left"]),
                "candidate_right_placed": bool(self.candidate_placed["candidate_right"]),
                "match_candidate_placed": bool(self.match_candidate_placed),
                "distractor_placed": bool(self.distractor_placed),
                "reference_touched": bool(self.reference_touched),
                "reference_lifted": bool(self.reference_lifted),
                "reference_tactile_signature_valid": bool(self.reference_tactile_signature_valid),
                "failure_stage": self.failure_stage,
                "task_phase": self.task_phase,
            }
        )
        for public_name, actor in self.public_candidate_actors.items():
            pose = actor.get_pose()
            self.metadata[f"{public_name}_pose"] = pose.tolist()
            self.metadata[f"{public_name}_on_match_slot"] = bool(self._is_candidate_placed(public_name))
        self.metadata["reference_pose"] = self.reference_object.get_pose().tolist()

    def _save_metadata(self):
        self._sync_metadata()
        super()._save_metadata()
        if not hasattr(self, "tactile_timeline"):
            return
        timeline_dir = self.save_root / "tactile_memory_match_timeline"
        timeline_dir.mkdir(parents=True, exist_ok=True)
        json_path = timeline_dir / f"{self.cfg.seed}.json"
        csv_path = timeline_dir / f"{self.cfg.seed}.csv"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.tactile_timeline, f, indent=2)
        if self.tactile_timeline:
            fieldnames = sorted({key for row in self.tactile_timeline for key in row.keys()})
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.tactile_timeline)

    def check_success(self):
        self._update_task_state()
        return bool(
            self.selection_correct
            and self.match_candidate_placed
            and not self.distractor_placed
        )

    @staticmethod
    def _stash_pose(idx: int) -> Pose:
        return Pose([-2.0, -2.0 - 0.12 * idx, 0.021], [1.0, 0.0, 0.0, 0.0])

    @staticmethod
    def _resting_z(variant: dict) -> float:
        diameter_cm = int(variant.get("diameter", 4))
        return 0.005 * float(diameter_cm) + 0.001

    @staticmethod
    def _public_variant(variant: dict) -> dict:
        return {
            "asset": str(variant["asset"]),
            "diameter": int(variant["diameter"]),
            "density": float(variant["density"]),
            "friction_ratio": float(variant["friction_ratio"]),
        }

    def _variant_for_public_candidate(self, public_name: str) -> dict:
        if public_name == self.match_candidate_public_name:
            return self.reference_class
        return self.distractor_class

    def _jitter_xy(self, center_xy: np.ndarray, amount: float) -> list[float]:
        jitter = self.rng.uniform(-float(amount), float(amount), size=2)
        xy = np.asarray(center_xy, dtype=np.float64) + jitter
        return [float(xy[0]), float(xy[1])]

    def _lift_can_style_grasp_pose(self, actor: Actor) -> Pose:
        return self._lift_can_style_grasp_pose_from_pose(actor.get_pose())

    def _lift_can_style_grasp_pose_from_pose(self, actor_pose: Pose) -> Pose:
        target_pose = actor_pose.add_bias([-0.065, 0.0, -0.008])
        target_mat = target_pose.to_transformation_matrix()
        x_axis = target_mat[:3, 0].reshape(-1)
        grasp_mat = np.vstack([x_axis, np.cross(x_axis, [0, 0, 1]), [0, 0, 1]])
        return construct_grasp_pose(target_pose.p, grasp_mat[:3, 2], grasp_mat[:3, 0])

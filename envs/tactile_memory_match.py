from ._base_task import *
import json
import numpy as np


_WEIGHT_VARIANTS = {"light": 300.0, "heavy": 650.0}
_ROUGHNESS_VARIANTS = {"smooth": 1.5, "rough": 2.8}
_HARDNESS_VARIANTS = ("rigid", "soft")

# Every physical class deliberately uses the same visible mesh.  The task never
# exposes this table: it belongs to scene construction and private scoring only.
TACTILE_CLASSES = {
    f"{weight}_{roughness}_{hardness}": {
        "label": f"{weight}-{roughness}-{hardness}",
        "asset": "Can_d4cm.usd",
        "diameter": 4,
        "length": 0.120,
        "weight": weight,
        "roughness": roughness,
        "hardness": hardness,
        "density": density,
        "friction_ratio": friction_ratio,
    }
    for weight, density in _WEIGHT_VARIANTS.items()
    for roughness, friction_ratio in _ROUGHNESS_VARIANTS.items()
    for hardness in _HARDNESS_VARIANTS
}

CLASS_KEYS = tuple(TACTILE_CLASSES)
ORDERED_CLASS_PAIRS = tuple(
    (reference_key, distractor_key)
    for reference_key in CLASS_KEYS
    for distractor_key in CLASS_KEYS
    if reference_key != distractor_key
)
TACTILE_PROBE_SCHEMA_VERSION = "tactile_probe.v3"
PUBLIC_PROBE_RESPONSE_SCHEMA_VERSION = "public_probe_response.v1"
PUBLIC_PROBE_PROTOCOL_ID = "symmetric_side_grasp_preload_lift_hold_release.v3"
TACTILE_PROBE_FIELDS = (
    "depth_mm",
    "marker_displacement_px",
    "marker_coherence",
)


@configclass
class TaskCfg(BaseTaskCfg):
    step_lim = 500
    use_adaptive_grasp = True
    adaptive_grasp_depth_threshold = 27.75
    # The balanced schedule maps five consecutive repetitions onto every
    # ordered pair among the eight hidden physical combinations.
    identity_pair_schedule: Literal["random", "balanced_ordered_pairs"] = "random"
    reference_class_key: str = "random"
    distractor_class_key: str = "random"
    probe_capture_steps: int = 6
    probe_min_bilateral_ratio: float = 0.8


class Task(BaseTask):
    reference_xy = np.array([0.68, 0.24], dtype=np.float64)
    candidate_xy = {
        "candidate_left": np.array([0.60, -0.345], dtype=np.float64),
        "candidate_right": np.array([0.60, -0.175], dtype=np.float64),
    }
    match_slot_xy = np.array([0.42, 0.00], dtype=np.float64)
    stash_xy = np.array([-2.0, -2.0], dtype=np.float64)
    can_rot = [1.0, 0.0, 0.0, 0.0]
    # Phase-one identity matching fixes scene geometry across seeds. The hidden
    # tactile class and the left/right match assignment remain randomized.
    xy_jitter = 0.0
    candidate_jitter = 0.0
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
    occlusion_enabled = False
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
        if cfg.identity_pair_schedule not in {"random", "balanced_ordered_pairs"}:
            raise ValueError("identity_pair_schedule must be 'random' or 'balanced_ordered_pairs'")
        if cfg.reference_class_key not in {"random", *CLASS_KEYS}:
            raise ValueError("reference_class_key must be 'random' or a known tactile class")
        if cfg.distractor_class_key not in {"random", *CLASS_KEYS}:
            raise ValueError("distractor_class_key must be 'random' or a known tactile class")
        if cfg.probe_capture_steps < 1:
            raise ValueError("probe_capture_steps must be positive")
        if not 0.0 < cfg.probe_min_bilateral_ratio <= 1.0:
            raise ValueError("probe_min_bilateral_ratio must be in (0, 1]")
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
                constitution_cfg=self._constitution_for_variant(class_cfg),
                density=float(class_cfg["density"]),
                friction_ratio=float(class_cfg["friction_ratio"]),
            )
            self.candidate_actors[class_key] = self._actor_manager.add_from_usd_file(
                name=f"candidate_variant_{idx}",
                asset_path=str(class_cfg["asset"]),
                pose=self._stash_pose(idx + len(TACTILE_CLASSES)),
                constitution_cfg=self._constitution_for_variant(class_cfg),
                density=float(class_cfg["density"]),
                friction_ratio=float(class_cfg["friction_ratio"]),
            )

    def _reset_actors(self):
        for idx, actor in enumerate(self.reference_actors.values()):
            actor.set_pose(self._stash_pose(idx))
        for idx, actor in enumerate(self.candidate_actors.values()):
            actor.set_pose(self._stash_pose(idx + len(TACTILE_CLASSES)))

        self.reference_class_key, self.distractor_class_key = self._choose_identity_pair()
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

    def _choose_identity_pair(self) -> tuple[str, str]:
        """Choose the hidden reference/distractor pair without exposing it to agents."""
        if self.cfg.identity_pair_schedule == "balanced_ordered_pairs":
            # Consecutive seed ranges divisible by twelve contain every ordered
            # class pair equally often.  This is useful for offline expert
            # audits while preserving the existing random schedule by default.
            return ORDERED_CLASS_PAIRS[int(self.cfg.seed) % len(ORDERED_CLASS_PAIRS)]

        if self.cfg.reference_class_key == "random":
            reference_key = str(self.rng.choice(CLASS_KEYS))
        else:
            reference_key = str(self.cfg.reference_class_key)

        if self.cfg.distractor_class_key == "random":
            distractor_choices = [key for key in CLASS_KEYS if key != reference_key]
            distractor_key = str(self.rng.choice(distractor_choices))
        else:
            distractor_key = str(self.cfg.distractor_class_key)
            if distractor_key == reference_key:
                raise ValueError("distractor_class_key must differ from reference_class_key")
        return reference_key, distractor_key

    def _reset_episode_state(self):
        self.task_phase = "reset"
        self.active_public_name = None
        self.selected_candidate = None
        self.physical_selected_candidate = None
        self.selection_correct = False
        self.reference_touched = False
        self.reference_lifted = False
        self.reference_tactile_probe_valid = False
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
        self.tactile_probes = {
            "reference": {},
            "candidate_left": {},
            "candidate_right": {},
        }
        self._probe_raw_records = {
            "reference": {},
            "candidate_left": {},
            "candidate_right": {},
        }
        self._active_probe_segment = None
        self.metadata.update(
            {
                # This complete dictionary is private and is written to
                # private_metadata.json only.  Public metadata is assembled in
                # _public_episode_record and contains no labels or poses.
                "reference_class": str(self.reference_class["label"]),
                "reference_class_key": self.reference_class_key,
                "distractor_class": str(self.distractor_class["label"]),
                "distractor_class_key": self.distractor_class_key,
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
                "physical_selected_candidate": None,
                "selection_correct": False,
                "candidate_left_touched": False,
                "candidate_right_touched": False,
                "candidate_left_placed": False,
                "candidate_right_placed": False,
                "match_candidate_placed": False,
                "distractor_placed": False,
                "reference_touched": False,
                "reference_lifted": False,
                "tactile_probe_schema_version": TACTILE_PROBE_SCHEMA_VERSION,
                "reference_tactile_probe_valid": False,
                "tactile_probe_reference": {},
                "tactile_probe_candidate_left": {},
                "tactile_probe_candidate_right": {},
                "occlusion_enabled": bool(self.occlusion_enabled),
                "occlusion_opacity": float(self.occlusion_opacity),
                "occlusion_wall_base_z": float(self.occlusion_wall_base_z),
                "occlusion_wall_height": float(self.occlusion_wall_height),
                "occlusion_inner_width": float(self.occlusion_inner_width),
                "occlusion_inner_depth": float(self.occlusion_inner_depth),
                "wrist_view_darkened": bool(self.occlusion_enabled),
                "expert": "hidden_match_validation",
                "expert_policy": "symmetric_public_probe_then_hidden_match_placement",
                "expert_selection_uses_tactile": False,
                "selection_method": "hidden_match_for_expert_validation",
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
        # The official expert may read the hidden match assignment only to
        # validate that the scene and transport chain are physically viable.
        # Tactile probes remain public measurements for offline or agent-owned
        # matching; task code never computes a tactile similarity score.
        self._select_and_place_candidate(self.match_candidate_public_name)
        self.delay(20, is_save=False)

    def _probe_reference(self) -> bool:
        response = self.run_public_probe("reference_object")
        probe = response["probe"]
        self.reference_touched = bool(probe["quality"]["preload_bilateral_contact_ratio"] > 0.0)
        self.reference_tactile_probe_valid = bool(probe["quality"]["valid"])
        self._sync_metadata()
        return bool(response["execution"]["approach_ok"])

    def _probe_candidate(self, public_name: str) -> bool:
        response = self.run_public_probe(public_name)
        probe = response["probe"]
        self.candidate_touched[public_name] = bool(probe["quality"]["preload_bilateral_contact_ratio"] > 0.0)
        self._sync_metadata()
        return bool(response["execution"]["approach_ok"])

    def get_public_probe_spec(self) -> dict:
        """Return the fixed, label-free measurement protocol for every object."""
        return {
            "schema_version": "public_probe_spec.v1",
            "protocol_id": PUBLIC_PROBE_PROTOCOL_ID,
            "side_grasp": True,
            "adaptive_close": True,
            "preload_steps": int(self.probe_delay_steps),
            "lift_height": float(self.weight_probe_lift_height),
            "hold_steps": int(self.weight_probe_hold_steps),
            "lower_settle_steps": int(self.weight_probe_return_steps),
            "min_bilateral_contact_ratio": float(self.cfg.probe_min_bilateral_ratio),
            "release_then_clearance": True,
        }

    def run_public_probe(self, object_name: str) -> dict:
        """Execute the benchmark probe and return raw tactile data plus v3 summary.

        This is a controlled measurement primitive, not a memory, scorer, pose
        service, or candidate selector.  It intentionally has no access to
        physical labels in its return value.
        """
        public_name = self._resolve_public_object_name(object_name)
        if public_name is None:
            raise KeyError(f"Unknown public probe object: {object_name!r}")
        key = "reference" if public_name == "reference_object" else public_name
        actor = self.public_actor_map[public_name]
        self.active_public_name = public_name
        previous_phase = self.task_phase
        self.task_phase = f"{key}_probe_approach"
        self._probe_raw_records[key] = {"preload": [], "lift_motion": [], "hold": []}

        approach_ok = self._move_to_actor_grasp(public_name, actor, tag=f"{key}_probe_grasp_actor")
        close_ok = bool(approach_ok and self._close_current_grasp(f"{key}_probe_close"))
        preload = self._capture_probe_stationary_segment(key, "preload", self.probe_delay_steps)
        bilateral_gate = bool(
            close_ok
            and preload["bilateral_contact_ratio"] >= float(self.cfg.probe_min_bilateral_ratio)
        )

        self.task_phase = f"{key}_probe_lift"
        self._begin_probe_segment(key, "lift_motion")
        lift_ok = bool(
            bilateral_gate
            and self._role_move(
                public_name,
                self.atom.move_by_displacement(z=float(self.weight_probe_lift_height), xyz_coord="world"),
                tag=f"{key}_probe_lift",
                time_dilation_factor=0.5,
                is_save=True,
                delay=False,
            )
        )
        lift_motion = self._end_probe_segment(key, "lift_motion")

        hold = self._capture_probe_stationary_segment(key, "hold", self.weight_probe_hold_steps)
        lower_ok = True
        if lift_ok:
            self.task_phase = f"{key}_probe_lower"
            lower_ok = bool(
                self._role_move(
                    public_name,
                    self.atom.move_by_displacement(z=-float(self.weight_probe_lift_height), xyz_coord="world"),
                    tag=f"{key}_probe_lower",
                    time_dilation_factor=0.5,
                    is_save=True,
                    delay=False,
                )
            )
            self.delay(self.weight_probe_return_steps, is_save=True)

        self.task_phase = f"{key}_probe_release"
        release_ok = bool(self._role_move(public_name, self.atom.open_gripper(1.0), tag=f"{key}_probe_open", is_save=True))
        self.delay(4, is_save=True)
        clearance_ok = self._move_gripper_center_to_z(self.safe_gripper_z, tag=f"{key}_probe_clearance")
        probe = self._build_tactile_probe(
            key,
            preload,
            lift_motion,
            hold,
            approach_ok=approach_ok,
            close_ok=close_ok,
            bilateral_gate=bilateral_gate,
            lift_ok=lift_ok,
            lower_ok=lower_ok,
            release_ok=release_ok,
            clearance_ok=clearance_ok,
        )
        self._store_tactile_probe(key, probe)
        self.task_phase = previous_phase
        self._sync_metadata()
        return {
            "schema_version": PUBLIC_PROBE_RESPONSE_SCHEMA_VERSION,
            "object_name": public_name,
            "protocol": self.get_public_probe_spec(),
            "probe": probe,
            "raw": self._probe_raw_records[key],
            "execution": {
                "approach_ok": bool(approach_ok),
                "close_ok": bool(close_ok),
                "bilateral_gate": bool(bilateral_gate),
                "lift_ok": bool(lift_ok),
                "lower_ok": bool(lower_ok),
                "release_ok": bool(release_ok),
                "clearance_ok": bool(clearance_ok),
            },
        }

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
        final_lift_delta = float(actor.get_pose().p[2] - start_z)
        self.metadata[f"{public_name}_final_lift_delta"] = final_lift_delta
        if final_lift_delta < 0.010:
            self._mark_failure(f"{public_name}_final_lift_not_following")
            return False

        self.task_phase = "candidate_place"
        # Soft bodies do not have a reliable rigid center while compressed.
        # Use the same single planned placement primitive that VitaForge's
        # soft/hard expert uses, rather than repeatedly chasing a Kabsch pose.
        place_actions = self.atom.place_actor(
            actor,
            target_pose=self.match_slot_pose,
            pre_dis=0.0,
            dis=0.0,
            is_open=False,
        )
        if not self._role_move(
            public_name,
            place_actions,
            tag=f"{public_name}_final_place_actor",
            time_dilation_factor=0.5,
            is_save=True,
        ):
            self._mark_failure(f"{public_name}_place_failed")
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
        active_probe_segment = getattr(self, "_active_probe_segment", None)
        if active_probe_segment is not None:
            key, segment = active_probe_segment
            self._probe_raw_records[key][segment].append(self._read_tactile_measurement(include_raw=True))
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
            self.candidate_placed[public_name] = bool(
                self.candidate_place_stable_count[public_name] >= self.placement_stable_steps_required
            )

        self.match_candidate_placed = bool(self.candidate_placed[self.match_candidate_public_name])
        self.distractor_placed = bool(self.candidate_placed[self.distractor_candidate_public_name])
        placed_candidates = [
            public_name
            for public_name in ("candidate_left", "candidate_right")
            if self.candidate_placed[public_name]
        ]
        self.physical_selected_candidate = placed_candidates[0] if len(placed_candidates) == 1 else None
        if self.selected_candidate is None and self.physical_selected_candidate is not None:
            self.selected_candidate = self.physical_selected_candidate
            self.metadata["selection_method"] = "physical_placement"
        elif len(placed_candidates) > 1:
            self.metadata["selection_method"] = "ambiguous_physical_placement"
        self.selection_correct = bool(self.match_candidate_placed and not self.distractor_placed)
        self._sync_metadata()

    def _is_candidate_placed(self, public_name: str) -> bool:
        actor = self.public_candidate_actors[public_name]
        actor_pose = actor.get_pose()
        xy_error = float(np.linalg.norm(actor_pose.p[:2] - self.match_slot_pose.p[:2]))
        z_error = float(abs(actor_pose.p[2] - self.match_slot_pose.p[2]))
        # The cylinder center rests above the pad center.  After release, it
        # only needs to settle below the commanded release height.
        support_z_ceiling = float(
            self.match_slot_pose.p[2]
            + self.release_z_clearance
            + self.placement_supported_z_threshold
        )
        supported = bool(float(actor_pose.p[2]) <= support_z_ceiling)
        gripper_open = bool(self._robot_manager.get_gripper_qpos() > 0.020)
        if str(self.placement_mode) == "pad_overlap":
            on_pad = self._object_footprint_overlaps_pad(actor_pose, self.match_slot_pose, self._variant_for_public_candidate(public_name))
        else:
            on_pad = xy_error < 0.035
        self.metadata[f"{public_name}_xy_error"] = xy_error
        self.metadata[f"{public_name}_z_error"] = z_error
        self.metadata[f"{public_name}_placement_support_z_ceiling"] = support_z_ceiling
        self.metadata[f"{public_name}_placement_supported"] = supported
        self.metadata[f"{public_name}_placement_gripper_open"] = gripper_open
        self.metadata[f"{public_name}_on_match_pad"] = bool(on_pad)
        # A projected footprint can overlap the pad while the robot is still
        # holding the cylinder above it. Count placement only after the object
        # is supported at the pad height and the gripper has released it.
        return bool(on_pad and supported and gripper_open)

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
        """No live actor poses are part of the composable-probe benchmark."""
        return {}

    def get_public_regions(self) -> dict:
        """Legacy region service is intentionally disabled for this benchmark."""
        return {}

    def get_public_grasp_actor(self, object_name: str):
        raise RuntimeError("tactile_memory_match exposes run_public_probe(), not live actors")

    def get_public_grasp_pose(self, object_name: str, *, grasp_height: float = 0.04):
        raise RuntimeError("tactile_memory_match exposes run_public_probe(), not grasp poses")

    def make_public_grasp_pose(self, object_name: str, actor: Actor | None = None, *, grasp_height: float = 0.04):
        raise RuntimeError("tactile_memory_match exposes run_public_probe(), not grasp poses")

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

    def _begin_probe_segment(self, key: str, segment: str):
        self._active_probe_segment = (key, segment)

    def _end_probe_segment(self, key: str, segment: str) -> dict:
        if self._active_probe_segment != (key, segment):
            raise RuntimeError(f"Probe segment state mismatch for {key}/{segment}")
        self._active_probe_segment = None
        frames = self._probe_raw_records[key][segment]
        return self._aggregate_tactile_window(frames) if frames else self._empty_tactile_window()

    def _capture_probe_stationary_segment(self, key: str, segment: str, steps: int) -> dict:
        self.task_phase = f"{key}_probe_{segment}"
        self._begin_probe_segment(key, segment)
        self.delay(int(steps), is_save=True)
        return self._end_probe_segment(key, segment)

    def _read_tactile_measurement(self, *, include_raw: bool = False) -> dict:
        try:
            tactile_obs = self._tactile_manager.get_observations(["depth", "marker"])
        except Exception:
            tactile_obs = {}
        left = self._tactile_hand_measurement(tactile_obs.get("left_tactile", {}))
        right = self._tactile_hand_measurement(tactile_obs.get("right_tactile", {}))
        measurement = {
            "step": int(self.step_count),
            "left": left,
            "right": right,
            "both_contact": bool(left["contact"] and right["contact"]),
            "gripper_qpos": float(self._robot_manager.get_gripper_qpos()),
            "control_frame": {"atom_id": int(self.atom_id), "atom_tag": str(self.atom_tag)},
        }
        if include_raw:
            measurement["raw"] = {}
            for hand, source in (("left", "left_tactile"), ("right", "right_tactile")):
                hand_obs = tactile_obs.get(source, {})
                for field in ("depth", "marker"):
                    value = self._as_numpy(hand_obs.get(field))
                    if value is not None:
                        measurement["raw"][f"{hand}_{field}"] = np.asarray(value).copy()
        return measurement

    def _tactile_hand_measurement(self, hand_obs: dict) -> dict:
        depth = self._as_numpy(hand_obs.get("depth"))
        marker = self._as_numpy(hand_obs.get("marker"))
        far_plane = float(self.cfg.robot.tactile_far_plane)
        depth_mm = 0.0
        contact_area = 0.0
        if depth is not None and depth.size:
            values = np.asarray(depth, dtype=np.float64)
            values = values[np.isfinite(values)]
            if values.size:
                depth_mm = max(0.0, far_plane - float(np.percentile(values, 5.0)))
                contact_area = float(np.mean(values < far_plane - 0.1))

        marker_displacement_px = 0.0
        marker_coherence = 0.0
        if marker is not None and marker.ndim >= 3 and marker.shape[0] >= 2 and marker.shape[-1] >= 2:
            flow = np.asarray(marker[-1, ..., :2] - marker[0, ..., :2], dtype=np.float64).reshape(-1, 2)
            flow = flow[np.isfinite(flow).all(axis=1)]
            if flow.size:
                magnitudes = np.linalg.norm(flow, axis=1)
                mean_magnitude = float(np.mean(magnitudes))
                marker_displacement_px = mean_magnitude
                if mean_magnitude > 1e-8:
                    marker_coherence = float(np.linalg.norm(np.mean(flow, axis=0)) / mean_magnitude)

        return {
            "contact": bool(depth_mm >= 0.5 and contact_area > 0.001),
            "depth_mm": float(depth_mm),
            "contact_area": float(contact_area),
            "marker_displacement_px": float(marker_displacement_px),
            "marker_coherence": float(np.clip(marker_coherence, 0.0, 1.0)),
        }

    @staticmethod
    def _as_numpy(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        if value is not None:
            return np.asarray(value)
        return None

    def _aggregate_tactile_window(self, frames: list[dict]) -> dict:
        if not frames:
            raise RuntimeError("Cannot aggregate an empty tactile window")

        def median(hand: str, field: str) -> float:
            values = [float(frame[hand][field]) for frame in frames]
            return float(np.median(values))

        def mad(hand: str, field: str) -> float:
            values = np.asarray([float(frame[hand][field]) for frame in frames], dtype=np.float64)
            return float(np.median(np.abs(values - np.median(values))))

        def field_stats(hand: str, field: str) -> dict:
            value = median(hand, field)
            noise = mad(hand, field)
            return {"value": value, "mad": noise, "snr": float(abs(value) / max(noise, 1e-6))}

        bilateral_ratio = float(np.mean([frame["both_contact"] for frame in frames]))
        return {
            "frame_count": len(frames),
            "start_step": int(frames[0]["step"]),
            "end_step": int(frames[-1]["step"]),
            "bilateral_contact_ratio": bilateral_ratio,
            "left": {field: median("left", field) for field in (*TACTILE_PROBE_FIELDS, "contact_area")},
            "right": {field: median("right", field) for field in (*TACTILE_PROBE_FIELDS, "contact_area")},
            "noise": {
                hand: {field: field_stats(hand, field) for field in (*TACTILE_PROBE_FIELDS, "contact_area")}
                for hand in ("left", "right")
            },
            "gripper_qpos": float(np.median([frame["gripper_qpos"] for frame in frames])),
        }

    @staticmethod
    def _empty_tactile_window() -> dict:
        fields = (*TACTILE_PROBE_FIELDS, "contact_area")
        return {
            "frame_count": 0,
            "start_step": None,
            "end_step": None,
            "bilateral_contact_ratio": 0.0,
            "left": {field: 0.0 for field in fields},
            "right": {field: 0.0 for field in fields},
            "noise": {
                hand: {field: {"value": 0.0, "mad": 0.0, "snr": 0.0} for field in fields}
                for hand in ("left", "right")
            },
            "gripper_qpos": 0.0,
        }

    def _build_tactile_probe(
        self,
        key: str,
        preload: dict,
        lift_motion: dict,
        hold: dict,
        *,
        approach_ok: bool,
        close_ok: bool,
        bilateral_gate: bool,
        lift_ok: bool,
        lower_ok: bool,
        release_ok: bool,
        clearance_ok: bool,
    ) -> dict:
        delta = {
            hand: {
                field: float(lift_motion[hand][field] - preload[hand][field])
                for field in TACTILE_PROBE_FIELDS
            }
            for hand in ("left", "right")
        }
        preload_ratio = float(preload["bilateral_contact_ratio"])
        lift_ratio = float(lift_motion["bilateral_contact_ratio"])
        hold_ratio = float(hold["bilateral_contact_ratio"])
        valid = bool(
            approach_ok
            and close_ok
            and bilateral_gate
            and lift_ok
            and preload_ratio >= float(self.cfg.probe_min_bilateral_ratio)
            and lift_ratio >= float(self.cfg.probe_min_bilateral_ratio)
        )
        return {
            "schema_version": TACTILE_PROBE_SCHEMA_VERSION,
            "object_name": key,
            "protocol_id": PUBLIC_PROBE_PROTOCOL_ID,
            "quality": {
                "valid": valid,
                "approach_ok": bool(approach_ok),
                "close_ok": bool(close_ok),
                "bilateral_gate": bool(bilateral_gate),
                "lift_command_ok": bool(lift_ok),
                "lower_ok": bool(lower_ok),
                "release_ok": bool(release_ok),
                "clearance_ok": bool(clearance_ok),
                "preload_bilateral_contact_ratio": preload_ratio,
                "lift_motion_bilateral_contact_ratio": lift_ratio,
                "hold_bilateral_contact_ratio": hold_ratio,
                "lift_motion_frame_count": int(lift_motion["frame_count"]),
            },
            "preload": preload,
            "lift_motion": lift_motion,
            "hold": hold,
            "lift_minus_preload": delta,
        }

    def _store_tactile_probe(self, key: str, probe: dict):
        if key not in self.tactile_probes:
            raise KeyError(f"Unknown tactile probe key: {key}")
        self.tactile_probes[key] = probe
        metadata_key = "reference" if key == "reference" else key
        self.metadata[f"tactile_probe_{metadata_key}"] = probe

    def _record_tactile_timeline(self):
        if not hasattr(self, "tactile_timeline") or self.step_count % self.timeline_frequency != 0:
            return
        measurement = self._read_tactile_measurement()
        row = {
            "step": measurement["step"],
            "phase": self.task_phase,
            "active_public_name": self.active_public_name,
            "selected_candidate": self.selected_candidate,
            "both_contact": measurement["both_contact"],
            "gripper_qpos": measurement["gripper_qpos"],
        }
        for hand in ("left", "right"):
            for field, value in measurement[hand].items():
                row[f"{hand}_{field}"] = value
        self.tactile_timeline.append(row)

    def _sync_metadata(self):
        if not hasattr(self, "public_candidate_actors"):
            return
        self.metadata.update(
            {
                "selected_candidate": self.selected_candidate,
                "physical_selected_candidate": self.physical_selected_candidate,
                "selection_correct": bool(self.selection_correct),
                "candidate_left_touched": bool(self.candidate_touched["candidate_left"]),
                "candidate_right_touched": bool(self.candidate_touched["candidate_right"]),
                "candidate_left_placed": bool(self.candidate_placed["candidate_left"]),
                "candidate_right_placed": bool(self.candidate_placed["candidate_right"]),
                "match_candidate_placed": bool(self.match_candidate_placed),
                "distractor_placed": bool(self.distractor_placed),
                "reference_touched": bool(self.reference_touched),
                "reference_lifted": bool(self.reference_lifted),
                "reference_tactile_probe_valid": bool(self.reference_tactile_probe_valid),
                "tactile_probe_reference": self.tactile_probes.get("reference", {}),
                "tactile_probe_candidate_left": self.tactile_probes.get("candidate_left", {}),
                "tactile_probe_candidate_right": self.tactile_probes.get("candidate_right", {}),
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
        raw_path = self._save_public_probe_raw()
        tactile_probes = getattr(
            self,
            "tactile_probes",
            {"reference": {}, "candidate_left": {}, "candidate_right": {}},
        )
        public_record = {
            "schema_version": "tactile_memory_match_public_episode.v3",
            "seed": int(self.cfg.seed),
            "task": "tactile_memory_match",
            "probe_spec": self.get_public_probe_spec(),
            "probes": tactile_probes,
            "raw_probe_path": str(raw_path.relative_to(self.save_root)),
        }
        self._update_seed_json(self.metadata_path, public_record)
        private_record = {
            "schema_version": "tactile_memory_match_private_episode.v3",
            "seed": int(self.cfg.seed),
            **self.metadata,
        }
        self._update_seed_json(self.save_root / "private_metadata.json", private_record)

    def _save_public_probe_raw(self) -> Path:
        """Persist only public raw probe segments, never the full episode cache."""
        raw_dir = self.save_root / "public_probe"
        raw_dir.mkdir(parents=True, exist_ok=True)
        raw_path = raw_dir / f"{self.cfg.seed}.npz"
        arrays: dict[str, np.ndarray] = {}
        # ``clean_cache(result='error')`` can be reached during a reset error,
        # before episode state initializes probe buffers.  Save an empty NPZ so
        # the original reset failure remains visible instead of being masked.
        probe_records = getattr(self, "_probe_raw_records", {})
        for object_key, segments in probe_records.items():
            for segment_name, frames in segments.items():
                prefix = f"{object_key}__{segment_name}"
                arrays[f"{prefix}__step"] = np.asarray([frame["step"] for frame in frames], dtype=np.int64)
                arrays[f"{prefix}__gripper_qpos"] = np.asarray(
                    [frame["gripper_qpos"] for frame in frames], dtype=np.float32
                )
                arrays[f"{prefix}__atom_id"] = np.asarray(
                    [frame["control_frame"]["atom_id"] for frame in frames], dtype=np.int64
                )
                arrays[f"{prefix}__atom_tag"] = np.asarray(
                    [frame["control_frame"]["atom_tag"] for frame in frames], dtype="U96"
                )
                for raw_name in ("left_depth", "right_depth", "left_marker", "right_marker"):
                    values = [frame.get("raw", {}).get(raw_name) for frame in frames]
                    values = [value for value in values if value is not None]
                    if values:
                        arrays[f"{prefix}__{raw_name}"] = np.stack(values).astype(np.float32, copy=False)
        np.savez_compressed(raw_path, **arrays)
        return raw_path

    @staticmethod
    def _update_seed_json(path: Path, record: dict):
        if path.exists():
            try:
                with path.open("r", encoding="utf-8") as handle:
                    payload = json.load(handle)
            except (OSError, ValueError):
                payload = {}
        else:
            payload = {}
        payload[str(record.get("seed", "unknown"))] = record
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)

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
    def _constitution_for_variant(variant: dict):
        """Return the declared material model without a geometry fallback."""
        if str(variant["hardness"]) == "soft":
            return UipcObjectCfg.StableNeoHookeanCfg(
                youngs_modulus=0.1,
                poisson_rate=0.45,
            )
        return None

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

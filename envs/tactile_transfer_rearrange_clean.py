from ._base_task import *
import csv
import json
import numpy as np


TACTILE_VARIANTS = {
    "light_rough": {
        "label": "light-rough",
        "a": {"asset": "Can_d4cm.usd", "diameter": 4, "density": 300.0, "friction_ratio": 2.5},
        "b": {"asset": "Can_d4cm.usd", "diameter": 4, "density": 450.0, "friction_ratio": 2.8},
    },
}


@configclass
class TaskCfg(BaseTaskCfg):
    step_lim = 600
    use_adaptive_grasp = True
    adaptive_grasp_depth_threshold = 27.75


class Task(BaseTask):
    start_xy = {
        "object_a": np.array([0.58, -0.26], dtype=np.float64),
        "object_b": np.array([0.68, 0.26], dtype=np.float64),
    }
    slot_xy = {
        "object_a": np.array([0.42, -0.16], dtype=np.float64),
        "object_b": np.array([0.42, 0.16], dtype=np.float64),
    }
    stash_xy = np.array([-1.0, -1.2], dtype=np.float64)
    can_rot = [1.0, 0.0, 0.0, 0.0]
    xy_jitter = 0.02
    placement_xy_threshold = 0.035
    placement_z_threshold = 0.030
    lift_height_threshold = 0.020
    hold_steps_required = 8
    placement_stable_steps_required = 12
    max_transfer_attempts = 1
    timeline_frequency = 5
    release_z_clearance = 0.018
    place_pre_dis = 0.040
    place_dis = 0.020
    safe_gripper_z = 0.160
    release_retreat_z = 0.140
    transport_xy_step = 0.025
    descend_z_step = 0.008
    mask_vision_after_object_a = False
    b_occlusion_box_enabled = True
    b_occlusion_inner_center_xy = np.array([0.68, 0.26], dtype=np.float64)
    b_occlusion_wall_height = 0.18
    b_occlusion_wall_thickness = 0.015
    b_occlusion_inner_width = 0.20
    b_occlusion_inner_depth = 0.20
    b_occlusion_lid_enabled = True
    b_occlusion_lid_height = 0.145
    b_occlusion_lid_thickness = 0.012
    b_occlusion_color = np.array([0.70, 0.72, 0.74], dtype=np.float32)
    reset_settle_steps_per_chunk = 40
    reset_settle_max_chunks = 8
    reset_settle_xy_threshold = 0.025
    reset_settle_z_threshold = 0.020
    capx_premove_role = "object_a"
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
        self.slot_a = self._actor_manager.add_from_usd_file(
            name="slot_a",
            asset_path="GreenPad.usd",
            pose=Pose([self.slot_xy["object_a"][0], self.slot_xy["object_a"][1], 0.01], [1, 0, 0, 0]),
            density=1e5,
        )
        self.slot_b = self._actor_manager.add_from_usd_file(
            name="slot_b",
            asset_path="GreenPad.usd",
            pose=Pose([self.slot_xy["object_b"][0], self.slot_xy["object_b"][1], 0.01], [1, 0, 0, 0]),
            density=1e5,
        )
        self._create_b_occlusion_box()

        self.variant_actors: dict[tuple[str, str], Actor] = {}
        stash_idx = 0
        for class_key, class_cfg in TACTILE_VARIANTS.items():
            for role_key, role_name in (("a", "object_a"), ("b", "object_b")):
                variant = class_cfg[role_key]
                if len(TACTILE_VARIANTS) == 1:
                    initial_xy = self.start_xy[role_name]
                    initial_z = self._resting_z(variant)
                else:
                    initial_xy = np.array(
                        [self.stash_xy[0], self.stash_xy[1] - 0.12 * stash_idx],
                        dtype=np.float64,
                    )
                    initial_z = 0.03
                actor = self._actor_manager.add_from_usd_file(
                    name=f"{role_name}_{class_key}",
                    asset_path=str(variant.get("asset", "Can_d4cm.usd")),
                    pose=Pose(
                        [initial_xy[0], initial_xy[1], initial_z],
                        self.can_rot,
                    ),
                    density=float(variant.get("density", 1000.0)),
                    friction_ratio=float(variant.get("friction_ratio", 1.0)),
                )
                self.variant_actors[(role_name, class_key)] = actor
                stash_idx += 1

    def _reset_actors(self):
        if len(TACTILE_VARIANTS) > 1:
            for idx, actor in enumerate(self.variant_actors.values()):
                actor.set_pose(Pose([self.stash_xy[0], self.stash_xy[1] - 0.12 * idx, 0.03], self.can_rot))

        self.selected_class_key = str(self.rng.choice(list(TACTILE_VARIANTS.keys())))
        self.selected_class = TACTILE_VARIANTS[self.selected_class_key]
        self.objects = {
            "object_a": self.variant_actors[("object_a", self.selected_class_key)],
            "object_b": self.variant_actors[("object_b", self.selected_class_key)],
        }
        self.variants = {
            "object_a": dict(self.selected_class["a"]),
            "object_b": dict(self.selected_class["b"]),
        }
        self.target_poses = {}
        self.start_poses = {}

        for role_name, actor in self.objects.items():
            variant = self.variants[role_name]
            jitter = self.rng.uniform(-self.xy_jitter, self.xy_jitter, size=2)
            start_xy = self.start_xy[role_name] + jitter
            self.start_poses[role_name] = Pose([start_xy[0], start_xy[1], self._resting_z(variant)], self.can_rot)
            actor.set_pose(self.start_poses[role_name])

            slot_xy = self.slot_xy[role_name]
            self.target_poses[role_name] = Pose([slot_xy[0], slot_xy[1], self._resting_z(variant)], self.can_rot)

        self._reset_episode_state()

    def _reset_episode_state(self):
        self.task_phase = "approach"
        self.active_role = None
        self.vision_disabled = False
        self.vision_disabled_step = None
        self.vision_disabled_reason = None
        self.sequence_violation = False
        self.failure_stage = None
        self.tactile_timeline = []
        self.object_initial_z = {role: float(actor.get_pose().p[2]) for role, actor in self.objects.items()}
        self.object_lift_hold_count = {"object_a": 0, "object_b": 0}
        self.object_place_stable_count = {"object_a": 0, "object_b": 0}
        self.object_lifted = {"object_a": False, "object_b": False}
        self.object_placed = {"object_a": False, "object_b": False}
        self.object_action_count = {"object_a": 0, "object_b": 0}
        self.object_regrasp_count = {"object_a": 0, "object_b": 0}

        self.metadata.update(
            {
                "selected_tactile_class": self.selected_class["label"],
                "object_a_variant": dict(self.variants["object_a"]),
                "object_b_variant": dict(self.variants["object_b"]),
                "object_a_start_pose": self.start_poses["object_a"].tolist(),
                "object_b_start_pose": self.start_poses["object_b"].tolist(),
                "object_a_slot_pose": self.target_poses["object_a"].tolist(),
                "object_b_slot_pose": self.target_poses["object_b"].tolist(),
                "object_start_distance": float(
                    np.linalg.norm(self.start_poses["object_a"].p[:2] - self.start_poses["object_b"].p[:2])
                ),
                "object_a_lifted": False,
                "object_a_placed": False,
                "object_b_lifted": False,
                "object_b_placed": False,
                "object_a_action_count": 0,
                "object_b_action_count": 0,
                "object_a_regrasp_count": 0,
                "object_b_regrasp_count": 0,
                "sequence_violation": False,
                "failure_stage": None,
                "expert": "lift_can_style_collect",
                "expert_policy": "single_try_grasp_move_release",
                "grasp_close_mode": "adaptive",
                "adaptive_grasp_depth_threshold": float(self.cfg.adaptive_grasp_depth_threshold),
                "xy_jitter": float(self.xy_jitter),
                "release_z_clearance": float(self.release_z_clearance),
                "place_pre_dis": float(self.place_pre_dis),
                "place_dis": float(self.place_dis),
                "safe_gripper_z": float(self.safe_gripper_z),
                "release_retreat_z": float(self.release_retreat_z),
                "transport_mode": "horizontal_then_descend",
                "transport_xy_step": float(self.transport_xy_step),
                "descend_z_step": float(self.descend_z_step),
                "vision_mask_policy": "geometry_occlusion_for_object_b",
                "b_occlusion_box_enabled": bool(self.b_occlusion_box_enabled),
                "b_occlusion_inner_center_xy": self.b_occlusion_inner_center_xy.tolist(),
                "b_occlusion_wall_height": float(self.b_occlusion_wall_height),
                "b_occlusion_wall_thickness": float(self.b_occlusion_wall_thickness),
                "b_occlusion_inner_width": float(self.b_occlusion_inner_width),
                "b_occlusion_inner_depth": float(self.b_occlusion_inner_depth),
                "b_occlusion_lid_enabled": bool(self.b_occlusion_lid_enabled),
                "b_occlusion_lid_height": float(self.b_occlusion_lid_height),
                "b_occlusion_lid_thickness": float(self.b_occlusion_lid_thickness),
                "vision_disabled": False,
                "vision_disabled_step": None,
                "vision_disabled_reason": None,
            }
        )

    def _create_b_occlusion_box(self):
        if not self.b_occlusion_box_enabled:
            self.b_occlusion_walls = []
            return

        center = np.asarray(self.b_occlusion_inner_center_xy, dtype=np.float64)
        height = float(self.b_occlusion_wall_height)
        thickness = float(self.b_occlusion_wall_thickness)
        inner_width = float(self.b_occlusion_inner_width)
        inner_depth = float(self.b_occlusion_inner_depth)
        z_center = 0.018 + 0.5 * height
        lid_z = float(self.b_occlusion_lid_height)
        lid_thickness = float(self.b_occlusion_lid_thickness)
        color = np.asarray(self.b_occlusion_color, dtype=np.float32)

        wall_specs = [
            (
                "b_occlusion_back",
                [center[0] + 0.5 * inner_depth + 0.5 * thickness, center[1], z_center],
                [thickness, inner_width + 2.0 * thickness, height],
            ),
            (
                "b_occlusion_left",
                [center[0], center[1] - 0.5 * inner_width - 0.5 * thickness, z_center],
                [inner_depth + thickness, thickness, height],
            ),
            (
                "b_occlusion_right",
                [center[0], center[1] + 0.5 * inner_width + 0.5 * thickness, z_center],
                [inner_depth + thickness, thickness, height],
            ),
            (
                "b_occlusion_camera_side_shield",
                [center[0] - 0.015, center[1] + 0.5 * inner_width + 1.5 * thickness, z_center],
                [inner_depth + 0.06, thickness, height],
            ),
        ]
        if self.b_occlusion_lid_enabled:
            wall_specs.append(
                (
                    "b_occlusion_lid",
                    [center[0], center[1], lid_z],
                    [inner_depth + thickness, inner_width + 2.0 * thickness, lid_thickness],
                )
            )

        self.b_occlusion_walls = []
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
            self.b_occlusion_walls.append(wall)

    def pre_move(self):
        self._settle_selected_objects_at_start()
        if self.plan_success:
            self._premove_near_object(self.capx_premove_role)
        self.delay(10)

    def _premove_near_object(self, role_name: str) -> bool:
        role_name = self._resolve_public_role(role_name)
        if role_name is None:
            return False

        self.active_role = role_name
        self.task_phase = f"{role_name}_premove"
        self._sync_metadata()

        if not self.move(self.atom.open_gripper(1.0), tag=f"{role_name}_premove_open", is_save=False):
            self._mark_failure(f"{role_name}_premove_open_failed")
            self.plan_success = False
            return False

        actor = self.objects[role_name]
        grasp_pose = self._lift_can_style_grasp_pose(actor)
        grasp_idx = actor.register_point(pose=grasp_pose, type="contact")
        ok = self.move(
            self.atom.grasp_actor(actor, contact_point_id=grasp_idx, is_close=False),
            tag=f"{role_name}_premove_anchor",
            time_dilation_factor=self.capx_premove_time_dilation_factor,
            is_save=False,
        )
        if not ok:
            self._mark_failure(f"{role_name}_premove_anchor_failed")
            self.plan_success = False
            return False

        self.metadata["capx_premove_role"] = role_name
        self.metadata["capx_premove_grasp_pose"] = grasp_pose.tolist()
        self._sync_metadata()
        return True

    def _settle_selected_objects_at_start(self):
        if not hasattr(self, "objects"):
            return

        max_xy_error = float("inf")
        max_z_error = float("inf")
        settle_steps = 0
        for chunk_idx in range(self.reset_settle_max_chunks):
            for role_name, actor in self.objects.items():
                actor.set_pose(self.start_poses[role_name])

            self.delay(self.reset_settle_steps_per_chunk, is_save=False)
            settle_steps += self.reset_settle_steps_per_chunk

            xy_errors = []
            z_errors = []
            for role_name, actor in self.objects.items():
                actual_pose = actor.get_pose()
                target_pose = self.start_poses[role_name]
                xy_errors.append(float(np.linalg.norm(actual_pose.p[:2] - target_pose.p[:2])))
                z_errors.append(float(abs(actual_pose.p[2] - target_pose.p[2])))
            max_xy_error = max(xy_errors) if xy_errors else 0.0
            max_z_error = max(z_errors) if z_errors else 0.0

            print(
                "[tactile-transfer] reset settle "
                f"chunk={chunk_idx + 1} steps={settle_steps} "
                f"max_xy_error={max_xy_error:.4f} max_z_error={max_z_error:.4f}"
            )
            if (
                max_xy_error <= self.reset_settle_xy_threshold
                and max_z_error <= self.reset_settle_z_threshold
            ):
                break

        for actor in self.objects.values():
            actor.remove_animate()

        self.tactile_timeline = []
        self.object_initial_z = {role: float(actor.get_pose().p[2]) for role, actor in self.objects.items()}
        self.object_lift_hold_count = {"object_a": 0, "object_b": 0}
        self.object_place_stable_count = {"object_a": 0, "object_b": 0}
        self.object_lifted = {"object_a": False, "object_b": False}
        self.object_placed = {"object_a": False, "object_b": False}

        actual_start_poses = {role: actor.get_pose() for role, actor in self.objects.items()}
        self.metadata.update(
            {
                "reset_settle_steps": int(settle_steps),
                "reset_settle_max_xy_error": float(max_xy_error),
                "reset_settle_max_z_error": float(max_z_error),
                "object_a_actual_start_pose": actual_start_poses["object_a"].tolist(),
                "object_b_actual_start_pose": actual_start_poses["object_b"].tolist(),
                "object_actual_start_distance": float(
                    np.linalg.norm(
                        actual_start_poses["object_a"].p[:2] - actual_start_poses["object_b"].p[:2]
                    )
                ),
            }
        )
        if (
            max_xy_error > self.reset_settle_xy_threshold
            or max_z_error > self.reset_settle_z_threshold
        ):
            self.plan_success = False
            self._mark_failure("reset_pose_settle_failed")
        else:
            self._sync_metadata()

    def _play_once(self):
        if not self._transfer_object("object_a"):
            return
        if not self._transfer_object("object_b"):
            return
        self.task_phase = "completed"
        self.delay(30, is_save=False)

    def _transfer_object(self, role_name: str) -> bool:
        self.active_role = role_name
        if role_name == "object_b" and self.mask_vision_after_object_a:
            self._enable_vision_mask("object_b_phase")

        for attempt in range(self.max_transfer_attempts):
            if attempt > 0:
                self.object_regrasp_count[role_name] += 1
                self._sync_metadata()

            self.task_phase = f"{role_name}_grasp"
            if not self._grasp_object(role_name):
                self._mark_failure(f"{role_name}_grasp_failed")
                continue

            self.task_phase = f"{role_name}_lift"
            if not self._lift_and_verify(role_name):
                self._mark_failure(f"{role_name}_lift_failed")
                self.move(self.atom.open_gripper(1.0), tag=f"{role_name}_recover_open", is_save=True)
                self.delay(15, is_save=True)
                continue

            self.task_phase = f"{role_name}_place"
            if not self._place_object(role_name):
                self._mark_failure(f"{role_name}_release_failed")
                continue

            self._update_task_state()
            if self.object_placed[role_name]:
                self.active_role = None
                self.failure_stage = None
                self._sync_metadata()
                return True

            self._mark_failure(f"{role_name}_wrong_slot")

        self.plan_success = False
        self.active_role = None
        self._sync_metadata()
        return False

    def _grasp_object(self, role_name: str) -> bool:
        actor = self.objects[role_name]
        if not self._move_gripper_center_to_z(role_name, self.safe_gripper_z, tag=f"{role_name}_pregrasp_safe_z"):
            return False
        self._role_move(role_name, self.atom.open_gripper(1.0), tag=f"{role_name}_open", is_save=True)

        grasp_pose = self._lift_can_style_grasp_pose(actor)
        grasp_idx = actor.register_point(pose=grasp_pose, type="contact")
        if not self._role_move(
            role_name,
            self.atom.grasp_actor(actor, contact_point_id=grasp_idx, is_close=False),
            tag=f"{role_name}_grasp_actor",
            time_dilation_factor=0.5,
            is_save=True,
        ):
            return False

        if not self._role_move(role_name, self.atom.close_gripper(), tag=f"{role_name}_adaptive_close", is_save=True):
            return False
        self.delay(20, is_save=True)
        return True

    def _lift_can_style_grasp_pose(self, actor: Actor) -> Pose:
        return self._lift_can_style_grasp_pose_from_pose(actor.get_pose())

    def _lift_can_style_grasp_pose_from_pose(self, actor_pose: Pose) -> Pose:
        target_pose = actor_pose.add_bias([-0.065, 0.0, -0.008])
        target_mat = target_pose.to_transformation_matrix()
        x_axis = target_mat[:3, 0].reshape(-1)
        grasp_mat = np.vstack([x_axis, np.cross(x_axis, [0, 0, 1]), [0, 0, 1]])
        return construct_grasp_pose(target_pose.p, grasp_mat[:3, 2], grasp_mat[:3, 0])

    def _resolve_public_role(self, object_name: str) -> str | None:
        key = str(object_name).strip().lower().replace(" ", "_")
        aliases = {
            "a": "object_a",
            "first": "object_a",
            "first_object": "object_a",
            "object_a": "object_a",
            "b": "object_b",
            "second": "object_b",
            "second_object": "object_b",
            "object_b": "object_b",
        }
        if key in {"current", "current_object", "object", "can"}:
            return "object_b" if self.object_placed.get("object_a", False) else "object_a"
        return aliases.get(key)

    def _resolve_public_slot(self, object_name: str) -> str | None:
        key = str(object_name).strip().lower().replace(" ", "_")
        aliases = {
            "slot_a": "slot_a",
            "target_a": "slot_a",
            "a_slot": "slot_a",
            "slot_b": "slot_b",
            "target_b": "slot_b",
            "b_slot": "slot_b",
        }
        if key in {"current_slot", "slot", "target"}:
            return "slot_b" if self.object_placed.get("object_a", False) else "slot_a"
        return aliases.get(key)

    def begin_capx_role(self, object_name: str) -> dict:
        role_name = self._resolve_public_role(object_name)
        if role_name is None:
            return {"ok": False, "message": f"unknown public role {object_name!r}"}
        self.active_role = role_name
        if not str(self.task_phase).startswith(role_name):
            self.task_phase = f"{role_name}_capx"
        if role_name == "object_b" and self.mask_vision_after_object_a and self.object_placed["object_a"]:
            self._enable_vision_mask("object_b_phase")
        self._sync_metadata()
        return {
            "ok": True,
            "role": role_name,
            "slot": "slot_b" if role_name == "object_b" else "slot_a",
            "vision_disabled": bool(self.vision_disabled),
        }

    def get_public_regions(self) -> dict:
        pickup_half_extents = [
            float(self.xy_jitter + 0.035),
            float(self.xy_jitter + 0.035),
        ]
        slot_half_extents = [0.045, 0.045]
        pickup_low_z = 0.018
        regions = {}
        target_poses = getattr(self, "target_poses", {})
        for role_name, pickup_name, slot_name in (
            ("object_a", "pickup_a_region", "slot_a_region"),
            ("object_b", "pickup_b_region", "slot_b_region"),
        ):
            start_xy = self.start_xy[role_name]
            slot_xy = self.slot_xy[role_name]
            target_pose = target_poses.get(role_name) if isinstance(target_poses, dict) else None
            release_z = (
                float(target_pose.p[2] + self.release_z_clearance)
                if target_pose is not None
                else float(0.021 + self.release_z_clearance)
            )
            regions[pickup_name] = {
                "kind": "pickup",
                "center_xy": [float(start_xy[0]), float(start_xy[1])],
                "half_extents": pickup_half_extents,
                "hover_z": float(self.safe_gripper_z),
                "search_z_range": [pickup_low_z, float(self.safe_gripper_z)],
                "description": f"coarse search region for {role_name}",
            }
            regions[slot_name] = {
                "kind": "place",
                "center_xy": [float(slot_xy[0]), float(slot_xy[1])],
                "half_extents": slot_half_extents,
                "hover_z": float(self.safe_gripper_z),
                "release_z": release_z,
                "search_z_range": [release_z, float(self.safe_gripper_z)],
                "description": f"public placement region for {role_name}",
            }
        return regions

    def get_public_grasp_actor(self, object_name: str):
        role_name = self._resolve_public_role(object_name)
        if role_name is None:
            return None
        return self.objects.get(role_name)

    def get_public_grasp_pose(self, object_name: str, *, grasp_height: float = 0.04):
        role_name = self._resolve_public_role(object_name)
        if role_name is None:
            raise KeyError(f"public grasp object {object_name!r} is not available")
        self.begin_capx_role(role_name)
        # Use the reset anchor instead of live actor tracking so B-stage visual
        # masking does not become simulator-GT tracking.
        anchor_pose = self.start_poses.get(role_name)
        if anchor_pose is None:
            anchor_pose = self.objects[role_name].get_pose()
        grasp_pose = self._lift_can_style_grasp_pose_from_pose(anchor_pose)
        return (
            np.asarray(grasp_pose.p, dtype=np.float32),
            np.asarray(grasp_pose.q, dtype=np.float32),
        )

    def make_public_grasp_pose(self, object_name: str, actor: Actor | None = None, *, grasp_height: float = 0.04):
        role_name = self._resolve_public_role(object_name)
        if role_name is None:
            raise KeyError(f"public grasp object {object_name!r} is not available")
        anchor_pose = self.start_poses.get(role_name)
        if anchor_pose is None:
            anchor_pose = self.objects[role_name].get_pose()
        return self._lift_can_style_grasp_pose_from_pose(anchor_pose)

    def _lift_and_verify(self, role_name: str) -> bool:
        actor = self.objects[role_name]
        start_z = float(actor.get_pose().p[2])
        if not self._role_move(
            role_name,
            self.atom.move_by_displacement(z=0.035),
            tag=f"{role_name}_lift",
            is_save=True,
            time_dilation_factor=0.5,
        ):
            return False
        self.delay(20, is_save=True)

        current_z = float(actor.get_pose().p[2])
        if current_z - start_z < self.lift_height_threshold:
            self.metadata[f"{role_name}_lift_start_z"] = start_z
            self.metadata[f"{role_name}_lift_current_z"] = current_z
            self._sync_metadata()
            return False
        self._update_task_state()
        return True

    def _place_object(self, role_name: str) -> bool:
        target_pose = self.target_poses[role_name]

        if not self._transport_held_object_xy(role_name, target_pose.p[:2]):
            self._mark_failure(f"{role_name}_transport_failed")
            return False

        release_z = float(target_pose.p[2] + self.release_z_clearance)
        if not self._descend_held_object_to_z(role_name, release_z):
            self._mark_failure(f"{role_name}_release_failed")
            return False

        if not self._role_move(role_name, self.atom.open_gripper(1.0), tag=f"{role_name}_release_open", is_save=True):
            return False
        self.delay(35, is_save=True)

        if not self._retreat_after_release(role_name):
            return False
        self.delay(20, is_save=True)
        self._update_task_state()
        return self.object_placed[role_name]

    def _transport_held_object_xy(self, role_name: str, target_xy: np.ndarray) -> bool:
        actor = self.objects[role_name]
        target_xy = np.asarray(target_xy, dtype=np.float64).reshape(2)
        max_iters = 32
        for idx in range(max_iters):
            actor_xy = np.asarray(actor.get_pose().p[:2], dtype=np.float64)
            delta_xy = target_xy - actor_xy
            xy_error = float(np.linalg.norm(delta_xy))
            if xy_error < 0.012:
                return True
            step_xy = np.zeros(2, dtype=np.float64)
            axis = int(np.argmax(np.abs(delta_xy)))
            step_xy[axis] = float(np.clip(delta_xy[axis], -self.transport_xy_step, self.transport_xy_step))
            if not self._role_move(
                role_name,
                self.atom.move_by_displacement(
                    x=float(step_xy[0]),
                    y=float(step_xy[1]),
                    z=0.0,
                    xyz_coord="world",
                ),
                tag=f"{role_name}_transport_xy_{idx}",
                time_dilation_factor=0.5,
                is_save=True,
            ):
                return False
            self.delay(8, is_save=True)

        actor_xy = np.asarray(actor.get_pose().p[:2], dtype=np.float64)
        self.metadata[f"{role_name}_transport_xy_error"] = float(np.linalg.norm(target_xy - actor_xy))
        self._sync_metadata()
        return float(np.linalg.norm(target_xy - actor_xy)) < self.placement_xy_threshold

    def _descend_held_object_to_z(self, role_name: str, release_z: float) -> bool:
        actor = self.objects[role_name]
        max_iters = 10
        for idx in range(max_iters):
            actor_z = float(actor.get_pose().p[2])
            delta_z = float(release_z - actor_z)
            if abs(delta_z) < 0.006:
                return True
            step_z = float(np.clip(delta_z, -self.descend_z_step, self.descend_z_step))
            if not self._role_move(
                role_name,
                self.atom.move_by_displacement(z=step_z, xyz_coord="world"),
                tag=f"{role_name}_descend_{idx}",
                time_dilation_factor=0.5,
                is_save=True,
            ):
                return False
            self.delay(8, is_save=True)

        actor_z = float(actor.get_pose().p[2])
        self.metadata[f"{role_name}_release_z_error"] = float(abs(release_z - actor_z))
        self._sync_metadata()
        return abs(release_z - actor_z) < self.placement_z_threshold

    def _retreat_after_release(self, role_name: str) -> bool:
        return self._move_gripper_center_to_z(role_name, self.release_retreat_z, tag="release_retreat")

    def _move_gripper_center_to_z(self, role_name: str, target_z: float, tag: str) -> bool:
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
            role_name,
            self.atom.move_to_pose(target_ee),
            tag=tag,
            time_dilation_factor=0.5,
            is_save=True,
        )

    def _role_move(self, role_name: str, actions: list[Action], **kwargs) -> bool:
        if actions is None:
            return False
        self.object_action_count[role_name] += 1
        self._sync_metadata()
        return self.move(actions, **kwargs)

    def _mark_failure(self, stage: str):
        self.failure_stage = stage
        self.metadata["failure_stage"] = stage
        self._sync_metadata()

    def _enable_vision_mask(self, reason: str):
        if getattr(self, "vision_disabled", False):
            return
        self.vision_disabled = True
        self.vision_disabled_step = int(self.step_count)
        self.vision_disabled_reason = reason
        print(f"[tactile-transfer] vision mask enabled step={self.vision_disabled_step} reason={reason}")
        self._sync_metadata()

    def _step(self, is_save: bool = True):
        ret = super()._step(is_save=is_save)
        self._update_task_state()
        self._record_tactile_timeline()
        return ret

    def _get_observations(self):
        obs = super()._get_observations()
        if getattr(self, "vision_disabled", False):
            self._mask_camera_observations(obs)
        return obs

    def _mask_camera_observations(self, obs: dict):
        camera_obs = obs.get("observation", {})
        for camera_name in list(camera_obs.keys()):
            for data_type, value in list(camera_obs[camera_name].items()):
                if isinstance(value, torch.Tensor):
                    camera_obs[camera_name][data_type] = torch.zeros_like(value)
                elif isinstance(value, np.ndarray):
                    camera_obs[camera_name][data_type] = np.zeros_like(value)

    def get_frame_shot(self, obs):
        if not getattr(self, "vision_disabled", False):
            return super().get_frame_shot(obs)
        try:
            unmasked_camera_obs = self._camera_manager.get_observations(["rgb"])
            frame_obs = dict(obs)
            frame_obs["observation"] = {
                "head": {"rgb": unmasked_camera_obs["head"]["rgb"]},
                "wrist": {"rgb": unmasked_camera_obs["wrist"]["rgb"]},
            }
            return super().get_frame_shot(frame_obs)
        except Exception:
            return super().get_frame_shot(obs)

    def _update_task_state(self):
        if not hasattr(self, "objects"):
            return

        for role_name, actor in self.objects.items():
            if self._is_currently_held(role_name):
                self.object_lift_hold_count[role_name] += 1
            else:
                self.object_lift_hold_count[role_name] = 0
            if self.object_lift_hold_count[role_name] >= self.hold_steps_required:
                self.object_lifted[role_name] = True

            if self._is_placed(role_name):
                self.object_place_stable_count[role_name] += 1
            else:
                self.object_place_stable_count[role_name] = 0
            if self.object_place_stable_count[role_name] >= self.placement_stable_steps_required:
                self.object_placed[role_name] = True

        b_lift = float(self.objects["object_b"].get_pose().p[2] - self.object_initial_z["object_b"])
        if self.active_role == "object_b" and not self.object_placed["object_a"] and b_lift > 0.03:
            self.sequence_violation = True

        if self.object_placed["object_a"] and self.object_placed["object_b"]:
            self.task_phase = "completed"

        self._sync_metadata()

    def _is_currently_held(self, role_name: str) -> bool:
        actor_pose = self.objects[role_name].get_pose()
        gripper_pose = self._robot_manager.get_gripper_center_pose()
        z_lift = float(actor_pose.p[2] - self.object_initial_z[role_name])
        inhand_xy = float(np.linalg.norm(actor_pose.p[:2] - gripper_pose.p[:2]))
        inhand_z = float(abs(actor_pose.p[2] - gripper_pose.p[2]))
        return z_lift >= self.lift_height_threshold and inhand_xy < 0.08 and inhand_z < 0.16

    def _is_placed(self, role_name: str) -> bool:
        actor_pose = self.objects[role_name].get_pose()
        target_pose = self.target_poses[role_name]
        xy_error = float(np.linalg.norm(actor_pose.p[:2] - target_pose.p[:2]))
        z_error = float(abs(actor_pose.p[2] - target_pose.p[2]))
        gripper_open = self._robot_manager.get_gripper_qpos() > 0.020
        return xy_error < self.placement_xy_threshold and z_error < self.placement_z_threshold and gripper_open

    def _sync_metadata(self):
        if not hasattr(self, "objects"):
            return
        self.metadata.update(
            {
                "object_a_lifted": bool(self.object_lifted["object_a"]),
                "object_a_placed": bool(self.object_placed["object_a"]),
                "object_b_lifted": bool(self.object_lifted["object_b"]),
                "object_b_placed": bool(self.object_placed["object_b"]),
                "object_a_action_count": int(self.object_action_count["object_a"]),
                "object_b_action_count": int(self.object_action_count["object_b"]),
                "object_a_regrasp_count": int(self.object_regrasp_count["object_a"]),
                "object_b_regrasp_count": int(self.object_regrasp_count["object_b"]),
                "sequence_violation": bool(self.sequence_violation),
                "failure_stage": self.failure_stage,
                "task_phase": self.task_phase,
                "vision_disabled": bool(getattr(self, "vision_disabled", False)),
                "vision_disabled_step": getattr(self, "vision_disabled_step", None),
                "vision_disabled_reason": getattr(self, "vision_disabled_reason", None),
                "object_live_distance": float(
                    np.linalg.norm(
                        self.objects["object_a"].get_pose().p[:2] - self.objects["object_b"].get_pose().p[:2]
                    )
                ),
            }
        )
        for role_name, actor in self.objects.items():
            pose = actor.get_pose()
            target = self.target_poses[role_name]
            self.metadata[f"{role_name}_pose"] = pose.tolist()
            self.metadata[f"{role_name}_xy_error"] = float(np.linalg.norm(pose.p[:2] - target.p[:2]))
            self.metadata[f"{role_name}_z_error"] = float(abs(pose.p[2] - target.p[2]))

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
            "active_role": self.active_role,
            "vision_disabled": bool(getattr(self, "vision_disabled", False)),
            "gripper_qpos": float(self._robot_manager.get_gripper_qpos()),
        }
        for role_name, actor in self.objects.items():
            pose = actor.get_pose()
            row[f"{role_name}_x"] = float(pose.p[0])
            row[f"{role_name}_y"] = float(pose.p[1])
            row[f"{role_name}_z"] = float(pose.p[2])
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
            "depth_min": None,
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
                stats["depth_min"] = float(finite_depth.min())
                far_plane = float(self.cfg.robot.tactile_far_plane)
                stats["contact_area"] = float(np.mean(finite_depth < far_plane - 0.1))

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

    def _save_metadata(self):
        self._sync_metadata()
        super()._save_metadata()
        if not hasattr(self, "tactile_timeline"):
            return
        timeline_dir = self.save_root / "tactile_transfer_timeline"
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

    def check_early_stop(self):
        if self.failure_stage is not None and not self.plan_success:
            self.metadata["early_stop"] = True
            return True
        return False

    def check_success(self):
        self._update_task_state()
        return (
            bool(self.object_lifted["object_a"])
            and bool(self.object_lifted["object_b"])
            and bool(self.object_placed["object_a"])
            and bool(self.object_placed["object_b"])
            and not bool(self.sequence_violation)
        )

    @staticmethod
    def _resting_z(variant: dict) -> float:
        diameter_cm = int(variant.get("diameter", 4))
        return 0.005 * float(diameter_cm) + 0.001

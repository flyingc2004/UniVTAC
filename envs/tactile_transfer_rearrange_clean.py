from ._base_task import *
import csv
import json
import numpy as np


TACTILE_VARIANTS = {
    # Start with one easy condition for task sanity. More material classes can be
    # enabled after the collect expert is consistently above the sanity bar.
    "light_rough": {
        "label": "light-rough",
        "a": {"asset": "Can_d4cm.usd", "diameter": 4, "density": 300.0, "friction_ratio": 2.5},
        "b": {"asset": "Can_d4cm.usd", "diameter": 4, "density": 300.0, "friction_ratio": 2.5},
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
    xy_jitter = 0.01
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
            }
        )

    def pre_move(self):
        self.delay(10)

    def _play_once(self):
        if not self._transfer_object("object_a"):
            return
        if not self._transfer_object("object_b"):
            return
        self.task_phase = "completed"
        self.delay(30, is_save=False)

    def _transfer_object(self, role_name: str) -> bool:
        self.active_role = role_name

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
        actor_pose = actor.get_pose()
        target_pose = actor_pose.add_bias([-0.065, 0.0, -0.008])
        target_mat = target_pose.to_transformation_matrix()
        x_axis = target_mat[:3, 0].reshape(-1)
        grasp_mat = np.vstack([x_axis, np.cross(x_axis, [0, 0, 1]), [0, 0, 1]])
        return construct_grasp_pose(target_pose.p, grasp_mat[:3, 2], grasp_mat[:3, 0])

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

    def _step(self, is_save: bool = True):
        ret = super()._step(is_save=is_save)
        self._update_task_state()
        self._record_tactile_timeline()
        return ret

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

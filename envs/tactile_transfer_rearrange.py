from ._base_task import *
import csv
import math
import numpy as np


TACTILE_VARIANTS = {
    "light_smooth": {
        "label": "light-smooth",
        "a": {"diameter": 4, "density": 500.0, "friction": 0.25},
        "b": {"diameter": 4, "density": 700.0, "friction": 0.35},
    },
    "light_rough": {
        "label": "light-rough",
        "a": {"diameter": 4, "density": 500.0, "friction": 1.00},
        "b": {"diameter": 4, "density": 700.0, "friction": 1.30},
    },
    "heavy_smooth": {
        "label": "heavy-smooth",
        "a": {"diameter": 4, "density": 2500.0, "friction": 0.25},
        "b": {"diameter": 4, "density": 3200.0, "friction": 0.35},
    },
    "heavy_rough": {
        "label": "heavy-rough",
        "a": {"diameter": 4, "density": 2500.0, "friction": 1.00},
        "b": {"diameter": 4, "density": 3200.0, "friction": 1.30},
    },
}


@configclass
class TaskCfg(BaseTaskCfg):
    step_lim = 600
    adaptive_grasp_depth_threshold = 27.75
    video_size = (1120, 320)
    max_save_frames = 900
    cameras = [
        CameraCfg(
            name="head",
            prim_path="/World/envs/env_.*/Camera",
            offset=CameraCfg.OffsetCfg(
                pos=(0.554, 1.0, 0.150),
                rot=(0, 0, 0.707, 0.707),
                convention="opengl",
            ),
            data_types=["rgb", "depth"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=1.94,
                focus_distance=1.0,
                horizontal_aperture=2.688,
                clipping_range=(0.01, 100.0),
            ),
            width=480,
            height=270,
            update_period=1 / 120,
        ),
        CameraCfg(
            name="wrist",
            prim_path="/World/envs/env_.*/Robot/WristCamera/Camera",
            data_types=["rgb", "depth"],
            spawn=None,
            width=480,
            height=270,
            update_period=1 / 120,
        ),
        CameraCfg(
            name="observer",
            prim_path="/World/envs/env_.*/ObserverCamera",
            offset=CameraCfg.OffsetCfg(
                pos=(0.58, 1.35, 0.58),
                rot=(0, 0, 0.707, 0.707),
                convention="opengl",
            ),
            data_types=["rgb", "depth"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=1.55,
                focus_distance=1.0,
                horizontal_aperture=5.0,
                clipping_range=(0.01, 100.0),
            ),
            width=480,
            height=270,
            update_period=1 / 120,
        ),
    ]


class Task(BaseTask):
    search_center = np.array([0.70, 0.00], dtype=np.float64)
    search_half_extents = np.array([0.18, 0.20], dtype=np.float64)
    vision_disable_z = 0.20
    slot_xy = {
        "object_a": np.array([0.45, -0.08], dtype=np.float64),
        "object_b": np.array([0.45, 0.08], dtype=np.float64),
    }
    start_xy = {
        "object_a": np.array([0.70, -0.08], dtype=np.float64),
        "object_b": np.array([0.70, 0.08], dtype=np.float64),
    }
    stash_xy = np.array([-1.0, -1.2], dtype=np.float64)
    placement_xy_threshold = 0.035
    placement_z_threshold = 0.035
    slot_hover_height = 0.11
    slot_release_clearance = 0.006
    slot_retreat_height = 0.08
    lift_height_threshold = 0.05
    b_sequence_lift_threshold = 0.03
    hold_steps_required = 10
    placement_stable_steps_required = 15
    timeline_frequency = 5

    def __init__(
        self,
        cfg: BaseTaskCfg,
        mode: Literal["collect", "eval"] = "collect",
        render_mode: str | None = None,
        **kwargs,
    ):
        cfg.sim.physics_material.dynamic_friction = 1.5
        cfg.sim.physics_material.static_friction = 1.5
        cfg.uipc_sim.contact.default_friction_ratio = 0.5
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
        stash_i = 0
        for class_key, class_cfg in TACTILE_VARIANTS.items():
            for role_key, role_name in (("a", "object_a"), ("b", "object_b")):
                variant = class_cfg[role_key]
                diameter = int(variant["diameter"])
                stash_pose = Pose(
                    [
                        self.stash_xy[0],
                        self.stash_xy[1] - 0.12 * stash_i,
                        self._resting_z(diameter),
                    ],
                    [1, 0, 0, 0],
                )
                actor = self._actor_manager.add_from_usd_file(
                    name=f"{role_name}_{class_key}",
                    asset_path=f"Can_d{diameter}cm.usd",
                    pose=stash_pose,
                    density=float(variant["density"]),
                    friction_ratio=float(variant["friction"]),
                )
                self.variant_actors[(role_name, class_key)] = actor
                stash_i += 1

    def _reset_actors(self):
        for idx, actor in enumerate(self.variant_actors.values()):
            actor.set_pose(
                Pose(
                    [
                        self.stash_xy[0],
                        self.stash_xy[1] - 0.12 * idx,
                        0.03,
                    ],
                    [1, 0, 0, 0],
                )
            )

        self.selected_class_key = str(self.rng.choice(list(TACTILE_VARIANTS.keys())))
        self.selected_class = TACTILE_VARIANTS[self.selected_class_key]
        self.object_a = self.variant_actors[("object_a", self.selected_class_key)]
        self.object_b = self.variant_actors[("object_b", self.selected_class_key)]
        self.objects = {"object_a": self.object_a, "object_b": self.object_b}
        self.variants = {
            "object_a": dict(self.selected_class["a"]),
            "object_b": dict(self.selected_class["b"]),
        }
        self.target_poses = {}

        for role_name, actor in self.objects.items():
            variant = self.variants[role_name]
            jitter = self.rng.uniform(-0.015, 0.015, size=2)
            start_xy = self.start_xy[role_name] + jitter
            diameter = int(variant["diameter"])
            actor.set_pose(Pose([start_xy[0], start_xy[1], self._resting_z(diameter)], [1, 0, 0, 0]))
            slot_xy = self.slot_xy[role_name]
            self.target_poses[role_name] = Pose(
                [slot_xy[0], slot_xy[1], self._resting_z(diameter)],
                [1, 0, 0, 0],
            )

        self._reset_episode_state()

    def _reset_episode_state(self):
        self.vision_disabled = False
        self.vision_disabled_step = None
        self.task_phase = "approach"
        self.active_role = None
        self.sequence_violation = False
        self.tactile_timeline = []
        self.object_initial_z = {
            role: float(actor.get_pose().p[2]) for role, actor in self.objects.items()
        }
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
                "vision_disabled_step": None,
                "object_a_lifted": False,
                "object_a_placed": False,
                "object_b_lifted": False,
                "object_b_placed": False,
                "object_a_action_count": 0,
                "object_b_action_count": 0,
                "object_a_regrasp_count": 0,
                "object_b_regrasp_count": 0,
                "sequence_violation": False,
            }
        )

    def pre_move(self):
        self.delay(10)
        self.move(self.atom.open_gripper(1.0), tag="pre_open")

    def _play_once(self):
        if not self._transfer_object("object_a"):
            return
        if not self._transfer_object("object_b"):
            return
        self.delay(30, is_save=False)

    def _transfer_object(self, role_name: str):
        actor = self.objects[role_name]
        self.active_role = role_name

        for attempt in range(3):
            if attempt > 0:
                self.object_regrasp_count[role_name] += 1
                self._sync_metadata()

            self.task_phase = f"{role_name}_search"
            self._role_move(role_name, self.atom.open_gripper(1.0), tag=f"{role_name}_open")
            grasp_pose = self._register_can_grasp(actor, self.variants[role_name], attempt)
            self._role_move(
                role_name,
                self.atom.grasp_actor(
                    actor,
                    contact_point_id=grasp_pose,
                    pre_dis=0.05,
                    dis=0.0,
                    is_close=False,
                ),
                tag=f"{role_name}_approach",
                time_dilation_factor=0.5,
            )

            self.task_phase = f"{role_name}_grasp"
            self._role_move(
                role_name,
                self.atom.close_gripper(0.0, depth_threshold=self._adaptive_threshold(role_name)),
                tag=f"{role_name}_adaptive_close",
            )
            self.delay(8, is_save=True)

            self.task_phase = f"{role_name}_lift"
            self._role_move(
                role_name,
                self.atom.move_by_displacement(z=self._lift_distance(role_name)),
                tag=f"{role_name}_lift",
                time_dilation_factor=self._lift_time_dilation(role_name),
            )
            self.delay(12, is_save=True)
            self._update_task_state()
            if self.object_lifted[role_name]:
                break
        else:
            self.plan_success = False
            self.metadata[f"{role_name}_pregrasp_failed"] = True
            self._sync_metadata()
            return False

        if not self._release_object_at_slot(role_name):
            self.plan_success = False
            self.metadata[f"{role_name}_place_failed"] = True
            self._sync_metadata()
            self.active_role = None
            return False
        self.active_role = None
        return True

    def _register_can_grasp(self, actor: Actor, variant: dict, attempt: int):
        can_pose = actor.get_pose()
        diameter = int(variant["diameter"])
        lateral_bias = -0.065 - 0.002 * (diameter - 5)
        height_bias = -0.008 - 0.002 * attempt
        target_pose = can_pose.add_bias([lateral_bias, 0.0, height_bias])
        target_mat = target_pose.to_transformation_matrix()
        x_axis = target_mat[:3, 0].reshape(-1)
        grasp_mat = np.vstack([x_axis, np.cross(x_axis, [0, 0, 1]), [0, 0, 1]])
        grasp_noise = Pose.create_noise(vec=[0.002, 0.002, 0.001], euler=[0, [-np.pi / 24, np.pi / 24], 0], rng=self.rng)
        grasp_pose = construct_grasp_pose(
            target_pose.p,
            grasp_mat[:3, 2],
            grasp_mat[:3, 0],
        ).add_offset(grasp_noise)
        return actor.register_point(pose=grasp_pose, type="contact")

    def _release_object_at_slot(self, role_name: str) -> bool:
        actor = self.objects[role_name]
        target_pose = self.target_poses[role_name]
        current_gripper = self._robot_manager.get_gripper_center_pose()
        current_actor = actor.get_pose()
        inhand_offset = np.asarray(current_gripper.p - current_actor.p, dtype=np.float64)
        target_xy = np.asarray(target_pose.p[:2], dtype=np.float64)
        target_z = float(target_pose.p[2])

        def move_gripper_center(tag: str, actor_target_p: np.ndarray) -> bool:
            gripper_target_p = np.asarray(actor_target_p, dtype=np.float64) + inhand_offset
            gripper_target_p[2] = max(float(gripper_target_p[2]), 0.075)
            gripper_target = Pose(gripper_target_p, current_gripper.q.copy())
            ee_target = self._robot_manager.gripper_center_to_ee(gripper_target)
            return self._role_move(
                role_name,
                self.atom.move_to_pose(ee_target),
                tag=tag,
                time_dilation_factor=0.5,
            )

        self.task_phase = f"{role_name}_place"
        hover_actor_p = np.array(
            [target_xy[0], target_xy[1], target_z + self.slot_hover_height],
            dtype=np.float64,
        )
        if not move_gripper_center(f"{role_name}_slot_hover", hover_actor_p):
            return False

        release_actor_p = np.array(
            [target_xy[0], target_xy[1], target_z + self.slot_release_clearance],
            dtype=np.float64,
        )
        if not move_gripper_center(f"{role_name}_slot_descend", release_actor_p):
            return False

        if not self._role_move(role_name, self.atom.open_gripper(1.0), tag=f"{role_name}_release_open"):
            return False
        self.delay(20, is_save=True)

        retreat_gripper = self._robot_manager.get_gripper_center_pose().add_bias(
            [0.0, 0.0, self.slot_retreat_height],
            coord="world",
        )
        retreat_ee = self._robot_manager.gripper_center_to_ee(retreat_gripper)
        if not self._role_move(
            role_name,
            self.atom.move_to_pose(retreat_ee),
            tag=f"{role_name}_release_retreat",
            time_dilation_factor=0.5,
        ):
            return False
        self.delay(25, is_save=True)
        self._update_task_state()
        return bool(self.object_placed[role_name])

    def _role_move(self, role_name: str, actions: list[Action], **kwargs):
        if actions is None:
            return False
        self.object_action_count[role_name] += 1
        self._sync_metadata()
        return self.move(actions, **kwargs)

    def _adaptive_threshold(self, role_name: str) -> float:
        variant = self.variants[role_name]
        heavy = float(variant["density"]) >= 2000.0
        rough = float(variant["friction"]) >= 0.8
        threshold = 27.75
        if heavy:
            threshold -= 0.45
        if not rough:
            threshold -= 0.20
        return threshold

    def _lift_distance(self, role_name: str) -> float:
        variant = self.variants[role_name]
        return 0.10 if float(variant["density"]) >= 2000.0 else 0.075

    def _lift_time_dilation(self, role_name: str) -> float:
        variant = self.variants[role_name]
        return 0.35 if float(variant["friction"]) < 0.8 else 0.5

    @staticmethod
    def _resting_z(diameter_cm: int) -> float:
        return 0.005 * float(diameter_cm) + 0.001

    def _step(self, is_save: bool = True):
        ret = super()._step(is_save=is_save)
        self._update_vision_gate()
        self._update_task_state()
        self._record_tactile_timeline()
        return ret

    def _get_observations(self):
        self._update_vision_gate()
        obs = super()._get_observations()

        observer_obs = obs.get("observation", {}).pop("observer", None)
        if observer_obs is not None:
            self._last_observer_obs = observer_obs

        if getattr(self, "vision_disabled", False) and "observation" in obs:
            for camera_name in ("head", "wrist"):
                if camera_name not in obs["observation"]:
                    continue
                for data_type, value in list(obs["observation"][camera_name].items()):
                    if isinstance(value, torch.Tensor):
                        obs["observation"][camera_name][data_type] = torch.zeros_like(value)

        if "actor" in obs and hasattr(self, "objects"):
            actor_obs = {}
            for role_name, actor in self.objects.items():
                actor_obs[role_name] = actor.get_pose().totensor(device=self.device)
            actor_obs["slot_a"] = self.slot_a.get_pose().totensor(device=self.device)
            actor_obs["slot_b"] = self.slot_b.get_pose().totensor(device=self.device)
            obs["actor"] = actor_obs
        return obs

    def get_frame_shot(self, obs):
        camera_obs = self._camera_manager.get_observations(["rgb"])
        observer = camera_obs.get("observer", camera_obs.get("head", {})).get("rgb")
        wrist = camera_obs.get("wrist", camera_obs.get("head", {})).get("rgb")
        if observer is None or wrist is None:
            return super().get_frame_shot(obs)

        tac_size = 160
        left_tac = torchvision.transforms.Resize((tac_size, tac_size))(
            obs["tactile"]["left_tactile"]["rgb_marker"].clone().permute(2, 0, 1)
        ).permute(1, 2, 0)
        right_tac = torchvision.transforms.Resize((tac_size, tac_size))(
            obs["tactile"]["right_tactile"]["rgb_marker"].clone().permute(2, 0, 1)
        ).permute(1, 2, 0)

        img = torch.zeros((320, 480 * 2 + 160, 3), dtype=observer.dtype, device=observer.device)
        img[:, :480, :] = torchvision.transforms.Resize((320, 480))(
            observer.permute(2, 0, 1)
        ).permute(1, 2, 0)
        img[:, 480:960, :] = torchvision.transforms.Resize((320, 480))(
            wrist.permute(2, 0, 1)
        ).permute(1, 2, 0)
        img[:tac_size, 960:, :] = left_tac
        img[tac_size:, 960:, :] = right_tac
        return img

    def _update_vision_gate(self):
        if getattr(self, "vision_disabled", False):
            return
        try:
            p = self._robot_manager.get_gripper_center_pose().p
        except Exception:
            return
        xy = np.asarray(p[:2], dtype=np.float64)
        in_region = np.all(np.abs(xy - self.search_center) <= self.search_half_extents)
        if in_region and float(p[2]) < self.vision_disable_z:
            self.vision_disabled = True
            self.vision_disabled_step = int(self.step_count)
            self.metadata["vision_disabled_step"] = int(self.step_count)

    def _update_task_state(self):
        if not hasattr(self, "objects"):
            return

        for role_name, actor in self.objects.items():
            pose = actor.get_pose()
            z_lift = float(pose.p[2] - self.object_initial_z[role_name])
            gripper_pose = self._robot_manager.get_gripper_center_pose()
            inhand_xy = float(np.linalg.norm(pose.p[:2] - gripper_pose.p[:2]))
            inhand_z = float(abs(pose.p[2] - gripper_pose.p[2]))
            held = z_lift >= self.lift_height_threshold and inhand_xy < 0.08 and inhand_z < 0.16
            if held:
                self.object_lift_hold_count[role_name] += 1
            else:
                self.object_lift_hold_count[role_name] = 0
            if self.object_lift_hold_count[role_name] >= self.hold_steps_required:
                self.object_lifted[role_name] = True

            placed = self._is_placed(role_name)
            if placed:
                self.object_place_stable_count[role_name] += 1
            else:
                self.object_place_stable_count[role_name] = 0
            if (
                self.object_lifted[role_name]
                and self.object_place_stable_count[role_name] >= self.placement_stable_steps_required
            ):
                self.object_placed[role_name] = True

        b_lift = self.object_b.get_pose().p[2] - self.object_initial_z["object_b"]
        if not self.object_placed["object_a"] and b_lift > self.b_sequence_lift_threshold:
            self.sequence_violation = True

        if self.object_placed["object_a"] and not self.object_placed["object_b"]:
            self.task_phase = "manipulate_object_b"
        elif self.object_placed["object_a"] and self.object_placed["object_b"]:
            self.task_phase = "completed"

        self._sync_metadata()

    def _is_placed(self, role_name: str) -> bool:
        actor = self.objects[role_name]
        pose = actor.get_pose()
        target = self.target_poses[role_name]
        xy_error = float(np.linalg.norm(pose.p[:2] - target.p[:2]))
        z_error = float(abs(pose.p[2] - target.p[2]))
        gripper_open = self._robot_manager.get_gripper_qpos() > 0.020
        return (
            xy_error < self.placement_xy_threshold
            and z_error < self.placement_z_threshold
            and gripper_open
        )

    @staticmethod
    def _quat_angle(q1, q2) -> float:
        q1 = np.asarray(q1, dtype=np.float64)
        q2 = np.asarray(q2, dtype=np.float64)
        q1 = q1 / max(np.linalg.norm(q1), 1e-8)
        q2 = q2 / max(np.linalg.norm(q2), 1e-8)
        dot = float(np.clip(abs(np.dot(q1, q2)), -1.0, 1.0))
        return 2.0 * math.acos(dot)

    def _sync_metadata(self):
        if not hasattr(self, "object_lifted"):
            return
        self.metadata.update(
            {
                "vision_disabled_step": self.vision_disabled_step,
                "object_a_lifted": bool(self.object_lifted["object_a"]),
                "object_a_placed": bool(self.object_placed["object_a"]),
                "object_b_lifted": bool(self.object_lifted["object_b"]),
                "object_b_placed": bool(self.object_placed["object_b"]),
                "object_a_action_count": int(self.object_action_count["object_a"]),
                "object_b_action_count": int(self.object_action_count["object_b"]),
                "object_a_regrasp_count": int(self.object_regrasp_count["object_a"]),
                "object_b_regrasp_count": int(self.object_regrasp_count["object_b"]),
                "sequence_violation": bool(self.sequence_violation),
                "task_phase": self.task_phase,
            }
        )

    def _record_tactile_timeline(self):
        if not hasattr(self, "tactile_timeline"):
            return
        if self.step_count % self.timeline_frequency != 0:
            return
        try:
            tactile_obs = self._tactile_manager.get_observations(["depth", "marker"])
        except Exception:
            tactile_obs = {}

        row = {
            "step": int(self.step_count),
            "phase": self.task_phase,
            "active_role": self.active_role,
            "vision_disabled": bool(self.vision_disabled),
            "gripper_qpos": float(self._robot_manager.get_gripper_qpos()),
        }
        for role_name, actor in getattr(self, "objects", {}).items():
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
        if len(self.tactile_timeline) > 0:
            fieldnames = sorted({key for row in self.tactile_timeline for key in row.keys()})
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.tactile_timeline)

    def check_success(self):
        self._update_task_state()
        return (
            bool(self.object_lifted["object_a"])
            and bool(self.object_lifted["object_b"])
            and bool(self.object_placed["object_a"])
            and bool(self.object_placed["object_b"])
            and not bool(self.sequence_violation)
        )

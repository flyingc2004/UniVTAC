from ._base_task import *
import numpy as np

@configclass
class TaskCfg(BaseTaskCfg):
    cameras = [
        CameraCfg(
            name="head",
            prim_path="/World/envs/env_.*/Camera",
            offset=CameraCfg.OffsetCfg(pos=(1, 0.0, 0.15), rot=(0.5, 0.5, 0.5, 0.5), convention="opengl"),
            data_types=["rgb", "depth"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=2.5, focus_distance=1.0, horizontal_aperture=3.6, clipping_range=(0.1, 100.0)
            ),
            width=480,
            height=270,
            update_period=1/120
        ),
        CameraCfg(
            name="wrist",
            prim_path="/World/envs/env_.*/Robot/WristCamera/Camera",
            data_types=["rgb", "depth"],
            spawn=None, # use existing camera
            width=480,
            height=270,
            update_period=1/120,
        )
    ]
    use_adaptive_grasp = False

class Task(BaseTask):
    def __init__(self, cfg: BaseTaskCfg, mode:Literal['collect', 'eval'] = 'collect', render_mode: str|None = None, **kwargs):
        cfg.sim.physics_material.dynamic_friction = 2.5
        cfg.sim.physics_material.static_friction = 2.5
        cfg.uipc_sim.contact.default_friction_ratio = 2.5
        self._deterministic_inhand_follow = False
        self.origin_inhand_pose = None
        # Pose of the prism center in the gripper-center frame during CaP-X
        # deterministic in-hand reset. Local +z points down through the fingers
        # for this top-down grasp; smaller z pulls the prism closer into the pads.
        self._pregrasp_inhand_max_error = 0.025
        super().__init__(cfg, mode, render_mode, **kwargs)

    def _pregrasp_state(self):
        prism_pose = self.prism.get_pose()
        gripper_pose = self._robot_manager.get_gripper_center_pose()
        xy_dist = float(np.linalg.norm(prism_pose.p[:2] - gripper_pose.p[:2]))
        prism_in_gripper = self.prism.get_pose().rebase(gripper_pose)
        target_inhand_pose = self.origin_inhand_pose
        if target_inhand_pose is None:
            target_inhand_pose = Pose([0.0, 0.0, 0.040], [1, 0, 0, 0])
        inhand_error = float(
            np.linalg.norm(prism_in_gripper.p - target_inhand_pose.p)
        )
        return {
            'prism_pose': prism_pose,
            'gripper_pose': gripper_pose,
            'prism_in_gripper': prism_in_gripper,
            'target_inhand_pose': target_inhand_pose,
            'inhand_error': inhand_error,
            'prism_z': float(prism_pose.p[2]),
            'xy_dist': xy_dist,
            'gripper_qpos': float(self._robot_manager.get_gripper_qpos()),
        }

    def _is_pregrasp_ok(
        self,
        min_z: float = 0.035,
        max_xy: float = 0.08,
        max_inhand_error: float | None = None,
    ) -> bool:
        state = self._pregrasp_state()
        prism_pose = state['prism_pose']
        gripper_pose = state['gripper_pose']
        if not (np.isfinite(prism_pose.p).all() and np.isfinite(gripper_pose.p).all()):
            return False
        if max_inhand_error is None:
            max_inhand_error = self._pregrasp_inhand_max_error
        return (
            state['prism_z'] > min_z
            and state['xy_dist'] < max_xy
            and state['inhand_error'] < max_inhand_error
        )

    def _safe_move(self, actions, **kwargs):
        prev_plan_success = self.plan_success
        ok = self.move(actions, **kwargs)
        if not ok and prev_plan_success:
            self.plan_success = prev_plan_success
        return ok

    def _log_pregrasp(self, prefix: str, attempt: int, extra: str = ""):
        state = self._pregrasp_state()
        msg = (
            f"[grasp_classify] {prefix} attempt={attempt} "
            f"prism_z={state['prism_z']:.4f} "
            f"xy_dist={state['xy_dist']:.4f} "
            f"gripper_qpos={state['gripper_qpos']:.4f} "
            f"inhand_p={np.array2string(state['prism_in_gripper'].p, precision=4)} "
            f"target_inhand_p={np.array2string(state['target_inhand_pose'].p, precision=4)} "
            f"inhand_error={state['inhand_error']:.4f}"
        )
        if extra:
            msg += f" {extra}"
        print(msg)

    def _set_prism_in_hand(self):
        gripper_pose = self._robot_manager.get_gripper_center_pose()
        inhand_pose = self._deterministic_inhand_pose
        target_pose = inhand_pose.rebase(from_coord=gripper_pose)
        self.prism.set_pose(target_pose)
        self.metadata['deterministic_inhand_reset'] = True
        self.metadata['deterministic_inhand_pose'] = inhand_pose.tolist()

        for _ in range(20):
            self._step(is_save=False)
        state = self._pregrasp_state()
        self.metadata['deterministic_inhand_locked_state'] = {
            'prism_pose': state['prism_pose'].tolist(),
            'gripper_pose': state['gripper_pose'].tolist(),
            'prism_in_gripper': state['prism_in_gripper'].tolist(),
            'target_inhand_pose': state['target_inhand_pose'].tolist(),
            'inhand_error': state['inhand_error'],
            'prism_z': state['prism_z'],
            'xy_dist': state['xy_dist'],
            'gripper_qpos': state['gripper_qpos'],
        }
        self._log_pregrasp("inhand_locked", 1)

    def _follow_prism_to_gripper(self):
        if not self._deterministic_inhand_follow:
            return
        if not hasattr(self, 'prism'):
            return
        gripper_pose = self._robot_manager.get_gripper_center_pose()
        target_pose = self._deterministic_inhand_pose.rebase(from_coord=gripper_pose)
        self.prism.set_pose(target_pose)

    def enable_deterministic_inhand_follow(self):
        self._deterministic_inhand_follow = True
        self.metadata['deterministic_inhand_follow'] = True
        self._follow_prism_to_gripper()
        print("[grasp_classify] deterministic in-hand follow enabled", flush=True)

    def release_deterministic_inhand_follow(self, settle_steps: int = 20):
        if not self._deterministic_inhand_follow:
            return
        self._follow_prism_to_gripper()
        self._deterministic_inhand_follow = False
        self.metadata['deterministic_inhand_follow_released'] = True
        self.prism.remove_animate()
        print("[grasp_classify] deterministic in-hand follow released", flush=True)
        for _ in range(max(0, int(settle_steps))):
            self._step(is_save=False)

    def _step(self, is_save: bool = True):
        self._follow_prism_to_gripper()
        ret = super()._step(is_save=is_save)
        self._follow_prism_to_gripper()
        return ret

    def _release_prism_animation(self, settle_steps: int = 10):
        self.prism.remove_animate()
        for _ in range(max(0, int(settle_steps))):
            self._step(is_save=False)
        state = self._pregrasp_state()
        self.metadata['deterministic_inhand_released_state'] = {
            'prism_pose': state['prism_pose'].tolist(),
            'gripper_pose': state['gripper_pose'].tolist(),
            'prism_in_gripper': state['prism_in_gripper'].tolist(),
            'target_inhand_pose': state['target_inhand_pose'].tolist(),
            'inhand_error': state['inhand_error'],
            'prism_z': state['prism_z'],
            'xy_dist': state['xy_dist'],
            'gripper_qpos': state['gripper_qpos'],
        }
        self._log_pregrasp("inhand_released", 1)

    def _do_pregrasp_attempt(self, attempt: int) -> bool:
        height_jitter = self.rng.uniform(-0.003, 0.003)
        grasp_height = [0.042, 0.038, 0.034][attempt] + height_jitter
        close_pos = float(
            np.clip([0.05, 0.02, 0.00][attempt] + self.rng.uniform(-0.003, 0.003), 0.0, 0.08)
        )

        self.metadata.setdefault('pregrasp_attempts', [])
        self.metadata['pregrasp_attempts'].append({
            'attempt': attempt + 1,
            'grasp_height': float(grasp_height),
            'close_pos': float(close_pos),
        })

        self._log_pregrasp("open", attempt + 1)
        if not self._safe_move(self.atom.open_gripper(0.5)):
            self._log_pregrasp("open_failed", attempt + 1)
            return False

        target_pose = self.prism.get_pose().add_bias([0.0, 0.0, grasp_height])
        cpose = construct_grasp_pose(
            target_pose.p,
            [0, 0, 1],
            [1, 0, 0]
        )
        cid = self.prism.register_point(cpose, type='contact')
        self._log_pregrasp("grasp", attempt + 1, extra=f"grasp_height={grasp_height:.4f} close_pos={close_pos:.4f}")
        if not self._safe_move(self.atom.grasp_actor(
            self.prism,
            contact_point_id=cid,
            pre_dis=0.04,
            dis=0.0,
            is_close=False
        )):
            self._log_pregrasp("grasp_failed", attempt + 1)
            return False

        self.delay(10, is_save=False)
        prev_use_adaptive_grasp = self.cfg.use_adaptive_grasp
        try:
            self.cfg.use_adaptive_grasp = False
            self._log_pregrasp("close", attempt + 1, extra="fixed_qpos")
            if not self._safe_move(self.atom.close_gripper(close_pos)):
                self._log_pregrasp("close_failed", attempt + 1)
                return False
        finally:
            self.cfg.use_adaptive_grasp = prev_use_adaptive_grasp

        self.origin_inhand_pose = self.prism.get_pose().rebase(
            self._robot_manager.get_gripper_center_pose()
        )
        self.metadata['origin_inhand_pose'] = self.origin_inhand_pose.tolist()
        self.cfg.keep_contact = True
        self.delay(15, is_save=False)

        self._log_pregrasp("lift", attempt + 1)
        if not self._safe_move(self.atom.move_by_displacement(z=0.05)):
            self._log_pregrasp("lift_failed", attempt + 1)
            return False

        self.delay(5, is_save=False)
        ok = self._is_pregrasp_ok()
        state = self._pregrasp_state()
        self.metadata['pregrasp_ok'] = bool(ok)
        self.metadata['pregrasp_state'] = {
            'prism_pose': state['prism_pose'].tolist(),
            'gripper_pose': state['gripper_pose'].tolist(),
            'prism_in_gripper': state['prism_in_gripper'].tolist(),
            'target_inhand_pose': state['target_inhand_pose'].tolist(),
            'inhand_error': state['inhand_error'],
            'prism_z': state['prism_z'],
            'xy_dist': state['xy_dist'],
            'gripper_qpos': state['gripper_qpos'],
        }
        self._log_pregrasp("check", attempt + 1, extra=f"ok={ok}")
        if ok:
            return True

        self._log_pregrasp("retry_recover", attempt + 1, extra="open_and_retract")
        self.origin_inhand_pose = None
        if not self._safe_move(self.atom.open_gripper(1.0)):
            self._log_pregrasp("retry_open_failed", attempt + 1)
            return False
        self._safe_move(self.atom.move_by_displacement(z=0.03))
        self.delay(3, is_save=False)
        return False

    def create_actors(self):
        green_pose = Pose([0.4, 0.08, 0.01], [1, 0, 0, 0])
        orange_pose = Pose([0.4, -0.08, 0.01], [1, 0, 0, 0])

        self.green_pad = self._actor_manager.add_from_usd_file(
            name='green_pad',
            asset_path="GreenPad.usd",
            pose=green_pose,
        )
        self.orange_pad = self._actor_manager.add_from_usd_file(
            name='orange_pad',
            asset_path="OrangePad.usd",
            pose=orange_pose,
        )
        
        # rough -> orange; plain -> green
        self.rough_prism = self._actor_manager.add_from_usd_file(
            name='rough_prism',
            asset_path="RoughPrism.usd",
            pose=Pose([0.35, 1.0, 0.01], [1, 0, 0, 0])
        )
        self.plain_prism = self._actor_manager.add_from_usd_file(
            name='plain_prism',
            asset_path="PlainPrism.usd",
            pose=Pose([0.35, -1.0, 0.01], [1, 0, 0, 0])
        )
    
    def _reset_actors(self):
        self.choice = self.rng.choice(['rough', 'plain'])
        start_pose = Pose([0.35, 0.0, 0.01], [1, 0, 0, 0])
        if self.choice == 'rough':
            self.prism = self.rough_prism
            self.target = self.orange_pad
            self.other_target = self.green_pad
        else:
            self.prism = self.plain_prism
            self.target = self.green_pad
            self.other_target = self.orange_pad
        self.prism.set_pose(start_pose)

    def pre_move(self):
        self.delay(10)

        self.target_pose = self.target.get_pose().add_bias([0.0, 0.0, 0.015])
        self.metadata['target_pose'] = self.target_pose.tolist()
        self.metadata['pregrasp_ok'] = False
        self.origin_inhand_pose = None
        print("[grasp_classify] pre_move physical grasp begin")
        for attempt in range(3):
            if self._do_pregrasp_attempt(attempt):
                self.metadata['pregrasp_ok'] = True
                self._log_pregrasp("physical_pregrasp", attempt + 1, extra="ok=True")
                break
        else:
            self.plan_success = False
            self.metadata['pregrasp_fail_reason'] = 'physical_pregrasp_failed_after_retries'
            print(
                "[grasp_classify] physical pre_move grasp failed after 3 retries",
                flush=True,
            )
            return

        self.target_pose = self.target.get_pose().add_bias([0.0, 0.0, 0.015])

    def _play_once(self):
        self.move(self.atom.place_actor(
            self.prism,
            target_pose=self.target_pose,
            pre_dis=0.0, dis=0.0,
            is_open=False
        ), time_dilation_factor=0.5)
        self.delay(20, is_save=False)

    def check_success(self):
        card_pose = self.prism.get_pose().rebase(self.target_pose)
        return np.all(np.abs(card_pose.p) < np.array([0.02, 0.02, 0.01])) and \
            np.dot(card_pose.to_transformation_matrix()[:3, 2], np.array([0, 0, 1])) > 0.965  # 15°

import os
import sys
import time
import yaml
import json
import argparse
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, Literal

sys.path.append('.')

# add argparse arguments
parser = argparse.ArgumentParser(
    description="Collect data"
)
parser.add_argument(
    "task",
    type=str,
    help="Task file name",
)
parser.add_argument(
    "config",
    type=str,
    help="Config file name",
    default="demo.yml"
)
parser.add_argument(
    "--episode_num",
    type=int,
    default=-1,
)
parser.add_argument(
    "--start_seed",
    type=int,
    default=-1,
)
parser.add_argument(
    "--max_seed",
    type=int,
    default=-1,
)
parser.add_argument(
    "--gpu",
    type=str,
    default=None,
)
parser.add_argument(
    "--seed_stride",
    type=int,
    default=1,
    help="Advance seeds by this stride after every episode.",
)
parser.add_argument(
    "--task_cfg_override",
    action="append",
    default=[],
    help="Repeatable key=value YAML override applied to TaskCfg before construction.",
)
parser.add_argument(
    "--run_name",
    type=str,
    default=None,
    help="Optional isolated subdirectory below this task configuration's output root.",
)

args_cli = parser.parse_args()
if args_cli.gpu is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = args_cli.gpu

from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)

# parse the arguments
args_cli.enable_cameras = True
args_cli.num_envs = 1

def get_config(file, default_root:Path, type:Literal['yaml', 'json']):
    if type == 'yaml':
        if file.endswith('.yml') or file.endswith('.yaml'):
            file = Path(file)
        else:
            file = default_root / f'{file}.yml'
        with open(file, 'r') as f:
            config = yaml.load(f.read(), Loader=yaml.FullLoader)
        return config, file
    else:
        if file.endswith('.json'):
            file = Path(file)
        else:
            file = default_root / f'{file}.json'
        with open(file, 'r') as f:
            config = json.load(f)
        return config, file

def get_bool_config(config: dict, key: str, default: bool = False) -> bool:
    value = os.environ.get(f"UNIVTAC_{key.upper()}", config.get(key, default))
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)

task_config, task_config_file = get_config(
    args_cli.config, 
    default_root=Path(__file__).parent.parent / 'task_config', 
    type='yaml'
)

for raw_override in args_cli.task_cfg_override:
    if "=" not in raw_override:
        parser.error("--task_cfg_override must use key=value syntax")
    key, value = raw_override.split("=", 1)
    key = key.strip()
    if not key:
        parser.error("--task_cfg_override key cannot be empty")
    task_config.setdefault("task_cfg_overrides", {})[key] = yaml.safe_load(value)

args_cli.headless = task_config.get("headless", getattr(args_cli, "headless", True))
args_cli.livestream = task_config.get("livestream", getattr(args_cli, "livestream", 0))

if task_config.get('render_frequency', 1) == 0 and "livestream" not in task_config:
    args_cli.livestream = 2

# launch omniverse app, must done before importing anything from omni.isaac
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import importlib
if TYPE_CHECKING:
    from envs._base_task import BaseTask, BaseTaskCfg

log_path = Path('./log')
def log(msg):
    global log_path
    log_path.parent.mkdir(parents=True, exist_ok=True)

    msg = f"[{time.strftime(r'%Y-%m-%d %H:%M:%S')}] {msg}"
    with open(log_path, 'a') as f:
        f.write(msg + '\n')
    print(msg)

def run(
    task: 'BaseTask',
    episode_num,
    use_seed,
    start_seed,
    max_seed,
    save_hdf5: bool = True,
    seed_stride: int = 1,
):
    if seed_stride < 1:
        raise ValueError("seed_stride must be positive")
    suc_num, seed = 0, 0
    attempted_num = 0
    suc_map_path = task.save_root / 'suc_map.txt'
    if suc_map_path.exists():
        with open(suc_map_path, 'r') as f:
            suc_map = [entry for entry in f.read().strip().split(' ') if entry]
    else:
        suc_map = []

    def record_seed_result(seed_value: int, result: str):
        """Keep a resumable seed-indexed map when collection is split."""
        if seed_value >= len(suc_map):
            suc_map.extend(['?'] * (seed_value - len(suc_map)))
            suc_map.append(result)
        else:
            suc_map[seed_value] = result
    
    if start_seed != -1:
        seed = start_seed
        log(f"Starting from seed {seed}.")
    elif use_seed:
        if suc_map_path.exists():
            suc_num = sum([1 for s in suc_map if s == '1'])
            seed = len(suc_map)
            log(f"Use seed with {suc_num} successful episodes. Starting from seed {seed}.")

    mean_steps = 0.0
    while suc_num < episode_num and (max_seed == -1 or seed <= max_seed):
        attempted_num += 1
        try:
            start_t = time.perf_counter()
            task.reset(seed=seed)
            task.play_once()
            cost_t = time.perf_counter() - start_t
        except Exception as e:
            log(f"[{suc_num:<3d}] Seed {seed} failed with error: {traceback.format_exc()}")
            record_seed_result(seed, '0')
            task.clean_cache(mean_steps=mean_steps, result='error')
        else:
            if task.plan_success and task.check_success() and not task.check_early_stop():
                if save_hdf5:
                    task.save_to_hdf5()
                log(f"[{suc_num:<3d}] Seed {seed} success in {cost_t:.2f} s.\n"
                    f"steps: {task.step_count:<5d}, save frames: {task.save_count:<5d}.\n")
                suc_num += 1
                record_seed_result(seed, '1')
                if mean_steps > 0: 
                    mean_steps = ((suc_num - 1) * mean_steps + task.step_count) / suc_num
                else:
                    mean_steps = task.step_count
                task.clean_cache(mean_steps=mean_steps, result='success')
            else:
                log(f"[{suc_num:<3d}] Seed {seed} failed in {cost_t:.2f} s.\n"
                    f"Plan {task.plan_success}, Check {task.check_success()}")
                record_seed_result(seed, '0')
                task.clean_cache(mean_steps=mean_steps, result='fail')
        
        with open(suc_map_path, 'w') as f:
            f.write(' '.join([s for s in suc_map]))
        
        seed += seed_stride
    
    success_rate = (suc_num / attempted_num * 100.0) if attempted_num else 0.0
    log(f'Complete collection, success rate: {suc_num}/{attempted_num} ({success_rate:.2f}%)')

    task.close()
    simulation_app.close()

def main():
    global args_cli, task_config, task_config_file, log_path
    task_file_name = args_cli.task

    episode_num = task_config.get("episode_num", -1)
    if args_cli.episode_num != -1:
        episode_num = args_cli.episode_num
    start_seed = task_config.get("start_seed", -1)
    if args_cli.start_seed != -1:
        start_seed = args_cli.start_seed
    max_seed = task_config.get("max_seed", -1)
    if args_cli.max_seed != -1:
        max_seed = args_cli.max_seed
    
    task_config.update({
        "episode_num": episode_num,
        "start_seed": start_seed,
        "max_seed": max_seed,
    })

    task_module = importlib.import_module(f"envs.{task_file_name}")
    env_cfg:'BaseTaskCfg' = task_module.TaskCfg()
    task_cfg_overrides = task_config.get("task_cfg_overrides", {})
    if not isinstance(task_cfg_overrides, dict):
        raise TypeError("task_cfg_overrides must be a mapping")
    for key, value in task_cfg_overrides.items():
        if not hasattr(env_cfg, key):
            raise KeyError(f"Unknown TaskCfg override for {task_file_name}: {key}")
        setattr(env_cfg, key, value)
    env_cfg.tactile_sensor_type = task_config.get('sensor_type', 'gsmini')
    env_cfg.save_dir = Path(task_config.get("save_dir", "./data")) / task_file_name / task_config_file.stem
    if args_cli.run_name:
        env_cfg.save_dir = env_cfg.save_dir / args_cli.run_name
    env_cfg.decimation = task_config.get("decimation", env_cfg.decimation)
    env_cfg.save_frequency = task_config.get("save_frequency", env_cfg.save_frequency)
    env_cfg.video_frequency = task_config.get("video_frequency", env_cfg.video_frequency)
    # A tactile-only collection config has no head/wrist RGB frames for the
    # composite video renderer. Never let an inherited video default abort raw
    # tactile collection with a KeyError while saving a frame.
    camera_observations = task_config.get("observations", {}).get("camera", [])
    if env_cfg.video_frequency > 0 and not camera_observations:
        print(
            "[collect-data] disabling video: observations.camera is empty; "
            "raw tactile collection will continue without RGB video."
        )
        env_cfg.video_frequency = 0
    env_cfg.preserve_raw_cache = get_bool_config(
        task_config, "preserve_raw_cache", env_cfg.preserve_raw_cache
    )
    env_cfg.render_frequency = task_config.get("render_frequency", env_cfg.render_frequency)
    env_cfg.obs_data_type = task_config.get("observations", {})
    env_cfg.random_texture = task_config.get("random_texture", False)
    env_cfg.reset_time_limit = task_config.get("reset_time_limit", env_cfg.reset_time_limit)
    env_cfg.step_lim = task_config.get("step_lim", env_cfg.step_lim)
    env_cfg.max_save_frames = task_config.get("max_save_frames", env_cfg.max_save_frames)
    env_cfg.live_preview_enabled = get_bool_config(task_config, "live_preview_enabled", env_cfg.live_preview_enabled)
    env_cfg.live_preview_path = os.environ.get(
        "UNIVTAC_LIVE_PREVIEW_PATH",
        task_config.get("live_preview_path", env_cfg.live_preview_path),
    )
    env_cfg.live_preview_stride = int(os.environ.get(
        "UNIVTAC_LIVE_PREVIEW_STRIDE",
        task_config.get("live_preview_stride", env_cfg.live_preview_stride),
    ))
    env_cfg.live_preview_jpeg_quality = int(os.environ.get(
        "UNIVTAC_LIVE_PREVIEW_JPEG_QUALITY",
        task_config.get("live_preview_jpeg_quality", env_cfg.live_preview_jpeg_quality),
    ))

    env_cfg.scene.num_envs = 1
    
    init_start = time.perf_counter()
    task:'BaseTask' = task_module.Task(env_cfg, mode='collect')
    init_cost = time.perf_counter() - init_start
    
    log_path = task.save_root / f"{time.strftime(r'%Y-%m-%d_%H:%M:%S')}.log"
    log(f"Task Name: {task_file_name}")
    log(f"Config Name: {task_config_file.stem}")
    log(f"Task Config: \n{json.dumps(task_config, ensure_ascii=False, indent=4)}\n{'-' * 20}\n")
    log(f"Env Config: \n{env_cfg}\n{'-' * 20}\n")
    log(f"Init cost {init_cost:.2f} seconds, devices: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    if env_cfg.live_preview_enabled:
        log(f"Live preview: {env_cfg.live_preview_path} stride={env_cfg.live_preview_stride}")
    log(f"Save HDF5: {get_bool_config(task_config, 'save_hdf5', True)}")
    run(
        task,
        episode_num=episode_num,
        use_seed=task_config.get("use_seed", True),
        start_seed=start_seed,
        max_seed=max_seed,
        save_hdf5=get_bool_config(task_config, "save_hdf5", True),
        seed_stride=args_cli.seed_stride,
    )

if __name__ == "__main__":
    main()

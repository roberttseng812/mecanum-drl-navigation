import os
import json
import time
import argparse
import math
import signal
from collections import deque

import numpy as np

def resolve_axis_bounds(axis_name: str, min_value, max_value, lim, default_lim: float):
    """Resolve rectangular bounds for one axis with legacy x/y_lim fallback."""
    lim_value = float(default_lim if lim is None else lim)
    lo = -lim_value if min_value is None else float(min_value)
    hi = lim_value if max_value is None else float(max_value)

    if not math.isfinite(lo) or not math.isfinite(hi):
        raise ValueError(f"{axis_name}-axis bounds must be finite, got ({lo}, {hi})")
    if lo >= hi:
        raise ValueError(f"{axis_name}-axis lower bound must be smaller than upper bound, got ({lo}, {hi})")
    return lo, hi


def add_rect_bounds_args(parser, default_x_lim: float, default_y_lim: float):
    parser.add_argument("--x-min", type=float, default=None,
                        help=f"World x lower bound. Default falls back to {-float(default_x_lim):.3f}.")
    parser.add_argument("--x-max", type=float, default=None,
                        help=f"World x upper bound. Default falls back to {float(default_x_lim):.3f}.")
    parser.add_argument("--y-min", type=float, default=None,
                        help=f"World y lower bound. Default falls back to {-float(default_y_lim):.3f}.")
    parser.add_argument("--y-max", type=float, default=None,
                        help=f"World y upper bound. Default falls back to {float(default_y_lim):.3f}.")
    parser.add_argument("--x-lim", type=float, default=None,
                        help="Legacy symmetric x half-range. Used only when x-min/x-max are not provided.")
    parser.add_argument("--y-lim", type=float, default=None,
                        help="Legacy symmetric y half-range. Used only when y-min/y-max are not provided.")


def finalize_rect_bounds_args(args, default_x_lim: float, default_y_lim: float):
    args.x_min, args.x_max = resolve_axis_bounds(
        "x",
        getattr(args, "x_min", None),
        getattr(args, "x_max", None),
        getattr(args, "x_lim", None),
        default_x_lim,
    )
    args.y_min, args.y_max = resolve_axis_bounds(
        "y",
        getattr(args, "y_min", None),
        getattr(args, "y_max", None),
        getattr(args, "y_lim", None),
        default_y_lim,
    )
    return args


import rclpy
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback

from amr_rl.amr_env_stage1 import AmrNavStage1Env


class StopTrainingOnSigintCallback(BaseCallback):
    """SIGINT 按下後，回傳 False 讓 SB3 learn() 正常停止。"""

    def __init__(self, interrupted_flag: dict, verbose: int = 0):
        super().__init__(verbose)
        self.interrupted_flag = interrupted_flag
        self._reported = False

    def _on_step(self) -> bool:
        if self.interrupted_flag.get("stop", False):
            if not self._reported:
                print("[INFO] SIGINT stop flag detected. Stopping SB3 learn() loop...")
                self._reported = True
            return False
        return True


class CheckpointWithReplayBufferCallback(BaseCallback):
    """定期存模型；若啟用 save_replay_buffer，則同時存 replay buffer。

    目的：
    - 讓 checkpoint / best / final / interrupt 都採用同一套 replay buffer 檔名規則
      ("<model>.zip" + "<model>_replay_buffer.pkl")，方便後續 resume。
    """

    def __init__(
        self,
        save_freq: int,
        save_path: str,
        name_prefix: str = "sac_stage1_ckpt",
        save_replay_buffer: bool = False,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.save_freq = int(save_freq)
        self.save_path = str(save_path)
        self.name_prefix = str(name_prefix)
        self.save_replay_buffer = bool(save_replay_buffer)
        os.makedirs(self.save_path, exist_ok=True)

    @staticmethod
    def _rb_path_from_model_path(model_zip_path: str) -> str:
        if model_zip_path.endswith(".zip"):
            return model_zip_path[:-4] + "_replay_buffer.pkl"
        return model_zip_path + "_replay_buffer.pkl"

    def _save(self, model_path: str):
        # 先存模型
        self.model.save(model_path)
        # 再存 replay buffer（如果需要）
        if self.save_replay_buffer:
            rb_path = self._rb_path_from_model_path(model_path)
            try:
                # Off-policy (SAC/TD3/DQN) 才有
                if hasattr(self.model, "save_replay_buffer"):
                    self.model.save_replay_buffer(rb_path)
                else:
                    print("[WARN] Model has no save_replay_buffer(). Did you load a non off-policy model?")
            except Exception as e:
                print(f"[WARN] Failed saving replay buffer: {e}")

    def _on_step(self) -> bool:
        if self.save_freq <= 0:
            return True

        # SB3 慣例：用 num_timesteps 來當 checkpoint 編號
        if self.num_timesteps > 0 and (self.num_timesteps % self.save_freq) == 0:
            model_path = os.path.join(self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps.zip")
            try:
                self._save(model_path)
                if self.verbose:
                    print(f"[INFO] Saved checkpoint: {model_path}")
            except Exception as e:
                print(f"[WARN] Failed saving checkpoint: {e}")

        return True


class MultiMetricEarlyStopCallback(BaseCallback):
    """沿用 Stage0/Stage1 的成功率、碰撞率 plateau 偵測。"""

    def __init__(
        self,
        model_save_dir: str,
        save_replay_buffer: bool = False,
        window_episodes: int = 100,
        min_episodes: int = 200,
        plateau_window: int = 20,
        sr_delta: float = 0.02,
        cr_delta: float = 0.01,
        ret_delta: float = 0.5,
        len_delta: float = 5.0,
        plateau_min_success: float = 0.90,
        plateau_max_collision: float = 0.05,
        max_steps: int = 240,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.model_save_dir = model_save_dir
        os.makedirs(self.model_save_dir, exist_ok=True)

        self.save_replay_buffer = bool(save_replay_buffer)

        self.window_episodes = int(window_episodes)
        self.min_episodes = int(min_episodes)

        self.plateau_window = int(plateau_window)
        self.sr_delta = float(sr_delta)
        self.cr_delta = float(cr_delta)
        self.ret_delta = float(ret_delta)
        self.len_delta = float(len_delta)
        self.plateau_min_success = float(plateau_min_success)
        self.plateau_max_collision = float(plateau_max_collision)
        self.max_steps = int(max_steps)

        self._ep_return = 0.0
        self._ep_len = 0

        self._success_hist = deque(maxlen=self.window_episodes)
        self._collision_hist = deque(maxlen=self.window_episodes)
        self._return_hist = deque(maxlen=self.window_episodes)
        self._len_hist = deque(maxlen=self.window_episodes)

        self._sr_series = deque(maxlen=self.plateau_window)
        self._cr_series = deque(maxlen=self.plateau_window)
        self._ret_series = deque(maxlen=self.plateau_window)
        self._len_series = deque(maxlen=self.plateau_window)

        self._episode_count = 0
        self._best_sr = -1.0

    @staticmethod
    def _rb_path_from_model_path(model_zip_path: str) -> str:
        if model_zip_path.endswith(".zip"):
            return model_zip_path[:-4] + "_replay_buffer.pkl"
        return model_zip_path + "_replay_buffer.pkl"

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards", None)
        dones = self.locals.get("dones", None)
        infos = self.locals.get("infos", None)
        if rewards is None or dones is None or infos is None:
            return True

        r = float(rewards[0])
        self._ep_return += r
        self._ep_len += 1

        if not bool(dones[0]):
            return True

        info = infos[0]
        self._episode_count += 1

        is_success = int(info.get("is_success", 0))
        is_collision = int(info.get("is_collision", 0))

        self._success_hist.append(is_success)
        self._collision_hist.append(is_collision)
        self._return_hist.append(self._ep_return)
        self._len_hist.append(self._ep_len)

        self._ep_return = 0.0
        self._ep_len = 0

        self.logger.record("rollout/episode_count", self._episode_count)
        if len(self._success_hist) < self.window_episodes:
            return True

        success_rate = sum(self._success_hist) / float(len(self._success_hist))
        collision_rate = sum(self._collision_hist) / float(len(self._collision_hist))
        ret_mean = sum(self._return_hist) / float(len(self._return_hist))
        len_mean = sum(self._len_hist) / float(len(self._len_hist))

        self.logger.record("rollout/success_rate", success_rate)
        self.logger.record("rollout/collision_rate", collision_rate)
        self.logger.record("rollout/ep_return_mean_window", ret_mean)
        self.logger.record("rollout/ep_len_mean_window", len_mean)

        self._sr_series.append(success_rate)
        self._cr_series.append(collision_rate)
        self._ret_series.append(ret_mean)
        self._len_series.append(len_mean)

        if success_rate > self._best_sr + 1e-6:
            self._best_sr = success_rate
            best_path = os.path.join(self.model_save_dir, "best_success.zip")
            try:
                self.model.save(best_path)
                if self.save_replay_buffer:
                    rb_path = self._rb_path_from_model_path(best_path)
                    try:
                        if hasattr(self.model, "save_replay_buffer"):
                            self.model.save_replay_buffer(rb_path)
                        else:
                            print("[WARN] Model has no save_replay_buffer(). Did you load a non off-policy model?")
                    except Exception as e:
                        print(f"[WARN] Failed saving best replay buffer: {e}")
                if self.verbose:
                    print(f"[INFO] New best success_rate={success_rate:.3f} saved: {best_path}")
            except Exception as e:
                print(f"[WARN] Failed saving best model: {e}")

        if self._episode_count >= self.min_episodes and len(self._sr_series) >= self.plateau_window:
            sr_var = max(self._sr_series) - min(self._sr_series)
            cr_var = max(self._cr_series) - min(self._cr_series)
            ret_var = max(self._ret_series) - min(self._ret_series)
            len_var = max(self._len_series) - min(self._len_series)

            sr_now = self._sr_series[-1]
            cr_now = self._cr_series[-1]

            if (
                sr_now >= self.plateau_min_success
                and cr_now <= self.plateau_max_collision
                and sr_var <= self.sr_delta
                and cr_var <= self.cr_delta
                and ret_var <= self.ret_delta
                and len_var <= self.len_delta
            ):
                print(
                    "[OK] Early stop (plateau): "
                    f"sr={sr_now:.3f} (var={sr_var:.3f}), "
                    f"cr={cr_now:.3f} (var={cr_var:.3f}), "
                    f"ret_var={ret_var:.3f}, len_var={len_var:.1f}"
                )
                return False

        return True


class SensorDebugCallback(BaseCallback):
    """把 LiDAR 健康度指標寫進 TensorBoard，用來驗證 Stage1 真的有用到 /scan。"""

    def __init__(self, window_episodes: int = 100, verbose: int = 0):
        super().__init__(verbose)
        self._ep_steps = 0
        self._ep_scan_new = 0
        self._ep_scan_present = 0
        self._ep_scan_fresh = 0
        self._ep_scan_age_sum = 0.0
        self._episode_count = 0
        self._scan_new_hist = deque(maxlen=int(window_episodes))
        self._scan_present_hist = deque(maxlen=int(window_episodes))
        self._scan_fresh_hist = deque(maxlen=int(window_episodes))
        self._scan_age_hist = deque(maxlen=int(window_episodes))

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", None)
        dones = self.locals.get("dones", None)
        if infos is None or dones is None:
            return True

        info = infos[0] if len(infos) > 0 else {}
        scan_present = int(info.get("scan_present", 0))
        scan_fresh = int(info.get("scan_fresh", 0))
        scan_new = int(info.get("scan_new", 0))
        scan_age = float(info.get("scan_age_sec", 0.0))
        laser_std = float(info.get("laser_std_norm", 0.0))

        # per-step diagnostics
        self.logger.record("sensor/scan_present", scan_present)
        self.logger.record("sensor/scan_fresh", scan_fresh)
        self.logger.record("sensor/scan_new", scan_new)
        self.logger.record("sensor/scan_age_sec", scan_age)
        self.logger.record("sensor/laser_std_norm", laser_std)

        self._ep_steps += 1
        self._ep_scan_new += scan_new
        self._ep_scan_present += scan_present
        self._ep_scan_fresh += scan_fresh
        self._ep_scan_age_sum += scan_age

        if not bool(dones[0]):
            return True

        # episode-level aggregates
        self._episode_count += 1
        steps = max(1, int(self._ep_steps))
        new_ratio = float(self._ep_scan_new) / float(steps)
        present_ratio = float(self._ep_scan_present) / float(steps)
        fresh_ratio = float(self._ep_scan_fresh) / float(steps)
        age_mean = float(self._ep_scan_age_sum) / float(steps)

        self._scan_new_hist.append(new_ratio)
        self._scan_present_hist.append(present_ratio)
        self._scan_fresh_hist.append(fresh_ratio)
        self._scan_age_hist.append(age_mean)

        self.logger.record("rollout/scan_new_ratio_ep", new_ratio)
        self.logger.record("rollout/scan_present_ratio_ep", present_ratio)
        self.logger.record("rollout/scan_fresh_ratio_ep", fresh_ratio)
        self.logger.record("rollout/scan_age_mean_ep", age_mean)

        if len(self._scan_present_hist) > 0:
            self.logger.record(
                "rollout/scan_present_ratio_window",
                sum(self._scan_present_hist) / float(len(self._scan_present_hist)),
            )
            self.logger.record(
                "rollout/scan_fresh_ratio_window",
                sum(self._scan_fresh_hist) / float(len(self._scan_fresh_hist)),
            )
            self.logger.record(
                "rollout/scan_new_ratio_window",
                sum(self._scan_new_hist) / float(len(self._scan_new_hist)),
            )
            self.logger.record(
                "rollout/scan_age_mean_window",
                sum(self._scan_age_hist) / float(len(self._scan_age_hist)),
            )

        # reset accumulators
        self._ep_steps = 0
        self._ep_scan_new = 0
        self._ep_scan_present = 0
        self._ep_scan_fresh = 0
        self._ep_scan_age_sum = 0.0
        return True


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--logdir", type=str, default="runs_stage1")
    p.add_argument("--modeldir", type=str, default="models_stage1")
    p.add_argument("--total-steps", type=int, default=400_000)
    p.add_argument("--seed", type=int, default=0)

    # ROS/Gazebo wiring
    p.add_argument("--set-entity-service", type=str, default="/set_entity_state")
    p.add_argument("--cmd-vel-topic", type=str, default="cmd_vel")
    p.add_argument("--odom-topic", type=str, default="odom")
    p.add_argument("--scan-topic", type=str, default="scan")

    # env params
    add_rect_bounds_args(p, default_x_lim=3.5, default_y_lim=3.5)
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--collision-threshold", type=float, default=0.57)
    p.add_argument(
        "--collision-confirm-steps",
        type=int,
        default=2,
        help="Require N consecutive LiDAR collision signals before terminating (reduces false positives).",
    )
    p.add_argument(
        "--collision-min-hits",
        type=int,
        default=2,
        help="Require >=K downsample buckets below threshold to count as a collision signal.",
    )
    p.add_argument("--control-dt", type=float, default=0.05)

    # action limits
    p.add_argument("--vx-min", type=float, default=-0.3)
    p.add_argument("--vx-max", type=float, default=0.6)
    p.add_argument("--vy-max", type=float, default=0.4)
    p.add_argument("--wz-max", type=float, default=1.0)

    # obstacle randomization
    p.add_argument("--min-obstacles", type=int, default=2)
    p.add_argument("--max-obstacles", type=int, default=5)
    p.add_argument("--obstacle-spawn-margin", type=float, default=0.85)
    p.add_argument("--keepout-robot", type=float, default=1.0)
    p.add_argument("--keepout-goal", type=float, default=1.0)
    p.add_argument("--keepout-between", type=float, default=2.0)

    # corridor obstacles (讓多數 episode 需要避障)
    p.add_argument("--corridor-prob", type=float, default=0.60)
    p.add_argument("--corridor-half-width", type=float, default=0.93)
    p.add_argument("--corridor-min-obstacles", type=int, default=1)
    p.add_argument("--corridor-max-obstacles", type=int, default=3)
    p.add_argument("--corridor-t-min", type=float, default=0.30)
    p.add_argument("--corridor-t-max", type=float, default=0.70)
    p.add_argument("--corridor-max-tries", type=int, default=400)

    # clearance reward shaping
    p.add_argument("--near-obstacle-threshold", type=float, default=0.87)
    p.add_argument("--near-obstacle-penalty", type=float, default=0.35)
    p.add_argument("--step-penalty", type=float, default=-0.05)

    # sensor health / debugging
    p.add_argument("--topic-timeout-sec", type=float, default=6.0)
    p.add_argument("--require-scan", type=int, default=1, choices=[0, 1])
    p.add_argument("--scan-max-age-sec", type=float, default=0.75)
    p.add_argument("--terminate-on-lidar-stale", type=int, default=0, choices=[0, 1])

    # optional load params json
    p.add_argument("--sac-params-json", type=str, default="")

    # SAC hyperparams (CLI override)
    p.add_argument("--learning-rate", type=float, default=None)
    p.add_argument("--buffer-size", type=int, default=None)
    p.add_argument("--learning-starts", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--gamma", type=float, default=None)
    p.add_argument("--tau", type=float, default=None)
    p.add_argument("--train-freq", type=int, default=None)
    p.add_argument("--gradient-steps", type=int, default=None)
    p.add_argument("--ent-coef", type=str, default=None)  # "auto" recommended

    # retraining / resume (SAC only)
    p.add_argument("--load-model", type=str, default="")
    p.add_argument("--reset-timesteps", action="store_true")
    p.add_argument(
        "--use-replay-buffer",
        type=int,
        default=0,
        choices=[0, 1],
        help="When resuming from --load-model, also load replay buffer if available.",
    )
    p.add_argument(
        "--replay-buffer-path",
        type=str,
        default="",
        help="Optional explicit replay buffer path (.pkl). If empty, infer from --load-model.",
    )

    # autosave
    p.add_argument("--save-every", type=int, default=20_000)
    p.add_argument("--save-replay-buffer", type=int, default=0, choices=[0, 1])

    # early stop
    p.add_argument("--window-episodes", type=int, default=100)
    p.add_argument("--min-episodes", type=int, default=200)
    p.add_argument("--plateau-window", type=int, default=20)
    p.add_argument("--sr-delta", type=float, default=0.02)
    p.add_argument("--cr-delta", type=float, default=0.02)
    p.add_argument("--ret-delta", type=float, default=0.5)
    p.add_argument("--len-delta", type=float, default=5.0)
    p.add_argument("--plateau-min-success", type=float, default=0.90)
    p.add_argument("--plateau-max-collision", type=float, default=0.05)

    args = p.parse_args()
    return finalize_rect_bounds_args(args, default_x_lim=3.5, default_y_lim=3.5)


def load_json(path: str):
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def make_env_fn(args, seed_offset=0):
    def _thunk():
        return AmrNavStage1Env(
            set_entity_service=args.set_entity_service,
            cmd_vel_topic=args.cmd_vel_topic,
            odom_topic=args.odom_topic,
            scan_topic=args.scan_topic,
            x_min=args.x_min,
            x_max=args.x_max,
            y_min=args.y_min,
            y_max=args.y_max,
            x_lim=args.x_lim,
            y_lim=args.y_lim,
            max_steps=args.max_steps,
            collision_threshold=args.collision_threshold,
            collision_confirm_steps=args.collision_confirm_steps,
            collision_min_hits=args.collision_min_hits,
            control_dt=args.control_dt,
            vx_min=args.vx_min,
            vx_max=args.vx_max,
            vy_max=args.vy_max,
            wz_max=args.wz_max,
            min_obstacles=args.min_obstacles,
            max_obstacles=args.max_obstacles,
            obstacle_spawn_margin=args.obstacle_spawn_margin,
            keepout_robot=args.keepout_robot,
            keepout_goal=args.keepout_goal,
            keepout_between=args.keepout_between,
            corridor_prob=args.corridor_prob,
            corridor_half_width=args.corridor_half_width,
            corridor_min_obstacles=args.corridor_min_obstacles,
            corridor_max_obstacles=args.corridor_max_obstacles,
            corridor_t_min=args.corridor_t_min,
            corridor_t_max=args.corridor_t_max,
            corridor_max_tries=args.corridor_max_tries,
            near_obstacle_threshold=args.near_obstacle_threshold,
            near_obstacle_penalty=args.near_obstacle_penalty,
            step_penalty=args.step_penalty,
            topic_timeout_sec=args.topic_timeout_sec,
            require_scan=bool(args.require_scan),
            scan_max_age_sec=args.scan_max_age_sec,
            terminate_on_lidar_stale=bool(args.terminate_on_lidar_stale),
            seed=args.seed + seed_offset,
        )

    return _thunk


def main():
    args = parse_args()
    os.makedirs(args.logdir, exist_ok=True)
    os.makedirs(args.modeldir, exist_ok=True)

    rclpy.init()

    env = DummyVecEnv([make_env_fn(args, seed_offset=0)])
    env = VecMonitor(env)

    # Quick preflight: show that LiDAR is present and not constant-all-ones fallback
    try:
        obs = env.reset()
        if isinstance(obs, dict) and ("laser" in obs):
            laser = np.array(obs["laser"])[0]
            print(
                "[SENSOR] reset laser stats: "
                f"std={float(np.std(laser)):.4f}, min={float(np.min(laser)):.3f}, max={float(np.max(laser)):.3f}"
            )
    except Exception as e:
        print(f"[WARN] Sensor preflight failed: {e}")

    # SAC defaults (single-env): conservative but stable
    sac_params = dict(
        learning_rate=3e-4,
        buffer_size=300_000,
        learning_starts=3_000,
        batch_size=256,
        gamma=0.99,
        tau=0.005,
        train_freq=1,
        gradient_steps=2,
        ent_coef="auto",
        verbose=1,
        tensorboard_log=args.logdir,
    )

    sac_params.update(load_json(args.sac_params_json))

    # CLI overrides (highest priority)
    if args.learning_rate is not None:
        sac_params["learning_rate"] = float(args.learning_rate)
    if args.buffer_size is not None:
        sac_params["buffer_size"] = int(args.buffer_size)
    if args.learning_starts is not None:
        sac_params["learning_starts"] = int(args.learning_starts)
    if args.batch_size is not None:
        sac_params["batch_size"] = int(args.batch_size)
    if args.gamma is not None:
        sac_params["gamma"] = float(args.gamma)
    if args.tau is not None:
        sac_params["tau"] = float(args.tau)
    if args.train_freq is not None:
        sac_params["train_freq"] = int(args.train_freq)
    if args.gradient_steps is not None:
        sac_params["gradient_steps"] = int(args.gradient_steps)
    if args.ent_coef is not None:
        sac_params["ent_coef"] = str(args.ent_coef)

    run_name = f"sac_stage1_seed{args.seed}_{time.strftime('%Y%m%d_%H%M%S')}"

    ckpt_cb = CheckpointWithReplayBufferCallback(
        save_freq=int(args.save_every),
        save_path=args.modeldir,
        name_prefix="sac_stage1_ckpt",
        save_replay_buffer=bool(args.save_replay_buffer),
        verbose=1,
    )

    interrupted_flag = {"stop": False}

    def _sigint_handler(sig, frame):
        if not interrupted_flag["stop"]:
            interrupted_flag["stop"] = True
            print("\n[WARN] SIGINT received. Will stop training and save model...")
        else:
            raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _sigint_handler)
    sigint_cb = StopTrainingOnSigintCallback(interrupted_flag)

    best_dir = os.path.join(args.modeldir, "best_sac")
    mm_cb = MultiMetricEarlyStopCallback(
        model_save_dir=best_dir,
        save_replay_buffer=bool(args.save_replay_buffer),
        window_episodes=args.window_episodes,
        min_episodes=args.min_episodes,
        plateau_window=args.plateau_window,
        sr_delta=args.sr_delta,
        cr_delta=args.cr_delta,
        ret_delta=args.ret_delta,
        len_delta=args.len_delta,
        plateau_min_success=args.plateau_min_success,
        plateau_max_collision=args.plateau_max_collision,
        max_steps=args.max_steps,
        verbose=1,
    )

    sensor_cb = SensorDebugCallback(window_episodes=args.window_episodes, verbose=0)

    if args.load_model:
        print(f"[INFO] Loading SAC model (resume): {args.load_model}")
        # 注意：SAC 無法直接載入 PPO 的模型檔，請確保 load_model 是 SAC 產生的 zip。
        model = SAC.load(args.load_model, env=env)

        if bool(args.use_replay_buffer):
            rb_path = str(args.replay_buffer_path).strip()
            if not rb_path:
                rb_path = CheckpointWithReplayBufferCallback._rb_path_from_model_path(args.load_model)

            if os.path.exists(rb_path):
                try:
                    if hasattr(model, "load_replay_buffer"):
                        model.load_replay_buffer(rb_path)
                        print(f"[OK] Loaded replay buffer: {rb_path}")
                    else:
                        print("[WARN] Model has no load_replay_buffer(). Did you load a non off-policy model?")
                except Exception as e:
                    print(f"[WARN] Failed loading replay buffer: {e}")
            else:
                print(f"[WARN] Replay buffer file not found, continuing without it: {rb_path}")
    else:
        model = SAC("MultiInputPolicy", env, seed=args.seed, **sac_params)

    try:
        model.learn(
            total_timesteps=int(args.total_steps),
            callback=[ckpt_cb, sigint_cb, mm_cb, sensor_cb],
            tb_log_name=run_name,
            reset_num_timesteps=bool(args.reset_timesteps),
        )
    except KeyboardInterrupt:
        interrupted_flag["stop"] = True
        print("\n[WARN] KeyboardInterrupt. Saving model...")

    out_path = os.path.join(args.modeldir, "sac_stage1.zip")
    if interrupted_flag["stop"]:
        out_path = os.path.join(args.modeldir, "sac_stage1_interrupt.zip")

    model.save(out_path)
    print(f"[OK] Saved model: {out_path}")

    if bool(args.save_replay_buffer):
        rb_path = CheckpointWithReplayBufferCallback._rb_path_from_model_path(out_path)
        try:
            if hasattr(model, "save_replay_buffer"):
                model.save_replay_buffer(rb_path)
                print(f"[OK] Saved replay buffer: {rb_path}")
            else:
                print("[WARN] Model has no save_replay_buffer(). Did you create a non off-policy model?")
        except Exception as e:
            print(f"[WARN] Failed saving replay buffer: {e}")

    env.close()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
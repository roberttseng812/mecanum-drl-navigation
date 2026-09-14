import os
import json
import time
import argparse
import math
import signal
from collections import deque

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
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

from amr_rl.amr_env_stage0 import AmrNavStage0Env


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


class MultiMetricEarlyStopCallback(BaseCallback):
    """
    每個 episode 結束時計算 rolling-window 指標並寫入 TensorBoard：
      - rollout/success_rate
      - rollout/collision_rate
      - rollout/ep_return_mean_window
      - rollout/ep_len_mean_window

    自動停止規則：
      A) 達標停止：success_rate >= target_sr 且 collision_rate <= target_cr 且 len_mean <= ratio*max_steps
      B) 多指標平台期：success_rate/return/len/collision 全都在 plateau_window 內變化很小，且 sr 高、cr 低
    """
    def __init__(
        self,
        model_save_dir: str,
        window_episodes: int = 100,
        min_episodes: int = 200,

        # stop rule A (performance)
        target_success_rate: float = 0.95,
        target_collision_rate: float = 0.02,
        target_ep_len_ratio: float = 0.6,

        # stop rule B (plateau)
        plateau_window: int = 20,
        sr_delta: float = 0.02,
        cr_delta: float = 0.01,
        ret_delta: float = 1.0,
        len_delta: float = 10.0,
        plateau_min_success: float = 0.90,
        plateau_max_collision: float = 0.03,

        max_steps: int = 400,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.model_save_dir = model_save_dir
        os.makedirs(self.model_save_dir, exist_ok=True)

        self.window_episodes = int(window_episodes)
        self.min_episodes = int(min_episodes)

        self.target_success_rate = float(target_success_rate)
        self.target_collision_rate = float(target_collision_rate)
        self.target_ep_len_ratio = float(target_ep_len_ratio)

        self.plateau_window = int(plateau_window)
        self.sr_delta = float(sr_delta)
        self.cr_delta = float(cr_delta)
        self.ret_delta = float(ret_delta)
        self.len_delta = float(len_delta)
        self.plateau_min_success = float(plateau_min_success)
        self.plateau_max_collision = float(plateau_max_collision)

        self.max_steps = int(max_steps)

        # per-episode accumulators (VecEnv=1 assumption)
        self._ep_return = 0.0
        self._ep_len = 0

        # rolling episode windows
        self._success_hist = deque(maxlen=self.window_episodes)
        self._collision_hist = deque(maxlen=self.window_episodes)
        self._return_hist = deque(maxlen=self.window_episodes)
        self._len_hist = deque(maxlen=self.window_episodes)

        # series of window means for plateau detection
        self._sr_series = deque(maxlen=self.plateau_window)
        self._cr_series = deque(maxlen=self.plateau_window)
        self._ret_series = deque(maxlen=self.plateau_window)
        self._len_series = deque(maxlen=self.plateau_window)

        self._episode_count = 0
        self._best_sr = -1.0

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards", None)
        dones = self.locals.get("dones", None)
        infos = self.locals.get("infos", None)

        if rewards is None or dones is None or infos is None:
            return True

        # VecEnv=1
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

        # reset accumulators for next episode
        self._ep_return = 0.0
        self._ep_len = 0

        self.logger.record("rollout/episode_count", self._episode_count)

        if len(self._success_hist) < self.window_episodes:
            return True

        success_rate = sum(self._success_hist) / float(len(self._success_hist))
        collision_rate = sum(self._collision_hist) / float(len(self._collision_hist))
        ret_mean = sum(self._return_hist) / float(len(self._return_hist))
        len_mean = sum(self._len_hist) / float(len(self._len_hist))

        # record to TensorBoard
        self.logger.record("rollout/success_rate", success_rate)
        self.logger.record("rollout/collision_rate", collision_rate)
        self.logger.record("rollout/ep_return_mean_window", ret_mean)
        self.logger.record("rollout/ep_len_mean_window", len_mean)

        # series for plateau detection
        self._sr_series.append(success_rate)
        self._cr_series.append(collision_rate)
        self._ret_series.append(ret_mean)
        self._len_series.append(len_mean)

        # save best by success_rate
        if success_rate > self._best_sr + 1e-6:
            self._best_sr = success_rate
            best_path = os.path.join(self.model_save_dir, "best_success.zip")
            try:
                self.model.save(best_path)
                if self.verbose:
                    print(f"[INFO] New best success_rate={success_rate:.3f} saved: {best_path}")
            except Exception as e:
                print(f"[WARN] Failed saving best model: {e}")

        # --- Stop Rule A: performance target ---
        # if self._episode_count >= self.min_episodes:
        #     if (success_rate >= self.target_success_rate and
        #         collision_rate <= self.target_collision_rate and
        #         len_mean <= self.target_ep_len_ratio * self.max_steps):
        #         print("[OK] Early stop (target reached): "
        #               f"sr={success_rate:.3f}, cr={collision_rate:.3f}, len_mean={len_mean:.1f}")
        #         return False

        # --- Stop Rule B: multi-metric plateau ---
        if self._episode_count >= self.min_episodes and len(self._sr_series) >= self.plateau_window:
            sr_var = max(self._sr_series) - min(self._sr_series)
            cr_var = max(self._cr_series) - min(self._cr_series)
            ret_var = max(self._ret_series) - min(self._ret_series)
            len_var = max(self._len_series) - min(self._len_series)

            sr_now = self._sr_series[-1]
            cr_now = self._cr_series[-1]

            if (sr_now >= self.plateau_min_success and
                cr_now <= self.plateau_max_collision and
                sr_var <= self.sr_delta and
                cr_var <= self.cr_delta and
                ret_var <= self.ret_delta and
                len_var <= self.len_delta):
                print("[OK] Early stop (plateau): "
                      f"sr={sr_now:.3f} (var={sr_var:.3f}), "
                      f"cr={cr_now:.3f} (var={cr_var:.3f}), "
                      f"ret_var={ret_var:.3f}, len_var={len_var:.1f}")
                return False

        return True


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--logdir", type=str, default="runs_stage0")
    p.add_argument("--modeldir", type=str, default="models_stage0")
    p.add_argument("--total-steps", type=int, default=300_000)
    p.add_argument("--seed", type=int, default=0)

    # ROS/Gazebo wiring
    p.add_argument("--set-entity-service", type=str, default="/set_entity_state")
    p.add_argument("--cmd-vel-topic", type=str, default="cmd_vel")
    p.add_argument("--odom-topic", type=str, default="odom")
    p.add_argument("--scan-topic", type=str, default="scan")

    # env params
    add_rect_bounds_args(p, default_x_lim=3.0, default_y_lim=3.0)
    p.add_argument("--max-steps", type=int, default=400)
    p.add_argument("--control-dt", type=float, default=0.05)

    # action limits
    p.add_argument("--vx-min", type=float, default=-0.3)
    p.add_argument("--vx-max", type=float, default=0.6)
    p.add_argument("--vy-max", type=float, default=0.4)
    p.add_argument("--wz-max", type=float, default=1.0)

    # optional load params json (e.g. optuna best params)
    p.add_argument("--ppo-params-json", type=str, default="")

    # retraining / resume
    p.add_argument("--load-model", type=str, default="")
    p.add_argument("--reset-timesteps", action="store_true")

    # autosave
    p.add_argument("--save-every", type=int, default=20_000)

    # multi-metric stop params
    p.add_argument("--window-episodes", type=int, default=100)
    p.add_argument("--min-episodes", type=int, default=200)

    p.add_argument("--target-success-rate", type=float, default=0.95)
    p.add_argument("--target-collision-rate", type=float, default=0.02)
    p.add_argument("--target-ep-len-ratio", type=float, default=0.6)

    p.add_argument("--plateau-window", type=int, default=20)
    p.add_argument("--sr-delta", type=float, default=0.01)
    p.add_argument("--cr-delta", type=float, default=0.01)
    p.add_argument("--ret-delta", type=float, default=0.1)
    p.add_argument("--len-delta", type=float, default=2.0)
    p.add_argument("--plateau-min-success", type=float, default=0.97)
    p.add_argument("--plateau-max-collision", type=float, default=0.03)

    args = p.parse_args()
    return finalize_rect_bounds_args(args, default_x_lim=3.0, default_y_lim=3.0)


def load_json(path: str):
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def make_env_fn(args, seed_offset=0):
    def _thunk():
        return AmrNavStage0Env(
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
            control_dt=args.control_dt,
            vx_min=args.vx_min,
            vx_max=args.vx_max,
            vy_max=args.vy_max,
            wz_max=args.wz_max,
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

    ppo_params = dict(
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        verbose=1,
        tensorboard_log=args.logdir,
    )
    ppo_params.update(load_json(args.ppo_params_json))

    run_name = f"ppo_stage0_seed{args.seed}_{time.strftime('%Y%m%d_%H%M%S')}"

    # checkpoint
    ckpt_cb = CheckpointCallback(
        save_freq=int(args.save_every),
        save_path=args.modeldir,
        name_prefix="ppo_stage0_ckpt",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    # SIGINT stop
    interrupted_flag = {"stop": False}

    def _sigint_handler(sig, frame):
        if not interrupted_flag["stop"]:
            interrupted_flag["stop"] = True
            print("\n[WARN] SIGINT received. Will stop training and save model...")
        else:
            raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _sigint_handler)
    sigint_cb = StopTrainingOnSigintCallback(interrupted_flag)

    # multi-metric stop
    best_dir = os.path.join(args.modeldir, "best")
    mm_cb = MultiMetricEarlyStopCallback(
        model_save_dir=best_dir,
        window_episodes=args.window_episodes,
        min_episodes=args.min_episodes,

        target_success_rate=args.target_success_rate,
        target_collision_rate=args.target_collision_rate,
        target_ep_len_ratio=args.target_ep_len_ratio,

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

    # create/load model
    if args.load_model:
        print(f"[INFO] Loading model: {args.load_model}")
        model = PPO.load(args.load_model, env=env, **ppo_params)
    else:
        model = PPO("MultiInputPolicy", env, seed=args.seed, **ppo_params)

    try:
        model.learn(
            total_timesteps=int(args.total_steps),
            callback=[ckpt_cb, sigint_cb, mm_cb],
            tb_log_name=run_name,
            reset_num_timesteps=bool(args.reset_timesteps),
        )
    except KeyboardInterrupt:
        interrupted_flag["stop"] = True
        print("\n[WARN] KeyboardInterrupt. Saving model...")

    # always save at end
    out_path = os.path.join(args.modeldir, "ppo_stage0.zip")
    if interrupted_flag["stop"]:
        out_path = os.path.join(args.modeldir, "ppo_stage0_interrupt.zip")

    model.save(out_path)
    print(f"[OK] Saved model: {out_path}")

    env.close()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

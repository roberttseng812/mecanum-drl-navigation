import argparse
import json
import os
import signal
import time
from collections import deque

import rclpy
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

try:
    from stable_baselines3.common.type_aliases import TrainFreq, TrainFrequencyUnit
except Exception:  # pragma: no cover
    TrainFreq, TrainFrequencyUnit = None, None

from amr_rl.amr_env_stage3 import AmrNavStage3Env
from amr_rl.custom_extractors import Stage3LaserConvExtractor


def load_json(path: str):
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class StopTrainingOnSigintCallback(BaseCallback):
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
    def __init__(
        self,
        save_freq: int,
        save_path: str,
        name_prefix: str = "sac_stage3_ckpt",
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
        self.model.save(model_path)
        if self.save_replay_buffer:
            rb_path = self._rb_path_from_model_path(model_path)
            try:
                if hasattr(self.model, "save_replay_buffer"):
                    self.model.save_replay_buffer(rb_path)
                else:
                    print("[WARN] Model has no save_replay_buffer(). Did you load a non off-policy model?")
            except Exception as e:
                print(f"[WARN] Failed saving replay buffer: {e}")

    def _on_step(self) -> bool:
        if self.save_freq <= 0:
            return True
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
    def __init__(
        self,
        model_save_dir: str,
        save_replay_buffer: bool = False,
        window_episodes: int = 100,
        min_episodes: int = 200,
        plateau_window: int = 20,
        sr_delta: float = 0.02,
        cr_delta: float = 0.02,
        ret_delta: float = 0.5,
        len_delta: float = 5.0,
        plateau_min_success: float = 0.95,
        plateau_max_collision: float = 0.05,
        max_steps: int = 300,
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


def make_env_fn(args, seed_offset=0):
    def _thunk():
        return AmrNavStage3Env(
            set_entity_service=args.set_entity_service,
            cmd_vel_topic=args.cmd_vel_topic,
            odom_topic=args.odom_topic,
            scan_topic=args.scan_topic,
            x_lim=None,
            y_lim=None,
            x_min=args.x_min,
            x_max=args.x_max,
            y_min=args.y_min,
            y_max=args.y_max,
            robot_spawn_margin=args.robot_spawn_margin,
            goal_spawn_margin=args.goal_spawn_margin,
            spawn_min_clearance=args.spawn_min_clearance,
            max_steps=args.max_steps,
            min_goal_dist=args.min_goal_dist,
            max_goal_dist=args.max_goal_dist,
            goal_threshold=args.goal_threshold,
            yaw_threshold=args.yaw_threshold,
            goal_yaw_sector_prob=args.goal_yaw_sector_prob,
            goal_yaw_sector_half_angle_deg=args.goal_yaw_sector_half_angle_deg,
            collision_threshold=args.collision_threshold,
            collision_confirm_steps=args.collision_confirm_steps,
            collision_min_hits=args.collision_min_hits,
            control_dt=args.control_dt,
            laser_downsample_n=args.laser_downsample_n,
            lidar_history_steps=args.lidar_history_steps,
            lidar_history_stride=args.lidar_history_stride,
            vx_min=args.vx_min,
            vx_max=args.vx_max,
            vy_max=args.vy_max,
            wz_max=args.wz_max,
            near_obstacle_threshold=args.near_obstacle_threshold,
            near_obstacle_penalty=args.near_obstacle_penalty,
            clearance_linear_weight=args.clearance_linear_weight,
            clearance_square_weight=args.clearance_square_weight,
            speed_near_obstacle_penalty=args.speed_near_obstacle_penalty,
            step_penalty=args.step_penalty,
            sensor_freshness_sec=args.sensor_freshness_sec,
            sensor_fail_penalty=args.sensor_fail_penalty,
            timeout_penalty=args.timeout_penalty,
            action_smooth_penalty=args.action_smooth_penalty,
            idle_penalty=args.idle_penalty,
            idle_speed_thresh=args.idle_speed_thresh,
            topic_timeout_sec=args.topic_timeout_sec,
            dynamic_route_prob=args.dynamic_route_prob,
            dynamic_route_mode=args.dynamic_route_mode,
            dynamic_headon_prob=args.dynamic_headon_prob,
            dynamic_samedir_prob=args.dynamic_samedir_prob,
            dynamic_route_t_margin=args.dynamic_route_t_margin,
            dynamic_route_trim_m=args.dynamic_route_trim_m,
            dynamic_corridor_half_width=args.dynamic_corridor_half_width,
            dynamic_route_jitter=args.dynamic_route_jitter,
            dynamic_route_yaw_noise_deg=args.dynamic_route_yaw_noise_deg,
            seed=args.seed + seed_offset,
            oob_mode=args.oob_mode,
            oob_radius=args.oob_radius,
        )
    return _thunk



def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--logdir", type=str, default="runs_stage3_1")
    p.add_argument("--modeldir", type=str, default="models_stage3_1")
    p.add_argument("--total-steps", type=int, default=300_000)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--set-entity-service", type=str, default="/set_entity_state")
    p.add_argument("--cmd-vel-topic", type=str, default="cmd_vel")
    p.add_argument("--odom-topic", type=str, default="odom")
    p.add_argument("--scan-topic", type=str, default="scan")

    p.add_argument("--x-min", type=float, default=-4.3)
    p.add_argument("--x-max", type=float, default=3.0)
    p.add_argument("--y-min", type=float, default=-3.2)
    p.add_argument("--y-max", type=float, default=2.0)
    p.add_argument("--robot-spawn-margin", type=float, default=0.85)
    p.add_argument("--goal-spawn-margin", type=float, default=0.85)
    p.add_argument("--oob-mode", type=str, default="both", choices=["world_box", "start_circle", "both"])
    p.add_argument("--oob-radius", type=float, default=7.0)
    p.add_argument("--max-steps", type=int, default=700)
    p.add_argument("--min-goal-dist", type=float, default=3.0)
    p.add_argument("--max-goal-dist", type=float, default=6.0)
    p.add_argument("--goal-threshold", type=float, default=0.30)
    p.add_argument("--yaw-threshold", type=float, default=0.20)
    p.add_argument("--goal-yaw-sector-prob", type=float, default=0.65)
    p.add_argument("--goal-yaw-sector-half-angle-deg", type=float, default=50.0)
    p.add_argument("--collision-threshold", type=float, default=0.57)
    p.add_argument("--collision-confirm-steps", type=int, default=2)
    p.add_argument("--collision-min-hits", type=int, default=2)
    p.add_argument("--control-dt", type=float, default=0.05)
    p.add_argument("--laser-downsample-n", type=int, default=240)
    p.add_argument("--lidar-history-steps", type=int, default=4)
    p.add_argument("--lidar-history-stride", type=int, default=3)
    p.add_argument("--vx-min", type=float, default=-0.3)
    p.add_argument("--vx-max", type=float, default=0.47)
    p.add_argument("--vy-max", type=float, default=0.35)
    p.add_argument("--wz-max", type=float, default=0.8)
    p.add_argument("--near-obstacle-threshold", type=float, default=1.4)
    p.add_argument("--near-obstacle-penalty", type=float, default=0.40)
    p.add_argument("--clearance-linear-weight", type=float, default=0.60)
    p.add_argument("--clearance-square-weight", type=float, default=0.40)
    p.add_argument("--speed-near-obstacle-penalty", type=float, default=0.15)
    p.add_argument("--timeout-penalty", type=float, default=6.5)
    p.add_argument("--action-smooth-penalty", type=float, default=0.01)
    p.add_argument("--idle-penalty", type=float, default=0.01)
    p.add_argument("--idle-speed-thresh", type=float, default=0.03)
    p.add_argument("--step-penalty", type=float, default=-0.02)
    p.add_argument("--sensor-freshness-sec", type=float, default=0.20)
    p.add_argument("--sensor-fail-penalty", type=float, default=2.0)
    p.add_argument("--topic-timeout-sec", type=float, default=6.0)
    p.add_argument("--spawn-min-clearance", type=float, default=1.3)
    p.add_argument("--dynamic-route-prob", type=float, default=0.90)
    p.add_argument(
        "--dynamic-route-mode",
        type=str,
        default="mixed",
        choices=["crossing", "head_on", "same_direction", "mixed"],
    )
    p.add_argument("--dynamic-headon-prob", type=float, default=0.45)
    p.add_argument("--dynamic-samedir-prob", type=float, default=0.20)
    p.add_argument("--dynamic-route-t-margin", type=float, default=0.0,
                   help="Normalized in-corridor longitudinal margin used when sampling along a corridor.")
    p.add_argument("--dynamic-route-trim-m", type=float, default=0.70,
                   help="Trim each actor straight segment by this many meters to avoid spawning near turns.")
    p.add_argument("--dynamic-corridor-half-width", type=float, default=0.25,
                   help="Fallback corridor half-width when a custom route segment has no width metadata.")
    p.add_argument("--dynamic-route-jitter", type=float, default=0.35)
    p.add_argument("--dynamic-route-yaw-noise-deg", type=float, default=30.0)

    p.add_argument("--sac-params-json", type=str, default="")
    p.add_argument("--load-model", type=str, default="")
    p.add_argument("--reset-timesteps", action="store_true")
    p.add_argument(
        "--apply-sac-params-on-load",
        type=int,
        default=1,
        choices=[0, 1],
        help="When resuming with --load-model, override selected SAC hyperparameters using sac_params/sac-params-json.",
    )
    p.add_argument("--use-replay-buffer", type=int, default=0, choices=[0, 1])
    p.add_argument("--replay-buffer-path", type=str, default="")
    p.add_argument("--save-every", type=int, default=100_000)
    p.add_argument("--save-replay-buffer", type=int, default=1, choices=[0, 1])

    p.add_argument("--window-episodes", type=int, default=100)
    p.add_argument("--min-episodes", type=int, default=200)
    p.add_argument("--plateau-window", type=int, default=20)
    p.add_argument("--sr-delta", type=float, default=0.02)
    p.add_argument("--cr-delta", type=float, default=0.02)
    p.add_argument("--ret-delta", type=float, default=0.5)
    p.add_argument("--len-delta", type=float, default=5.0)
    p.add_argument("--plateau-min-success", type=float, default=0.95)
    p.add_argument("--plateau-max-collision", type=float, default=0.05)

    return p.parse_args()



def apply_sac_overrides(model: SAC, sac_params: dict, args) -> None:
    applied = {}

    if hasattr(model, "tensorboard_log"):
        model.tensorboard_log = args.logdir

    if "learning_rate" in sac_params and sac_params["learning_rate"] is not None:
        lr = sac_params["learning_rate"]
        if callable(lr):
            model.lr_schedule = lr
            applied["learning_rate(schedule)"] = "callable"
        else:
            lr = float(lr)
            model.learning_rate = lr
            model.lr_schedule = (lambda _progress_remaining, _lr=lr: _lr)
            for opt in [getattr(model.actor, "optimizer", None), getattr(model.critic, "optimizer", None)]:
                if opt is None:
                    continue
                for pg in opt.param_groups:
                    pg["lr"] = lr
            ent_opt = getattr(model, "ent_coef_optimizer", None)
            if ent_opt is not None:
                for pg in ent_opt.param_groups:
                    pg["lr"] = lr
            applied["learning_rate"] = lr

    for k in ("gamma", "tau", "batch_size", "gradient_steps", "learning_starts"):
        if k in sac_params and sac_params[k] is not None and hasattr(model, k):
            v = sac_params[k]
            if k in ("batch_size", "gradient_steps", "learning_starts"):
                v = int(v)
            else:
                v = float(v)
            setattr(model, k, v)
            applied[k] = v

    if "train_freq" in sac_params and sac_params["train_freq"] is not None:
        tf = sac_params["train_freq"]
        try:
            if isinstance(tf, (tuple, list)) and len(tf) == 2:
                freq = int(tf[0])
                unit_raw = tf[1]
                if TrainFrequencyUnit is not None:
                    unit = TrainFrequencyUnit.STEP if str(unit_raw).lower().startswith("step") else TrainFrequencyUnit.EPISODE
                    model.train_freq = TrainFreq(freq, unit)
                    applied["train_freq"] = f"{freq} {unit.name.lower()}"
            else:
                freq = int(tf)
                if TrainFrequencyUnit is not None and TrainFreq is not None:
                    model.train_freq = TrainFreq(freq, TrainFrequencyUnit.STEP)
                    applied["train_freq"] = f"{freq} step"
        except Exception:
            pass

    if "buffer_size" in sac_params and sac_params["buffer_size"] is not None:
        new_bs = int(sac_params["buffer_size"])
        cur_rb = getattr(model, "replay_buffer", None)
        cur_bs = getattr(cur_rb, "buffer_size", None) if cur_rb is not None else None
        if cur_bs is not None and int(cur_bs) != new_bs:
            print(
                f"[WARN] buffer_size in sac_params={new_bs}, but loaded replay_buffer.buffer_size={cur_bs}. "
                f"SB3 does not support resizing the replay buffer safely when resuming; keeping {cur_bs}."
            )

    if "ent_coef" in sac_params and sac_params["ent_coef"] is not None:
        desired = sac_params["ent_coef"]
        if isinstance(desired, (int, float)):
            print(
                "[WARN] ent_coef was set to a fixed value in sac_params, but changing auto<->fixed after loading "
                "is not supported in this script. Keeping loaded ent_coef configuration."
            )
        elif isinstance(desired, str) and desired != "auto":
            print("[WARN] ent_coef override requested, but only 'auto' is supported safely here. Keeping loaded ent_coef.")

    if applied:
        print("[OK] Applied SAC overrides after loading model:")
        for k, v in applied.items():
            print(f"  - {k}: {v}")
    else:
        print("[INFO] No SAC overrides were applied after loading (either disabled or nothing matched).")



def resolve_replay_buffer_path(load_model_path: str, explicit_rb_path: str = "") -> str:
    explicit_rb_path = (explicit_rb_path or "").strip()
    if explicit_rb_path:
        return explicit_rb_path
    return CheckpointWithReplayBufferCallback._rb_path_from_model_path(load_model_path)



def main():
    args = parse_args()
    os.makedirs(args.logdir, exist_ok=True)
    os.makedirs(args.modeldir, exist_ok=True)

    rclpy.init()

    env = DummyVecEnv([make_env_fn(args, seed_offset=0)])
    env = VecMonitor(env)

    policy_kwargs = dict(
        features_extractor_class=Stage3LaserConvExtractor,
        features_extractor_kwargs=dict(
            laser_num_beams=args.laser_downsample_n,
            lidar_history_steps=args.lidar_history_steps,
            conv_channels=32,
            lidar_latent_dim=256,
            fused_hidden_dim=128,
        ),
        net_arch=dict(pi=[256, 128], qf=[256, 128]),
    )

    sac_params = dict(
        learning_rate=2e-4,
        buffer_size=400000,
        learning_starts=5000,
        batch_size=256,
        gamma=0.99,
        tau=0.005,
        train_freq=2,
        gradient_steps=1,
        ent_coef="auto",
        verbose=1,
        tensorboard_log=args.logdir,
        policy_kwargs=policy_kwargs,
    )
    sac_params.update(load_json(args.sac_params_json))

    print(
        f"[INFO] Stage3 bounds: x=[{args.x_min}, {args.x_max}], y=[{args.y_min}, {args.y_max}], "
        f"oob_mode={args.oob_mode}, oob_radius={args.oob_radius}"
    )
    print(
        f"[INFO] Stage3 goal settings: min_goal_dist={args.min_goal_dist}, max_goal_dist={args.max_goal_dist}, "
        f"goal_threshold={args.goal_threshold}, yaw_threshold={args.yaw_threshold}, "
        f"goal_yaw_sector_prob={args.goal_yaw_sector_prob}, "
        f"goal_yaw_sector_half_angle_deg={args.goal_yaw_sector_half_angle_deg}"
    )
    print(
        f"[INFO] Stage3 dynamic route settings: prob={args.dynamic_route_prob}, mode={args.dynamic_route_mode}, "
        f"head_on={args.dynamic_headon_prob}, same_dir={args.dynamic_samedir_prob}, "
        f"trim_m={args.dynamic_route_trim_m}, t_margin={args.dynamic_route_t_margin}, "
        f"corridor_half_width={args.dynamic_corridor_half_width}"
    )
    print(
        f"[INFO] Stage3 reward settings: near_thr={args.near_obstacle_threshold}, near_pen={args.near_obstacle_penalty}, "
        f"clearance_mix=({args.clearance_linear_weight:.2f},{args.clearance_square_weight:.2f}), "
        f"speed_near_pen={args.speed_near_obstacle_penalty}, timeout_pen={args.timeout_penalty}, "
        f"smooth_pen={args.action_smooth_penalty}, idle_pen={args.idle_penalty}"
    )
    print(
        f"[INFO] Stage3 lidar settings: laser_downsample_n={args.laser_downsample_n}, "
        f"lidar_history_steps={args.lidar_history_steps}, "
        f"lidar_history_stride={args.lidar_history_stride}, "
        f"effective_lidar_span_sec={(args.lidar_history_steps - 1) * args.lidar_history_stride * args.control_dt:.3f}"
    )

    run_name = f"sac_stage3_seed{args.seed}_{time.strftime('%Y%m%d_%H%M%S')}"

    ckpt_cb = CheckpointWithReplayBufferCallback(
        save_freq=int(args.save_every),
        save_path=args.modeldir,
        name_prefix="sac_stage3_ckpt",
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
        save_replay_buffer=False,
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

    if args.load_model:
        model = SAC.load(args.load_model, env=env)
        if bool(args.apply_sac_params_on_load):
            apply_sac_overrides(model, sac_params, args)

        print("[WARN] --load-model is set. Policy architecture changes do not apply to an already-saved model; ")
        print("       start a fresh run if you want to use the new conv-based LiDAR extractor.")

        if bool(args.use_replay_buffer):
            rb_path = resolve_replay_buffer_path(args.load_model, args.replay_buffer_path)
            if os.path.isfile(rb_path):
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
        if bool(args.use_replay_buffer):
            print("[WARN] --use-replay-buffer=1 was set, but --load-model is empty. Ignoring replay buffer load.")

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

    out_path = os.path.join(args.modeldir, "sac_stage3.zip")
    if interrupted_flag["stop"]:
        out_path = os.path.join(args.modeldir, "sac_stage3_interrupt.zip")

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
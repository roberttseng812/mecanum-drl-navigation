import os
import json
import time
import argparse
import signal

import numpy as np
import rclpy
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
try:
    from stable_baselines3.common.type_aliases import TrainFreq, TrainFrequencyUnit
except Exception:  # pragma: no cover
    TrainFreq, TrainFrequencyUnit = None, None

from amr_rl.amr_env_stage2 import AmrNavStage2Env
from amr_rl.train_stage1 import StopTrainingOnSigintCallback, CheckpointWithReplayBufferCallback, MultiMetricEarlyStopCallback, SensorDebugCallback


def load_json(path: str):
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def make_env_fn(args, seed_offset=0):
    def _thunk():
        return AmrNavStage2Env(
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
            max_steps=args.max_steps,
            goal_threshold=args.goal_threshold,
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
            yaw_threshold=args.yaw_threshold,
            goal_yaw_sector_prob=args.goal_yaw_sector_prob,
            goal_yaw_sector_half_angle_deg=args.goal_yaw_sector_half_angle_deg,
            oob_mode=args.oob_mode,
            oob_radius=args.oob_radius,
        )
    return _thunk


def parse_args():
    p = argparse.ArgumentParser()
    # 將所有儲存路徑改為 stage2
    p.add_argument("--logdir", type=str, default="runs_stage2")
    p.add_argument("--modeldir", type=str, default="models_stage2")
    p.add_argument("--total-steps", type=int, default=400_000)
    p.add_argument("--seed", type=int, default=0)

    # ROS/Gazebo wiring
    p.add_argument("--set-entity-service", type=str, default="/set_entity_state")
    p.add_argument("--cmd-vel-topic", type=str, default="cmd_vel")
    p.add_argument("--odom-topic", type=str, default="odom")
    p.add_argument("--scan-topic", type=str, default="scan")

    # Env Params
    # Rectangular world bounds. Edit these four defaults directly when your map changes.
    p.add_argument("--x-min", type=float, default=-3.5)
    p.add_argument("--x-max", type=float, default=3.5)
    p.add_argument("--y-min", type=float, default=-2.5)
    p.add_argument("--y-max", type=float, default=2.5)
    # reset safety: keep spawn points away from walls
    p.add_argument("--robot-spawn-margin", type=float, default=0.85)
    p.add_argument("--goal-spawn-margin", type=float, default=0.85)
    # 出界判定：Stage2 預設用「以起點為圓心」的半徑限制
    p.add_argument("--oob-mode", type=str, default="start_circle", choices=["world_box", "start_circle", "both"])
    p.add_argument("--oob-radius", type=float, default=2.5)  # 3.0
    p.add_argument("--max-steps", type=int, default=300)  # 300
    p.add_argument("--goal-threshold", type=float, default=0.30)
    p.add_argument("--yaw-threshold", type=float, default=0.20)
    p.add_argument("--goal-yaw-sector-prob", type=float, default=0.60)
    p.add_argument("--goal-yaw-sector-half-angle-deg", type=float, default=60.0)
    p.add_argument("--collision-threshold", type=float, default=0.57)
    p.add_argument("--collision-confirm-steps", type=int, default=2)
    p.add_argument("--collision-min-hits", type=int, default=2)
    p.add_argument("--control-dt", type=float, default=0.05)
    p.add_argument("--vx-min", type=float, default=-0.3)
    p.add_argument("--vx-max", type=float, default=0.5)
    p.add_argument("--vy-max", type=float, default=0.4)
    p.add_argument("--wz-max", type=float, default=0.8)
    p.add_argument("--min-obstacles", type=int, default=1)
    p.add_argument("--max-obstacles", type=int, default=4)
    p.add_argument("--obstacle-spawn-margin", type=float, default=0.85)
    p.add_argument("--keepout-robot", type=float, default=1.0)
    p.add_argument("--keepout-goal", type=float, default=1.0)
    p.add_argument("--keepout-between", type=float, default=2.0)
    p.add_argument("--corridor-prob", type=float, default=0.75)
    p.add_argument("--corridor-half-width", type=float, default=0.90)
    p.add_argument("--corridor-min-obstacles", type=int, default=1)
    p.add_argument("--corridor-max-obstacles", type=int, default=3)
    p.add_argument("--corridor-t-min", type=float, default=0.35)
    p.add_argument("--corridor-t-max", type=float, default=0.65)
    p.add_argument("--corridor-max-tries", type=int, default=400)
    p.add_argument("--near-obstacle-threshold", type=float, default=0.80)
    p.add_argument("--near-obstacle-penalty", type=float, default=0.30)
    p.add_argument("--step-penalty", type=float, default=-0.02)
    p.add_argument("--topic-timeout-sec", type=float, default=6.0)
    p.add_argument("--require-scan", type=int, default=1, choices=[0, 1])
    p.add_argument("--scan-max-age-sec", type=float, default=0.75)
    p.add_argument("--terminate-on-lidar-stale", type=int, default=0, choices=[0, 1])

    # SAC Params
    p.add_argument("--sac-params-json", type=str, default="")
    p.add_argument("--load-model", type=str, default="")
    p.add_argument("--reset-timesteps", action="store_true")
    p.add_argument(
        "--apply-sac-params-on-load",
        type=int,
        default=1,
        choices=[0, 1],
        help="When resuming with --load-model, override selected SAC hyperparameters (e.g., learning_rate, gamma, tau, batch_size, train_freq, gradient_steps) using sac_params/sac-params-json.",
    )
    p.add_argument("--use-replay-buffer", type=int, default=0, choices=[0, 1])
    p.add_argument("--replay-buffer-path", type=str, default="")
    p.add_argument("--save-every", type=int, default=20_000)
    p.add_argument("--save-replay-buffer", type=int, default=1, choices=[0, 1])

    # Early stop
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
    """Override selected SAC hyperparameters after SAC.load().

    Notes:
    - learning_rate: applied to lr_schedule + actor/critic/ent optimizers.
    - train_freq: kept as TrainFreq if available.
    - buffer_size: cannot be resized safely without rebuilding the replay buffer; we only warn.
    - ent_coef: changing between auto/fixed after loading is non-trivial; we warn and skip.
    """
    applied = {}

    # Make sure tensorboard logdir is current
    if hasattr(model, "tensorboard_log"):
        model.tensorboard_log = args.logdir

    # learning rate
    if "learning_rate" in sac_params and sac_params["learning_rate"] is not None:
        lr = sac_params["learning_rate"]
        if callable(lr):
            # keep schedule as-is (rare in this codebase)
            model.lr_schedule = lr
            applied["learning_rate(schedule)"] = "callable"
        else:
            lr = float(lr)
            model.learning_rate = lr
            model.lr_schedule = (lambda _progress_remaining, _lr=lr: _lr)
            # Update optimizers' param groups
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

    # core scalars
    for k in ("gamma", "tau", "batch_size", "gradient_steps", "learning_starts"):
        if k in sac_params and sac_params[k] is not None and hasattr(model, k):
            v = sac_params[k]
            if k in ("batch_size", "gradient_steps", "learning_starts"):
                v = int(v)
            else:
                v = float(v)
            setattr(model, k, v)
            applied[k] = v

    # train_freq (SB3 expects TrainFreq(frequency, unit))
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
            # keep existing train_freq
            pass

    # buffer_size (cannot safely resize)
    if "buffer_size" in sac_params and sac_params["buffer_size"] is not None:
        new_bs = int(sac_params["buffer_size"])
        cur_rb = getattr(model, "replay_buffer", None)
        cur_bs = getattr(cur_rb, "buffer_size", None) if cur_rb is not None else None
        if cur_bs is not None and int(cur_bs) != new_bs:
            print(
                f"[WARN] buffer_size in sac_params={new_bs}, but loaded replay_buffer.buffer_size={cur_bs}. "
                f"SB3 does not support resizing the replay buffer safely when resuming; keeping {cur_bs}."
            )

    # ent_coef changes are tricky post-load
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
    """Return replay buffer path for resume.

    Priority:
    1) --replay-buffer-path if user provided it
    2) infer from --load-model using the same naming rule as checkpoints/final saves
    """
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

    sac_params = dict(
        learning_rate=3e-4,
        buffer_size=400_000,
        learning_starts=3_000,
        batch_size=256,
        gamma=0.99,
        tau=0.005,
        train_freq=2,
        gradient_steps=1,
        ent_coef="auto",
        verbose=1,
        tensorboard_log=args.logdir,
    )
    sac_params.update(load_json(args.sac_params_json))

    print(
        f"[INFO] Stage2 bounds: x=[{args.x_min}, {args.x_max}], y=[{args.y_min}, {args.y_max}], "
        f"oob_mode={args.oob_mode}, oob_radius={args.oob_radius}"
    )
    print(
        f"[INFO] Stage2 goal settings: goal_threshold={args.goal_threshold}, "
        f"yaw_threshold={args.yaw_threshold}, "
        f"goal_yaw_sector_prob={args.goal_yaw_sector_prob}, "
        f"goal_yaw_sector_half_angle_deg={args.goal_yaw_sector_half_angle_deg}"
    )

    run_name = f"sac_stage2_seed{args.seed}_{time.strftime('%Y%m%d_%H%M%S')}"

    ckpt_cb = CheckpointWithReplayBufferCallback(
        save_freq=int(args.save_every),
        save_path=args.modeldir,
        name_prefix="sac_stage2_ckpt",
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
        model = SAC.load(args.load_model, env=env)
        # Optionally override selected SAC hyperparameters when resuming
        if bool(args.apply_sac_params_on_load):
            apply_sac_overrides(model, sac_params, args)

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
            callback=[ckpt_cb, sigint_cb, mm_cb, sensor_cb],
            tb_log_name=run_name,
            reset_num_timesteps=bool(args.reset_timesteps),
        )
    except KeyboardInterrupt:
        interrupted_flag["stop"] = True
        print("\n[WARN] KeyboardInterrupt. Saving model...")

    out_path = os.path.join(args.modeldir, "sac_stage2.zip")
    if interrupted_flag["stop"]:
        out_path = os.path.join(args.modeldir, "sac_stage2_interrupt.zip")

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
#!/usr/bin/env python3
import os
import json
import argparse
import math
from dataclasses import dataclass

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
from stable_baselines3 import SAC, PPO

from amr_rl.amr_env_stage1 import AmrNavStage1Env


@dataclass
class EpResult:
    episode: int
    ep_return: float
    ep_len: int
    termination_reason: str
    is_success: int
    is_collision: int
    dist_to_goal: float
    n_obstacles: int
    corridor_used: int
    corridor_n: int
    reset_attempts: int
    reset_reason: str


def parse_args():
    p = argparse.ArgumentParser()

    def add_bool_flag(flag: str, default: bool, help_text: str):
        """Python3.8+ compatible boolean flag: --flag / --no-flag."""
        dest = flag.replace('-', '_')
        g = p.add_mutually_exclusive_group(required=False)
        g.add_argument(f"--{flag}", dest=dest, action="store_true", help=help_text)
        g.add_argument(f"--no-{flag}", dest=dest, action="store_false", help=f"Disable: {help_text}")
        p.set_defaults(**{dest: bool(default)})

    # basic
    p.add_argument("--algo", type=str, default="sac", choices=["sac", "ppo"], help="Algorithm used to train the model")
    p.add_argument("--model", type=str, default="models_stage1/sac_stage1.zip")
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--seed", type=int, default=None, help="If set, reset() will use seed+episode")

    # ROS/Gazebo wiring
    p.add_argument("--set-entity-service", type=str, default="/set_entity_state")
    p.add_argument("--cmd-vel-topic", type=str, default="cmd_vel")
    p.add_argument("--odom-topic", type=str, default="odom")
    p.add_argument("--scan-topic", type=str, default="scan")
    # Stage1 幾乎都在 Gazebo /clock 下跑，預設開啟。
    add_bool_flag("use-sim-time", default=True, help_text="Use /clock (sim time) for this node")

    # entity names
    p.add_argument("--robot-name", type=str, default="amr")
    p.add_argument("--target-name", type=str, default="Target")
    p.add_argument(
        "--obstacle-names",
        type=str,
        default="",
        help="Comma-separated gazebo model names (e.g. 'Obstacle_0,Obstacle_1,...'). Leave empty to use default Obstacle_0..Obstacle_7.",
    )

    # env params (match Stage1 defaults)
    add_rect_bounds_args(p, default_x_lim=3.5, default_y_lim=3.5)
    # Default values aligned to train_stage1.py defaults (so eval distribution matches training unless you override).
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--control-dt", type=float, default=0.05)
    p.add_argument("--min-goal-dist", type=float, default=1.5)
    p.add_argument("--max-goal-dist", type=float, default=3.5)
    p.add_argument("--goal-threshold", type=float, default=0.35)
    p.add_argument("--collision-threshold", type=float, default=0.57)
    p.add_argument("--collision-confirm-steps", type=int, default=2)
    p.add_argument("--collision-min-hits", type=int, default=2)
    p.add_argument("--laser-downsample-n", type=int, default=120)

    # action limits
    p.add_argument("--vx-min", type=float, default=-0.3)
    p.add_argument("--vx-max", type=float, default=0.6)
    p.add_argument("--vy-max", type=float, default=0.4)
    p.add_argument("--wz-max", type=float, default=1.0)

    # obstacle randomization knobs
    p.add_argument("--min-obstacles", type=int, default=0)
    p.add_argument("--max-obstacles", type=int, default=0)
    p.add_argument("--obstacle-spawn-margin", type=float, default=0.85)
    p.add_argument("--keepout-robot", type=float, default=1.0)
    p.add_argument("--keepout-goal", type=float, default=1.0)
    p.add_argument("--keepout-between", type=float, default=2.0)

    p.add_argument("--corridor-prob", type=float, default=0.80)
    p.add_argument("--corridor-half-width", type=float, default=0.93)
    p.add_argument("--corridor-min-obstacles", type=int, default=1)
    p.add_argument("--corridor-max-obstacles", type=int, default=3)
    p.add_argument("--corridor-t-min", type=float, default=0.20)
    p.add_argument("--corridor-t-max", type=float, default=0.80)
    p.add_argument("--corridor-max-tries", type=int, default=400)

    # reward shaping (only affects returned episode reward; does not change observation)
    p.add_argument("--near-obstacle-threshold", type=float, default=0.87)
    p.add_argument("--near-obstacle-penalty", type=float, default=0.35)
    p.add_argument("--step-penalty", type=float, default=-0.05)

    # reset safety / sensor health
    p.add_argument("--topic-timeout-sec", type=float, default=6.0)
    # Stage1 觀測高度依賴 LiDAR，預設要求 scan 一定要存在。
    # 若你只想 best-effort 跑起來，可用：--no-require-scan
    add_bool_flag(
        "require-scan",
        default=True,
        help_text="Require LiDAR scan during reset/step",
    )
    p.add_argument("--scan-max-age-sec", type=float, default=0.75)
    add_bool_flag(
        "terminate-on-lidar-stale",
        default=False,
        help_text="Truncate episode if LiDAR becomes stale",
    )
    p.add_argument("--max-reset-tries", type=int, default=25)

    # output
    p.add_argument("--out-json", type=str, default="")

    args = p.parse_args()
    return finalize_rect_bounds_args(args, default_x_lim=3.5, default_y_lim=3.5)


def _parse_obstacle_names(s: str) -> list[str] | None:
    s = (s or "").strip()
    if not s:
        return None
    parts = [p.strip() for p in s.split(",")]
    parts = [p for p in parts if p]
    return parts or None


def main():
    args = parse_args()

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model not found: {args.model}")

    obstacle_names = _parse_obstacle_names(args.obstacle_names)

    rclpy.init()

    env = None
    try:
        env = AmrNavStage1Env(
            robot_name=args.robot_name,
            target_name=args.target_name,
            obstacle_names=obstacle_names,
            cmd_vel_topic=args.cmd_vel_topic,
            odom_topic=args.odom_topic,
            scan_topic=args.scan_topic,
            set_entity_service=args.set_entity_service,
            use_sim_time=bool(args.use_sim_time),
            control_dt=args.control_dt,
            max_steps=args.max_steps,
            x_min=args.x_min,
            x_max=args.x_max,
            y_min=args.y_min,
            y_max=args.y_max,
            x_lim=args.x_lim,
            y_lim=args.y_lim,
            min_goal_dist=args.min_goal_dist,
            max_goal_dist=args.max_goal_dist,
            goal_threshold=args.goal_threshold,
            collision_threshold=args.collision_threshold,
            collision_confirm_steps=args.collision_confirm_steps,
            collision_min_hits=args.collision_min_hits,
            laser_downsample_n=args.laser_downsample_n,
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
            max_reset_tries=args.max_reset_tries,
            seed=args.seed,
        )

        algo = str(args.algo).lower()
        if algo == "ppo":
            model = PPO.load(args.model)
        elif algo == "sac":
            model = SAC.load(args.model)
        else:
            raise ValueError(f"Unknown --algo: {args.algo}")

        results: list[EpResult] = []
        success = 0
        collision = 0
        sum_ret = 0.0
        sum_len = 0

        for ep in range(int(args.episodes)):
            if args.seed is None:
                obs, info0 = env.reset()
            else:
                obs, info0 = env.reset(seed=int(args.seed) + int(ep))

            ep_ret = 0.0
            ep_len = 0

            while True:
                action, _ = model.predict(obs, deterministic=bool(args.deterministic))
                obs, reward, terminated, truncated, info = env.step(action)
                ep_ret += float(reward)
                ep_len += 1

                if terminated or truncated:
                    res = EpResult(
                        episode=int(ep),
                        ep_return=float(ep_ret),
                        ep_len=int(ep_len),
                        termination_reason=str(info.get("termination_reason", "unknown")),
                        is_success=int(info.get("is_success", 0)),
                        is_collision=int(info.get("is_collision", 0)),
                        dist_to_goal=float(info.get("dist_to_goal", float("nan"))),
                        n_obstacles=int(info0.get("n_obstacles", 0)),
                        corridor_used=int(info0.get("corridor_used", 0)),
                        corridor_n=int(info0.get("corridor_n", 0)),
                        reset_attempts=int(info0.get("reset_attempts", 0)),
                        reset_reason=str(info0.get("reset_reason", "")),
                    )
                    results.append(res)

                    success += res.is_success
                    collision += res.is_collision
                    sum_ret += res.ep_return
                    sum_len += res.ep_len

                    print(
                        f"[EP {ep:03d}] "
                        f"ret={res.ep_return:8.2f} len={res.ep_len:4d} "
                        f"reason={res.termination_reason:>13s} "
                        f"dist={res.dist_to_goal:6.2f} "
                        f"obs={res.n_obstacles} corridor={res.corridor_used}:{res.corridor_n} "
                        f"reset={res.reset_attempts}({res.reset_reason})"
                    )
                    break

        n = max(len(results), 1)
        print("\n=== Summary ===")
        print(f"model: {args.model}")
        print(f"episodes: {n}")
        print(f"success_rate: {success/n:.3f} ({success}/{n})")
        print(f"collision_rate: {collision/n:.3f} ({collision}/{n})")
        print(f"avg_return: {sum_ret/n:.2f}")
        print(f"avg_len: {sum_len/n:.1f}")

        if args.out_json:
            payload = {
                "model": args.model,
                "episodes": n,
                "success": success,
                "collision": collision,
                "success_rate": success / n,
                "collision_rate": collision / n,
                "avg_return": sum_ret / n,
                "avg_len": sum_len / n,
                "params": {
                    "min_obstacles": int(args.min_obstacles),
                    "max_obstacles": int(args.max_obstacles),
                    "corridor_prob": float(args.corridor_prob),
                    "corridor_half_width": float(args.corridor_half_width),
                },
                "per_episode": [r.__dict__ for r in results],
            }
            with open(args.out_json, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            print(f"saved: {args.out_json}")

    finally:
        if env is not None:
            env.close()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

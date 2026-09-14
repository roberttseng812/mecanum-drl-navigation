#!/usr/bin/env python3
import os
import json
import math
import argparse
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
from stable_baselines3 import PPO

from amr_rl.amr_env_stage0 import AmrNavStage0Env


def wrap_to_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def pose_compose(p1, p2):
    """
    2D pose compose: p = p1 ⊕ p2
    p = (x, y, yaw)
    """
    x1, y1, th1 = p1
    x2, y2, th2 = p2
    c = math.cos(th1)
    s = math.sin(th1)
    x = x1 + c * x2 - s * y2
    y = y1 + s * x2 + c * y2
    th = wrap_to_pi(th1 + th2)
    return (x, y, th)


def pose_inverse(p):
    """
    2D pose inverse
    """
    x, y, th = p
    c = math.cos(th)
    s = math.sin(th)
    # inverse of rotation + translation
    xi = -c * x - s * y
    yi =  s * x - c * y
    thi = wrap_to_pi(-th)
    return (xi, yi, thi)


class CalibratedStage0Env(AmrNavStage0Env):
    """
    在每次 reset teleport 之後，用「teleport 的真值 pose」對齊輪速里程計 odom：
      world_pose = world_T_odom ⊕ odom_pose
    以解決 teleport 後 odom 不重置造成的位姿錯位。
    """
    def __init__(self, *args, calibrate_odom: bool = True, topic_timeout: float = 6.0, **kwargs):
        super().__init__(*args, **kwargs)

        self._calibrate_odom = bool(calibrate_odom)
        self._topic_timeout = float(topic_timeout)

        self._world_T_odom = None  # (x,y,yaw)

        # 保留原始 odom 讀法（輪速里程計的原生輸出）
        self._raw_get_pose2d = self.ros.get_robot_pose2d

        # 把 env 內所有 pose 讀取都換成「校正後」
        if self._calibrate_odom:
            self.ros.get_robot_pose2d = self._get_pose2d_calibrated

    def _get_pose2d_calibrated(self):
        raw = self._raw_get_pose2d()
        if raw is None or self._world_T_odom is None:
            return raw
        return pose_compose(self._world_T_odom, raw)

    def _calibrate_from_world_pose(self, world_pose):
        """
        用當下 raw odom pose 推出 world_T_odom，使得：
          world_pose = world_T_odom ⊕ odom_pose
        => world_T_odom = world_pose ⊕ inv(odom_pose)
        """
        raw = self._raw_get_pose2d()
        if raw is None:
            # 還沒收到 odom，先不校正
            self._world_T_odom = None
            return False

        self._world_T_odom = pose_compose(world_pose, pose_inverse(raw))
        return True

    def reset(self, *, seed=None, options=None):
        # 這段基本沿用 AmrNavStage0Env.reset，但插入「odom 校正」
        super().reset(seed=seed)

        self._step_count = 0
        self.ros.stop()

        # teleport robot
        rx, ry, ryaw = self._sample_start_pose()
        self.ros.set_entity_state(self.robot_name, rx, ry, yaw=ryaw, z=0.0)

        # sample goal + teleport target (for visualization)
        gx, gy = self._sample_goal_pose(rx, ry)
        self._goal_world[:] = [gx, gy]
        self.ros.set_entity_state(self.target_name, gx, gy, yaw=0.0, z=0.05)

        # warmup + wait topics (mecanum_controller 前 2 秒可能不出 odom，這裡拉長等待)
        self.ros.spin_for(0.3)
        self.ros.wait_for_topics(timeout_sec=self._topic_timeout)

        # === 關鍵：teleport 後做一次 odom 對齊 ===
        if self._calibrate_odom:
            # 用「你 teleport 的真值 pose」作為 world_pose
            ok = self._calibrate_from_world_pose((float(rx), float(ry), float(ryaw)))
            if not ok:
                # 若 odom 還沒來，先不校正；後續 episode 內 pose 會是 raw odom
                pass

        dist = self._compute_dist()
        self._prev_dist = float(dist) if dist is not None else 999.0

        obs = self._get_obs()
        info = {
            "goal_x": float(gx),
            "goal_y": float(gy),
            "start_x": float(rx),
            "start_y": float(ry),
            "start_yaw": float(ryaw),
            "odom_calibrated": int(self._world_T_odom is not None) if self._calibrate_odom else 0,
        }
        return obs, info


@dataclass
class EpResult:
    episode: int
    ep_return: float
    ep_len: int
    termination_reason: str
    is_success: int
    is_collision: int
    dist_to_goal: float


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--model", type=str, default="models_stage0/ppo_stage0.zip")
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--deterministic", action="store_true")

    # ROS/Gazebo wiring
    p.add_argument("--set-entity-service", type=str, default="/set_entity_state")
    p.add_argument("--cmd-vel-topic", type=str, default="cmd_vel")
    p.add_argument("--odom-topic", type=str, default="odom")
    p.add_argument("--scan-topic", type=str, default="scan")
    p.add_argument("--use-sim-time", action="store_true")

    # entity names
    p.add_argument("--robot-name", type=str, default="amr")
    p.add_argument("--target-name", type=str, default="Target")

    # env params
    add_rect_bounds_args(p, default_x_lim=3.0, default_y_lim=3.0)
    p.add_argument("--max-steps", type=int, default=400)
    p.add_argument("--control-dt", type=float, default=0.05)
    p.add_argument("--goal-threshold", type=float, default=0.35)
    p.add_argument("--collision-threshold", type=float, default=0.20)
    p.add_argument("--laser-downsample-n", type=int, default=120)

    # action limits
    p.add_argument("--vx-min", type=float, default=-0.3)
    p.add_argument("--vx-max", type=float, default=0.6)
    p.add_argument("--vy-max", type=float, default=0.4)
    p.add_argument("--wz-max", type=float, default=1.0)

    # wheel-odom + teleport reset 專用：每回合自動 odom 對齊
    p.add_argument("--calibrate-odom", action="store_true")
    p.add_argument("--topic-timeout", type=float, default=6.0)

    # output
    p.add_argument("--out-json", type=str, default="")

    args = p.parse_args()
    return finalize_rect_bounds_args(args, default_x_lim=3.0, default_y_lim=3.0)


def main():
    args = parse_args()

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model not found: {args.model}")

    rclpy.init()

    env = None
    try:
        env = CalibratedStage0Env(
            robot_name=args.robot_name,
            target_name=args.target_name,
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
            goal_threshold=args.goal_threshold,
            collision_threshold=args.collision_threshold,
            laser_downsample_n=args.laser_downsample_n,
            vx_min=args.vx_min,
            vx_max=args.vx_max,
            vy_max=args.vy_max,
            wz_max=args.wz_max,
            calibrate_odom=bool(args.calibrate_odom),
            topic_timeout=float(args.topic_timeout),
        )

        model = PPO.load(args.model)

        results = []
        success = 0
        collision = 0
        sum_ret = 0.0
        sum_len = 0

        for ep in range(int(args.episodes)):
            obs, info0 = env.reset()
            ep_ret = 0.0
            ep_len = 0

            while True:
                action, _ = model.predict(obs, deterministic=bool(args.deterministic))
                obs, reward, terminated, truncated, info = env.step(action)
                ep_ret += float(reward)
                ep_len += 1

                if terminated or truncated:
                    res = EpResult(
                        episode=ep,
                        ep_return=float(ep_ret),
                        ep_len=int(ep_len),
                        termination_reason=str(info.get("termination_reason", "unknown")),
                        is_success=int(info.get("is_success", 0)),
                        is_collision=int(info.get("is_collision", 0)),
                        dist_to_goal=float(info.get("dist_to_goal", float("nan"))),
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
                        f"(odom_calib={info0.get('odom_calibrated', 0)})"
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

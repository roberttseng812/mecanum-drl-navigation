#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Stage3 DRL inference node.

This node is intentionally aligned with amr_rl.amr_env_stage3:
  - agent obs: [x_rel/max_goal_dist, y_rel/max_goal_dist, theta/pi, yaw_err/pi]
  - laser obs: stacked LiDAR history sampled at the control tick cadence
  - action denorm: (vx, vy, wz) using Stage3 train ranges

It does NOT overwrite the existing stage2 inference bridge.
"""

import math
from collections import deque
from typing import Optional, Tuple

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import PoseStamped, Twist
from sensor_msgs.msg import LaserScan


def downsample_min_pool(ranges: np.ndarray, n: int) -> np.ndarray:
    """Match amr_rl.amr_ros_interface.get_laser_downsample()."""
    if n <= 0:
        raise ValueError("n must be > 0")
    if ranges.size == 0:
        return np.ones((n,), dtype=np.float32)

    m = int(ranges.size)
    if m == 1:
        return np.full((n,), float(ranges[0]), dtype=np.float32)

    edges = np.linspace(0, m, n + 1).astype(np.int32)
    out = np.empty((n,), dtype=np.float32)
    for i in range(n):
        start = int(edges[i])
        end = int(edges[i + 1])
        if end <= start:
            out[i] = float(ranges[min(start, m - 1)])
        else:
            out[i] = float(np.min(ranges[start:end]))
    return out


def quat_to_yaw(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


class DrlInferenceNodeStage3(Node):
    def __init__(self):
        super().__init__("drl_inference_stage3")

        # ===== Parameters =====
        self.declare_parameter("model_path", "models_stage3/sac_stage3.zip")
        self.declare_parameter("algo", "SAC")

        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("goal_topic", "drl/local_goal")
        self.declare_parameter("cmd_topic", "drl/cmd_vel")

        # Training-aligned defaults; exact laser/history layout is auto-inferred from the model when possible.
        self.declare_parameter("laser_downsample_n", 240)
        self.declare_parameter("lidar_history_steps", 4)
        self.declare_parameter("lidar_history_stride", 3)
        self.declare_parameter("max_goal_dist", 6.0)
        self.declare_parameter("goal_threshold", 0.30)
        self.declare_parameter("yaw_threshold", 0.20)

        self.declare_parameter("vx_min", -0.3)
        self.declare_parameter("vx_max", 0.47)
        self.declare_parameter("vy_abs_max", 0.35)
        self.declare_parameter("wz_abs_max", 0.8)

        self.declare_parameter("control_hz", 20.0)
        self.declare_parameter("debug_log_enabled", True)
        self.declare_parameter("debug_log_hz", 1.0)

        self.model_path = str(self.get_parameter("model_path").value)
        self.algo = str(self.get_parameter("algo").value).upper()
        self.scan_topic = str(self.get_parameter("scan_topic").value)
        self.goal_topic = str(self.get_parameter("goal_topic").value)
        self.cmd_topic = str(self.get_parameter("cmd_topic").value)

        self.max_goal_dist = float(self.get_parameter("max_goal_dist").value)
        self.goal_threshold = float(self.get_parameter("goal_threshold").value)
        self.yaw_threshold = float(self.get_parameter("yaw_threshold").value)
        self.control_hz = float(self.get_parameter("control_hz").value)
        self.debug_log_enabled = bool(self.get_parameter("debug_log_enabled").value)
        self.debug_log_hz = float(self.get_parameter("debug_log_hz").value)

        self.vx_min = float(self.get_parameter("vx_min").value)
        self.vx_max = float(self.get_parameter("vx_max").value)
        self.vy_abs_max = float(self.get_parameter("vy_abs_max").value)
        self.wz_abs_max = float(self.get_parameter("wz_abs_max").value)

        self._param_laser_downsample_n = int(self.get_parameter("laser_downsample_n").value)
        self._param_history_steps = int(self.get_parameter("lidar_history_steps").value)
        self._history_stride = int(self.get_parameter("lidar_history_stride").value)

        self.model = self._load_model(self.model_path, self.algo)
        self._infer_observation_layout()

        self._last_scan: Optional[LaserScan] = None
        self._last_goal: Optional[PoseStamped] = None
        self._laser_history = deque(maxlen=self._raw_history_len)

        self.create_subscription(LaserScan, self.scan_topic, self._scan_cb, qos_profile_sensor_data)
        self.create_subscription(PoseStamped, self.goal_topic, self._goal_cb, 10)
        self.cmd_pub = self.create_publisher(Twist, self.cmd_topic, 10)

        period = 1.0 / max(self.control_hz, 1.0)
        self.create_timer(period, self._control_tick)
        if self.debug_log_enabled:
            self.create_timer(1.0 / max(self.debug_log_hz, 1.0e-3), self._debug_tick)

        self.get_logger().info(
            "Loaded %s Stage3 model: %s | agent_dim=%d laser_total_dim=%d beams=%d hist_steps=%d stride=%d span=%.3fs"
            % (
                self.algo,
                self.model_path,
                self._agent_dim,
                self._laser_total_dim,
                self._laser_downsample_n,
                self._history_steps,
                self._history_stride,
                (self._history_steps - 1) * self._history_stride * period,
            )
        )

    def _load_model(self, model_path: str, algo: str):
        # Register custom extractor class before deserialization.
        try:
            from amr_rl.custom_extractors import Stage3LaserConvExtractor  # noqa: F401
        except Exception as exc:
            self.get_logger().warn(f"Failed to import Stage3 custom extractor before model load: {exc}")

        try:
            if algo == "SAC":
                from stable_baselines3 import SAC
                return SAC.load(model_path)
            if algo == "PPO":
                from stable_baselines3 import PPO
                return PPO.load(model_path)
            raise ValueError(f"Unsupported algo: {algo}")
        except Exception as exc:
            self.get_logger().error(
                "Failed to load model. Check stable-baselines3/torch installation and model_path."
            )
            raise exc

    def _infer_observation_layout(self):
        obs_space = getattr(self.model, "observation_space", None)
        if obs_space is None or "agent" not in obs_space.spaces or "laser" not in obs_space.spaces:
            raise RuntimeError("Loaded model does not expose the expected Dict observation space with 'agent' and 'laser'.")

        self._agent_dim = int(np.prod(obs_space.spaces["agent"].shape))
        self._laser_total_dim = int(np.prod(obs_space.spaces["laser"].shape))

        hist_steps = None
        beams = None
        extractor = getattr(getattr(self.model, "policy", None), "features_extractor", None)
        if extractor is not None:
            hist_steps = getattr(extractor, "lidar_history_steps", None)
            beams = getattr(extractor, "laser_num_beams", None)

        if hist_steps is not None and beams is not None:
            hist_steps = int(hist_steps)
            beams = int(beams)
            if hist_steps * beams != self._laser_total_dim:
                hist_steps = None
                beams = None

        if hist_steps is None or beams is None:
            hist_steps = int(max(1, self._param_history_steps))
            if self._laser_total_dim % hist_steps == 0:
                beams = int(self._laser_total_dim // hist_steps)
            elif self._param_laser_downsample_n > 0 and (self._laser_total_dim % self._param_laser_downsample_n == 0):
                beams = int(self._param_laser_downsample_n)
                hist_steps = int(self._laser_total_dim // beams)
            else:
                raise RuntimeError(
                    "Cannot infer Stage3 laser layout from model. "
                    f"laser_total_dim={self._laser_total_dim}, requested history_steps={hist_steps}, "
                    f"requested beams={self._param_laser_downsample_n}."
                )

        self._history_steps = int(hist_steps)
        self._laser_downsample_n = int(beams)
        self._raw_history_len = 1 + (self._history_steps - 1) * int(max(1, self._history_stride))

        if self._agent_dim < 4:
            self.get_logger().warn(
                f"Loaded model agent_dim={self._agent_dim}. Stage3 training expects 4-D agent obs; will truncate accordingly."
            )
        elif self._agent_dim > 4:
            self.get_logger().warn(
                f"Loaded model agent_dim={self._agent_dim}. Extra dims will be zero-padded after Stage3 base features."
            )

    def _scan_cb(self, msg: LaserScan):
        self._last_scan = msg

    def _goal_cb(self, msg: PoseStamped):
        self._last_goal = msg

    def _sanitize_and_downsample_scan(self, scan: LaserScan) -> np.ndarray:
        # Keep preprocessing aligned with amr_rl.amr_ros_interface.get_laser_downsample().
        ranges = np.asarray(scan.ranges, dtype=np.float32)
        rmax = float(scan.range_max) if scan.range_max > 0.0 else 10.0
        if ranges.size == 0:
            return np.ones((self._laser_downsample_n,), dtype=np.float32)
        ranges[np.isinf(ranges)] = rmax
        ranges[np.isnan(ranges)] = rmax
        ds = downsample_min_pool(ranges, self._laser_downsample_n)
        return np.clip(ds / max(rmax, 1e-6), 0.0, 1.0).astype(np.float32)

    def _append_current_scan_to_history(self) -> bool:
        if self._last_scan is None:
            return False
        laser = self._sanitize_and_downsample_scan(self._last_scan)
        self._laser_history.append(laser.copy())
        return True

    def _prime_laser_history(self) -> bool:
        if self._last_scan is None:
            return False
        laser = self._sanitize_and_downsample_scan(self._last_scan)
        self._laser_history.clear()
        for _ in range(self._raw_history_len):
            self._laser_history.append(laser.copy())
        return True

    def _stack_laser_history(self) -> Optional[np.ndarray]:
        if self._last_scan is None:
            return None
        if len(self._laser_history) <= 0:
            if not self._prime_laser_history():
                return None
        while len(self._laser_history) < self._raw_history_len:
            self._laser_history.appendleft(self._laser_history[0].copy())
        hist_list = list(self._laser_history)
        sampled = []
        for i in range(self._history_steps):
            sampled.append(hist_list[i * self._history_stride])
        laser = np.concatenate(sampled, axis=0).astype(np.float32)
        if laser.size != self._laser_total_dim:
            raise RuntimeError(
                f"Built laser obs dim={laser.size}, but model expects {self._laser_total_dim}."
            )
        return laser

    def _build_agent_obs(self) -> Optional[np.ndarray]:
        if self._last_goal is None:
            return None

        goal = self._last_goal
        x_rel = float(goal.pose.position.x)
        y_rel = float(goal.pose.position.y)
        dist = math.hypot(x_rel, y_rel)
        theta = math.atan2(y_rel, x_rel) if dist > 1e-6 else 0.0

        goal_yaw_rel = quat_to_yaw(
            goal.pose.orientation.x,
            goal.pose.orientation.y,
            goal.pose.orientation.z,
            goal.pose.orientation.w,
        )

        base_agent = np.array(
            [
                np.clip(x_rel / max(self.max_goal_dist, 1e-6), -1.0, 1.0),
                np.clip(y_rel / max(self.max_goal_dist, 1e-6), -1.0, 1.0),
                np.clip(theta / math.pi, -1.0, 1.0),
                np.clip(goal_yaw_rel / math.pi, -1.0, 1.0),
            ],
            dtype=np.float32,
        )

        if self._agent_dim == 4:
            return base_agent
        if self._agent_dim < 4:
            return base_agent[: self._agent_dim].astype(np.float32)

        agent = np.zeros((self._agent_dim,), dtype=np.float32)
        agent[:4] = base_agent
        return agent

    def _build_obs(self) -> Optional[dict]:
        if self._last_goal is None or self._last_scan is None:
            return None

        if len(self._laser_history) <= 0:
            if not self._prime_laser_history():
                return None
        else:
            self._append_current_scan_to_history()

        laser = self._stack_laser_history()
        agent = self._build_agent_obs()
        if laser is None or agent is None:
            return None
        return {"agent": agent, "laser": laser}

    def _goal_metrics(self) -> Optional[Tuple[float, float, float]]:
        if self._last_goal is None:
            return None
        goal = self._last_goal
        x_rel = float(goal.pose.position.x)
        y_rel = float(goal.pose.position.y)
        dist = math.hypot(x_rel, y_rel)
        heading_err = math.atan2(y_rel, x_rel) if dist > 1e-6 else 0.0
        goal_yaw_err = quat_to_yaw(
            goal.pose.orientation.x,
            goal.pose.orientation.y,
            goal.pose.orientation.z,
            goal.pose.orientation.w,
        )
        return dist, heading_err, goal_yaw_err

    def _debug_tick(self):
        metrics = self._goal_metrics()
        if metrics is None:
            self.get_logger().info("[STAGE3_DEBUG] waiting for scan/local_goal ...")
            return

        dist, heading_err, goal_yaw_err = metrics
        hist_ready = len(self._laser_history)
        self.get_logger().info(
            "[STAGE3_DEBUG] dist=%.3f m | heading_err=%.1f deg | goal_yaw_err=%.1f deg | hist=%d/%d | stop=%s"
            % (
                dist,
                math.degrees(heading_err),
                math.degrees(goal_yaw_err),
                hist_ready,
                self._raw_history_len,
                "YES" if (dist < self.goal_threshold and abs(goal_yaw_err) < self.yaw_threshold) else "NO",
            )
        )

    def _denorm_action(self, action: np.ndarray) -> Tuple[float, float, float]:
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        if a.size != 3:
            raise ValueError(f"Expected action dim=3, got {a.size}")
        vx = float(np.interp(a[0], [-1.0, 1.0], [self.vx_min, self.vx_max]))
        vy = float(a[1] * self.vy_abs_max)
        wz = float(a[2] * self.wz_abs_max)
        return vx, vy, wz

    def _control_tick(self):
        metrics = self._goal_metrics()
        if metrics is None or self._last_scan is None:
            self._publish_zero()
            return

        dist, _, goal_yaw_err = metrics
        if dist < self.goal_threshold and abs(goal_yaw_err) < self.yaw_threshold:
            self._publish_zero()
            return

        obs = self._build_obs()
        if obs is None:
            self._publish_zero()
            return

        try:
            action, _ = self.model.predict(obs, deterministic=True)
            vx, vy, wz = self._denorm_action(action)
        except Exception as exc:
            self.get_logger().warn(f"Model inference failed: {exc}")
            self._publish_zero()
            return

        msg = Twist()
        msg.linear.x = float(vx)
        msg.linear.y = float(vy)
        msg.angular.z = float(wz)
        self.cmd_pub.publish(msg)

    def _publish_zero(self):
        self.cmd_pub.publish(Twist())


def main():
    rclpy.init()
    node = DrlInferenceNodeStage3()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._publish_zero()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

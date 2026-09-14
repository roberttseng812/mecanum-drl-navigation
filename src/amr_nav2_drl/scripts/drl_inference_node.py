#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Nav2 DRL inference node

Subscribes:
  - /scan (sensor_msgs/LaserScan)
  - drl/local_goal (geometry_msgs/PoseStamped)  [goal already in base_link frame]

Publishes:
  - drl/cmd_vel (geometry_msgs/Twist)

This is intentionally kept very close to your Stage1 observation/action conventions:
  obs = {
    "agent": [x_rel/max_goal_dist, y_rel/max_goal_dist, theta/pi],
    "laser": downsampled_minpool(scan)/range_max
  }

Action is denormalized to (vx, vy, wz) with the same ranges as Stage1.
"""

import math
from typing import Optional, Tuple

import numpy as np

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist
from sensor_msgs.msg import LaserScan
from rclpy.qos import qos_profile_sensor_data


def downsample_min_pool(ranges: np.ndarray, n: int) -> np.ndarray:
    """Min-pool downsampling to n bins (matching your AmrRosInterface logic)."""
    if n <= 0:
        raise ValueError("n must be > 0")
    if ranges.size == 0:
        return np.ones((n,), dtype=np.float32)

    m = ranges.size
    if m <= n:
        # If scan has fewer points than requested, pad by repeating edges
        idx = np.linspace(0, m - 1, n).astype(np.int32)
        return ranges[idx]

    seg = m // n
    out = np.empty((n,), dtype=np.float32)
    for i in range(n):
        start = i * seg
        end = (i + 1) * seg if i < n - 1 else m
        out[i] = float(np.min(ranges[start:end]))
    return out


class DrlInferenceNode(Node):
    def __init__(self):
        super().__init__("drl_inference")

        # ===== Parameters =====
        self.declare_parameter("model_path", "models_stage1/sac_stage1.zip")
        self.declare_parameter("algo", "SAC")  # SAC | PPO

        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("goal_topic", "drl/local_goal")
        self.declare_parameter("cmd_topic", "drl/cmd_vel")

        self.declare_parameter("laser_downsample_n", 120)
        self.declare_parameter("max_goal_dist", 2.5)
        self.declare_parameter("stop_dist", 0.35)
        self.declare_parameter("stop_yaw", 0.25)   # Stage2: 距離到後還要看角度

        # Action denorm (match your Stage1 defaults)
        self.declare_parameter("vx_min", -0.3)
        self.declare_parameter("vx_max", 0.5)
        self.declare_parameter("vy_abs_max", 0.4)
        self.declare_parameter("wz_abs_max", 0.8)

        self.declare_parameter("control_hz", 20.0)
        self.declare_parameter("debug_log_enabled", True)
        self.declare_parameter("debug_log_hz", 1.0)

        self.model_path = str(self.get_parameter("model_path").value)
        self.algo = str(self.get_parameter("algo").value).upper()

        self.scan_topic = str(self.get_parameter("scan_topic").value)
        self.goal_topic = str(self.get_parameter("goal_topic").value)
        self.cmd_topic = str(self.get_parameter("cmd_topic").value)

        self.laser_downsample_n = int(self.get_parameter("laser_downsample_n").value)
        self.max_goal_dist = float(self.get_parameter("max_goal_dist").value)
        self.stop_dist = float(self.get_parameter("stop_dist").value)
        self.stop_yaw = float(self.get_parameter("stop_yaw").value)

        self.vx_min = float(self.get_parameter("vx_min").value)
        self.vx_max = float(self.get_parameter("vx_max").value)
        self.vy_abs_max = float(self.get_parameter("vy_abs_max").value)
        self.wz_abs_max = float(self.get_parameter("wz_abs_max").value)

        self.control_hz = float(self.get_parameter("control_hz").value)
        self.debug_log_enabled = bool(self.get_parameter("debug_log_enabled").value)
        self.debug_log_hz = float(self.get_parameter("debug_log_hz").value)

        # ===== Model =====
        self.model = self._load_model(self.model_path, self.algo)

        # ===== State =====
        self._last_scan: Optional[LaserScan] = None
        self._last_goal: Optional[PoseStamped] = None

        # ===== ROS IO =====
        self.create_subscription(
            LaserScan,
            self.scan_topic,
            self._scan_cb,
            qos_profile_sensor_data)
        self.create_subscription(PoseStamped, self.goal_topic, self._goal_cb, 10)
        self.cmd_pub = self.create_publisher(Twist, self.cmd_topic, 10)

        period = 1.0 / max(self.control_hz, 1.0)
        self.create_timer(period, self._control_tick)
        if self.debug_log_enabled:
            debug_period = 1.0 / max(self.debug_log_hz, 1.0e-3)
            self.create_timer(debug_period, self._debug_tick)

        self.get_logger().info(
            f"Loaded {self.algo} model: {self.model_path} | scan={self.scan_topic} goal={self.goal_topic} cmd={self.cmd_topic}"
        )

    def _load_model(self, model_path: str, algo: str):
        try:
            if algo == "SAC":
                from stable_baselines3 import SAC

                return SAC.load(model_path)
            if algo == "PPO":
                from stable_baselines3 import PPO

                return PPO.load(model_path)
            raise ValueError(f"Unsupported algo: {algo}")
        except Exception as e:
            self.get_logger().error(
                "Failed to load model. Make sure stable-baselines3 + torch are installed and model_path is correct."
            )
            raise e

    def _scan_cb(self, msg: LaserScan):
        self._last_scan = msg

    def _goal_cb(self, msg: PoseStamped):
        self._last_goal = msg

    # def _build_obs(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:#stage1
    #     if self._last_scan is None or self._last_goal is None:
    #         return None

    #     scan = self._last_scan
    #     goal = self._last_goal

    #     # Goal is expected in base_link frame (controller plugin publishes it that way)
    #     x_rel = float(goal.pose.position.x)
    #     y_rel = float(goal.pose.position.y)

    #     dist = math.hypot(x_rel, y_rel)
    #     theta = math.atan2(y_rel, x_rel) if dist > 1e-6 else 0.0

    #     x_n = max(-1.0, min(1.0, x_rel / self.max_goal_dist))
    #     y_n = max(-1.0, min(1.0, y_rel / self.max_goal_dist))
    #     th_n = max(-1.0, min(1.0, theta / math.pi))

    #     agent = np.array([x_n, y_n, th_n], dtype=np.float32)

    #     ranges = np.array(scan.ranges, dtype=np.float32)
    #     rmin = float(scan.range_min)
    #     rmax = float(scan.range_max) if scan.range_max > 0.0 else 10.0

    #     if ranges.size == 0:
    #         laser = np.ones((self.laser_downsample_n,), dtype=np.float32)
    #     else:
    #         # Match your stage1 preprocessing
    #         ranges[np.isinf(ranges)] = rmax
    #         ranges[np.isnan(ranges)] = rmax
    #         ranges[ranges <= 0.0] = rmax
    #         ranges[ranges < rmin] = rmin

    #         ds = downsample_min_pool(ranges, self.laser_downsample_n)
    #         laser = (ds / rmax).astype(np.float32)

    #     return agent, laser

    def _build_obs(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if self._last_scan is None or self._last_goal is None:
            return None

        scan = self._last_scan
        goal = self._last_goal

        # Goal is expected in base_link frame (Nav2 controller 轉好的)
        x_rel = float(goal.pose.position.x)
        y_rel = float(goal.pose.position.y)

        dist = math.hypot(x_rel, y_rel)
        theta = math.atan2(y_rel, x_rel) if dist > 1e-6 else 0.0

        x_n = max(-1.0, min(1.0, x_rel / self.max_goal_dist))
        y_n = max(-1.0, min(1.0, y_rel / self.max_goal_dist))
        th_n = max(-1.0, min(1.0, theta / math.pi))

        # --- 新增: 計算 Nav2 傳遞過來的 目標 Yaw 與車體的相對誤差 ---
        q = goal.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        target_yaw_rel = math.atan2(siny_cosp, cosy_cosp)
        
        yaw_err_n = max(-1.0, min(1.0, target_yaw_rel / math.pi))

        # --- 自動判斷模型是 Stage 1 (3維) 還是 Stage 2 (4維) ---
        agent_shape = self.model.observation_space.spaces["agent"].shape[0]
        
        if agent_shape == 4:
            # Stage 2: [x, y, theta, yaw_err]
            agent = np.array([x_n, y_n, th_n, yaw_err_n], dtype=np.float32)
        else:
            # Stage 1: [x, y, theta]
            agent = np.array([x_n, y_n, th_n], dtype=np.float32)

        # 處理雷射下採樣
        ranges = np.array(scan.ranges, dtype=np.float32)
        rmin = float(scan.range_min)
        rmax = float(scan.range_max) if scan.range_max > 0.0 else 10.0

        if ranges.size == 0:
            laser = np.ones((self.laser_downsample_n,), dtype=np.float32)
        else:
            ranges[np.isinf(ranges)] = rmax
            ranges[np.isnan(ranges)] = rmax
            ranges[ranges <= 0.0] = rmax
            ranges[ranges < rmin] = rmin

            ds = downsample_min_pool(ranges, self.laser_downsample_n)
            laser = (ds / rmax).astype(np.float32)

        return agent, laser

    def _goal_metrics(self) -> Optional[Tuple[float, float, float]]:
        if self._last_goal is None:
            return None

        goal = self._last_goal
        x_rel = float(goal.pose.position.x)
        y_rel = float(goal.pose.position.y)

        dist = math.hypot(x_rel, y_rel)
        heading_err = math.atan2(y_rel, x_rel) if dist > 1e-6 else 0.0

        q = goal.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        goal_yaw_err = math.atan2(siny_cosp, cosy_cosp)

        return dist, heading_err, goal_yaw_err

    def _debug_tick(self):
        metrics = self._goal_metrics()
        if metrics is None:
            self.get_logger().info("[NAV_DEBUG] waiting for local goal ...")
            return

        dist, heading_err, goal_yaw_err = metrics
        is_stage2 = False
        try:
            agent_shape = self.model.observation_space.spaces["agent"].shape[0]
            is_stage2 = (agent_shape >= 4)
        except Exception:
            pass

        if is_stage2:
            stop_ok = (dist < self.stop_dist) and (abs(goal_yaw_err) < self.stop_yaw)
        else:
            stop_ok = (dist < self.stop_dist)

        self.get_logger().info(
            "[NAV_DEBUG] dist=%.3f m | heading_err=%.1f deg | goal_yaw_err=%.1f deg | stop=%s"
            % (
                dist,
                math.degrees(heading_err),
                math.degrees(goal_yaw_err),
                "YES" if stop_ok else "NO",
            )
        )

    def _denorm_action(self, a: np.ndarray) -> Tuple[float, float, float]:
        a = np.asarray(a, dtype=np.float32).reshape(-1)
        if a.size != 3:
            raise ValueError(f"Expected action dim=3, got {a.size}")

        vx = float(np.interp(a[0], [-1.0, 1.0], [self.vx_min, self.vx_max]))
        vy = float(a[1] * self.vy_abs_max)
        wz = float(a[2] * self.wz_abs_max)
        return vx, vy, wz

    def _control_tick(self):
        obs_pair = self._build_obs()
        if obs_pair is None:
            self._publish_zero()
            return

        agent, laser = obs_pair

        # stop near goal (lets Nav2 goal checker succeed)
        # --- Stop condition ---
        # Stage1: 只看距離
        # Stage2: 距離 + yaw 都達標才停，避免提早截斷最終對齊
        dist = float(math.hypot(agent[0] * self.max_goal_dist, agent[1] * self.max_goal_dist))

        is_stage2 = (agent.shape[0] >= 4)
        if is_stage2:
            # agent[3] = yaw_err / pi  (在 _build_obs() 裡建的)
            yaw_err = float(agent[3] * math.pi)
            if dist < self.stop_dist and abs(yaw_err) < self.stop_yaw:
                self._publish_zero()
                return
        else:
            if dist < self.stop_dist:
                self._publish_zero()
                return

        obs = {"agent": agent, "laser": laser}

        try:
            action, _ = self.model.predict(obs, deterministic=True)
            vx, vy, wz = self._denorm_action(action)
        except Exception as e:
            self.get_logger().warn(f"Model inference failed: {e}")
            self._publish_zero()
            return

        msg = Twist()
        msg.linear.x = vx
        msg.linear.y = vy
        msg.angular.z = wz
        self.cmd_pub.publish(msg)

    def _publish_zero(self):
        msg = Twist()
        self.cmd_pub.publish(msg)


def main():
    rclpy.init()
    node = DrlInferenceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
import time
import math
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan

from gazebo_msgs.srv import SetEntityState
from gazebo_msgs.msg import EntityState

from rclpy.qos import qos_profile_sensor_data, QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy


def _stamp_to_float_sec(stamp) -> float:
    """Convert ROS2 builtin_interfaces/Time to float seconds."""
    try:
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9
    except Exception:
        return float('nan')


def yaw_from_quat(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class AmrRosInterface(Node):
    """
    封裝：
    - publish cmd_vel
    - subscribe odom + scan
    - service call: /set_entity_state (or /gazebo/set_entity_state)
    """
    def __init__(
        self,
        cmd_vel_topic: str = 'cmd_vel',
        odom_topic: str = 'odom',
        scan_topic: str = 'scan',
        set_entity_service: str = '/set_entity_state',
        use_sim_time: bool = True,
        laser_downsample_n: int = 120,
    ):
        super().__init__('amr_rl_interface')

        # sim time
        self.set_parameters([rclpy.parameter.Parameter('use_sim_time',
                                                       rclpy.Parameter.Type.BOOL,
                                                       use_sim_time)])

        self._cmd_pub = self.create_publisher(Twist, cmd_vel_topic, 10)

        # Odom 通常是 RELIABLE，這裡明確指定一個 QoS（避免某些情況下也出現不相容）
        odom_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE
        )

        # Scan 必須用 sensor data QoS（BEST_EFFORT）
        scan_qos = qos_profile_sensor_data

        self._odom_sub = self.create_subscription(Odometry, odom_topic, self._odom_cb, odom_qos)
        self._scan_sub = self.create_subscription(LaserScan, scan_topic, self._scan_cb, scan_qos)

        self._set_state_cli = self.create_client(SetEntityState, set_entity_service)
        self._wait_for_service(self._set_state_cli, set_entity_service)

        # latest data
        self._has_odom = False
        self._has_scan = False
        self._odom = None
        self._scan = None

        # counters / timing (for debugging sensor health)
        self._odom_cb_count = 0
        self._scan_cb_count = 0
        self._last_odom_wall_time = None
        self._last_scan_wall_time = None
        self._last_odom_stamp_ros_sec = float('nan')
        self._last_scan_stamp_ros_sec = float('nan')

        self._laser_downsample_n = int(laser_downsample_n)
        self._last_scan_max = 10.0  # fallback

    # ---------- Time helpers (ROS time / sim time) ----------
    def _now_ros(self):
        """Return current ROS time (uses /clock when use_sim_time=True)."""
        return self.get_clock().now()

    def _now_ros_sec(self) -> float:
        now = self._now_ros()
        return float(now.nanoseconds) * 1e-9

    def _deadline_sim(self, timeout_sec: float):
        """Return a ROS-time deadline (now + timeout)."""
        return self._now_ros() + Duration(seconds=float(timeout_sec))

    def _warn_if_sim_stuck(self, start_wall: float, wall_timeout_sec: float, what: str) -> bool:
        """Return True if we exceeded wall-time watchdog (sim time may be paused / /clock missing)."""
        if wall_timeout_sec is None:
            return False
        if (time.time() - float(start_wall)) > float(wall_timeout_sec):
            self.get_logger().warn(f"{what} timed out: sim time did not advance within {wall_timeout_sec:.2f}s wall time")
            return True
        return False


    def _wait_for_service(self, cli, name: str, timeout_sec: float = 5.0):
        t0 = time.time()
        while not cli.wait_for_service(timeout_sec=0.2):
            if time.time() - t0 > timeout_sec:
                raise RuntimeError(
                    f"Service not available: {name}. "
                    f"Check: ros2 service list | grep set_entity_state"
                )
        self.get_logger().info(f"Connected to service: {name}")

    def _odom_cb(self, msg: Odometry):
        self._odom = msg
        self._has_odom = True
        self._odom_cb_count += 1
        self._last_odom_wall_time = time.time()
        try:
            self._last_odom_stamp_ros_sec = _stamp_to_float_sec(msg.header.stamp)
        except Exception:
            self._last_odom_stamp_ros_sec = float('nan')

    def _scan_cb(self, msg: LaserScan):
        self._scan = msg
        self._has_scan = True
        self._last_scan_max = float(msg.range_max)
        self._scan_cb_count += 1
        self._last_scan_wall_time = time.time()
        try:
            self._last_scan_stamp_ros_sec = _stamp_to_float_sec(msg.header.stamp)
        except Exception:
            self._last_scan_stamp_ros_sec = float('nan')


    def get_odom_status(self, max_age_sec: float = 0.5) -> dict:
        """Return odom health/status for debugging.

        Note: age_sec is computed in ROS time (sim time when use_sim_time=True).
        """
        present = bool(self._has_odom and (self._odom is not None))

        now_ros = self._now_ros_sec()
        stamp_ros = float(self._last_odom_stamp_ros_sec)
        age_ros = float('inf')
        if np.isfinite(stamp_ros):
            age_ros = max(0.0, float(now_ros - stamp_ros))

        fresh = bool(present and age_ros <= float(max_age_sec))

        # Keep wall-time age for diagnosing /clock stalls or callback starvation.
        now_wall = time.time()
        age_wall = float('inf') if self._last_odom_wall_time is None else float(now_wall - self._last_odom_wall_time)

        return {
            "present": present,
            "fresh": fresh,
            "age_sec": float(age_ros),
            "age_wall_sec": float(age_wall),
            "cb_count": int(self._odom_cb_count),
            "stamp_ros_sec": float(self._last_odom_stamp_ros_sec),
        }


    def get_scan_status(self, max_age_sec: float = 0.5) -> dict:
        """Return scan health/status for debugging.

        Note: age_sec is computed in ROS time (sim time when use_sim_time=True).
        """
        present = bool(
            self._has_scan
            and (self._scan is not None)
            and (hasattr(self._scan, "ranges"))
            and (len(self._scan.ranges) > 0)
        )

        now_ros = self._now_ros_sec()
        stamp_ros = float(self._last_scan_stamp_ros_sec)
        age_ros = float('inf')
        if np.isfinite(stamp_ros):
            age_ros = max(0.0, float(now_ros - stamp_ros))

        fresh = bool(present and age_ros <= float(max_age_sec))

        n_ranges = int(len(self._scan.ranges)) if present else 0

        # Keep wall-time age for diagnosing /clock stalls or callback starvation.
        now_wall = time.time()
        age_wall = float('inf') if self._last_scan_wall_time is None else float(now_wall - self._last_scan_wall_time)

        return {
            "present": present,
            "fresh": fresh,
            "age_sec": float(age_ros),
            "age_wall_sec": float(age_wall),
            "cb_count": int(self._scan_cb_count),
            "n_ranges": int(n_ranges),
            "range_max": float(self._last_scan_max),
            "stamp_ros_sec": float(self._last_scan_stamp_ros_sec),
        }


    def wait_for_new_scan(self, prev_count: int, timeout_sec: float = 2.0, wall_timeout_sec: float = 5.0) -> bool:
        """Block until a new LaserScan message arrives (scan callback count increases).

        timeout_sec is interpreted in ROS time (sim time when use_sim_time=True).
        wall_timeout_sec prevents deadlock if /clock stops (paused) or not published.
        """
        prev = int(prev_count)
        deadline = self._deadline_sim(timeout_sec)

        t0_wall = time.time()
        while rclpy.ok() and self._now_ros() < deadline:
            # Waiting here is in wall time, but completion condition is ROS time.
            rclpy.spin_once(self, timeout_sec=0.05)
            if int(self._scan_cb_count) > prev:
                return True
            if self._warn_if_sim_stuck(t0_wall, wall_timeout_sec, "wait_for_new_scan"):
                return False
        return False


    def wait_for_new_odom(self, prev_count: int, timeout_sec: float = 2.0, wall_timeout_sec: float = 5.0) -> bool:
        """Block until a new Odometry message arrives (odom callback count increases).

        timeout_sec is interpreted in ROS time (sim time when use_sim_time=True).
        wall_timeout_sec prevents deadlock if /clock stops (paused) or not published.
        """
        prev = int(prev_count)
        deadline = self._deadline_sim(timeout_sec)

        t0_wall = time.time()
        while rclpy.ok() and self._now_ros() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            if int(self._odom_cb_count) > prev:
                return True
            if self._warn_if_sim_stuck(t0_wall, wall_timeout_sec, "wait_for_new_odom"):
                return False
        return False



    def wait_for_new_data(
        self,
        prev_scan_count: int,
        prev_odom_count: int,
        timeout_sec: float = 2.0,
        wall_timeout_sec: float = 5.0,
        require_scan: bool = True,
        require_odom: bool = True,
    ) -> tuple[bool, bool]:
        """Spin until *new* scan/odom callbacks arrive (example-style stepping).

        This is the same idea as the reference example's `spin()` loop (spin until both
        pose and laser callbacks have fired once after applying the action), but with:

        - `timeout_sec` interpreted in ROS time (sim time when use_sim_time=True)
        - `wall_timeout_sec` watchdog to avoid deadlock if /clock stalls or sim is paused

        Returns:
            (got_scan, got_odom) where each indicates whether that topic callback count increased.
        """
        prev_scan = int(prev_scan_count)
        prev_odom = int(prev_odom_count)

        got_scan = False if bool(require_scan) else True
        got_odom = False if bool(require_odom) else True

        deadline = self._deadline_sim(timeout_sec)
        t0_wall = time.time()

        while rclpy.ok() and self._now_ros() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)

            if (not got_scan) and int(self._scan_cb_count) > prev_scan:
                got_scan = True
            if (not got_odom) and int(self._odom_cb_count) > prev_odom:
                got_odom = True

            if got_scan and got_odom:
                break

            if self._warn_if_sim_stuck(t0_wall, wall_timeout_sec, "wait_for_new_data"):
                break

        return bool(got_scan), bool(got_odom)


    def spin_for(self, duration_sec: float, sleep_step: float = 0.001, wall_timeout_sec: float = 5.0):
        """Spin the node for a given duration of ROS time (sim time when use_sim_time=True).

        - duration_sec: measured in ROS time (/clock) when use_sim_time=True
        - sleep_step: kept for backward compatibility; used as an upper bound for spin_once timeout
        - wall_timeout_sec: watchdog to avoid deadlock if /clock stalls
        """
        duration_sec = float(duration_sec)
        if duration_sec <= 0.0:
            rclpy.spin_once(self, timeout_sec=0.0)
            return

        # Use ROS-time deadline; do not use wall-time sleeps to define "dt".
        deadline = self._deadline_sim(duration_sec)
        t0_wall = time.time()

        # Use a small wall-time wait to reduce busy-looping.
        step_timeout = float(max(0.0, min(0.05, float(sleep_step) if sleep_step is not None else 0.01)))
        if step_timeout == 0.0:
            step_timeout = 0.01

        while rclpy.ok() and self._now_ros() < deadline:
            rclpy.spin_once(self, timeout_sec=step_timeout)
            if self._warn_if_sim_stuck(t0_wall, wall_timeout_sec, "spin_for"):
                break


    def wait_for_topics(self, timeout_sec: float = 2.0, wall_timeout_sec: float = 10.0):
        """Wait until both odom and scan have been received.

        timeout_sec is interpreted in ROS time (sim time when use_sim_time=True).
        wall_timeout_sec prevents deadlock during startup if /clock is not running yet.
        """
        deadline = self._deadline_sim(timeout_sec)
        t0_wall = time.time()
        while rclpy.ok() and self._now_ros() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self._has_odom and self._has_scan:
                return True
            if self._warn_if_sim_stuck(t0_wall, wall_timeout_sec, "wait_for_topics"):
                break
        return bool(self._has_odom and self._has_scan)

    def publish_cmd(self, vx: float, vy: float, wz: float):
        msg = Twist()
        msg.linear.x = float(vx)
        msg.linear.y = float(vy)
        msg.angular.z = float(wz)
        self._cmd_pub.publish(msg)

    def stop(self, repeat: int = 3, dt: float = 0.02):
        for _ in range(int(repeat)):
            self.publish_cmd(0.0, 0.0, 0.0)
            self.spin_for(dt)

    def get_robot_pose2d(self):
        if not self._has_odom or self._odom is None:
            return None
        p = self._odom.pose.pose.position
        q = self._odom.pose.pose.orientation
        return float(p.x), float(p.y), float(yaw_from_quat(q))

    def get_laser_downsample(self):
        if (not self._has_scan) or (self._scan is None) or (len(self._scan.ranges) == 0):
            n = self._laser_downsample_n
            return np.ones((n,), dtype=np.float32) * self._last_scan_max, self._last_scan_max

        ranges = np.array(self._scan.ranges, dtype=np.float32)
        rmax = float(self._scan.range_max)

        ranges[np.isinf(ranges)] = rmax
        ranges[np.isnan(ranges)] = rmax

        n = self._laser_downsample_n

        # 分成 n 個桶，每個桶取最小值（min-pooling），不會漏掉最近障礙
        N = len(ranges)
        edges = np.linspace(0, N, n + 1).astype(np.int32)

        out = np.empty((n,), dtype=np.float32)
        for i in range(n):
            s = edges[i]
            e = edges[i + 1]
            if e <= s:
                out[i] = ranges[min(s, N - 1)]
            else:
                out[i] = float(np.min(ranges[s:e]))

        return out, rmax

    def set_entity_state(
        self,
        name: str,
        x: float,
        y: float,
        yaw: float = 0.0,
        z: float = 0.0,
        ref_frame: str = 'world'
    ):
        req = SetEntityState.Request()
        state = EntityState()
        state.name = str(name)
        state.reference_frame = str(ref_frame)

        state.pose.position.x = float(x)
        state.pose.position.y = float(y)
        state.pose.position.z = float(z)

        half = float(yaw) * 0.5
        state.pose.orientation.x = 0.0
        state.pose.orientation.y = 0.0
        state.pose.orientation.z = math.sin(half)
        state.pose.orientation.w = math.cos(half)

        state.twist.linear.x = 0.0
        state.twist.linear.y = 0.0
        state.twist.linear.z = 0.0
        state.twist.angular.z = 0.0

        req.state = state
        fut = self._set_state_cli.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=2.0)
        if fut.result() is None:
            raise RuntimeError(
                f"SetEntityState failed for {name}. "
                f"Check service name: {self._set_state_cli.srv_name}"
            )
        return fut.result()
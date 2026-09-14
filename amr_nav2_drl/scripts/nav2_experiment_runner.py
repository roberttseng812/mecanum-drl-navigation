#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Send one exact Nav2 goal and record motion-based metrics for comparison.

Recording window:
  - Start recording only after the robot starts moving according to odometry.
  - Normal stop: NavigateToPose succeeds and the robot remains stopped for `stop_duration_sec`.
  - Failure stop: the robot remains continuously stopped for `failure_stop_duration_sec` without
    satisfying the normal success condition. Metrics are trimmed to `stop_duration_sec` after
    the beginning of that stopped period.

Recorded data:
  - path trajectory in goal_frame/map frame: pose_x, pose_y
  - odom_speed_xy = sqrt(odom_vx^2 + odom_vy^2)
  - odom_wz
  - total navigation time inside the motion-based recording window
  - total navigation distance inside the motion-based recording window

File naming:
  - Metrics files are stored in OUTPUT_DIR/METHOD/ROUTE/.
  - Metrics CSV files are named METHOD-TRIAL.csv. Example: DWB/1/DWB-1.csv.
  - If the selected file name already exists in that route folder, TRIAL is
    automatically increased until a free file name is found.
"""

import csv
import math
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import Odometry
from rclpy.action import ActionClient
from rclpy.node import Node
from tf2_ros import Buffer, TransformException, TransformListener


STATUS_TEXT = {
    GoalStatus.STATUS_UNKNOWN: "UNKNOWN",
    GoalStatus.STATUS_ACCEPTED: "ACCEPTED",
    GoalStatus.STATUS_EXECUTING: "EXECUTING",
    GoalStatus.STATUS_CANCELING: "CANCELING",
    GoalStatus.STATUS_SUCCEEDED: "SUCCEEDED",
    GoalStatus.STATUS_CANCELED: "CANCELED",
    GoalStatus.STATUS_ABORTED: "ABORTED",
}


def sanitize_name(text: str) -> str:
    text = text.strip() or "run"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def build_available_output_paths(
    output_dir: Path,
    controller_name: str,
    route_index: int,
    requested_trial_index: int,
) -> Tuple[Path, Path, Path, int]:
    """Return unique output paths using OUTPUT_DIR/METHOD/ROUTE/METHOD-TRIAL.csv naming."""
    method = sanitize_name(controller_name)
    route = max(1, int(route_index))
    trial = max(1, int(requested_trial_index))

    route_dir = output_dir / method / str(route)
    route_dir.mkdir(parents=True, exist_ok=True)

    while True:
        output_prefix = route_dir / f"{method}-{trial}"
        metrics_csv_path = output_prefix.with_suffix(".csv")
        summary_txt_path = output_prefix.with_suffix(".summary.txt")
        if not metrics_csv_path.exists() and not summary_txt_path.exists():
            return output_prefix, metrics_csv_path, summary_txt_path, trial
        trial += 1


def yaw_to_quat(yaw: float) -> Tuple[float, float, float, float]:
    half = 0.5 * yaw
    return 0.0, 0.0, math.sin(half), math.cos(half)


def quat_to_yaw(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


class Nav2ExperimentRunner(Node):
    def __init__(self) -> None:
        super().__init__("nav2_experiment_runner")

        # Experiment identity / goal
        self.declare_parameter("controller_name", "DWB")
        self.declare_parameter("run_id", "")
        self.declare_parameter("route_index", 1)
        self.declare_parameter("trial_index", 1)
        self.declare_parameter("goal_x", 0.0)
        self.declare_parameter("goal_y", 0.0)
        self.declare_parameter("goal_yaw", 0.0)
        self.declare_parameter("goal_frame", "map")
        self.declare_parameter("behavior_tree", "")

        # Topics / frames
        self.declare_parameter("navigate_action", "navigate_to_pose")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("base_frame", "base_link")

        # Runtime / output
        self.declare_parameter("sample_rate_hz", 20.0)
        self.declare_parameter("max_wait_action_server_sec", 30.0)
        self.declare_parameter("movement_start_speed_threshold", 0.01)
        self.declare_parameter("movement_start_wz_threshold", 0.01)
        self.declare_parameter("stopped_speed_threshold", 0.01)
        self.declare_parameter("stopped_wz_threshold", 0.01)
        self.declare_parameter("stop_duration_sec", 0.5)
        self.declare_parameter("failure_stop_duration_sec", 5.0)
        self.declare_parameter("show_plots", True)
        self.declare_parameter("save_files", True)
        self.declare_parameter("output_dir", "~/sac_ws/src/amr_nav2_experiments")

        self.controller_name = str(self.get_parameter("controller_name").value)
        self.run_id = str(self.get_parameter("run_id").value).strip()
        self.route_index = int(self.get_parameter("route_index").value)
        self.trial_index = int(self.get_parameter("trial_index").value)

        self.goal_x = float(self.get_parameter("goal_x").value)
        self.goal_y = float(self.get_parameter("goal_y").value)
        self.goal_yaw = float(self.get_parameter("goal_yaw").value)
        self.goal_frame = str(self.get_parameter("goal_frame").value)
        self.behavior_tree = str(self.get_parameter("behavior_tree").value)

        self.navigate_action = str(self.get_parameter("navigate_action").value)
        self.odom_topic = str(self.get_parameter("odom_topic").value)
        self.base_frame = str(self.get_parameter("base_frame").value)

        self.sample_rate_hz = max(1.0, float(self.get_parameter("sample_rate_hz").value))
        self.max_wait_action_server_sec = float(self.get_parameter("max_wait_action_server_sec").value)
        self.movement_start_speed_threshold = max(0.0, float(
            self.get_parameter("movement_start_speed_threshold").value
        ))
        self.movement_start_wz_threshold = max(0.0, float(
            self.get_parameter("movement_start_wz_threshold").value
        ))
        self.stopped_speed_threshold = max(0.0, float(
            self.get_parameter("stopped_speed_threshold").value
        ))
        self.stopped_wz_threshold = max(0.0, float(
            self.get_parameter("stopped_wz_threshold").value
        ))
        self.stop_duration_sec = max(0.0, float(self.get_parameter("stop_duration_sec").value))
        self.failure_stop_duration_sec = max(
            self.stop_duration_sec,
            float(self.get_parameter("failure_stop_duration_sec").value),
        )
        self.show_plots = bool(self.get_parameter("show_plots").value)
        self.save_files = bool(self.get_parameter("save_files").value)

        output_dir = Path(os.path.expanduser(str(self.get_parameter("output_dir").value)))
        (
            self.output_prefix,
            self.metrics_csv_path,
            self.summary_txt_path,
            self.trial_index,
        ) = build_available_output_paths(
            output_dir,
            self.controller_name,
            self.route_index,
            self.trial_index,
        )

        self.action_client = ActionClient(self, NavigateToPose, self.navigate_action)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.odom_sub = self.create_subscription(Odometry, self.odom_topic, self._odom_cb, 50)

        self.last_odom: Optional[Odometry] = None
        self.last_odom_receive_sec: Optional[float] = None
        self.rows: List[Dict[str, Optional[float]]] = []

        self._sent_goal = False
        self._accepted = False
        self._action_done = False
        self.finished = False

        self.recording_started = False
        self.recording_active = False
        self._stopped_since_sec: Optional[float] = None
        self._recording_cutoff_sec: Optional[float] = None

        self._wait_start_sec: Optional[float] = None
        self.navigation_start_sec: Optional[float] = None
        self.navigation_end_sec: Optional[float] = None
        self._result_status: int = GoalStatus.STATUS_UNKNOWN
        self._result_error_msg: Optional[str] = None
        self.summary: Dict[str, object] = {}

        self.sample_timer = self.create_timer(1.0 / self.sample_rate_hz, self._sample)
        self.start_timer = self.create_timer(0.5, self._try_start)

        self.get_logger().info(
            "Experiment prepared: file=%s, output_dir=%s, controller=%s, route=%d, trial=%d, "
            "goal=(%.3f, %.3f, %.3f) in %s, recording starts when robot moves, "
            "finishes after success + %.3f s stopped, or fails after %.3f s stopped"
            % (
                self.metrics_csv_path.name,
                str(self.metrics_csv_path.parent),
                self.controller_name,
                self.route_index,
                self.trial_index,
                self.goal_x,
                self.goal_y,
                self.goal_yaw,
                self.goal_frame,
                self.stop_duration_sec,
                self.failure_stop_duration_sec,
            )
        )

    def _odom_cb(self, msg: Odometry) -> None:
        self.last_odom = msg
        self.last_odom_receive_sec = self._now_sec()

    def _now_sec(self) -> float:
        return self.get_clock().now().nanoseconds * 1.0e-9

    def _make_goal_msg(self) -> NavigateToPose.Goal:
        goal = NavigateToPose.Goal()
        goal.pose = PoseStamped()
        goal.pose.header.frame_id = self.goal_frame
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = self.goal_x
        goal.pose.pose.position.y = self.goal_y
        goal.pose.pose.position.z = 0.0
        qx, qy, qz, qw = yaw_to_quat(self.goal_yaw)
        goal.pose.pose.orientation.x = qx
        goal.pose.pose.orientation.y = qy
        goal.pose.pose.orientation.z = qz
        goal.pose.pose.orientation.w = qw
        if self.behavior_tree:
            goal.behavior_tree = self.behavior_tree
        return goal

    def _try_start(self) -> None:
        if self._sent_goal:
            return

        now = self._now_sec()
        if self._wait_start_sec is None:
            self._wait_start_sec = now

        if not self.action_client.wait_for_server(timeout_sec=0.0):
            waited = now - self._wait_start_sec
            if waited > self.max_wait_action_server_sec:
                self.get_logger().error(
                    f"Action server '{self.navigate_action}' not available after {waited:.1f} s."
                )
                self._finish(GoalStatus.STATUS_ABORTED, "Action server unavailable")
            return

        self._sent_goal = True
        goal_msg = self._make_goal_msg()
        self.get_logger().info(
            f"Sending NavigateToPose goal: x={self.goal_x:.3f}, y={self.goal_y:.3f}, "
            f"yaw={self.goal_yaw:.3f}, frame={self.goal_frame}"
        )
        send_future = self.action_client.send_goal_async(goal_msg)
        send_future.add_done_callback(self._goal_response_cb)

    def _goal_response_cb(self, future) -> None:
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("NavigateToPose goal rejected.")
            self._finish(GoalStatus.STATUS_ABORTED, "Goal rejected")
            return

        self._accepted = True
        self.get_logger().info(
            "NavigateToPose goal accepted. Waiting for odom movement before recording metrics."
        )

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._result_cb)

    def _result_cb(self, future) -> None:
        wrapped = future.result()
        status = int(getattr(wrapped, "status", GoalStatus.STATUS_UNKNOWN))
        result = getattr(wrapped, "result", None)
        error_msg = getattr(result, "error_msg", None) if result is not None else None

        self._action_done = True
        self._result_status = status
        self._result_error_msg = str(error_msg) if error_msg else None

        if not self.recording_started:
            self._finish(status, self._result_error_msg or "Action finished before robot started moving")
            return

        self.get_logger().info(
            "NavigateToPose action result received: status=%s. Normal recording stop requires "
            "SUCCEEDED + %.3f s stopped. Failure stop occurs after %.3f s continuously stopped."
            % (
                STATUS_TEXT.get(status, str(status)),
                self.stop_duration_sec,
                self.failure_stop_duration_sec,
            )
        )

    def _lookup_pose_in_goal_frame(self) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        try:
            tf = self.tf_buffer.lookup_transform(
                self.goal_frame,
                self.base_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.01),
            )
            tr = tf.transform.translation
            q = tf.transform.rotation
            return float(tr.x), float(tr.y), quat_to_yaw(q.x, q.y, q.z, q.w)
        except TransformException:
            return None, None, None

    def _odom_velocity(self) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
        if self.last_odom is None:
            return None, None, None, None
        t = self.last_odom.twist.twist
        vx = float(t.linear.x)
        vy = float(t.linear.y)
        wz = float(t.angular.z)
        speed_xy = math.hypot(vx, vy)
        return vx, vy, speed_xy, wz

    def _is_moving_for_start(
        self,
        odom_speed_xy: Optional[float],
        odom_wz: Optional[float],
    ) -> bool:
        if odom_speed_xy is None or odom_wz is None:
            return False
        return (
            abs(float(odom_speed_xy)) >= self.movement_start_speed_threshold
            or abs(float(odom_wz)) >= self.movement_start_wz_threshold
        )

    def _is_stopped(
        self,
        odom_speed_xy: Optional[float],
        odom_wz: Optional[float],
    ) -> bool:
        if odom_speed_xy is None or odom_wz is None:
            return False
        return (
            abs(float(odom_speed_xy)) <= self.stopped_speed_threshold
            and abs(float(odom_wz)) <= self.stopped_wz_threshold
        )

    def _mark_recording_cutoff_after_stop(self) -> None:
        """Keep metrics only until `stop_duration_sec` after the current stopped period began."""
        if self._stopped_since_sec is None:
            return
        self._recording_cutoff_sec = self._stopped_since_sec + self.stop_duration_sec

    def _update_recording_window(self, now: float) -> None:
        if self.finished or not self._accepted:
            return

        _, _, odom_speed_xy, odom_wz = self._odom_velocity()
        moving_for_start = self._is_moving_for_start(odom_speed_xy, odom_wz)
        stopped = self._is_stopped(odom_speed_xy, odom_wz)

        if not self.recording_started:
            if moving_for_start:
                self.recording_started = True
                self.recording_active = True
                self.navigation_start_sec = now
                self._stopped_since_sec = None
                self._recording_cutoff_sec = None
                self.get_logger().info(
                    "Robot movement detected. Start recording metrics: "
                    "speed_xy=%s m/s, wz=%s rad/s"
                    % (
                        "N/A" if odom_speed_xy is None else f"{float(odom_speed_xy):.4f}",
                        "N/A" if odom_wz is None else f"{float(odom_wz):.4f}",
                    )
                )
            elif self._action_done:
                self._finish(
                    self._result_status,
                    self._result_error_msg or "Action finished before robot started moving",
                )
            return

        if not stopped:
            self._stopped_since_sec = None
            self._recording_cutoff_sec = None
            return

        if self._stopped_since_sec is None:
            self._stopped_since_sec = now
            return

        stopped_duration = now - self._stopped_since_sec

        success_condition = (
            self._action_done
            and self._result_status == GoalStatus.STATUS_SUCCEEDED
            and stopped_duration >= self.stop_duration_sec
        )
        failure_condition = stopped_duration >= self.failure_stop_duration_sec

        if success_condition:
            self._mark_recording_cutoff_after_stop()
            reason = (
                f"NavigateToPose SUCCEEDED and robot stopped for {stopped_duration:.3f} s "
                f"(success threshold={self.stop_duration_sec:.3f} s)"
            )
            self._finish(GoalStatus.STATUS_SUCCEEDED, reason)
            return

        if failure_condition:
            self._mark_recording_cutoff_after_stop()
            reason = (
                f"Robot remained continuously stopped for {stopped_duration:.3f} s without "
                f"satisfying success condition; failure threshold="
                f"{self.failure_stop_duration_sec:.3f} s. Metrics kept only until "
                f"{self.stop_duration_sec:.3f} s after stop began."
            )
            if self._action_done and self._result_error_msg:
                reason = f"{self._result_error_msg}; {reason}"
            self._finish(GoalStatus.STATUS_ABORTED, reason)

    def _append_sample(self, now: float) -> None:
        if self.navigation_start_sec is None:
            return

        pose_x, pose_y, pose_yaw = self._lookup_pose_in_goal_frame()
        odom_vx, odom_vy, odom_speed_xy, odom_wz = self._odom_velocity()

        self.rows.append({
            "elapsed_sec": now - self.navigation_start_sec,
            "pose_x": pose_x,
            "pose_y": pose_y,
            "pose_yaw": pose_yaw,
            "odom_vx": odom_vx,
            "odom_vy": odom_vy,
            "odom_speed_xy": odom_speed_xy,
            "odom_wz": odom_wz,
        })

    def _sample(self) -> None:
        if self.finished:
            return

        now = self._now_sec()
        self._update_recording_window(now)

        if self.finished or not self.recording_active or self.navigation_start_sec is None:
            return

        self._append_sample(now)

    def _trajectory_distance(self) -> Optional[float]:
        total = 0.0
        prev = None
        used = 0
        for row in self.rows:
            x = row.get("pose_x")
            y = row.get("pose_y")
            if x is None or y is None:
                continue
            x = float(x)
            y = float(y)
            if not (math.isfinite(x) and math.isfinite(y)):
                continue
            if prev is not None:
                total += math.hypot(x - prev[0], y - prev[1])
            prev = (x, y)
            used += 1
        return total if used >= 2 else None

    def _trim_rows_to_recording_cutoff(self) -> None:
        if self._recording_cutoff_sec is None or self.navigation_start_sec is None:
            return

        cutoff_elapsed_sec = max(0.0, self._recording_cutoff_sec - self.navigation_start_sec)
        self.rows = [
            row for row in self.rows
            if row.get("elapsed_sec") is not None
            and float(row["elapsed_sec"]) <= cutoff_elapsed_sec + 1.0e-9
        ]

    def _write_files(self) -> None:
        if not self.save_files:
            return

        with open(self.metrics_csv_path, "w", newline="", encoding="utf-8") as f:
            keys = ["elapsed_sec", "pose_x", "pose_y", "pose_yaw",
                    "odom_vx", "odom_vy", "odom_speed_xy", "odom_wz"]
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.rows)

        with open(self.summary_txt_path, "w", encoding="utf-8") as f:
            f.write("Nav2 experiment summary\n")
            f.write("=======================\n")
            f.write(f"controller_name: {self.controller_name}\n")
            f.write(f"run_id: {self.run_id}\n")
            f.write(f"route_index: {self.route_index}\n")
            f.write(f"trial_index: {self.trial_index}\n")
            f.write("recording_start_condition: odom speed/wz above movement thresholds\n")
            f.write(f"movement_start_speed_threshold: {self.movement_start_speed_threshold:.6f}\n")
            f.write(f"movement_start_wz_threshold: {self.movement_start_wz_threshold:.6f}\n")
            f.write(
                "recording_stop_condition: NavigateToPose SUCCEEDED and odom speed/wz "
                "below stopped thresholds for stop_duration_sec\n"
            )
            f.write(
                "failure_stop_condition: odom speed/wz below stopped thresholds "
                "for failure_stop_duration_sec without satisfying success condition\n"
            )
            f.write(f"stopped_speed_threshold: {self.stopped_speed_threshold:.6f}\n")
            f.write(f"stopped_wz_threshold: {self.stopped_wz_threshold:.6f}\n")
            f.write(f"stop_duration_sec: {self.stop_duration_sec:.6f}\n")
            f.write(f"failure_stop_duration_sec: {self.failure_stop_duration_sec:.6f}\n")
            f.write(f"goal_frame: {self.goal_frame}\n")
            f.write(f"goal_x: {self.goal_x:.6f}\n")
            f.write(f"goal_y: {self.goal_y:.6f}\n")
            f.write(f"goal_yaw: {self.goal_yaw:.6f}\n")
            f.write(f"status: {self.summary.get('status_text')}\n")
            f.write(f"navigation_total_time_sec: {self.summary.get('navigation_total_time_sec')}\n")
            f.write(f"navigation_total_distance_m: {self.summary.get('navigation_total_distance_m')}\n")
            f.write(f"metrics_csv: {self.metrics_csv_path}\n")

    def _build_summary(self, status: int, error_msg: Optional[str]) -> Dict[str, object]:
        self.navigation_end_sec = self._recording_cutoff_sec or self._now_sec()
        total_time = None
        if self.navigation_start_sec is not None:
            total_time = self.navigation_end_sec - self.navigation_start_sec

        total_distance = self._trajectory_distance()

        return {
            "controller_name": self.controller_name,
            "run_id": self.run_id,
            "route_index": self.route_index,
            "trial_index": self.trial_index,
            "movement_start_speed_threshold": self.movement_start_speed_threshold,
            "movement_start_wz_threshold": self.movement_start_wz_threshold,
            "stopped_speed_threshold": self.stopped_speed_threshold,
            "stopped_wz_threshold": self.stopped_wz_threshold,
            "stop_duration_sec": self.stop_duration_sec,
            "failure_stop_duration_sec": self.failure_stop_duration_sec,
            "status_code": int(status),
            "status_text": STATUS_TEXT.get(int(status), str(status)),
            "result_error_msg": error_msg,
            "navigation_total_time_sec": total_time,
            "navigation_total_distance_m": total_distance,
            "sample_count": len(self.rows),
            "sample_rate_hz": self.sample_rate_hz,
        }

    def _show_plots(self) -> None:
        if not self.show_plots:
            return
        if not self.rows:
            self.get_logger().warn("No sampled data available. Skipping plots.")
            return

        xs = [float(r["pose_x"]) for r in self.rows if r.get("pose_x") is not None and r.get("pose_y") is not None]
        ys = [float(r["pose_y"]) for r in self.rows if r.get("pose_x") is not None and r.get("pose_y") is not None]

        t_speed = [float(r["elapsed_sec"]) for r in self.rows
                   if r.get("elapsed_sec") is not None and r.get("odom_speed_xy") is not None]
        speed_xy = [float(r["odom_speed_xy"]) for r in self.rows
                    if r.get("elapsed_sec") is not None and r.get("odom_speed_xy") is not None]

        t_wz = [float(r["elapsed_sec"]) for r in self.rows
                if r.get("elapsed_sec") is not None and r.get("odom_wz") is not None]
        wz = [float(r["odom_wz"]) for r in self.rows
              if r.get("elapsed_sec") is not None and r.get("odom_wz") is not None]

        total_time = self.summary.get("navigation_total_time_sec")
        total_dist = self.summary.get("navigation_total_distance_m")
        time_text = "N/A" if total_time is None else f"{float(total_time):.3f} s"
        dist_text = "N/A" if total_dist is None else f"{float(total_dist):.3f} m"

        if xs and ys:
            fig1, ax1 = plt.subplots(figsize=(7, 7))
            ax1.plot(xs, ys, label="trajectory")
            ax1.scatter([xs[0]], [ys[0]], marker="o", s=80, label="start")
            ax1.scatter([self.goal_x], [self.goal_y], marker="x", s=100, label="goal")
            ax1.set_xlabel(f"x [{self.goal_frame}]")
            ax1.set_ylabel(f"y [{self.goal_frame}]")
            ax1.set_title(
                f"Trajectory in {self.goal_frame}\n"
                f"controller={self.controller_name}, time={time_text}, distance={dist_text}"
            )
            ax1.axis("equal")
            ax1.grid(True)
            ax1.legend()

        if t_speed and speed_xy:
            fig2, ax2 = plt.subplots(figsize=(8, 4.5))
            ax2.plot(t_speed, speed_xy, label="odom_speed_xy")
            ax2.set_xlabel("elapsed time [s]")
            ax2.set_ylabel("odom_speed_xy [m/s]")
            ax2.set_title(f"Linear speed profile\nnavigation time={time_text}")
            ax2.grid(True)
            ax2.legend()

        if t_wz and wz:
            fig3, ax3 = plt.subplots(figsize=(8, 4.5))
            ax3.plot(t_wz, wz, label="odom_wz")
            ax3.axhline(0.0, linewidth=0.8)
            ax3.set_xlabel("elapsed time [s]")
            ax3.set_ylabel("odom_wz [rad/s]")
            ax3.set_title(f"Angular velocity profile\nnavigation time={time_text}")
            ax3.grid(True)
            ax3.legend()

        if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
            self.get_logger().warn(
                "No GUI display detected. If plots do not pop up, run in a desktop session / ssh -X."
            )

        plt.show()

    def _finish(self, status: int, error_msg: Optional[str] = None) -> None:
        if self.finished:
            return

        if self._recording_cutoff_sec is None:
            if self.recording_active and self.navigation_start_sec is not None:
                self._append_sample(self._now_sec())
        else:
            self._trim_rows_to_recording_cutoff()

        self.recording_active = False
        self._result_status = int(status)
        self._result_error_msg = error_msg
        self.summary = self._build_summary(status, error_msg)

        self.get_logger().info(
            "Recording finished: status=%s, total_time=%s, total_distance=%s, samples=%d"
            % (
                self.summary["status_text"],
                "N/A" if self.summary["navigation_total_time_sec"] is None else
                f"{float(self.summary['navigation_total_time_sec']):.3f} s",
                "N/A" if self.summary["navigation_total_distance_m"] is None else
                f"{float(self.summary['navigation_total_distance_m']):.3f} m",
                len(self.rows),
            )
        )

        self._write_files()
        if self.save_files:
            self.get_logger().info(f"Saved metrics: {self.metrics_csv_path}")
            self.get_logger().info(f"Saved summary: {self.summary_txt_path}")

        self.finished = True


def main(args=None) -> None:
    rclpy.init(args=args)
    node = Nav2ExperimentRunner()
    try:
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        node.get_logger().warn("Interrupted. Finishing with partial data.")
        node._finish(GoalStatus.STATUS_CANCELED, "Interrupted by user")
    finally:
        try:
            node._show_plots()
        except Exception as e:
            node.get_logger().error(f"Failed to show plots: {e}")
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
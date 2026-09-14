import math
from collections import deque

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from amr_rl.amr_ros_interface import AmrRosInterface


def wrap_to_pi(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


# Default dynamic-human corridors extracted from straight actor motion in
# school_8f_square.world.  These are intentionally the straight portions of the
# actor trajectories; turn corners are trimmed later so robot start/goal are not
# placed right on the bends.
#
# Each item contains:
#   - p0/p1: corridor centerline endpoints in world coordinates
#   - half_width: lateral sampling half-width used to generate offset start/goal
DEFAULT_SCHOOL_8F_ROUTE_SEGMENTS = [
    {"name": "top_hall", "p0": (-5.0, 0.6), "p1": (2.3, 0.6), "half_width": 0.45},
    {"name": "diag_down", "p0": (2.511, -0.139), "p1": (-2.661, -2.288), "half_width": 0.40},
    {"name": "bottom_hall", "p0": (-4.0, -1.7), "p1": (2.5, -1.7), "half_width": 0.45},
    {"name": "diag_up", "p0": (2.679, -1.0), "p1": (-3.54, 0.804), "half_width": 0.40},
]

class AmrNavStage3Env(gym.Env):
    """
    Stage 3（動態避障版）
    --------------------
    設計目標：
    - 不繼承 Stage1 / Stage2，所有邏輯整合在這個檔案內。
    - 不再生成隨機靜態障礙物；動態障礙由 Gazebo world 自身提供。
    - Observation 使用：
        1) 相對目標資訊 [x_rel, y_rel, theta, yaw_err]
        2) 多幀 LiDAR history（預設 4 x 240，flatten）
           - 可用 lidar_history_stride 拉開歷史幀間距，在固定 20 Hz control 下保留時間感
        3) 不再把 odom 速度放進 obs，避免模擬/實機速度估計落差
    - Reward 沿用 Stage1 + Stage2 的方法，但直接整合在這份 env。
    - 每一步都檢查 scan / odom 是否有新資料且仍然 fresh，避免把 stale transition 寫進 replay buffer。
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        robot_name: str = "amr",
        target_name: str = "Target",
        cmd_vel_topic: str = "cmd_vel",
        odom_topic: str = "odom",
        scan_topic: str = "scan",
        set_entity_service: str = "/set_entity_state",
        use_sim_time: bool = True,
        control_dt: float = 0.05,
        max_steps: int = 300,
        x_lim: float | None = 3.5,
        y_lim: float | None = 3.5,
        x_min: float | None = None,
        x_max: float | None = None,
        y_min: float | None = None,
        y_max: float | None = None,
        oob_mode: str = "both",
        oob_radius: float = 5.5,
        min_goal_dist: float = 2.2,
        max_goal_dist: float = 4.8,
        goal_threshold: float = 0.30,
        yaw_threshold: float = 0.20,
        goal_yaw_sector_prob: float = 0.60,
        goal_yaw_sector_half_angle_deg: float = 60.0,
        collision_threshold: float = 0.57,
        collision_confirm_steps: int = 2,
        collision_min_hits: int = 2,
        laser_downsample_n: int = 240,
        lidar_history_steps: int = 4,
        lidar_history_stride: int = 3,
        vx_min: float = -0.3,
        vx_max: float = 0.5,
        vy_max: float = 0.4,
        wz_max: float = 0.8,
        near_obstacle_threshold: float = 1.00,
        near_obstacle_penalty: float = 0.28,
        clearance_linear_weight: float = 0.60,
        clearance_square_weight: float = 0.40,
        speed_near_obstacle_penalty: float = 0.15,
        step_penalty: float = -0.02,
        stage2_step_penalty: float | None = None,
        sensor_freshness_sec: float | None = None,
        sensor_fail_penalty: float = 2.0,
        clearance_square: bool = True,
        timeout_penalty: float = 6.5,
        near_goal_mult: float = 2.0,
        ring_bonus: float = 0.01,
        inside_goal_bonus: float = 0.02,
        yaw_abs_penalty: float = 0.06,
        exit_goal_penalty: float = 0.20,
        action_smooth_penalty: float = 0.01,
        idle_penalty: float = 0.01,
        idle_speed_thresh: float = 0.03,
        robot_spawn_margin: float = 0.85,
        goal_spawn_margin: float = 0.85,
        spawn_min_clearance: float = 1.3,
        max_reset_tries: int = 25,
        topic_timeout_sec: float = 6.0,
        dynamic_route_prob: float = 0.85,
        dynamic_route_mode: str = "mixed",
        dynamic_headon_prob: float = 0.40,
        dynamic_samedir_prob: float = 0.22,
        dynamic_route_t_margin: float = 0.18,
        dynamic_route_trim_m: float = 0.90,
        dynamic_corridor_half_width: float = 0.42,
        dynamic_route_jitter: float = 0.35,
        dynamic_route_yaw_noise_deg: float = 20.0,
        dynamic_route_segments: list[tuple[tuple[float, float], tuple[float, float]]] | None = None,
        seed: int | None = None,
    ):
        super().__init__()

        self.robot_name = str(robot_name)
        self.target_name = str(target_name)
        self.control_dt = float(control_dt)
        self.max_steps = int(max_steps)

        self.min_goal_dist = float(min_goal_dist)
        self.max_goal_dist = float(max_goal_dist)
        self.goal_threshold = float(goal_threshold)
        self.yaw_threshold = float(yaw_threshold)
        self.goal_yaw_sector_prob = float(np.clip(goal_yaw_sector_prob, 0.0, 1.0))
        self.goal_yaw_sector_half_angle_deg = float(goal_yaw_sector_half_angle_deg)
        self.goal_yaw_sector_half_angle = math.radians(self.goal_yaw_sector_half_angle_deg)

        self.collision_threshold = float(collision_threshold)
        self.collision_confirm_steps = max(1, int(collision_confirm_steps))
        self.collision_min_hits = max(1, int(collision_min_hits))

        self.vx_min = float(vx_min)
        self.vx_max = float(vx_max)
        self.vy_max = float(vy_max)
        self.wz_max = float(wz_max)

        self.near_obstacle_threshold = float(near_obstacle_threshold)
        self.near_obstacle_penalty = float(near_obstacle_penalty)
        self.clearance_linear_weight = max(0.0, float(clearance_linear_weight))
        self.clearance_square_weight = max(0.0, float(clearance_square_weight))
        self.speed_near_obstacle_penalty = max(0.0, float(speed_near_obstacle_penalty))
        self.step_penalty = float(step_penalty)
        self.stage2_step_penalty = float(stage2_step_penalty) if stage2_step_penalty is not None else None
        if (self.stage2_step_penalty is not None) and (not np.isclose(self.stage2_step_penalty, self.step_penalty)):
            print(
                "[WARN] stage2_step_penalty is deprecated in Stage3Env and no longer overrides reward. "
                f"Using step_penalty={self.step_penalty:.4f} (received deprecated stage2_step_penalty={self.stage2_step_penalty:.4f})."
            )
        self.sensor_freshness_sec = float(
            sensor_freshness_sec if sensor_freshness_sec is not None else max(0.20, 3.0 * float(control_dt))
        )
        self.sensor_fail_penalty = float(sensor_fail_penalty)
        self.clearance_square = bool(clearance_square)
        self.timeout_penalty = float(timeout_penalty)
        self.near_goal_mult = float(near_goal_mult)
        self.ring_bonus = float(ring_bonus)
        self.inside_goal_bonus = float(inside_goal_bonus)
        self.yaw_abs_penalty = float(yaw_abs_penalty)
        self.exit_goal_penalty = float(exit_goal_penalty)
        self.action_smooth_penalty = float(action_smooth_penalty)
        self.idle_penalty = float(idle_penalty)
        self.idle_speed_thresh = float(idle_speed_thresh)

        self.robot_spawn_margin = float(robot_spawn_margin)
        self.goal_spawn_margin = float(goal_spawn_margin)
        self.spawn_min_clearance = float(spawn_min_clearance)
        self.max_reset_tries = int(max_reset_tries)
        self.topic_timeout_sec = float(topic_timeout_sec)
        self.dynamic_route_prob = float(np.clip(dynamic_route_prob, 0.0, 1.0))
        self.dynamic_route_mode = str(dynamic_route_mode).strip().lower()
        if self.dynamic_route_mode not in {"crossing", "head_on", "same_direction", "mixed"}:
            raise ValueError(
                "dynamic_route_mode must be one of {'crossing', 'head_on', 'same_direction', 'mixed'}"
            )
        self.dynamic_headon_prob = float(np.clip(dynamic_headon_prob, 0.0, 1.0))
        self.dynamic_samedir_prob = float(np.clip(dynamic_samedir_prob, 0.0, 1.0))
        self.dynamic_route_t_margin = float(np.clip(dynamic_route_t_margin, 0.0, 0.45))
        self.dynamic_route_trim_m = max(0.0, float(dynamic_route_trim_m))
        self.dynamic_corridor_half_width = max(0.10, float(dynamic_corridor_half_width))
        self.dynamic_route_jitter = max(0.0, float(dynamic_route_jitter))
        self.dynamic_route_yaw_noise_deg = max(0.0, float(dynamic_route_yaw_noise_deg))
        self._dynamic_route_segments_src = list(dynamic_route_segments) if dynamic_route_segments is not None else list(DEFAULT_SCHOOL_8F_ROUTE_SEGMENTS)

        self.oob_mode = str(oob_mode)
        self.oob_radius = float(oob_radius)

        self._rng = np.random.default_rng(seed)

        def _resolve_axis_bounds(axis_min, axis_max, axis_lim, axis_name: str):
            if axis_min is None and axis_max is None:
                lim = float(axis_lim if axis_lim is not None else 3.5)
                lo, hi = -lim, lim
            elif axis_min is None:
                hi = float(axis_max)
                lim = float(axis_lim if axis_lim is not None else max(abs(hi), 0.1))
                lo = -lim
            elif axis_max is None:
                lo = float(axis_min)
                lim = float(axis_lim if axis_lim is not None else max(abs(lo), 0.1))
                hi = lim
            else:
                lo, hi = float(axis_min), float(axis_max)

            if not np.isfinite(lo) or not np.isfinite(hi):
                raise ValueError(f"{axis_name}_min/{axis_name}_max must be finite")
            if hi <= lo:
                raise ValueError(f"{axis_name}_max must be greater than {axis_name}_min")
            return float(lo), float(hi)

        self.x_min, self.x_max = _resolve_axis_bounds(x_min, x_max, x_lim, "x")
        self.y_min, self.y_max = _resolve_axis_bounds(y_min, y_max, y_lim, "y")
        self.x_lim = max(abs(self.x_min), abs(self.x_max))
        self.y_lim = max(abs(self.y_min), abs(self.y_max))
        self._dynamic_route_segments = self._build_dynamic_route_segments()

        self.ros = AmrRosInterface(
            cmd_vel_topic=cmd_vel_topic,
            odom_topic=odom_topic,
            scan_topic=scan_topic,
            set_entity_service=set_entity_service,
            use_sim_time=use_sim_time,
            laser_downsample_n=int(laser_downsample_n),
        )

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        self._laser_n = int(laser_downsample_n)
        self._lidar_history_steps = int(max(1, lidar_history_steps))
        self._lidar_history_stride = int(max(1, lidar_history_stride))
        self._lidar_raw_history_len = 1 + (self._lidar_history_steps - 1) * self._lidar_history_stride
        self._lidar_history = deque(maxlen=self._lidar_raw_history_len)
        self.observation_space = spaces.Dict({
            "agent": spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),
            "laser": spaces.Box(
                low=0.0,
                high=1.0,
                shape=(self._lidar_history_steps * self._laser_n,),
                dtype=np.float32,
            ),
        })

        self._step_count = 0
        self._collision_streak = 0
        self._goal_world = np.array([0.0, 0.0], dtype=np.float32)
        self._start_world = np.array([0.0, 0.0], dtype=np.float32)
        self._target_yaw_world = 0.0
        self._prev_dist = None
        self._prev_yaw_err = None
        self._prev_action = None
        self._was_in_goal_zone = False
        self._last_obs = None

    # ---------------------------
    # helpers
    # ---------------------------
    def _spawn_inner_bounds(self) -> tuple[float, float, float, float]:
        m = max(float(self.robot_spawn_margin), float(self.goal_spawn_margin))
        m = min(m, max(0.0, 0.5 * (self.x_max - self.x_min) - 0.05))
        m = min(m, max(0.0, 0.5 * (self.y_max - self.y_min) - 0.05))
        return (
            float(self.x_min + m),
            float(self.x_max - m),
            float(self.y_min + m),
            float(self.y_max - m),
        )

    @staticmethod
    def _clip_segment_to_box(
        p0: tuple[float, float],
        p1: tuple[float, float],
        x_lo: float,
        x_hi: float,
        y_lo: float,
        y_hi: float,
    ) -> tuple[tuple[float, float], tuple[float, float]] | None:
        x0, y0 = float(p0[0]), float(p0[1])
        x1, y1 = float(p1[0]), float(p1[1])
        dx = x1 - x0
        dy = y1 - y0
        t0, t1 = 0.0, 1.0

        for p, q in ((-dx, x0 - x_lo), (dx, x_hi - x0), (-dy, y0 - y_lo), (dy, y_hi - y0)):
            if abs(p) < 1e-12:
                if q < 0.0:
                    return None
                continue
            r = q / p
            if p < 0.0:
                if r > t1:
                    return None
                if r > t0:
                    t0 = r
            else:
                if r < t0:
                    return None
                if r < t1:
                    t1 = r

        if t1 <= t0:
            return None
        c0 = (x0 + t0 * dx, y0 + t0 * dy)
        c1 = (x0 + t1 * dx, y0 + t1 * dy)
        return (c0, c1)

    @staticmethod
    def _trim_segment_by_distance(
        p0: tuple[float, float],
        p1: tuple[float, float],
        trim_m: float,
    ) -> tuple[tuple[float, float], tuple[float, float]] | None:
        x0, y0 = float(p0[0]), float(p0[1])
        x1, y1 = float(p1[0]), float(p1[1])
        dx = x1 - x0
        dy = y1 - y0
        L = math.hypot(dx, dy)
        trim = max(0.0, float(trim_m))
        if L <= (2.0 * trim + 1e-6):
            return None
        tx = dx / L
        ty = dy / L
        return (x0 + trim * tx, y0 + trim * ty), (x1 - trim * tx, y1 - trim * ty)

    def _build_dynamic_route_segments(self) -> list[dict]:
        x_lo, x_hi, y_lo, y_hi = self._spawn_inner_bounds()
        segs: list[dict] = []
        for idx, item in enumerate(self._dynamic_route_segments_src):
            if isinstance(item, dict):
                p0 = tuple(item["p0"])
                p1 = tuple(item["p1"])
                half_width = float(item.get("half_width", self.dynamic_corridor_half_width))
                name = str(item.get("name", f"corridor_{idx}"))
            else:
                p0, p1 = item
                half_width = float(self.dynamic_corridor_half_width)
                name = f"corridor_{idx}"

            clipped = self._clip_segment_to_box(p0, p1, x_lo, x_hi, y_lo, y_hi)
            if clipped is None:
                continue
            trimmed = self._trim_segment_by_distance(clipped[0], clipped[1], self.dynamic_route_trim_m)
            if trimmed is None:
                continue
            (cx0, cy0), (cx1, cy1) = trimmed
            vx = float(cx1 - cx0)
            vy = float(cy1 - cy0)
            L = math.hypot(vx, vy)
            if L < max(1.8, 0.8 * self.min_goal_dist):
                continue
            tx = vx / L
            ty = vy / L
            nx = -ty
            ny = tx
            segs.append({
                "id": int(idx),
                "name": str(name),
                "p0": (float(cx0), float(cy0)),
                "p1": (float(cx1), float(cy1)),
                "length": float(L),
                "tx": float(tx),
                "ty": float(ty),
                "nx": float(nx),
                "ny": float(ny),
                "half_width": float(max(0.12, half_width)),
            })
        return segs

    def _dynamic_route_mode_probs(self) -> tuple[np.ndarray, list[str]]:
        if self.dynamic_route_mode != "mixed":
            return np.asarray([1.0], dtype=np.float64), [self.dynamic_route_mode]

        p_head = float(np.clip(self.dynamic_headon_prob, 0.0, 1.0))
        p_same = float(np.clip(self.dynamic_samedir_prob, 0.0, 1.0))
        p_cross = max(0.0, 1.0 - p_head - p_same)
        if p_cross <= 1e-9 and (p_head + p_same) > 1.0:
            s = p_head + p_same
            p_head /= s
            p_same /= s
            p_cross = 0.0

        probs = np.asarray([p_cross, p_head, p_same], dtype=np.float64)
        labels = ["crossing", "head_on", "same_direction"]
        total = float(np.sum(probs))
        if total <= 1e-12:
            return np.asarray([1.0], dtype=np.float64), ["crossing"]
        probs /= total
        keep = probs > 1e-9
        return probs[keep], [labels[i] for i in range(len(labels)) if keep[i]]

    def _sample_point_on_corridor(self, seg: dict, s: float, lateral: float) -> tuple[float, float]:
        x0, y0 = seg["p0"]
        return (
            float(x0 + s * seg["tx"] + lateral * seg["nx"]),
            float(y0 + s * seg["ty"] + lateral * seg["ny"]),
        )

    def _sample_dynamic_route_pair(self):
        if self.dynamic_route_prob <= 0.0 or not self._dynamic_route_segments:
            return None

        lengths = np.asarray([max(1e-6, s["length"]) for s in self._dynamic_route_segments], dtype=np.float64)
        seg_probs = lengths / np.sum(lengths)
        mode_probs, mode_labels = self._dynamic_route_mode_probs()
        yaw_noise = math.radians(self.dynamic_route_yaw_noise_deg)
        x_lo, x_hi, y_lo, y_hi = self._spawn_inner_bounds()

        for _ in range(180):
            seg = self._dynamic_route_segments[int(self._rng.choice(len(self._dynamic_route_segments), p=seg_probs))]
            route_mode = str(self._rng.choice(mode_labels, p=mode_probs))
            L = float(seg["length"])
            half_width = float(seg["half_width"])
            s_margin = max(0.25, float(self.dynamic_route_t_margin) * L)

            if route_mode in {"head_on", "same_direction"}:
                sep_lo = max(1.8, self.min_goal_dist)
                sep_hi = min(self.max_goal_dist, L - 2.0 * s_margin)
                if sep_hi <= sep_lo:
                    continue
                sep = float(self._rng.uniform(sep_lo, sep_hi))
                s_center_lo = s_margin + 0.5 * sep
                s_center_hi = L - s_margin - 0.5 * sep
                if s_center_hi <= s_center_lo:
                    continue
                s_center = float(self._rng.uniform(s_center_lo, s_center_hi))
                if self.dynamic_route_jitter > 1e-6:
                    max_j = min(self.dynamic_route_jitter, s_center - s_center_lo, s_center_hi - s_center)
                    if max_j > 1e-6:
                        s_center += float(self._rng.uniform(-max_j, max_j))
                lat0 = float(self._rng.uniform(-0.60 * half_width, 0.60 * half_width))
                lat1 = float(np.clip(lat0 + self._rng.uniform(-0.18, 0.18), -0.75 * half_width, 0.75 * half_width))

                if route_mode == "head_on":
                    s_start = s_center + 0.5 * sep
                    s_goal = s_center - 0.5 * sep
                else:
                    s_start = s_center - 0.5 * sep
                    s_goal = s_center + 0.5 * sep

                rx, ry = self._sample_point_on_corridor(seg, s_start, lat0)
                gx, gy = self._sample_point_on_corridor(seg, s_goal, lat1)
                cx, cy = self._sample_point_on_corridor(seg, s_center, 0.5 * (lat0 + lat1))

            else:  # crossing / cut-in corridor mode
                s_center = float(self._rng.uniform(s_margin, L - s_margin))
                if self.dynamic_route_jitter > 1e-6:
                    max_j = min(self.dynamic_route_jitter, s_center - s_margin, (L - s_margin) - s_center)
                    if max_j > 1e-6:
                        s_center += float(self._rng.uniform(-max_j, max_j))
                lat_span = float(self._rng.uniform(max(0.95, 2.2 * half_width), min(1.65, self.max_goal_dist)))
                forward_span = float(self._rng.uniform(0.8, min(1.9, self.max_goal_dist)))
                sign = -1.0 if self._rng.random() < 0.5 else 1.0
                start_lat = -sign * lat_span
                goal_lat = sign * lat_span
                start_s = s_center - 0.5 * forward_span
                goal_s = s_center + 0.5 * forward_span
                if start_s < 0.10 or goal_s > (L - 0.10):
                    continue
                rx, ry = self._sample_point_on_corridor(seg, start_s, start_lat)
                gx, gy = self._sample_point_on_corridor(seg, goal_s, goal_lat)
                cx, cy = self._sample_point_on_corridor(seg, s_center, 0.0)

            if not (x_lo <= rx <= x_hi and y_lo <= ry <= y_hi and x_lo <= gx <= x_hi and y_lo <= gy <= y_hi):
                continue

            total_dist = math.hypot(gx - rx, gy - ry)
            if not (self.min_goal_dist <= total_dist <= self.max_goal_dist):
                continue

            goal_line_yaw = math.atan2(gy - ry, gx - rx)
            ryaw = wrap_to_pi(goal_line_yaw + float(self._rng.uniform(-yaw_noise, yaw_noise)))

            return {
                "start_x": float(rx),
                "start_y": float(ry),
                "start_yaw": float(ryaw),
                "goal_x": float(gx),
                "goal_y": float(gy),
                "route_mode": str(route_mode),
                "route_segment_id": int(seg["id"]),
                "route_segment_name": str(seg["name"]),
                "route_cross_x": float(cx),
                "route_cross_y": float(cy),
                "route_length": float(total_dist),
            }

        return None

    def _sample_start_pose(self):
        m = float(self.robot_spawn_margin)
        m = min(m, max(0.0, 0.5 * (self.x_max - self.x_min) - 0.05))
        m = min(m, max(0.0, 0.5 * (self.y_max - self.y_min) - 0.05))

        x = self._rng.uniform(self.x_min + m, self.x_max - m)
        y = self._rng.uniform(self.y_min + m, self.y_max - m)
        yaw = self._rng.uniform(-math.pi, math.pi)
        return float(x), float(y), float(yaw)

    def _sample_goal_pose(self, rx, ry):
        m = float(self.goal_spawn_margin)
        m = min(m, max(0.0, 0.5 * (self.x_max - self.x_min) - 0.05))
        m = min(m, max(0.0, 0.5 * (self.y_max - self.y_min) - 0.05))

        for _ in range(200):
            r = self._rng.uniform(self.min_goal_dist, self.max_goal_dist)
            ang = self._rng.uniform(-math.pi, math.pi)
            gx = rx + r * math.cos(ang)
            gy = ry + r * math.sin(ang)
            if (self.x_min + m) <= gx <= (self.x_max - m) and (self.y_min + m) <= gy <= (self.y_max - m):
                return float(gx), float(gy)

        gx = float(np.clip(rx + self.max_goal_dist, self.x_min + m, self.x_max - m))
        gy = float(np.clip(ry, self.y_min + m, self.y_max - m))
        return gx, gy

    def _sample_target_yaw_world(self, sx: float, sy: float, gx: float, gy: float):
        goal_line_yaw = math.atan2(gy - sy, gx - sx)
        use_sector = bool(
            (self.goal_yaw_sector_prob > 0.0)
            and (self._rng.random() < self.goal_yaw_sector_prob)
        )

        if use_sector:
            delta = float(self._rng.uniform(-self.goal_yaw_sector_half_angle, self.goal_yaw_sector_half_angle))
            target_yaw = wrap_to_pi(goal_line_yaw + delta)
            mode = "sector"
        else:
            target_yaw = float(self._rng.uniform(-math.pi, math.pi))
            mode = "random"

        return float(target_yaw), float(goal_line_yaw), str(mode)

    def _world_to_robot_frame(self, rx, ry, ryaw, gx, gy):
        dx = gx - rx
        dy = gy - ry
        c = math.cos(-ryaw)
        s = math.sin(-ryaw)
        x_rel = c * dx - s * dy
        y_rel = s * dx + c * dy
        return x_rel, y_rel

    def _compute_dist(self):
        pose = self.ros.get_robot_pose2d()
        if pose is None:
            return None
        rx, ry, _ = pose
        dx = float(self._goal_world[0]) - rx
        dy = float(self._goal_world[1]) - ry
        return math.sqrt(dx * dx + dy * dy)

    def _compute_yaw_err(self):
        pose = self.ros.get_robot_pose2d()
        if pose is None:
            return None
        _, _, ryaw = pose
        return wrap_to_pi(self._target_yaw_world - ryaw)

    def _out_of_bounds(self):
        pose = self.ros.get_robot_pose2d()
        if pose is None:
            return False
        rx, ry, _ = pose

        mode = str(self.oob_mode).lower().strip()
        world_box = (rx < self.x_min) or (rx > self.x_max) or (ry < self.y_min) or (ry > self.y_max)

        dx = float(rx) - float(self._start_world[0])
        dy = float(ry) - float(self._start_world[1])
        start_circle = (dx * dx + dy * dy) > (self.oob_radius * self.oob_radius)

        if mode == "start_circle":
            return bool(start_circle)
        if mode == "both":
            return bool(world_box or start_circle)
        return bool(world_box)

    def _collision_signal(self, ranges: np.ndarray) -> tuple[bool, float, int]:
        if ranges is None:
            return False, float("inf"), 0

        r = np.asarray(ranges, dtype=np.float32)
        if r.size == 0:
            return False, float("inf"), 0

        min_r = float(np.min(r))
        hits = int(np.sum(r < self.collision_threshold))
        signal = bool((min_r < self.collision_threshold) and (hits >= self.collision_min_hits))
        return signal, min_r, hits

    @staticmethod
    def _kth_smallest(x: np.ndarray, k: int) -> float:
        if x is None:
            return float("inf")
        a = np.asarray(x, dtype=np.float32)
        if a.size == 0:
            return float("inf")
        kk = int(max(0, min(int(k), int(a.size) - 1)))
        return float(np.partition(a, kk)[kk])

    def _denorm_action(self, a: np.ndarray):
        a = np.clip(a, -1.0, 1.0).astype(np.float32)
        vx = (a[0] + 1.0) * 0.5 * (self.vx_max - self.vx_min) + self.vx_min
        vy = a[1] * self.vy_max
        wz = a[2] * self.wz_max
        return float(vx), float(vy), float(wz)

    def _get_robot_velocity(self):
        odom = getattr(self.ros, "_odom", None)
        if odom is None:
            return 0.0, 0.0, 0.0
        twist = odom.twist.twist
        return float(twist.linear.x), float(twist.linear.y), float(twist.angular.z)

    def _normalize_robot_velocity(self, vx: float, vy: float, wz: float):
        if vx >= 0.0:
            vx_den = max(abs(self.vx_max), 1e-6)
        else:
            vx_den = max(abs(self.vx_min), 1e-6)
        vx_n = float(np.clip(vx / vx_den, -1.0, 1.0))
        vy_n = float(np.clip(vy / max(abs(self.vy_max), 1e-6), -1.0, 1.0))
        wz_n = float(np.clip(wz / max(abs(self.wz_max), 1e-6), -1.0, 1.0))
        return vx_n, vy_n, wz_n

    def _get_current_laser_norm(self) -> np.ndarray:
        ranges, rmax = self.ros.get_laser_downsample()
        return np.clip(ranges / max(rmax, 1e-6), 0.0, 1.0).astype(np.float32)

    @staticmethod
    def _clone_obs(obs):
        if obs is None:
            return None
        return {k: np.array(v, copy=True) for k, v in obs.items()}

    def _get_sensor_status(self):
        scan_status = self.ros.get_scan_status(max_age_sec=self.sensor_freshness_sec)
        odom_status = self.ros.get_odom_status(max_age_sec=self.sensor_freshness_sec)
        sensor_ok = bool(scan_status.get("fresh", False) and odom_status.get("fresh", False))
        return sensor_ok, scan_status, odom_status

    def _prime_lidar_history(self):
        self._lidar_history.clear()
        laser = self._get_current_laser_norm()
        self._lidar_history.append(laser.copy())
        if self._lidar_raw_history_len <= 1:
            return
        for _ in range(1, self._lidar_raw_history_len):
            self.ros.spin_for(self.control_dt, sleep_step=min(0.01, max(0.002, 0.25 * self.control_dt)))
            self._lidar_history.append(self._get_current_laser_norm().copy())

    def _append_lidar_history(self):
        laser = self._get_current_laser_norm()
        self._lidar_history.append(laser.copy())

    def _stack_lidar_history(self) -> np.ndarray:
        if len(self._lidar_history) <= 0:
            self._prime_lidar_history()
        while len(self._lidar_history) < self._lidar_raw_history_len:
            self._lidar_history.appendleft(self._lidar_history[0].copy())

        hist_list = list(self._lidar_history)
        sampled = []
        for i in range(self._lidar_history_steps):
            idx = i * self._lidar_history_stride
            sampled.append(hist_list[idx])
        return np.concatenate(sampled, axis=0).astype(np.float32)

    def _get_obs(self):
        pose = self.ros.get_robot_pose2d()
        if pose is None:
            agent = np.zeros((4,), dtype=np.float32)
        else:
            rx, ry, ryaw = pose
            x_rel, y_rel = self._world_to_robot_frame(
                rx, ry, ryaw, float(self._goal_world[0]), float(self._goal_world[1])
            )
            x_rel_n = np.clip(x_rel / self.max_goal_dist, -1.0, 1.0)
            y_rel_n = np.clip(y_rel / self.max_goal_dist, -1.0, 1.0)
            theta = math.atan2(y_rel, x_rel)
            theta_n = float(np.clip(theta / math.pi, -1.0, 1.0))

            yaw_err = self._compute_yaw_err()
            if yaw_err is None:
                yaw_err = 0.0
            yaw_err_n = float(np.clip(yaw_err / math.pi, -1.0, 1.0))

            agent = np.array([x_rel_n, y_rel_n, theta_n, yaw_err_n], dtype=np.float32)

        laser_hist = self._stack_lidar_history()
        return {"agent": agent, "laser": laser_hist}

    # ---------------------------
    # gym API
    # ---------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._step_count = 0
        self._collision_streak = 0
        self._prev_dist = None
        self._prev_yaw_err = None
        self._prev_action = None
        self._was_in_goal_zone = False
        self._last_obs = None

        self.ros.stop()

        if not self.ros.wait_for_topics(timeout_sec=self.topic_timeout_sec):
            raise RuntimeError(
                f"[Stage3Env] No odom/scan received within {self.topic_timeout_sec:.1f}s. "
                "Please verify /odom and /scan topics before training."
            )

        chosen = None
        last_reason = "ok"
        reset_route_mode = "uniform"
        route_segment_id = -1
        route_segment_name = "uniform"
        route_cross_x = float("nan")
        route_cross_y = float("nan")

        for attempt in range(self.max_reset_tries):
            scan_cnt0 = int(getattr(self.ros, "_scan_cb_count", 0))
            odom_cnt0 = int(getattr(self.ros, "_odom_cb_count", 0))

            route_sample = None
            if self.dynamic_route_prob > 0.0 and (self._rng.random() < self.dynamic_route_prob):
                route_sample = self._sample_dynamic_route_pair()

            if route_sample is None:
                rx, ry, ryaw = self._sample_start_pose()
                gx, gy = self._sample_goal_pose(rx, ry)
                reset_route_mode = "uniform"
                route_segment_id = -1
                route_cross_x = float("nan")
                route_cross_y = float("nan")
                route_segment_name = "uniform"
            else:
                rx = float(route_sample["start_x"])
                ry = float(route_sample["start_y"])
                ryaw = float(route_sample["start_yaw"])
                gx = float(route_sample["goal_x"])
                gy = float(route_sample["goal_y"])
                reset_route_mode = str(route_sample["route_mode"])
                route_segment_id = int(route_sample["route_segment_id"])
                route_cross_x = float(route_sample["route_cross_x"])
                route_cross_y = float(route_sample["route_cross_y"])
                route_segment_name = str(route_sample.get("route_segment_name", "unknown"))

            self.ros.set_entity_state(self.robot_name, rx, ry, yaw=ryaw, z=0.0)

            self.ros.spin_for(0.08)
            got_scan = self.ros.wait_for_new_scan(scan_cnt0, timeout_sec=1.0)
            got_odom = self.ros.wait_for_new_odom(odom_cnt0, timeout_sec=1.0)
            if not got_scan:
                last_reason = "no_scan_update_after_teleport"
                continue
            if not got_odom:
                last_reason = "no_odom_update_after_teleport"
                continue

            ranges0, _ = self.ros.get_laser_downsample()
            min_r0 = self._kth_smallest(ranges0, 1)
            if min_r0 < self.spawn_min_clearance:
                last_reason = f"start_clearance<{self.spawn_min_clearance:.2f}"
                continue

            target_yaw_world, goal_line_yaw, goal_yaw_mode = self._sample_target_yaw_world(rx, ry, gx, gy)

            self._start_world[:] = [float(rx), float(ry)]
            self._goal_world[:] = [float(gx), float(gy)]
            self._target_yaw_world = float(target_yaw_world)

            self.ros.set_entity_state(
                self.target_name,
                float(gx),
                float(gy),
                yaw=self._target_yaw_world,
                z=0.05,
            )

            scan_cnt1 = int(getattr(self.ros, "_scan_cb_count", 0))
            self.ros.spin_for(0.20)
            got_scan2 = self.ros.wait_for_new_scan(scan_cnt1, timeout_sec=1.5)
            if not got_scan2:
                last_reason = "no_scan_update_after_goal"
                continue

            ranges1, _ = self.ros.get_laser_downsample()
            min_r1 = self._kth_smallest(ranges1, 1)
            if min_r1 < self.collision_threshold:
                last_reason = "immediate_too_close_after_goal"
                continue

            chosen = (
                rx,
                ry,
                ryaw,
                gx,
                gy,
                goal_line_yaw,
                goal_yaw_mode,
                attempt + 1,
                last_reason,
                min_r0,
                min_r1,
            )
            break

        if chosen is None:
            rx, ry, ryaw = 0.0, 0.0, 0.0
            self.ros.set_entity_state(self.robot_name, rx, ry, yaw=ryaw, z=0.0)
            gx, gy = self._sample_goal_pose(rx, ry)
            self._start_world[:] = [float(rx), float(ry)]
            self._goal_world[:] = [float(gx), float(gy)]
            self._target_yaw_world, goal_line_yaw, goal_yaw_mode = self._sample_target_yaw_world(rx, ry, gx, gy)
            self.ros.set_entity_state(self.target_name, gx, gy, yaw=self._target_yaw_world, z=0.05)
            self.ros.spin_for(0.25)
            ranges_f, _ = self.ros.get_laser_downsample()
            min_r0 = self._kth_smallest(ranges_f, 1)
            min_r1 = min_r0
            reset_attempts = self.max_reset_tries
            last_reason = "fallback_center"
            reset_route_mode = "fallback_uniform"
            route_segment_id = -1
            route_segment_name = "fallback_center"
            route_cross_x = float("nan")
            route_cross_y = float("nan")
        else:
            (
                rx,
                ry,
                ryaw,
                gx,
                gy,
                goal_line_yaw,
                goal_yaw_mode,
                reset_attempts,
                last_reason,
                min_r0,
                min_r1,
            ) = chosen

        dist = self._compute_dist()
        self._prev_dist = float(dist) if dist is not None else 999.0

        yaw_err = self._compute_yaw_err()
        self._prev_yaw_err = float(yaw_err) if yaw_err is not None else math.pi

        sensor_ok, scan_status, odom_status = self._get_sensor_status()
        if not sensor_ok:
            raise RuntimeError(
                "[Stage3Env] Sensor data is not fresh after reset. "
                f"scan_fresh={scan_status.get('fresh', False)}, odom_fresh={odom_status.get('fresh', False)}, "
                f"max_age={self.sensor_freshness_sec:.3f}s"
            )

        self._prime_lidar_history()
        obs = self._get_obs()
        self._last_obs = self._clone_obs(obs)

        info = {
            "start_x": float(rx),
            "start_y": float(ry),
            "goal_x": float(gx),
            "goal_y": float(gy),
            "oob_mode": str(self.oob_mode),
            "oob_radius": float(self.oob_radius),
            "goal_yaw": float(self._target_yaw_world),
            "goal_line_yaw": float(goal_line_yaw),
            "reset_route_mode": str(reset_route_mode),
            "route_segment_id": int(route_segment_id),
            "route_cross_x": float(route_cross_x),
            "route_cross_y": float(route_cross_y),
            "route_segment_name": str(route_segment_name),
            "dynamic_route_prob": float(self.dynamic_route_prob),
            "dynamic_route_mode": str(self.dynamic_route_mode),
            "dynamic_headon_prob": float(self.dynamic_headon_prob),
            "dynamic_samedir_prob": float(self.dynamic_samedir_prob),
            "dynamic_route_segment_count": int(len(self._dynamic_route_segments)),
            "dynamic_route_trim_m": float(self.dynamic_route_trim_m),
            "goal_yaw_mode": str(goal_yaw_mode),
            "goal_yaw_sector_prob": float(self.goal_yaw_sector_prob),
            "goal_yaw_sector_half_angle_deg": float(self.goal_yaw_sector_half_angle_deg),
            "yaw_err": float(self._prev_yaw_err),
            "reset_attempts": int(reset_attempts),
            "reset_reason": str(last_reason),
            "reset_min_laser_before": float(min_r0),
            "reset_min_laser_after": float(min_r1),
            "sensor_freshness_sec": float(self.sensor_freshness_sec),
            "scan_fresh": int(bool(scan_status.get("fresh", False))),
            "odom_fresh": int(bool(odom_status.get("fresh", False))),
            "scan_age_sec": float(scan_status.get("age_sec", float("inf"))),
            "odom_age_sec": float(odom_status.get("age_sec", float("inf"))),
            "lidar_history_steps": int(self._lidar_history_steps),
            "lidar_history_stride": int(self._lidar_history_stride),
            "lidar_history_span_sec": float((self._lidar_history_steps - 1) * self._lidar_history_stride * self.control_dt),
            "laser_downsample_n": int(self._laser_n),
            "control_dt": float(self.control_dt),
        }
        return obs, info

    def step(self, action):
        self._step_count += 1

        cmd_vx, cmd_vy, cmd_wz = self._denorm_action(np.asarray(action, dtype=np.float32))

        scan_cnt0 = int(getattr(self.ros, "_scan_cb_count", 0))
        odom_cnt0 = int(getattr(self.ros, "_odom_cb_count", 0))

        self.ros.publish_cmd(cmd_vx, cmd_vy, cmd_wz)

        step_timeout = max(0.35, 4.0 * float(self.control_dt))
        self.ros.spin_for(
            self.control_dt,
            sleep_step=min(0.01, max(0.002, 0.25 * float(self.control_dt))),
            wall_timeout_sec=5.0,
        )
        got_scan = int(getattr(self.ros, "_scan_cb_count", 0)) > scan_cnt0
        got_odom = int(getattr(self.ros, "_odom_cb_count", 0)) > odom_cnt0
        if not (got_scan and got_odom):
            got_scan2, got_odom2 = self.ros.wait_for_new_data(
                prev_scan_count=scan_cnt0,
                prev_odom_count=odom_cnt0,
                timeout_sec=max(0.05, step_timeout - float(self.control_dt)),
                wall_timeout_sec=5.0,
                require_scan=not got_scan,
                require_odom=not got_odom,
            )
            got_scan = bool(got_scan or got_scan2)
            got_odom = bool(got_odom or got_odom2)

        sensor_ok, scan_status, odom_status = self._get_sensor_status()
        if not (got_scan and got_odom and sensor_ok):
            reward = float(self.step_penalty) - float(self.sensor_fail_penalty)
            termination_reason = "sensor_update_timeout"
            if got_scan and got_odom and (not sensor_ok):
                termination_reason = "sensor_stale"

            self.ros.stop(repeat=2, dt=min(0.02, max(0.01, 0.5 * float(self.control_dt))))
            obs = self._clone_obs(self._last_obs)
            if obs is None:
                self._prime_lidar_history()
                obs = self._get_obs()
                self._last_obs = self._clone_obs(obs)

            info = {
                "dist_to_goal": float(self._prev_dist) if self._prev_dist is not None else 999.0,
                "dist_progress": 0.0,
                "dist_from_start": float("inf"),
                "oob_mode": str(self.oob_mode),
                "oob_radius": float(self.oob_radius),
                "step": int(self._step_count),
                "vx": float(cmd_vx),
                "vy": float(cmd_vy),
                "wz": float(cmd_wz),
                "odom_vx": 0.0,
                "odom_vy": 0.0,
                "odom_wz": 0.0,
                "min_laser": float("inf"),
                "min_laser_k1": float("inf"),
                "collision_signal": 0,
                "collision_hits": 0,
                "collision_streak": int(self._collision_streak),
                "collision_confirm_steps": int(self.collision_confirm_steps),
                "yaw_err": float(self._prev_yaw_err) if self._prev_yaw_err is not None else math.pi,
                "yaw_progress": 0.0,
                "r_step_penalty": float(self.step_penalty),
                "r_step_delta": 0.0,
                "r_sensor_fail": -float(self.sensor_fail_penalty),
                "r_clearance_linear": 0.0,
                "r_clearance_square": 0.0,
                "r_speed_near": 0.0,
                "r_idle_penalty": 0.0,
                "r_smooth_penalty": 0.0,
                "r_timeout_penalty": 0.0,
                "r_yaw_reward": 0.0,
                "r_ring_bonus": 0.0,
                "r_yaw_abs": 0.0,
                "r_inside_goal": 0.0,
                "r_exit_goal": 0.0,
                "is_success": 0,
                "termination_reason": str(termination_reason),
                "is_collision": 0,
                "sensor_freshness_sec": float(self.sensor_freshness_sec),
                "got_scan_update": int(bool(got_scan)),
                "got_odom_update": int(bool(got_odom)),
                "scan_fresh": int(bool(scan_status.get("fresh", False))),
                "odom_fresh": int(bool(odom_status.get("fresh", False))),
                "scan_age_sec": float(scan_status.get("age_sec", float("inf"))),
                "odom_age_sec": float(odom_status.get("age_sec", float("inf"))),
                "lidar_history_steps": int(self._lidar_history_steps),
                "lidar_history_stride": int(self._lidar_history_stride),
                "lidar_history_span_sec": float((self._lidar_history_steps - 1) * self._lidar_history_stride * self.control_dt),
                "laser_downsample_n": int(self._laser_n),
                "control_dt": float(self.control_dt),
            }
            print(
                f"[RESET] reason={termination_reason}, step={self._step_count}, "
                f"got_scan={got_scan}, got_odom={got_odom}, "
                f"scan_fresh={scan_status.get('fresh', False)}, odom_fresh={odom_status.get('fresh', False)}"
            )
            return obs, float(reward), False, True, info

        self._append_lidar_history()
        obs = self._get_obs()
        self._last_obs = self._clone_obs(obs)

        dist = self._compute_dist()
        if dist is None:
            dist = float(self._prev_dist) if self._prev_dist is not None else 999.0

        prev_dist = float(self._prev_dist) if self._prev_dist is not None else float(dist)
        progress = float(prev_dist - dist)
        self._prev_dist = float(dist)

        reward = 0.0
        if progress > 0.0:
            reward += 2.0 * progress
        else:
            reward += 0.2 * progress

        reward += float(self.step_penalty)
        r_step_penalty = float(self.step_penalty)
        r_step_delta = 0.0
        r_sensor_fail = 0.0

        reward -= 0.003 * (cmd_vx * cmd_vx + cmd_vy * cmd_vy + cmd_wz * cmd_wz)

        ranges, _ = self.ros.get_laser_downsample()
        collision_signal, min_r, collision_hits = self._collision_signal(ranges)
        min_r_k1 = self._kth_smallest(ranges, 1)

        thr = float(self.near_obstacle_threshold)
        closeness = 0.0
        if min_r_k1 < thr and thr > 1e-6:
            closeness = float((thr - min_r_k1) / thr)
            lin = float(self.near_obstacle_penalty) * (thr - min_r_k1)
            sq = float(self.near_obstacle_penalty) * (closeness ** 2)
            if self.clearance_square:
                clearance_pen = float(self.clearance_linear_weight) * lin + float(self.clearance_square_weight) * sq
            else:
                clearance_pen = lin
            reward -= clearance_pen
            r_clearance_linear = -float(float(self.clearance_linear_weight) * lin if self.clearance_square else lin)
            r_clearance_square = -float(float(self.clearance_square_weight) * sq if self.clearance_square else 0.0)
        else:
            r_clearance_linear = 0.0
            r_clearance_square = 0.0

        if collision_signal:
            self._collision_streak += 1
        else:
            self._collision_streak = 0
        collision_now = bool(self._collision_streak >= self.collision_confirm_steps)

        terminated = False
        truncated = False
        termination_reason = "running"
        is_success = 0

        pose_now = self.ros.get_robot_pose2d()
        if pose_now is None:
            dist_from_start = float("inf")
        else:
            rx_now, ry_now, _ = pose_now
            dxs = float(rx_now) - float(self._start_world[0])
            dys = float(ry_now) - float(self._start_world[1])
            dist_from_start = math.sqrt(dxs * dxs + dys * dys)

        if self._out_of_bounds():
            terminated = True
            reward -= 10.0
            termination_reason = "out_of_bounds"

        if collision_now:
            terminated = True
            reward -= 15.0
            termination_reason = "collision"

        if self._step_count >= self.max_steps:
            truncated = True
            if termination_reason == "running":
                termination_reason = "timeout"

        near_goal_r = float(self.goal_threshold) * float(self.near_goal_mult)
        v_xy_cmd = math.sqrt(cmd_vx * cmd_vx + cmd_vy * cmd_vy)
        if closeness > 0.0 and not (terminated or truncated):
            speed_near_pen = float(self.speed_near_obstacle_penalty) * float(v_xy_cmd) * float(closeness)
            reward -= speed_near_pen
            r_speed_near = -float(speed_near_pen)
        else:
            r_speed_near = 0.0

        if (dist > near_goal_r) and (v_xy_cmd < float(self.idle_speed_thresh)) and not (terminated or truncated):
            reward -= float(self.idle_penalty)
            r_idle_penalty = -float(self.idle_penalty)
        else:
            r_idle_penalty = 0.0

        try:
            a = np.asarray(action, dtype=np.float32).reshape(-1)
            if self._prev_action is not None:
                da = a - np.asarray(self._prev_action, dtype=np.float32).reshape(-1)
                smooth_pen = float(self.action_smooth_penalty) * float(np.sum(da * da))
                reward -= smooth_pen
                r_smooth_penalty = -float(smooth_pen)
            else:
                r_smooth_penalty = 0.0
            self._prev_action = a.copy()
        except Exception:
            r_smooth_penalty = 0.0

        if truncated and termination_reason == "timeout":
            reward -= float(self.timeout_penalty)
            r_timeout_penalty = -float(self.timeout_penalty)
        else:
            r_timeout_penalty = 0.0

        yaw_err = self._compute_yaw_err()
        if yaw_err is None:
            yaw_err = float(self._prev_yaw_err) if self._prev_yaw_err is not None else math.pi

        prev_yaw_err = float(self._prev_yaw_err) if self._prev_yaw_err is not None else float(yaw_err)
        yaw_progress = abs(prev_yaw_err) - abs(float(yaw_err))
        self._prev_yaw_err = float(yaw_err)

        yaw_reward = 1.5 * float(yaw_progress)
        reward += float(yaw_reward)

        if not (terminated or truncated):
            in_goal = bool(dist < float(self.goal_threshold))
            if dist < near_goal_r:
                if not in_goal:
                    reward += float(self.ring_bonus)
                    r_ring_bonus = float(self.ring_bonus)
                else:
                    r_ring_bonus = 0.0

                if in_goal:
                    yaw_abs_term = float(self.yaw_abs_penalty) * abs(float(yaw_err))
                    reward -= yaw_abs_term
                    r_yaw_abs = -float(yaw_abs_term)
                    reward += float(self.inside_goal_bonus)
                    r_inside_goal = float(self.inside_goal_bonus)
                else:
                    r_yaw_abs = 0.0
                    r_inside_goal = 0.0

                if bool(self._was_in_goal_zone) and (not in_goal):
                    reward -= float(self.exit_goal_penalty)
                    r_exit_goal = -float(self.exit_goal_penalty)
                else:
                    r_exit_goal = 0.0

                self._was_in_goal_zone = in_goal
            else:
                r_ring_bonus = 0.0
                r_yaw_abs = 0.0
                r_inside_goal = 0.0
                r_exit_goal = 0.0
                self._was_in_goal_zone = bool(dist < float(self.goal_threshold))
        else:
            r_ring_bonus = 0.0
            r_yaw_abs = 0.0
            r_inside_goal = 0.0
            r_exit_goal = 0.0

        if not (terminated or truncated) and dist < self.goal_threshold:
            if abs(float(yaw_err)) <= self.yaw_threshold:
                terminated = True
                reward += 20.0
                termination_reason = "perfect_success"
                is_success = 1

        if terminated or truncated:
            self.ros.stop()
            msg = (
                f"[RESET] reason={termination_reason}, step={self._step_count}, "
                f"dist={dist:.2f}, min_laser={min_r:.2f}, yaw_err={float(yaw_err):.2f}, "
                f"terminated={terminated}, truncated={truncated}"
            )
            print(msg)

        odom_vx, odom_vy, odom_wz = self._get_robot_velocity()

        info = {
            "dist_to_goal": float(dist),
            "dist_progress": float(progress),
            "dist_from_start": float(dist_from_start),
            "oob_mode": str(self.oob_mode),
            "oob_radius": float(self.oob_radius),
            "step": int(self._step_count),
            "vx": float(cmd_vx),
            "vy": float(cmd_vy),
            "wz": float(cmd_wz),
            "odom_vx": float(odom_vx),
            "odom_vy": float(odom_vy),
            "odom_wz": float(odom_wz),
            "min_laser": float(min_r),
            "min_laser_k1": float(min_r_k1),
            "collision_signal": int(bool(collision_signal)),
            "collision_hits": int(collision_hits),
            "collision_streak": int(self._collision_streak),
            "collision_confirm_steps": int(self.collision_confirm_steps),
            "yaw_err": float(yaw_err),
            "yaw_progress": float(yaw_progress),
            "r_step_penalty": float(r_step_penalty),
            "r_step_delta": float(r_step_delta),
            "r_sensor_fail": float(r_sensor_fail),
            "r_clearance_linear": float(r_clearance_linear),
            "r_clearance_square": float(r_clearance_square),
            "r_speed_near": float(r_speed_near),
            "r_idle_penalty": float(r_idle_penalty),
            "r_smooth_penalty": float(r_smooth_penalty),
            "r_timeout_penalty": float(r_timeout_penalty),
            "r_yaw_reward": float(yaw_reward),
            "r_ring_bonus": float(r_ring_bonus),
            "r_yaw_abs": float(r_yaw_abs),
            "r_inside_goal": float(r_inside_goal),
            "r_exit_goal": float(r_exit_goal),
            "is_success": int(is_success),
            "termination_reason": str(termination_reason),
            "is_collision": int(termination_reason == "collision"),
            "sensor_freshness_sec": float(self.sensor_freshness_sec),
            "got_scan_update": int(bool(got_scan)),
            "got_odom_update": int(bool(got_odom)),
            "scan_fresh": int(bool(scan_status.get("fresh", False))),
            "odom_fresh": int(bool(odom_status.get("fresh", False))),
            "scan_age_sec": float(scan_status.get("age_sec", float("inf"))),
            "odom_age_sec": float(odom_status.get("age_sec", float("inf"))),
            "lidar_history_steps": int(self._lidar_history_steps),
            "lidar_history_stride": int(self._lidar_history_stride),
            "lidar_history_span_sec": float((self._lidar_history_steps - 1) * self._lidar_history_stride * self.control_dt),
            "laser_downsample_n": int(self._laser_n),
            "control_dt": float(self.control_dt),
        }
        return obs, float(reward), terminated, truncated, info

    def close(self):
        try:
            self.ros.stop()
            self.ros.destroy_node()
        except Exception:
            pass
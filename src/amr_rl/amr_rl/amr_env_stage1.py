import math
from dataclasses import dataclass

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from amr_rl.amr_ros_interface import AmrRosInterface


@dataclass(frozen=True)
class ObstacleSpec:
    """Gazebo model 名稱與 sampling 用的近似半徑。"""
    name: str
    radius: float = 0.35


class AmrNavStage1Env(gym.Env):
    """
    Stage 1（避障版）：
    - square.world（同 Stage0）
    - 另外在 Gazebo 先 spawn 一組障礙物模型（例如 Obstacle_0..Obstacle_7）
    - reset 時用 /set_entity_state 隨機擺放障礙物
    - reward 增加「靠近障礙」的連續懲罰，讓 PPO 學會保持 clearance

    重要：
    - 這個環境仍然是「短距離子目標」（local planner/controller 訓練）
    - 全域繞路交給 A*，後續再把 goal 換成 A* path lookahead waypoint
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        robot_name: str = "amr",
        target_name: str = "Target",
        obstacle_names: list[str] | None = None,
        cmd_vel_topic: str = "cmd_vel",
        odom_topic: str = "odom",
        scan_topic: str = "scan",
        set_entity_service: str = "/set_entity_state",
        use_sim_time: bool = True,
        control_dt: float = 0.05,
        max_steps: int = 240,
        x_lim: float | None = 3.5,
        y_lim: float | None = 3.5,
        x_min: float | None = None,
        x_max: float | None = None,
        y_min: float | None = None,
        y_max: float | None = None,
        # --- out-of-bounds logic ---
        # world_box: abs(x)>x_lim OR abs(y)>y_lim (legacy)
        # start_circle: ||p - p_start|| > oob_radius
        # both: world_box OR start_circle
        oob_mode: str = "world_box",
        oob_radius: float = 3.0,
        min_goal_dist: float = 1.3,
        max_goal_dist: float = 2.5,
        goal_threshold: float = 0.35,
        collision_threshold: float = 0.57,
        # collision robustness (reduce false positives from single-beam LiDAR spikes)
        collision_confirm_steps: int = 2,         # require N consecutive collision signals to terminate
        collision_min_hits: int = 2,              # require >=K downsample buckets below threshold (1 keeps thin obstacles)
        laser_downsample_n: int = 120,
        # action limits (mecanum)
        vx_min: float = -0.3,
        vx_max: float = 0.6,
        vy_max: float = 0.4,
        wz_max: float = 1.0,
        # obstacle randomization
        min_obstacles: int = 3,
        max_obstacles: int = 6,
        obstacle_spawn_margin: float = 0.85,
        keepout_robot: float = 0.9,
        keepout_goal: float = 0.9,
        keepout_between: float = 1.7,
        # corridor obstacles (force more avoidance experiences)
        corridor_prob: float = 0.80,
        corridor_half_width: float = 0.60,
        corridor_min_obstacles: int = 1,
        corridor_max_obstacles: int = 3,
        corridor_t_min: float = 0.20,
        corridor_t_max: float = 0.80,
        corridor_max_tries: int = 400,
        obstacle_hide_xy: float = 50.0,
        # reward shaping for clearance
        near_obstacle_threshold: float = 1.0,
        near_obstacle_penalty: float = 0.8,
        step_penalty: float = -0.05,
        # --- reset safety (避免剛開始翻車) ---
        robot_spawn_margin: float = 0.25,         # robot 起始位置離牆至少留這個 margin
        goal_spawn_margin: float = 0.25,          # goal 生成點離牆至少留這個 margin
        spawn_min_clearance: float = 0.5,        # reset 後 min_laser 低於此值就重抽起始點（避免貼牆/角落）
        max_reset_tries: int = 25,                # reset 最多嘗試次數（避免卡死）
        # --- sensor health / debugging ---
        topic_timeout_sec: float = 6.0,            # reset 時等待 odom/scan 的最長時間（秒）
        require_scan: bool = True,                 # reset/step 是否要求 scan 一定要存在（否則直接中止/報錯）
        scan_max_age_sec: float = 0.75,            # step 時 scan 最長允許「沒有更新」的時間（牆鐘秒數）
        terminate_on_lidar_stale: bool = False,    # 若 scan 長時間不更新，是否直接 truncate 這個 episode
        seed: int | None = None,
    ):
        super().__init__()

        self.robot_name = robot_name
        self.target_name = target_name

        # obstacles
        if obstacle_names is None:
            obstacle_names = [f"Obstacle_{i}" for i in range(8)]
        self.obstacles = [ObstacleSpec(name=n) for n in obstacle_names]

        self.min_obstacles = int(min_obstacles)
        self.max_obstacles = int(max_obstacles)
        if self.max_obstacles > len(self.obstacles):
            self.max_obstacles = len(self.obstacles)
        if self.min_obstacles < 0:
            self.min_obstacles = 0
        if self.min_obstacles > self.max_obstacles:
            self.min_obstacles = self.max_obstacles

        self.obstacle_spawn_margin = float(obstacle_spawn_margin)
        self.keepout_robot = float(keepout_robot)
        self.keepout_goal = float(keepout_goal)
        self.keepout_between = float(keepout_between)

        # corridor placement settings
        self.corridor_prob = float(np.clip(corridor_prob, 0.0, 1.0))
        self.corridor_half_width = float(max(0.05, corridor_half_width))
        self.corridor_min_obstacles = int(max(0, corridor_min_obstacles))
        self.corridor_max_obstacles = int(max(0, corridor_max_obstacles))
        if self.corridor_max_obstacles < self.corridor_min_obstacles:
            self.corridor_max_obstacles = self.corridor_min_obstacles
        # t in [0,1] along (robot->goal) segment
        self.corridor_t_min = float(np.clip(corridor_t_min, 0.0, 1.0))
        self.corridor_t_max = float(np.clip(corridor_t_max, 0.0, 1.0))
        if self.corridor_t_max < self.corridor_t_min:
            self.corridor_t_max, self.corridor_t_min = self.corridor_t_min, self.corridor_t_max
        self.corridor_max_tries = int(max(50, corridor_max_tries))

        # last episode corridor stats (for logging)
        self._last_corridor_used = False
        self._last_corridor_n = 0

        self.obstacle_hide_xy = float(obstacle_hide_xy)

        self.near_obstacle_threshold = float(near_obstacle_threshold)
        self.near_obstacle_penalty = float(near_obstacle_penalty)
        self.step_penalty = float(step_penalty)

        self.control_dt = float(control_dt)
        self.max_steps = int(max_steps)

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
        # Legacy aliases kept for compatibility with older code/logging.
        self.x_lim = max(abs(self.x_min), abs(self.x_max))
        self.y_lim = max(abs(self.y_min), abs(self.y_max))

        self.oob_mode = str(oob_mode)
        self.oob_radius = float(oob_radius)

        self.min_goal_dist = float(min_goal_dist)
        self.max_goal_dist = float(max_goal_dist)
        self.goal_threshold = float(goal_threshold)
        self.collision_threshold = float(collision_threshold)

        # collision robustness
        self.collision_confirm_steps = max(1, int(collision_confirm_steps))
        self.collision_min_hits = max(1, int(collision_min_hits))
        self._collision_streak = 0

        self.vx_min = float(vx_min)
        self.vx_max = float(vx_max)
        self.vy_max = float(vy_max)
        self.wz_max = float(wz_max)

        # reset safety
        self.robot_spawn_margin = float(robot_spawn_margin)
        self.goal_spawn_margin = float(goal_spawn_margin)
        self.spawn_min_clearance = float(spawn_min_clearance)
        self.max_reset_tries = int(max_reset_tries)

        # sensor/debug
        self.topic_timeout_sec = float(topic_timeout_sec)
        self.require_scan = bool(require_scan)
        self.scan_max_age_sec = float(scan_max_age_sec)
        self.terminate_on_lidar_stale = bool(terminate_on_lidar_stale)

        self._rng = np.random.default_rng(seed)

        self.ros = AmrRosInterface(
            cmd_vel_topic=cmd_vel_topic,
            odom_topic=odom_topic,
            scan_topic=scan_topic,
            set_entity_service=set_entity_service,
            use_sim_time=use_sim_time,
            laser_downsample_n=laser_downsample_n,
        )

        # spaces
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        self._laser_n = int(laser_downsample_n)
        self.observation_space = spaces.Dict({
            "agent": spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
            "laser": spaces.Box(low=0.0, high=1.0, shape=(self._laser_n,), dtype=np.float32),
        })

        # state
        self._step_count = 0
        self._goal_world = np.array([0.0, 0.0], dtype=np.float32)
        self._prev_dist = None
        self._start_world = np.array([0.0, 0.0], dtype=np.float32)
        # sensor debug
        self._last_step_scan_cb_count = 0

        # sensor status (for debugging)
        self._last_step_scan_count = 0

    # ---------------------------
    # helpers
    # ---------------------------
    def _hide_all_obstacles(self):
        """把所有障礙物移到場外，避免新回合一開始就跟舊位置重疊。"""
        if len(self.obstacles) == 0:
            return
        hx = float(self.obstacle_hide_xy)
        hy = float(self.obstacle_hide_xy)
        for obs in self.obstacles:
            self.ros.set_entity_state(obs.name, hx, hy, yaw=0.0, z=0.0)

    def _sample_start_pose(self):
        # 讓 robot 起始點不要太貼近牆
        m = float(self.robot_spawn_margin)
        # 防止 m 太大導致上下界反轉
        m = min(m, max(0.0, 0.5 * (self.x_max - self.x_min) - 0.05))
        m = min(m, max(0.0, 0.5 * (self.y_max - self.y_min) - 0.05))

        x = self._rng.uniform(self.x_min + m, self.x_max - m)
        y = self._rng.uniform(self.y_min + m, self.y_max - m)
        yaw = self._rng.uniform(-math.pi, math.pi)
        return float(x), float(y), float(yaw)

    def _sample_goal_pose(self, rx, ry):
        # 讓 goal 生成點不要太貼近牆（與 robot_spawn_margin 對稱）
        m = float(self.goal_spawn_margin)
        m = min(m, max(0.0, 0.5 * (self.x_max - self.x_min) - 0.05))
        m = min(m, max(0.0, 0.5 * (self.y_max - self.y_min) - 0.05))

        for _ in range(200):
            r = self._rng.uniform(self.min_goal_dist, self.max_goal_dist)
            ang = self._rng.uniform(-math.pi, math.pi)
            gx = rx + r * math.cos(ang)
            gy = ry + r * math.sin(ang)
            # 目標點必須落在可用範圍內，並且離牆至少留 margin
            if (self.x_min + m) <= gx <= (self.x_max - m) and (self.y_min + m) <= gy <= (self.y_max - m):
                return float(gx), float(gy)

        # fallback: 若抽樣一直失敗，就把 goal 往 +x 方向推，但也要套用 margin
        gx = float(np.clip(rx + self.max_goal_dist, self.x_min + m, self.x_max - m))
        gy = float(np.clip(ry, self.y_min + m, self.y_max - m))
        return gx, gy

    def _world_to_robot_frame(self, rx, ry, ryaw, gx, gy):
        dx = gx - rx
        dy = gy - ry
        c = math.cos(-ryaw)
        s = math.sin(-ryaw)
        x_rel = c * dx - s * dy
        y_rel = s * dx + c * dy
        return x_rel, y_rel

    def _get_obs(self):
        pose = self.ros.get_robot_pose2d()
        if pose is None:
            agent = np.zeros((3,), dtype=np.float32)
        else:
            rx, ry, ryaw = pose
            x_rel, y_rel = self._world_to_robot_frame(
                rx, ry, ryaw, float(self._goal_world[0]), float(self._goal_world[1])
            )
            x_rel_n = np.clip(x_rel / self.max_goal_dist, -1.0, 1.0)
            y_rel_n = np.clip(y_rel / self.max_goal_dist, -1.0, 1.0)

            theta = math.atan2(y_rel, x_rel)
            theta_n = float(np.clip(theta / math.pi, -1.0, 1.0))

            agent = np.array([x_rel_n, y_rel_n, theta_n], dtype=np.float32)

        ranges, rmax = self.ros.get_laser_downsample()
        laser = np.clip(ranges / max(rmax, 1e-6), 0.0, 1.0).astype(np.float32)

        return {"agent": agent, "laser": laser}

    def _compute_dist(self):
        pose = self.ros.get_robot_pose2d()
        if pose is None:
            return None
        rx, ry, _ = pose
        dx = float(self._goal_world[0]) - rx
        dy = float(self._goal_world[1]) - ry
        return math.sqrt(dx * dx + dy * dy)

    def _out_of_bounds(self):
        pose = self.ros.get_robot_pose2d()
        if pose is None:
            return False
        rx, ry, _ = pose

        mode = str(self.oob_mode).lower().strip()

        # legacy world-aligned square (centered at world origin)
        world_box = (rx < self.x_min) or (rx > self.x_max) or (ry < self.y_min) or (ry > self.y_max)

        # circle centered at episode start pose
        dx = float(rx) - float(self._start_world[0])
        dy = float(ry) - float(self._start_world[1])
        start_circle = (dx * dx + dy * dy) > (self.oob_radius * self.oob_radius)

        if mode == "start_circle":
            return bool(start_circle)
        if mode == "both":
            return bool(world_box or start_circle)
        # default: world_box
        return bool(world_box)

    def _collision_signal(self, ranges: np.ndarray) -> tuple[bool, float, int]:
        """Compute a *raw* collision signal from downsampled LiDAR.

        IMPORTANT:
        - A single-beam spike (one bucket suddenly very small) can falsely trigger collision if you use min().
        - To make training stable, we gate collision on:
            (a) at least K buckets below threshold (collision_min_hits)
            (b) optionally N consecutive steps (collision_confirm_steps) handled in step().

        Returns: (signal, min_range, hits)
        """
        if ranges is None:
            return False, float('inf'), 0

        r = np.asarray(ranges, dtype=np.float32)
        if r.size == 0:
            return False, float('inf'), 0

        min_r = float(np.min(r))
        hits = int(np.sum(r < self.collision_threshold))
        signal = bool((min_r < self.collision_threshold) and (hits >= self.collision_min_hits))
        return signal, min_r, hits

    @staticmethod
    def _kth_smallest(x: np.ndarray, k: int) -> float:
        """Return the k-th smallest value (k=0 is min) robustly.

        This is used to avoid single-sample LiDAR spikes poisoning reset-time clearance checks.
        """
        if x is None:
            return float('inf')
        a = np.asarray(x, dtype=np.float32)
        if a.size == 0:
            return float('inf')
        kk = int(max(0, min(int(k), int(a.size) - 1)))
        return float(np.partition(a, kk)[kk])

    def _denorm_action(self, a: np.ndarray):
        a = np.clip(a, -1.0, 1.0).astype(np.float32)
        vx = (a[0] + 1.0) * 0.5 * (self.vx_max - self.vx_min) + self.vx_min
        vy = a[1] * self.vy_max
        wz = a[2] * self.wz_max
        return float(vx), float(vy), float(wz)

    def _rand_free_xy(self, rx, ry, gx, gy, existing_xy: list[tuple[float, float]], max_tries: int = 300):
        """在邊界內採樣一個不違反 keepout 的點。"""
        margin = self.obstacle_spawn_margin
        x_lo, x_hi = self.x_min + margin, self.x_max - margin
        y_lo, y_hi = self.y_min + margin, self.y_max - margin

        for _ in range(int(max_tries)):
            x = float(self._rng.uniform(x_lo, x_hi))
            y = float(self._rng.uniform(y_lo, y_hi))

            if (x - rx) ** 2 + (y - ry) ** 2 < self.keepout_robot ** 2:
                continue
            if (x - gx) ** 2 + (y - gy) ** 2 < self.keepout_goal ** 2:
                continue

            ok = True
            for (ex, ey) in existing_xy:
                if (x - ex) ** 2 + (y - ey) ** 2 < self.keepout_between ** 2:
                    ok = False
                    break
            if not ok:
                continue

            return x, y

        # fallback: 丟到場外
        return self.obstacle_hide_xy, self.obstacle_hide_xy

    def _rand_corridor_xy(
        self,
        rx: float,
        ry: float,
        gx: float,
        gy: float,
        existing_xy: list[tuple[float, float]],
        max_tries: int | None = None,
    ) -> tuple[float, float] | None:
        """Sample an obstacle position *near* the straight-line path (robot->goal).

        We sample a point in a rectangular corridor around the segment:
          P(t, d) = R + t*(G-R) + d*n
        where n is the unit normal to the segment.

        Returns None if sampling fails.
        """
        tries = int(max_tries) if max_tries is not None else int(self.corridor_max_tries)

        vx = float(gx - rx)
        vy = float(gy - ry)
        L = math.sqrt(vx * vx + vy * vy)
        if L < 1e-6:
            return None

        # unit normal
        nx = -vy / L
        ny = vx / L

        margin = float(self.obstacle_spawn_margin)
        x_lo, x_hi = self.x_min + margin, self.x_max - margin
        y_lo, y_hi = self.y_min + margin, self.y_max - margin

        t_lo = float(self.corridor_t_min)
        t_hi = float(self.corridor_t_max)
        # avoid degenerate ranges
        if t_hi - t_lo < 1e-6:
            t_lo = 0.2
            t_hi = 0.8

        for _ in range(int(tries)):
            t = float(self._rng.uniform(t_lo, t_hi))
            d = float(self._rng.uniform(-self.corridor_half_width, self.corridor_half_width))

            x = float(rx + t * vx + d * nx)
            y = float(ry + t * vy + d * ny)

            if (x < x_lo) or (x > x_hi) or (y < y_lo) or (y > y_hi):
                continue

            if (x - rx) ** 2 + (y - ry) ** 2 < self.keepout_robot ** 2:
                continue
            if (x - gx) ** 2 + (y - gy) ** 2 < self.keepout_goal ** 2:
                continue

            ok = True
            for (ex, ey) in existing_xy:
                if (x - ex) ** 2 + (y - ey) ** 2 < self.keepout_between ** 2:
                    ok = False
                    break
            if not ok:
                continue

            return x, y

        return None

    def _randomize_obstacles(self, rx: float, ry: float, gx: float, gy: float):
        """
        reset 時呼叫，隨機擺放障礙物；

        與原版不同：
        - 以 corridor_prob 的機率，把 1~N 個障礙放在『robot->goal 直線走廊』附近
          （走廊寬度 corridor_half_width，沿線位置 t in [corridor_t_min, corridor_t_max]）。
        - 其餘障礙依舊用均勻隨機方式擺放。

        目的：讓多數 episode 都必須做避障決策，提升避障樣本密度。
        """
        self._last_corridor_used = False
        self._last_corridor_n = 0

        if len(self.obstacles) == 0:
            return 0

        n_active = int(self._rng.integers(self.min_obstacles, self.max_obstacles + 1))
        n_active = max(0, min(n_active, len(self.obstacles)))

        # 先全部丟到場外（非常重要）
        self._hide_all_obstacles()

        # 隨機選 active 的障礙模型（避免每次都用 Obstacle_0..）
        if n_active <= 0:
            return 0
        idxs = self._rng.choice(len(self.obstacles), size=n_active, replace=False)
        active = [self.obstacles[int(i)] for i in idxs]

        # decide corridor mode
        corridor_mode = bool((self.corridor_prob > 0.0) and (self._rng.random() < self.corridor_prob))
        corridor_min = int(min(self.corridor_min_obstacles, n_active))
        corridor_max = int(min(self.corridor_max_obstacles, n_active))
        if corridor_max < corridor_min:
            corridor_max = corridor_min
        n_corr = 0
        if corridor_mode and corridor_max > 0:
            # sample corridor obstacle count
            lo = int(max(0, corridor_min))
            hi = int(max(lo, corridor_max))
            n_corr = int(self._rng.integers(lo, hi + 1)) if hi > 0 else 0
            n_corr = int(max(lo, min(n_corr, n_active)))

        placed: list[tuple[float, float]] = []

        # place corridor obstacles first
        if n_corr > 0:
            self._last_corridor_used = True
            self._last_corridor_n = int(n_corr)
            for i in range(n_corr):
                obs = active[i]
                xy = self._rand_corridor_xy(rx, ry, gx, gy, placed)
                if xy is None:
                    x, y = self._rand_free_xy(rx, ry, gx, gy, placed)
                else:
                    x, y = xy
                yaw = float(self._rng.uniform(-math.pi, math.pi))
                placed.append((x, y))
                self.ros.set_entity_state(obs.name, x, y, yaw=yaw, z=0.0)

        # place remaining obstacles uniformly
        for i in range(n_corr, n_active):
            obs = active[i]
            x, y = self._rand_free_xy(rx, ry, gx, gy, placed)
            yaw = float(self._rng.uniform(-math.pi, math.pi))
            placed.append((x, y))
            self.ros.set_entity_state(obs.name, x, y, yaw=yaw, z=0.0)

        return n_active

    # ---------------------------
    # gym API
    # ---------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self._step_count = 0
        self._collision_streak = 0

        # 先停車，避免上一回合 cmd_vel 殘留影響
        self.ros.stop()

        # 確保至少收到一次 odom/scan（避免 reset 初期用到 fallback scan）
        if not self.ros.wait_for_topics(timeout_sec=self.topic_timeout_sec):
            msg = (
                f"[Stage1Env] No odom/scan received within {self.topic_timeout_sec:.1f}s. "
                "Training will likely be meaningless (laser fallback is constant). "
                "Please verify /odom and /scan topics (and QoS compatibility)."
            )
            if self.require_scan:
                raise RuntimeError(msg)
            # best-effort: keep going, but warn once per reset
            print("[WARN] " + msg)

        # 關鍵：每次 reset 一開始先把障礙物全部移到場外
        self._hide_all_obstacles()
        self.ros.spin_for(0.05)

        last_reason = "ok"
        chosen = None

        # 嘗試多次，避免抽到貼牆/不安全的起點
        for attempt in range(self.max_reset_tries):
            scan_cnt0 = int(self.ros.get_scan_status(max_age_sec=999.0).get("cb_count", 0))
            odom_cnt0 = int(self.ros.get_odom_status(max_age_sec=999.0).get("cb_count", 0))

            # teleport robot
            rx, ry, ryaw = self._sample_start_pose()
            self.ros.set_entity_state(self.robot_name, rx, ry, yaw=ryaw, z=0.0)

            # 讓感測器更新一下
            self.ros.spin_for(0.08)
            # 等待至少一個新 scan/odom（避免一直用 reset 前的舊資料）
            got_scan = self.ros.wait_for_new_scan(scan_cnt0, timeout_sec=1.0)
            got_odom = self.ros.wait_for_new_odom(odom_cnt0, timeout_sec=1.0)
            if self.require_scan and (not got_scan):
                last_reason = "no_scan_update_after_teleport"
                continue
            if not got_odom:
                last_reason = "no_odom_update_after_teleport"
                continue

            # 起點 clearance 檢查（避免貼牆/角落）
            ranges, _ = self.ros.get_laser_downsample()
            # Use 2nd smallest to avoid occasional single-beam spikes triggering false rejection
            min_r0 = self._kth_smallest(ranges, 1)
            if min_r0 < self.spawn_min_clearance:
                last_reason = f"start_clearance<{self.spawn_min_clearance:.2f}"
                continue

            # sample goal + teleport target
            gx, gy = self._sample_goal_pose(rx, ry)
            self._goal_world[:] = [gx, gy]
            self.ros.set_entity_state(self.target_name, gx, gy, yaw=0.0, z=0.05)

            # randomize obstacles AFTER start/goal are fixed
            n_active = self._randomize_obstacles(rx, ry, gx, gy)

            # warmup
            # 再確認 scan 有更新（障礙物擺放會影響 scan，如果 scan 沒更新，觀測會不含障礙資訊）
            scan_cnt1 = int(self.ros.get_scan_status(max_age_sec=999.0).get("cb_count", 0))
            self.ros.spin_for(0.20)
            got_scan2 = self.ros.wait_for_new_scan(scan_cnt1, timeout_sec=1.5)
            if self.require_scan and (not got_scan2):
                # 先把障礙移走再重抽，避免下一次嘗試被舊位置影響
                self.ros.stop()
                self._hide_all_obstacles()
                self.ros.spin_for(0.05)
                last_reason = "no_scan_update_after_obstacles"
                continue

            # 再檢查一次：如果擺完障礙後已經極近/疑似重疊，就重抽（避免一開始爆）
            ranges2, _ = self.ros.get_laser_downsample()
            min_r1 = self._kth_smallest(ranges2, 1)
            if min_r1 < self.collision_threshold:
                # 再次把障礙物移走，避免下一次嘗試被舊位置影響
                self.ros.stop()
                self._hide_all_obstacles()
                self.ros.spin_for(0.05)
                last_reason = "immediate_too_close_after_obstacles"
                continue

            # ok
            chosen = (rx, ry, ryaw, gx, gy, n_active, attempt + 1, last_reason, min_r0, min_r1)
            break

        # 如果多次嘗試仍失敗，退而求其次：把障礙全移走，起點放中心，至少不會翻車卡死
        if chosen is None:
            rx, ry, ryaw = 0.0, 0.0, 0.0
            self.ros.set_entity_state(self.robot_name, rx, ry, yaw=ryaw, z=0.0)
            gx, gy = self._sample_goal_pose(rx, ry)
            self._goal_world[:] = [gx, gy]
            self.ros.set_entity_state(self.target_name, gx, gy, yaw=0.0, z=0.05)
            self._hide_all_obstacles()
            self.ros.spin_for(0.25)
            self.ros.wait_for_topics(timeout_sec=1.5)
            n_active = 0
            reset_attempts = self.max_reset_tries
            last_reason = "fallback_center_no_obstacles"
            ranges_f, _ = self.ros.get_laser_downsample()
            min_r0 = self._kth_smallest(ranges_f, 1)
            min_r1 = min_r0
        else:
            rx, ry, ryaw, gx, gy, n_active, reset_attempts, last_reason, min_r0, min_r1 = chosen

        # record episode start position for out-of-bounds checks
        self._start_world[:] = [float(rx), float(ry)]

        dist = self._compute_dist()
        self._prev_dist = float(dist) if dist is not None else 999.0

        # initialize scan counter baseline for step()
        self._last_step_scan_cb_count = int(self.ros.get_scan_status(max_age_sec=999.0).get("cb_count", 0))

        obs = self._get_obs()

        scan_status = self.ros.get_scan_status(max_age_sec=self.scan_max_age_sec)
        laser_std = float(np.std(obs["laser"]))
        laser_min = float(np.min(obs["laser"]))
        laser_mean = float(np.mean(obs["laser"]))
        laser_max = float(np.max(obs["laser"]))

        info = {
            "start_x": float(rx),
            "start_y": float(ry),
            "oob_mode": str(self.oob_mode),
            "oob_radius": float(self.oob_radius),
            "goal_x": float(gx),
            "goal_y": float(gy),
            "n_obstacles": int(n_active),
            "corridor_used": int(bool(self._last_corridor_used)),
            "corridor_n": int(self._last_corridor_n),
            "corridor_prob": float(self.corridor_prob),
            "corridor_half_width": float(self.corridor_half_width),
            "corridor_t_min": float(self.corridor_t_min),
            "corridor_t_max": float(self.corridor_t_max),
            "reset_attempts": int(reset_attempts),
            "reset_reason": str(last_reason),
            "reset_min_laser_before": float(min_r0),
            "reset_min_laser_after": float(min_r1),
            # sensor debug
            "scan_present": int(bool(scan_status.get("present", False))),
            "scan_fresh": int(bool(scan_status.get("fresh", False))),
            "scan_age_sec": float(scan_status.get("age_sec", float('inf'))),
            "scan_cb_count": int(scan_status.get("cb_count", 0)),
            "scan_n_ranges": int(scan_status.get("n_ranges", 0)),
            "laser_std_norm": float(laser_std),
            "laser_min_norm": float(laser_min),
            "laser_mean_norm": float(laser_mean),
            "laser_max_norm": float(laser_max),
        }
        return obs, info

    def step(self, action):
        self._step_count += 1

        vx, vy, wz = self._denorm_action(np.array(action, dtype=np.float32))

        # --- Example-style stepping (sensor-driven) ---
        # The reference example spins until it receives *new* odom + laser callbacks after applying the action.
        # This makes the environment robust to Gazebo update-rate changes (real_time_update_rate / sensor update rates).
        scan_cnt0 = int(self.ros.get_scan_status(max_age_sec=999.0).get("cb_count", 0))
        odom_cnt0 = int(self.ros.get_odom_status(max_age_sec=999.0).get("cb_count", 0))

        self.ros.publish_cmd(vx, vy, wz)

        # Wait for new sensor data in ROS time (sim time when use_sim_time=True).
        # Keep a small wall-time watchdog to avoid deadlock if /clock stalls (paused) or callbacks starve.
        step_timeout = max(0.25, 2.0 * float(self.control_dt))
        got_scan, got_odom = self.ros.wait_for_new_data(
            prev_scan_count=scan_cnt0,
            prev_odom_count=odom_cnt0,
            timeout_sec=step_timeout,
            wall_timeout_sec=5.0,
            require_scan=self.require_scan,
            require_odom=True,
        )

        # Best-effort fallback: if we did not get updates, still spin a bit so callbacks can be serviced.
        if (not got_odom) or (self.require_scan and (not got_scan)):
            self.ros.spin_for(float(self.control_dt))

        obs = self._get_obs()

        # sensor debug (computed early so stale scan can be diagnosed even when episode ends)
        scan_status = self.ros.get_scan_status(max_age_sec=self.scan_max_age_sec)
        scan_cb = int(scan_status.get("cb_count", 0))
        scan_new = int(scan_cb > int(scan_cnt0))
        self._last_step_scan_cb_count = scan_cb

        odom_status = self.ros.get_odom_status(max_age_sec=self.scan_max_age_sec)
        odom_cb = int(odom_status.get("cb_count", 0))
        odom_new = int(odom_cb > int(odom_cnt0))

        laser_std = float(np.std(obs["laser"]))
        laser_min = float(np.min(obs["laser"]))
        laser_mean = float(np.mean(obs["laser"]))
        laser_max = float(np.max(obs["laser"]))

        terminated = False
        truncated = False
        reward = 0.0

        dist = self._compute_dist()
        if dist is None:
            dist = float(self._prev_dist) if self._prev_dist is not None else 999.0

        prev = float(self._prev_dist) if self._prev_dist is not None else dist
        progress = float(prev - dist)
        self._prev_dist = float(dist)

        # ---------------- reward design ----------------
        if progress > 0:
            reward += 2.0 * progress
        else:
            reward += 0.2 * progress
        reward += self.step_penalty  # step penalty
        reward -= 0.003 * (vx * vx + vy * vy + wz * wz)

        # clearance shaping: 越靠近障礙越扣分（連續）
        # NOTE: we intentionally reuse the same LiDAR read for both shaping and collision gating.
        ranges, _ = self.ros.get_laser_downsample()
        collision_signal, min_r, collision_hits = self._collision_signal(ranges)
        # use 2nd smallest for shaping to reduce sporadic single-bucket spikes
        min_r_k1 = self._kth_smallest(ranges, 1)
        if min_r_k1 < self.near_obstacle_threshold:
            reward -= self.near_obstacle_penalty * (self.near_obstacle_threshold - min_r_k1)

        # collision debounce / confirmation: avoid terminating on a single spurious spike
        if collision_signal:
            self._collision_streak += 1
        else:
            self._collision_streak = 0
        collision_now = bool(self._collision_streak >= self.collision_confirm_steps)

        termination_reason = "running"
        is_success = 0

        # distance to episode start (debug / optional termination logic)
        pose_now = self.ros.get_robot_pose2d()
        if pose_now is None:
            dist_from_start = float("inf")
        else:
            rx_now, ry_now, _ = pose_now
            dxs = float(rx_now) - float(self._start_world[0])
            dys = float(ry_now) - float(self._start_world[1])
            dist_from_start = math.sqrt(dxs * dxs + dys * dys)

        # success
        if dist < self.goal_threshold:
            terminated = True
            reward += 15.0
            termination_reason = "success"
            is_success = 1

        # safety
        if self._out_of_bounds():
            terminated = True
            reward -= 10.0
            termination_reason = "out_of_bounds"
            is_success = 0

        if collision_now:
            terminated = True
            reward -= 15.0
            termination_reason = "collision"
            is_success = 0

        # time limit
        if self._step_count >= self.max_steps:
            truncated = True
            if termination_reason == "running":
                termination_reason = "timeout"
            is_success = 0

        # sensor health termination (optional)
        if self.terminate_on_lidar_stale and (termination_reason == "running"):
            if not bool(scan_status.get("fresh", False)):
                truncated = True
                termination_reason = "lidar_stale"
                # small penalty so it is visible in returns without dominating learning
                reward -= 2.0

        if terminated or truncated:
            self.ros.stop()
            print( f"[RESET] reason={termination_reason}, " f"step={self._step_count}, " f"dist={dist:.2f}, " f"min_laser={min_r:.2f}, " f"terminated={terminated}, truncated={truncated}" )

        info = {
            "dist_to_goal": float(dist),
            "dist_from_start": float(dist_from_start),
            "oob_mode": str(self.oob_mode),
            "oob_radius": float(self.oob_radius),
            "step": int(self._step_count),
            "vx": float(vx),
            "vy": float(vy),
            "wz": float(wz),
            "min_laser": float(min_r),
            "min_laser_k1": float(min_r_k1),
            "collision_signal": int(bool(collision_signal)),
            "collision_hits": int(collision_hits),
            "collision_streak": int(self._collision_streak),
            "collision_confirm_steps": int(self.collision_confirm_steps),
            "is_success": int(is_success),
            "termination_reason": str(termination_reason),
            "is_collision": int(termination_reason == "collision"),
            # sensor debug
            "scan_present": int(bool(scan_status.get("present", False))),
            "scan_fresh": int(bool(scan_status.get("fresh", False))),
            "scan_age_sec": float(scan_status.get("age_sec", float('inf'))),
            "scan_cb_count": int(scan_cb),
            "scan_n_ranges": int(scan_status.get("n_ranges", 0)),
            "scan_new": int(scan_new),
            "odom_present": int(bool(odom_status.get("present", False))),
            "odom_fresh": int(bool(odom_status.get("fresh", False))),
            "odom_age_sec": float(odom_status.get("age_sec", float('inf'))),
            "odom_cb_count": int(odom_cb),
            "odom_new": int(odom_new),
            "laser_std_norm": float(laser_std),
            "laser_min_norm": float(laser_min),
            "laser_mean_norm": float(laser_mean),
            "laser_max_norm": float(laser_max),
        }
        return obs, float(reward), terminated, truncated, info

    def close(self):
        try:
            self.ros.stop()
            self.ros.destroy_node()
        except Exception:
            pass
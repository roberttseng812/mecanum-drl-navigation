import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from amr_rl.amr_ros_interface import AmrRosInterface


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


class AmrNavStage0Env(gym.Env):
    """
    Stage 0：
    - square.world（基本空場）
    - obs: goal relative (robot frame) + laser(占位，後續 stage 可直接用)
    - action: [vx, vy, wz] normalized in [-1,1]
    - reward: progress shaping + goal bonus + small action penalty
    - done: reached goal / out of bounds / collision / time limit
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
        max_steps: int = 200,
        x_min: float | None = None,
        x_max: float | None = None,
        y_min: float | None = None,
        y_max: float | None = None,
        x_lim: float | None = 3.0,
        y_lim: float | None = 3.0,
        min_goal_dist: float = 1.0,
        max_goal_dist: float = 3.0,
        goal_threshold: float = 0.35,
        collision_threshold: float = 0.20,
        laser_downsample_n: int = 120,
        # action limits (mecanum)
        vx_min: float = -0.3,
        vx_max: float = 0.6,
        vy_max: float = 0.4,
        wz_max: float = 1.0,
        seed: int | None = None,
    ):
        super().__init__()

        self.robot_name = robot_name
        self.target_name = target_name

        self.control_dt = float(control_dt)
        self.max_steps = int(max_steps)

        self.x_min, self.x_max = resolve_axis_bounds("x", x_min, x_max, x_lim, 3.0)
        self.y_min, self.y_max = resolve_axis_bounds("y", y_min, y_max, y_lim, 3.0)
        # Backward-compatible aliases for code that still reads symmetric limits.
        self.x_lim = max(abs(self.x_min), abs(self.x_max))
        self.y_lim = max(abs(self.y_min), abs(self.y_max))

        self.min_goal_dist = float(min_goal_dist)
        self.max_goal_dist = float(max_goal_dist)
        self.goal_threshold = float(goal_threshold)

        self.collision_threshold = float(collision_threshold)

        self.vx_min = float(vx_min)
        self.vx_max = float(vx_max)
        self.vy_max = float(vy_max)
        self.wz_max = float(wz_max)

        self._rng = np.random.default_rng(seed)

        # ROS interface node (QoS 修正請在 AmrRosInterface 內處理 scan QoS)
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

    # ---------------------------
    # helpers
    # ---------------------------
    def _inner_bounds(self, margin: float = 0.0):
        margin = max(0.0, float(margin))
        max_margin_x = max(0.0, 0.5 * (self.x_max - self.x_min) - 0.05)
        max_margin_y = max(0.0, 0.5 * (self.y_max - self.y_min) - 0.05)
        mx = min(margin, max_margin_x)
        my = min(margin, max_margin_y)
        return (
            self.x_min + mx,
            self.x_max - mx,
            self.y_min + my,
            self.y_max - my,
        )

    def _sample_start_pose(self):
        x_lo, x_hi, y_lo, y_hi = self._inner_bounds(0.0)
        x = self._rng.uniform(x_lo, x_hi)
        y = self._rng.uniform(y_lo, y_hi)
        yaw = self._rng.uniform(-math.pi, math.pi)
        return float(x), float(y), float(yaw)

    def _sample_goal_pose(self, rx, ry):
        x_lo, x_hi, y_lo, y_hi = self._inner_bounds(0.0)
        for _ in range(200):
            r = self._rng.uniform(self.min_goal_dist, self.max_goal_dist)
            ang = self._rng.uniform(-math.pi, math.pi)
            gx = rx + r * math.cos(ang)
            gy = ry + r * math.sin(ang)
            if (x_lo <= gx <= x_hi) and (y_lo <= gy <= y_hi):
                return float(gx), float(gy)

        gx = float(np.clip(rx + self.max_goal_dist, x_lo, x_hi))
        gy = float(np.clip(ry, y_lo, y_hi))
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
        return (rx < self.x_min) or (rx > self.x_max) or (ry < self.y_min) or (ry > self.y_max)

    def _collision(self):
        ranges, _ = self.ros.get_laser_downsample()
        return float(np.min(ranges)) < self.collision_threshold

    def _denorm_action(self, a: np.ndarray):
        a = np.clip(a, -1.0, 1.0).astype(np.float32)

        # vx: asymmetric map to [vx_min, vx_max]
        vx = (a[0] + 1.0) * 0.5 * (self.vx_max - self.vx_min) + self.vx_min
        # vy, wz: symmetric
        vy = a[1] * self.vy_max
        wz = a[2] * self.wz_max
        return float(vx), float(vy), float(wz)

    # ---------------------------
    # gym API
    # ---------------------------
    def reset(self, *, seed=None, options=None):
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

        # warmup
        self.ros.spin_for(0.2)
        self.ros.wait_for_topics(timeout_sec=1.0)

        dist = self._compute_dist()
        self._prev_dist = float(dist) if dist is not None else 999.0

        obs = self._get_obs()
        info = {
            "goal_x": float(gx),
            "goal_y": float(gy),
            "x_min": float(self.x_min),
            "x_max": float(self.x_max),
            "y_min": float(self.y_min),
            "y_max": float(self.y_max),
        }
        return obs, info

    def step(self, action):
        self._step_count += 1

        vx, vy, wz = self._denorm_action(np.array(action, dtype=np.float32))
        self.ros.publish_cmd(vx, vy, wz)
        self.ros.spin_for(self.control_dt)

        obs = self._get_obs()

        terminated = False
        truncated = False
        reward = 0.0

        dist = self._compute_dist()
        if dist is None:
            dist = float(self._prev_dist) if self._prev_dist is not None else 999.0

        prev = float(self._prev_dist) if self._prev_dist is not None else dist
        progress = float(prev - dist)
        self._prev_dist = float(dist)

        # reward design
        reward += 2.0 * progress
        reward -= 0.01  # step penalty
        reward -= 0.001 * (vx * vx + vy * vy + wz * wz)

        termination_reason = "running"
        is_success = 0

        # success
        if dist < self.goal_threshold:
            terminated = True
            reward += 10.0
            termination_reason = "success"
            is_success = 1

        # safety
        if self._out_of_bounds():
            terminated = True
            reward -= 10.0
            termination_reason = "out_of_bounds"
            is_success = 0

        if self._collision():
            terminated = True
            reward -= 20.0
            termination_reason = "collision"
            is_success = 0

        # time limit
        if self._step_count >= self.max_steps:
            truncated = True
            if termination_reason == "running":
                termination_reason = "timeout"
            is_success = 0

        if terminated or truncated:
            self.ros.stop()

        info = {
            "dist_to_goal": float(dist),
            "step": int(self._step_count),
            "vx": float(vx),
            "vy": float(vy),
            "wz": float(wz),

            # for metrics/callbacks
            "is_success": int(is_success),
            "termination_reason": str(termination_reason),
            "is_collision": int(termination_reason == "collision"),
        }
        return obs, float(reward), terminated, truncated, info

    def close(self):
        try:
            self.ros.stop()
            self.ros.destroy_node()
        except Exception:
            pass

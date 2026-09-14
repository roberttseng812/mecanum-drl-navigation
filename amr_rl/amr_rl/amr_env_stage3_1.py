import math
from dataclasses import dataclass

import numpy as np

from amr_rl.amr_env_stage3 import AmrNavStage3Env


@dataclass(frozen=True)
class ObstacleSpec:
    """Gazebo static obstacle model name and approximate radius."""
    name: str
    radius: float = 0.35


class AmrNavStage31Env(AmrNavStage3Env):
    """
    Stage 3.1
    ---------
    - Inherits the observation / reward / dynamic-actor route sampling from Stage 3.
    - Adds a small set of Stage-2-style static obstacles (up to 3) without modifying Stage 2.
    - Supports per-episode obstacle modes:
        * actor_only
        * actor_plus_static
        * mixed  (sampled per episode by actor_plus_static_prob)
    """

    def __init__(
        self,
        obstacle_names: list[str] | None = None,
        min_static_obstacles: int = 1,
        max_static_obstacles: int = 3,
        obstacle_spawn_margin: float = 0.85,
        keepout_robot: float = 0.95,
        keepout_goal: float = 0.95,
        keepout_between: float = 1.60,
        corridor_prob: float = 0.80,
        corridor_half_width: float = 0.55,
        corridor_min_obstacles: int = 1,
        corridor_max_obstacles: int = 2,
        corridor_t_min: float = 0.20,
        corridor_t_max: float = 0.80,
        corridor_max_tries: int = 400,
        obstacle_hide_xy: float = 50.0,
        obstacle_episode_mode: str = 'mixed',
        actor_plus_static_prob: float = 0.65,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if obstacle_names is None:
            obstacle_names = [f'Obstacle_{i}' for i in range(3)]
        self.obstacles = [ObstacleSpec(name=n) for n in obstacle_names]

        self.min_static_obstacles = max(0, int(min_static_obstacles))
        self.max_static_obstacles = max(0, int(max_static_obstacles))
        if self.max_static_obstacles > len(self.obstacles):
            self.max_static_obstacles = len(self.obstacles)
        if self.min_static_obstacles > self.max_static_obstacles:
            self.min_static_obstacles = self.max_static_obstacles

        self.obstacle_spawn_margin = float(obstacle_spawn_margin)
        self.keepout_robot = float(keepout_robot)
        self.keepout_goal = float(keepout_goal)
        self.keepout_between = float(keepout_between)

        self.corridor_prob = float(np.clip(corridor_prob, 0.0, 1.0))
        self.corridor_half_width = float(max(0.05, corridor_half_width))
        self.corridor_min_obstacles = int(max(0, corridor_min_obstacles))
        self.corridor_max_obstacles = int(max(0, corridor_max_obstacles))
        if self.corridor_max_obstacles < self.corridor_min_obstacles:
            self.corridor_max_obstacles = self.corridor_min_obstacles
        self.corridor_t_min = float(np.clip(corridor_t_min, 0.0, 1.0))
        self.corridor_t_max = float(np.clip(corridor_t_max, 0.0, 1.0))
        if self.corridor_t_max < self.corridor_t_min:
            self.corridor_t_min, self.corridor_t_max = self.corridor_t_max, self.corridor_t_min
        self.corridor_max_tries = int(max(50, corridor_max_tries))
        self.obstacle_hide_xy = float(obstacle_hide_xy)

        self.obstacle_episode_mode = str(obstacle_episode_mode).strip().lower()
        if self.obstacle_episode_mode not in {'actor_only', 'actor_plus_static', 'mixed'}:
            raise ValueError("obstacle_episode_mode must be one of {'actor_only', 'actor_plus_static', 'mixed'}")
        self.actor_plus_static_prob = float(np.clip(actor_plus_static_prob, 0.0, 1.0))

        self._last_corridor_used = False
        self._last_corridor_n = 0
        self._last_episode_obstacle_mode = 'actor_only'
        self._last_static_obstacle_count = 0

    def _hide_all_obstacles(self):
        if len(self.obstacles) == 0:
            return
        hx = float(self.obstacle_hide_xy)
        hy = float(self.obstacle_hide_xy)
        for obs in self.obstacles:
            self.ros.set_entity_state(obs.name, hx, hy, yaw=0.0, z=0.0)

    def _rand_free_xy(
        self,
        rx: float,
        ry: float,
        gx: float,
        gy: float,
        existing_xy: list[tuple[float, float]],
        max_tries: int = 300,
    ) -> tuple[float, float] | None:
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

        return None

    def _rand_corridor_xy(
        self,
        rx: float,
        ry: float,
        gx: float,
        gy: float,
        existing_xy: list[tuple[float, float]],
        max_tries: int | None = None,
    ) -> tuple[float, float] | None:
        tries = int(max_tries) if max_tries is not None else int(self.corridor_max_tries)

        vx = float(gx - rx)
        vy = float(gy - ry)
        L = math.sqrt(vx * vx + vy * vy)
        if L < 1e-6:
            return None

        nx = -vy / L
        ny = vx / L

        margin = float(self.obstacle_spawn_margin)
        x_lo, x_hi = self.x_min + margin, self.x_max - margin
        y_lo, y_hi = self.y_min + margin, self.y_max - margin

        t_lo = float(self.corridor_t_min)
        t_hi = float(self.corridor_t_max)
        if t_hi - t_lo < 1e-6:
            t_lo, t_hi = 0.2, 0.8

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

    def _randomize_obstacles(self, rx: float, ry: float, gx: float, gy: float) -> int:
        self._last_corridor_used = False
        self._last_corridor_n = 0

        if len(self.obstacles) == 0 or self.max_static_obstacles <= 0:
            return 0

        n_active = int(self._rng.integers(self.min_static_obstacles, self.max_static_obstacles + 1))
        n_active = max(0, min(n_active, len(self.obstacles)))

        self._hide_all_obstacles()
        if n_active <= 0:
            return 0

        idxs = self._rng.choice(len(self.obstacles), size=n_active, replace=False)
        active = [self.obstacles[int(i)] for i in idxs]

        corridor_mode = bool((self.corridor_prob > 0.0) and (self._rng.random() < self.corridor_prob))
        corridor_min = int(min(self.corridor_min_obstacles, n_active))
        corridor_max = int(min(self.corridor_max_obstacles, n_active))
        if corridor_max < corridor_min:
            corridor_max = corridor_min

        n_corr_target = 0
        if corridor_mode and corridor_max > 0:
            lo = int(max(0, corridor_min))
            hi = int(max(lo, corridor_max))
            n_corr_target = int(self._rng.integers(lo, hi + 1)) if hi > 0 else 0
            n_corr_target = int(max(lo, min(n_corr_target, n_active)))

        placed: list[tuple[float, float]] = []
        corridor_placed = 0

        if n_corr_target > 0:
            for i in range(n_corr_target):
                obs = active[i]
                xy = self._rand_corridor_xy(rx, ry, gx, gy, placed)
                if xy is None:
                    xy = self._rand_free_xy(rx, ry, gx, gy, placed)
                if xy is None:
                    continue

                x, y = xy
                yaw = float(self._rng.uniform(-math.pi, math.pi))
                placed.append((x, y))
                corridor_placed += 1
                self.ros.set_entity_state(obs.name, x, y, yaw=yaw, z=0.0)

        for i in range(n_corr_target, n_active):
            obs = active[i]
            xy = self._rand_free_xy(rx, ry, gx, gy, placed)
            if xy is None:
                continue

            x, y = xy
            yaw = float(self._rng.uniform(-math.pi, math.pi))
            placed.append((x, y))
            self.ros.set_entity_state(obs.name, x, y, yaw=yaw, z=0.0)

        self._last_corridor_n = int(corridor_placed)
        self._last_corridor_used = bool(corridor_placed > 0)
        return int(len(placed))

    def _sample_episode_obstacle_mode(self) -> str:
        mode = str(self.obstacle_episode_mode)
        if mode == 'mixed':
            return 'actor_plus_static' if (self._rng.random() < self.actor_plus_static_prob) else 'actor_only'
        return mode

    def reset(self, *, seed=None, options=None):
        super(AmrNavStage3Env, self).reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._step_count = 0
        self._collision_streak = 0
        self._prev_dist = None
        self._prev_yaw_err = None
        self._prev_action = None
        self._was_in_goal_zone = False
        self._last_obs = None

        self._hide_all_obstacles()
        self.ros.stop()

        if not self.ros.wait_for_topics(timeout_sec=self.topic_timeout_sec):
            raise RuntimeError(
                f'[Stage31Env] No odom/scan received within {self.topic_timeout_sec:.1f}s. '
                'Please verify /odom and /scan topics before training.'
            )

        chosen = None
        last_reason = 'ok'
        reset_route_mode = 'uniform'
        route_segment_id = -1
        route_segment_name = 'uniform'
        route_cross_x = float('nan')
        route_cross_y = float('nan')
        episode_obstacle_mode = self._sample_episode_obstacle_mode()
        self._last_episode_obstacle_mode = str(episode_obstacle_mode)

        for attempt in range(self.max_reset_tries):
            last_reason = 'ok'
            self._hide_all_obstacles()
            scan_cnt0 = int(getattr(self.ros, '_scan_cb_count', 0))
            odom_cnt0 = int(getattr(self.ros, '_odom_cb_count', 0))

            route_sample = None
            if self.dynamic_route_prob > 0.0 and (self._rng.random() < self.dynamic_route_prob):
                route_sample = self._sample_dynamic_route_pair()

            if route_sample is None:
                rx, ry, ryaw = self._sample_start_pose()
                gx, gy = self._sample_goal_pose(rx, ry)
                reset_route_mode = 'uniform'
                route_segment_id = -1
                route_cross_x = float('nan')
                route_cross_y = float('nan')
                route_segment_name = 'uniform'
            else:
                rx = float(route_sample['start_x'])
                ry = float(route_sample['start_y'])
                ryaw = float(route_sample['start_yaw'])
                gx = float(route_sample['goal_x'])
                gy = float(route_sample['goal_y'])
                reset_route_mode = str(route_sample['route_mode'])
                route_segment_id = int(route_sample['route_segment_id'])
                route_cross_x = float(route_sample['route_cross_x'])
                route_cross_y = float(route_sample['route_cross_y'])
                route_segment_name = str(route_sample.get('route_segment_name', 'unknown'))

            self.ros.set_entity_state(self.robot_name, rx, ry, yaw=ryaw, z=0.0)

            self.ros.spin_for(0.08)
            got_scan = self.ros.wait_for_new_scan(scan_cnt0, timeout_sec=1.0)
            got_odom = self.ros.wait_for_new_odom(odom_cnt0, timeout_sec=1.0)
            if not got_scan:
                last_reason = 'no_scan_update_after_teleport'
                continue
            if not got_odom:
                last_reason = 'no_odom_update_after_teleport'
                continue

            ranges0, _ = self.ros.get_laser_downsample()
            min_r0 = self._kth_smallest(ranges0, 1)
            if min_r0 < self.spawn_min_clearance:
                last_reason = f'start_clearance<{self.spawn_min_clearance:.2f}'
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

            static_obstacle_count = 0
            if episode_obstacle_mode == 'actor_plus_static':
                static_obstacle_count = int(self._randomize_obstacles(rx, ry, gx, gy))
            else:
                self._hide_all_obstacles()

            scan_cnt1 = int(getattr(self.ros, '_scan_cb_count', 0))
            self.ros.spin_for(0.20)
            got_scan2 = self.ros.wait_for_new_scan(scan_cnt1, timeout_sec=1.5)
            if not got_scan2:
                last_reason = 'no_scan_update_after_goal'
                continue

            ranges1, _ = self.ros.get_laser_downsample()
            min_r1 = self._kth_smallest(ranges1, 1)
            if min_r1 < self.collision_threshold:
                last_reason = 'immediate_too_close_after_goal'
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
                static_obstacle_count,
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
            static_obstacle_count = 0
            if episode_obstacle_mode == 'actor_plus_static':
                static_obstacle_count = int(self._randomize_obstacles(rx, ry, gx, gy))
            self.ros.spin_for(0.25)
            ranges_f, _ = self.ros.get_laser_downsample()
            min_r0 = self._kth_smallest(ranges_f, 1)
            min_r1 = min_r0
            reset_attempts = self.max_reset_tries
            last_reason = 'fallback_center'
            reset_route_mode = 'fallback_uniform'
            route_segment_id = -1
            route_segment_name = 'fallback_center'
            route_cross_x = float('nan')
            route_cross_y = float('nan')
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
                static_obstacle_count,
            ) = chosen

        self._last_static_obstacle_count = int(static_obstacle_count)

        dist = self._compute_dist()
        self._prev_dist = float(dist) if dist is not None else 999.0

        yaw_err = self._compute_yaw_err()
        self._prev_yaw_err = float(yaw_err) if yaw_err is not None else math.pi

        sensor_ok, scan_status, odom_status = self._get_sensor_status()
        if not sensor_ok:
            raise RuntimeError(
                '[Stage31Env] Sensor data is not fresh after reset. '
                f"scan_fresh={scan_status.get('fresh', False)}, odom_fresh={odom_status.get('fresh', False)}, "
                f'max_age={self.sensor_freshness_sec:.3f}s'
            )

        self._prime_lidar_history()
        obs = self._get_obs()
        self._last_obs = self._clone_obs(obs)

        info = {
            'start_x': float(rx),
            'start_y': float(ry),
            'goal_x': float(gx),
            'goal_y': float(gy),
            'oob_mode': str(self.oob_mode),
            'oob_radius': float(self.oob_radius),
            'goal_yaw': float(self._target_yaw_world),
            'goal_line_yaw': float(goal_line_yaw),
            'reset_route_mode': str(reset_route_mode),
            'route_segment_id': int(route_segment_id),
            'route_cross_x': float(route_cross_x),
            'route_cross_y': float(route_cross_y),
            'route_segment_name': str(route_segment_name),
            'dynamic_route_prob': float(self.dynamic_route_prob),
            'dynamic_route_mode': str(self.dynamic_route_mode),
            'dynamic_headon_prob': float(self.dynamic_headon_prob),
            'dynamic_samedir_prob': float(self.dynamic_samedir_prob),
            'dynamic_route_segment_count': int(len(self._dynamic_route_segments)),
            'dynamic_route_trim_m': float(self.dynamic_route_trim_m),
            'goal_yaw_mode': str(goal_yaw_mode),
            'goal_yaw_sector_prob': float(self.goal_yaw_sector_prob),
            'goal_yaw_sector_half_angle_deg': float(self.goal_yaw_sector_half_angle_deg),
            'yaw_err': float(self._prev_yaw_err),
            'reset_attempts': int(reset_attempts),
            'reset_reason': str(last_reason),
            'reset_min_laser_before': float(min_r0),
            'reset_min_laser_after': float(min_r1),
            'sensor_freshness_sec': float(self.sensor_freshness_sec),
            'scan_fresh': int(bool(scan_status.get('fresh', False))),
            'odom_fresh': int(bool(odom_status.get('fresh', False))),
            'scan_age_sec': float(scan_status.get('age_sec', float('inf'))),
            'odom_age_sec': float(odom_status.get('age_sec', float('inf'))),
            'lidar_history_steps': int(self._lidar_history_steps),
            'lidar_history_stride': int(self._lidar_history_stride),
            'lidar_history_span_sec': float((self._lidar_history_steps - 1) * self._lidar_history_stride * self.control_dt),
            'laser_downsample_n': int(self._laser_n),
            'control_dt': float(self.control_dt),
            'obstacle_episode_mode_cfg': str(self.obstacle_episode_mode),
            'episode_obstacle_mode': str(self._last_episode_obstacle_mode),
            'actor_plus_static_prob': float(self.actor_plus_static_prob),
            'static_obstacle_count': int(self._last_static_obstacle_count),
            'static_obstacle_count_max': int(self.max_static_obstacles),
            'static_corridor_used': int(bool(self._last_corridor_used)),
            'static_corridor_count': int(self._last_corridor_n),
            'static_corridor_prob': float(self.corridor_prob),
        }
        return obs, info

    def close(self):
        try:
            self._hide_all_obstacles()
        except Exception:
            pass
        super().close()

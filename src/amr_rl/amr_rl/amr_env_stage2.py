import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# 繼承 Stage1 環境的避障與基礎邏輯
from amr_rl.amr_env_stage1 import AmrNavStage1Env

def wrap_to_pi(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi

class AmrNavStage2Env(AmrNavStage1Env):
    """
    Stage 2（位姿對齊版）：
    - 繼承 Stage 1 的避障邏輯。
    - Observation `agent` 從 3 維改為 4 維：加入 `yaw_err_n`。
    - Reward 增加對 Target Yaw 角度對齊的獎勵。
    - Success 條件：同時滿足位置靠近目標 且 abs(yaw_err) < yaw_threshold。
    """
    def __init__(
        self,
        *args,
        yaw_threshold: float = 0.25,
        goal_yaw_sector_prob: float = 0.60,
        goal_yaw_sector_half_angle_deg: float = 60.0,
        stage2_step_penalty: float = -0.02,
        clearance_square: bool = True,
        timeout_penalty: float = 5.0,
        near_goal_mult: float = 2.0,
        ring_bonus: float = 0.01,
        inside_goal_bonus: float = 0.02,
        yaw_abs_penalty: float = 0.06,
        exit_goal_penalty: float = 0.20,
        action_smooth_penalty: float = 0.01,
        idle_penalty: float = 0.01,
        idle_speed_thresh: float = 0.03,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        
        self.yaw_threshold = float(yaw_threshold)
        self.goal_yaw_sector_prob = float(np.clip(goal_yaw_sector_prob, 0.0, 1.0))
        self.goal_yaw_sector_half_angle_deg = float(goal_yaw_sector_half_angle_deg)
        self.goal_yaw_sector_half_angle = math.radians(self.goal_yaw_sector_half_angle_deg)
        
        # 擴充 Observation Space，將 agent 狀態變成 4 維 [x_rel, y_rel, theta, yaw_err]
        self.observation_space = spaces.Dict({
            "agent": spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),
            "laser": spaces.Box(low=0.0, high=1.0, shape=(self._laser_n,), dtype=np.float32),
        })

        # 新增目標朝向的狀態記錄
        self._target_yaw_world = 0.0
        self._prev_yaw_err = None
        
        # --- Stage2 reward tweaks (dense obstacles / anti-dither) ---
        # Override Stage1 step penalty to make "keep moving" more attractive.
        self.stage2_step_penalty = float(stage2_step_penalty) if stage2_step_penalty is not None else None
        
        # Replace Stage1 linear clearance penalty with a squared penalty (less punitive near the threshold,
        # still discourages very small clearance).
        self.clearance_square = bool(clearance_square)
        self.timeout_penalty = float(timeout_penalty)
        
        # Near-goal "ring" shaping: stabilize pose alignment without distance jitter.
        self.near_goal_mult = float(near_goal_mult)
        self.ring_bonus = float(ring_bonus)  # bonus when inside near-goal ring but not yet in goal circle
        self.inside_goal_bonus = float(inside_goal_bonus)
        self.yaw_abs_penalty = float(yaw_abs_penalty)
        self.exit_goal_penalty = float(exit_goal_penalty)
        
        # Reduce in-place oscillation / action chattering.
        self.action_smooth_penalty = float(action_smooth_penalty)
        self.idle_penalty = float(idle_penalty)
        self.idle_speed_thresh = float(idle_speed_thresh)
        
        # per-episode state
        self._prev_action = None
        self._was_in_goal_zone = False

    def _compute_yaw_err(self):
        pose = self.ros.get_robot_pose2d()
        if pose is None:
            return None
        _, _, ryaw = pose
        # 計算最短旋轉角度
        return wrap_to_pi(self._target_yaw_world - ryaw)

    def _sample_target_yaw_world(self):
        """Sample target yaw for Stage2.

        With probability `goal_yaw_sector_prob`, sample within ±half_angle around the
        start->goal line heading. Otherwise sample from full [-pi, pi].
        """
        sx, sy = float(self._start_world[0]), float(self._start_world[1])
        gx, gy = float(self._goal_world[0]), float(self._goal_world[1])

        goal_line_yaw = math.atan2(gy - sy, gx - sx)

        use_sector = bool(
            (self.goal_yaw_sector_prob > 0.0) and
            (self._rng.random() < self.goal_yaw_sector_prob)
        )

        if use_sector:
            delta = float(self._rng.uniform(
                -self.goal_yaw_sector_half_angle,
                self.goal_yaw_sector_half_angle,
            ))
            target_yaw = wrap_to_pi(goal_line_yaw + delta)
            mode = "sector"
        else:
            target_yaw = float(self._rng.uniform(-math.pi, math.pi))
            mode = "random"

        return float(target_yaw), float(goal_line_yaw), str(mode)

    def _get_obs(self):
        # 先取得 Stage 1 原本的 3D agent obs 與 雷射
        obs = super()._get_obs()
        agent_3d = obs["agent"]
        
        # 追加 Stage 2 的第 4 維: Yaw Error
        yaw_err = self._compute_yaw_err()
        if yaw_err is None:
            yaw_err = 0.0
            
        # 將誤差正規化到 [-1, 1]
        yaw_err_n = float(np.clip(yaw_err / math.pi, -1.0, 1.0))
        
        # 覆寫為 4D array
        agent_4d = np.array([agent_3d[0], agent_3d[1], agent_3d[2], yaw_err_n], dtype=np.float32)
        obs["agent"] = agent_4d
        return obs

    def reset(self, *, seed=None, options=None):
        # 攔截 reset，等待 Stage 1 將車子擺放完畢
        obs, info = super().reset(seed=seed, options=options)

        # 依機率決定終點朝向：
        # - sector: 以 start->goal 連線方向為中心，在左右 half-angle 內抽樣
        # - random: 維持全域隨機 [-pi, pi]
        self._target_yaw_world, goal_line_yaw, goal_yaw_mode = self._sample_target_yaw_world()

        # 更新視覺化：讓 Gazebo 中的 Target 轉向我們剛剛產生的角度
        self.ros.set_entity_state(
            self.target_name,
            float(self._goal_world[0]),
            float(self._goal_world[1]),
            yaw=self._target_yaw_world,
            z=0.05,
        )

        # 重新計算並記錄初始誤差
        yaw_err = self._compute_yaw_err()
        self._prev_yaw_err = float(yaw_err) if yaw_err is not None else math.pi

        # reset Stage2 shaping state
        self._prev_action = None
        self._was_in_goal_zone = False

        # 因為改變了 Target Yaw，必須更新 observation 重新回傳
        obs = self._get_obs()
        info["goal_yaw"] = float(self._target_yaw_world)
        info["goal_line_yaw"] = float(goal_line_yaw)
        info["goal_yaw_mode"] = str(goal_yaw_mode)
        info["goal_yaw_sector_prob"] = float(self.goal_yaw_sector_prob)
        info["goal_yaw_sector_half_angle_deg"] = float(self.goal_yaw_sector_half_angle_deg)
        info["yaw_err"] = float(self._prev_yaw_err)

        return obs, info

    def step(self, action):
        # 1. 暫時把 Stage 1 的距離門檻設為負數，這樣 Stage 1 就不會觸發「提早煞車與印出Log」
        orig_goal_threshold = self.goal_threshold
        self.goal_threshold = -1.0  

        # 2. 呼叫 Stage 1 的 step (只會處理移動、避障、出界、超時等邏輯)
        # For anti-dither shaping we want the same distance progress Stage1 used.
        prev_dist_before = float(self._prev_dist) if self._prev_dist is not None else None

        obs, base_reward, terminated, truncated, info = super().step(action)

        # 3. 恢復正確的距離門檻
        self.goal_threshold = orig_goal_threshold
        
        # 3.5) --- Stage 2 reward shaping adjustments (dense obstacles) ---
        dist = float(info.get("dist_to_goal", float("inf")))
        progress = (float(prev_dist_before) - dist) if prev_dist_before is not None else 0.0
        info["dist_progress"] = float(progress)
        
        # (a) Replace Stage1 linear clearance penalty with squared penalty
        if self.clearance_square:
            thr = float(self.near_obstacle_threshold)
            min_r_k1 = float(info.get("min_laser_k1", info.get("min_laser", float("inf"))))
            if min_r_k1 < thr and thr > 1e-6:
                lin = float(self.near_obstacle_penalty) * (thr - min_r_k1)
                sq = float(self.near_obstacle_penalty) * ((thr - min_r_k1) / thr) ** 2
                # Stage1 already subtracted `lin`; we add it back then apply `sq`.
                base_reward = float(base_reward) + lin - sq
                info["r_clearance_linear"] = -float(lin)
                info["r_clearance_square"] = -float(sq)
            else:
                info["r_clearance_linear"] = 0.0
                info["r_clearance_square"] = 0.0
        
        # (b) Override per-step penalty (Stage1 already added self.step_penalty)
        if self.stage2_step_penalty is not None:
            step_delta = float(self.stage2_step_penalty) - float(self.step_penalty)
            base_reward = float(base_reward) + step_delta
            info["r_step_delta"] = float(step_delta)
        else:
            info["r_step_delta"] = 0.0
        
        # (c) Mild penalty for being "idle" far from the goal to prevent oscillating-in-place + timeout
        near_goal_r = float(self.goal_threshold) * float(self.near_goal_mult)
        vx = float(info.get("vx", 0.0)); vy = float(info.get("vy", 0.0))
        v_xy = math.sqrt(vx * vx + vy * vy)
        if (dist > near_goal_r) and (v_xy < float(self.idle_speed_thresh)) and not (terminated or truncated):
            base_reward = float(base_reward) - float(self.idle_penalty)
            info["r_idle_penalty"] = -float(self.idle_penalty)
        else:
            info["r_idle_penalty"] = 0.0
        
        # (d) Action smoothness penalty to reduce dithering / chattering
        try:
            a = np.asarray(action, dtype=np.float32).reshape(-1)
            if self._prev_action is not None:
                da = a - np.asarray(self._prev_action, dtype=np.float32).reshape(-1)
                smooth_pen = float(self.action_smooth_penalty) * float(np.sum(da * da))
                base_reward = float(base_reward) - smooth_pen
                info["r_smooth_penalty"] = -float(smooth_pen)
            else:
                info["r_smooth_penalty"] = 0.0
            self._prev_action = a.copy()
        except Exception:
            info["r_smooth_penalty"] = 0.0
        
        # (e) Make timeout clearly worse than trying (reduces "freeze and time out" policy)
        if truncated and str(info.get("termination_reason", "")) == "timeout":
            base_reward = float(base_reward) - float(self.timeout_penalty)
            info["r_timeout_penalty"] = -float(self.timeout_penalty)
        else:
            info["r_timeout_penalty"] = 0.0
        
        # 4. --- Stage 2 專屬：Yaw 朝向獎勵與成功判定 ---
        yaw_err = self._compute_yaw_err()
        if yaw_err is None:
            yaw_err = float(self._prev_yaw_err) if self._prev_yaw_err is not None else math.pi
            
        prev_yaw_err = float(self._prev_yaw_err) if self._prev_yaw_err is not None else yaw_err
        
        # 角度誤差縮小了多少
        yaw_progress = abs(prev_yaw_err) - abs(yaw_err)
        self._prev_yaw_err = float(yaw_err)
        
        # 給予進度獎勵 
        yaw_reward = 1.5 * yaw_progress
        reward = float(base_reward) + float(yaw_reward)

        dist = info["dist_to_goal"]
        
        # --- Near-goal ring shaping: keep it in the goal zone while aligning yaw ---
        if not (terminated or truncated):
            near_goal_r = float(self.goal_threshold) * float(self.near_goal_mult)
            in_goal = bool(dist < float(self.goal_threshold))

            if dist < near_goal_r:
                # (方案一) Ring bonus: encourage entering the near-goal ring (but not yet the final goal circle)
                if not in_goal:
                    reward += float(self.ring_bonus)
                    info["r_ring_bonus"] = float(self.ring_bonus)
                else:
                    info["r_ring_bonus"] = 0.0

                # (方案二) Apply yaw_abs penalty ONLY inside the goal circle to avoid deterring entry
                if in_goal:
                    reward -= float(self.yaw_abs_penalty) * abs(float(yaw_err))
                    info["r_yaw_abs"] = -float(self.yaw_abs_penalty) * abs(float(yaw_err))
                else:
                    info["r_yaw_abs"] = 0.0

                # Encourage staying inside the goal circle while rotating to align
                if in_goal:
                    reward += float(self.inside_goal_bonus)
                    info["r_inside_goal"] = float(self.inside_goal_bonus)
                else:
                    info["r_inside_goal"] = 0.0

                # Penalize leaving the goal zone after entering (reduces "in-out wobble")
                if bool(self._was_in_goal_zone) and (not in_goal):
                    reward -= float(self.exit_goal_penalty)
                    info["r_exit_goal"] = -float(self.exit_goal_penalty)
                else:
                    info["r_exit_goal"] = 0.0

                self._was_in_goal_zone = in_goal
            else:
                info["r_ring_bonus"] = 0.0
                info["r_yaw_abs"] = 0.0
                info["r_inside_goal"] = 0.0
                info["r_exit_goal"] = 0.0
                self._was_in_goal_zone = bool(dist < float(self.goal_threshold))
        else:
            info["r_ring_bonus"] = 0.0
            info["r_yaw_abs"] = 0.0
            info["r_inside_goal"] = 0.0
            info["r_exit_goal"] = 0.0

        # --- Stage 2 自己的成功判定邏輯 ---
        # 如果還沒因為碰撞或超時結束，且進入了距離範圍
        if not (terminated or truncated) and dist < self.goal_threshold:
            # 檢查角度是否對齊
            if abs(yaw_err) <= self.yaw_threshold:
                # 距離與角度都達標 -> 真正成功
                terminated = True
                reward += 20.0  # 完美對齊獎勵
                info["termination_reason"] = "perfect_success"
                info["is_success"] = 1
                
                # 手動呼叫煞車與印出 Log (因為我們跳過了 Stage 1 的成功判定)
                self.ros.stop()
                print(f"[RESET] reason=perfect_success, step={self._step_count}, dist={dist:.2f}, yaw_err={yaw_err:.2f}")
            else:
                # 距離到了但角度不對：不結束，給予一點原地旋轉的動力 (不扣分，讓它專心拿 yaw_progress 獎勵)
                pass 

        info["yaw_err"] = float(yaw_err)

        return obs, float(reward), terminated, truncated, info
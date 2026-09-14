# amr_nav2_drl

This package integrates your Stage1 DRL controller into Nav2:

- `amr_nav2_drl::DrlController` (C++)
  - A Nav2 Controller plugin that publishes a **local goal** (lookahead point on the global plan) in `base_link` frame.
  - It then uses the latest cmd from a Python inference node.

- `drl_inference_node.py` (Python)
  - Loads a Stable-Baselines3 model (default: SAC)
  - Subscribes to `/scan` and `drl/local_goal`
  - Publishes `drl/cmd_vel` (`geometry_msgs/Twist`)

## Quick start (Gazebo + Nav2 + DRL)

```bash
# In your workspace
cd ~/sac_ws
colcon build --symlink-install
source install/setup.bash

# Start simulation + Nav2 + DRL
ros2 launch amr_nav2_drl amr_nav2_drl_sim.launch.py \
  model:=/absolute/path/to/sac_stage1.zip
```

Then in RViz2:
- Click **2D Pose Estimate** to set initial pose
- Click **Nav2 Goal** (or **2D Goal Pose**, depending on RViz panel) to send a goal

## Notes

- You must provide **/odom** and the TF **odom -> base_link**.
  - In simulation this typically comes from your Gazebo plugin (e.g., p3d).
  - On the real robot you can use wheel odometry or `robot_localization`.

- This controller ignores final yaw alignment by default (see goal checker params in `config/nav2_params_drl.yaml`).


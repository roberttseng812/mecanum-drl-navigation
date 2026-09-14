# amr_nav2_drl

Nav2 integration for the AMR150M SAC navigation policy.

- `amr_nav2_drl::DrlController`: selects a local goal from the global path and consumes inferred velocity commands.
- `drl_inference_node_stage3.py`: Stage 3 SAC inference using LiDAR history and goal state.
- `nav2_experiment_runner.py`: navigation goal execution and experiment logging.

See the [repository README](../README.md) for dependencies, simulation setup, model requirements, training, and real-robot navigation.

With dependencies and odometry configured, start Stage 3 simulation navigation using a compatible trained model:

```bash
ros2 launch amr_nav2_drl amr_nav2_drl_sim_stage3.launch.py \
  model:=/absolute/path/to/sac_stage3.zip
```

Model weights are not included. See [inference settings](config/drl_inference_stage3.yaml) and [Nav2 configuration](config/nav2_params_drl_stage3.yaml).

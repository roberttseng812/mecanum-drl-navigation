import numpy as np
import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class Stage3LaserConvExtractor(BaseFeaturesExtractor):
    """Conv-based feature extractor for Stage3 LiDAR history.

    Observation format:
      obs = {
        "agent": [4],
        "laser": [lidar_history_steps * laser_num_beams]
      }

    The laser vector is reshaped to [B, history_steps, laser_num_beams] and processed
    by Conv1d over the angular dimension. This keeps temporal history as channels,
    which is much closer to the MCAL paper than SB3's default flat MLP path.
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        laser_num_beams: int,
        lidar_history_steps: int,
        conv_channels: int = 32,
        lidar_latent_dim: int = 256,
        fused_hidden_dim: int = 128,
    ):
        if not isinstance(observation_space, spaces.Dict):
            raise TypeError("Stage3LaserConvExtractor expects a Dict observation space")
        if "agent" not in observation_space.spaces or "laser" not in observation_space.spaces:
            raise KeyError("Observation space must contain 'agent' and 'laser'")

        self.laser_num_beams = int(laser_num_beams)
        self.lidar_history_steps = int(lidar_history_steps)
        self.conv_channels = int(conv_channels)
        self.lidar_latent_dim = int(lidar_latent_dim)
        self.fused_hidden_dim = int(fused_hidden_dim)

        laser_shape = observation_space.spaces["laser"].shape
        agent_shape = observation_space.spaces["agent"].shape
        laser_dim = int(np.prod(laser_shape))
        agent_dim = int(np.prod(agent_shape))
        expected_laser_dim = self.laser_num_beams * self.lidar_history_steps
        if laser_dim != expected_laser_dim:
            raise ValueError(
                f"laser obs dim mismatch: got {laser_dim}, expected {expected_laser_dim} "
                f"(history_steps={self.lidar_history_steps}, beams={self.laser_num_beams})"
            )

        super().__init__(observation_space, features_dim=self.fused_hidden_dim)

        self.laser_encoder = nn.Sequential(
            nn.Conv1d(self.lidar_history_steps, self.conv_channels, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(self.conv_channels, self.conv_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with th.no_grad():
            dummy = th.zeros(1, self.lidar_history_steps, self.laser_num_beams)
            laser_flatten_dim = int(self.laser_encoder(dummy).shape[1])

        self.laser_projection = nn.Sequential(
            nn.Linear(laser_flatten_dim, self.lidar_latent_dim),
            nn.ReLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(self.lidar_latent_dim + agent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, self.fused_hidden_dim),
            nn.ReLU(),
        )

    def forward(self, observations) -> th.Tensor:
        laser = observations["laser"].view(-1, self.lidar_history_steps, self.laser_num_beams)
        laser_feat = self.laser_projection(self.laser_encoder(laser))
        agent = observations["agent"].view(observations["agent"].shape[0], -1)
        return self.fusion(th.cat([laser_feat, agent], dim=1))
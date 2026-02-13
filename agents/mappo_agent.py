import torch
import torch.nn as nn

class CentralizedCritic(nn.Module):
    """
    Centralized critic for MAPPO (CTDE)
    Menerima joint observation dari semua agen
    """

    def __init__(self, joint_obs_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(joint_obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, joint_obs):
        return self.net(joint_obs)

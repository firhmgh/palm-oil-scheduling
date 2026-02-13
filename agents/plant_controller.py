import torch
import torch.nn as nn

class PlantAgent(nn.Module):
    """
    Plant Controller Agent (MAPPO-compatible)
    Mengatur keputusan slot bongkar di pabrik
    """

    def __init__(self, obs_dim, action_dim):
        super().__init__()

        self.policy = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, obs):
        """
        obs: tensor shape (batch, obs_dim)
        return: logits untuk distribusi aksi
        """
        return self.policy(obs)

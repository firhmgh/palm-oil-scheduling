import torch
import torch.nn as nn

class SchedulerAgent(nn.Module):
    def __init__(self, obs_dim, hidden_dim, action_dim):
        super().__init__()
        self.lstm = nn.LSTM(obs_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, action_dim)
        self.hidden = None

    def forward(self, obs):
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)

        out, self.hidden = self.lstm(obs, self.hidden)

        # ⛔ penting: putus graph lama
        self.hidden = (
            self.hidden[0].detach(),
            self.hidden[1].detach()
        )

        logits = self.fc(out[:, -1, :])
        return logits

    def reset_hidden(self):
        self.hidden = None

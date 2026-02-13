import os
import csv

class Logger:
    def __init__(self, log_dir="logs", filename="training_log.csv"):
        os.makedirs(log_dir, exist_ok=True)
        self.filepath = os.path.join(log_dir, filename)

        with open(self.filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "episode",
                "reward",
                "policy_loss",
                "value_loss",
                "entropy"
            ])

    def log(self, episode, reward, policy_loss, value_loss, entropy):
        with open(self.filepath, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                episode,
                reward,
                policy_loss,
                value_loss,
                entropy
            ])

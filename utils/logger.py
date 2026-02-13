import os
import csv
import numpy as np

class Logger:
    def __init__(self, log_dir="logs", filename="training_log.csv"):
        os.makedirs(log_dir, exist_ok=True)
        self.filepath = os.path.join(log_dir, filename)

        # Header disesuaikan dengan Target Kinerja 
        self.headers = [
            "episode", 
            "reward", 
            "policy_loss", 
            "value_loss", 
            "entropy",
            "total_delay_minutes",   # Target: Turun 35%
            "total_throughput_tons", # Target: Naik 15%
            "total_operation_cost",  # Target: Naik max 5%
            "avg_queue_time"         # Target: < 45 menit
        ]

        with open(self.filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(self.headers)

    def log(self, episode, reward, policy_loss, value_loss, entropy, metrics_dict):
        """
        Menerima metrics_dict yang dikirim dari env.step(info)
        """
        with open(self.filepath, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                episode,
                round(reward, 4),
                round(policy_loss, 6),
                round(value_loss, 6),
                round(entropy, 6),
                round(metrics_dict.get('total_delay_minutes', 0), 2),
                round(metrics_dict.get('total_throughput_tons', 0), 2),
                round(metrics_dict.get('total_operation_cost', 2), 2),
                round(metrics_dict.get('avg_queue_time', 0), 2)
            ])
        
        # Print ke konsol agar bisa dimonitor saat training
        print(f"Ep {episode} | Reward: {reward:.2f} | Delay: {metrics_dict.get('total_delay_minutes', 0):.1f}m | Throughput: {metrics_dict.get('total_throughput_tons', 0):.1f}t")
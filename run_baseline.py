import sys
import os
import numpy as np
import csv

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from env.palm_env import PalmEnv
from baseline.heuristic import fifo_policy, oldest_first

def evaluate_baseline(policy_mode='fifo', episodes=5):
    print(f"📊 Mengevaluasi Baseline Mode: {policy_mode.upper()}...")
    
    # Inisialisasi Lingkungan (Gunakan domain_random=False agar hasil stabil untuk baseline)
    env = PalmEnv(domain_random=False)
    
    all_episode_rewards = []
    os.makedirs('logs', exist_ok=True)
    baseline_csv = 'logs/baseline_results.csv'

    # Tulis header jika file baru dibuat
    if not os.path.exists(baseline_csv):
        with open(baseline_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Mode", "Episode", "Delay_Minutes", "Throughput_Tons", "Avg_Queue_Time", "Total_Reward"])

    for ep in range(episodes):
        env.reset()
        done = False
        ep_reward = 0
        final_info = {}

        while not done:
            # Ambil keputusan berdasarkan policy
            if policy_mode == 'fifo':
                block_idx = fifo_policy(env.blocks)
            else:
                block_idx = oldest_first(env.blocks)

            # Kirim aksi ke lingkungan (Format action_dict sesuai PalmEnv terbaru)
            action_dict = {
                "scheduler": block_idx,
                "dispatcher": 0, # Default truk pertama
                "plant": 0       # Default prioritas normal
            }

            _, reward, done, info = env.step(action_dict)
            ep_reward += reward
            final_info = info

        all_episode_rewards.append(ep_reward)

        # Simpan hasil episode ini ke CSV
        with open(baseline_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                policy_mode, 
                ep, 
                round(final_info['total_delay_minutes'], 2), 
                round(final_info['total_throughput_tons'], 2), 
                round(final_info['avg_queue_time'], 2),
                round(ep_reward, 2)
            ])

        print(f"   Ep {ep} | Reward: {ep_reward:.2f} | Delay: {final_info['total_delay_minutes']:.1f}")

    np.save(f"logs/baseline_{policy_mode}_reward.npy", all_episode_rewards)
    
    print(f"✅ Baseline {policy_mode.upper()} selesai. Data disimpan di logs/baseline_results.csv\n")

if __name__ == "__main__":
    # Jalankan kedua metode pembanding
    evaluate_baseline('fifo', episodes=5)
    evaluate_baseline('oldest', episodes=5)
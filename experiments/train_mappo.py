import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
from torch.distributions import Categorical

from env.palm_env import PalmEnv
from agents.mappo_agent import CentralizedCritic
from agents.scheduler import SchedulerAgent
from agents.dispatcher import DispatcherAgent
from agents.plant_controller import PlantAgent
from utils.logger import Logger

def train_mappo(episodes=500): # Jumlah episode bisa ditambah sesuai waktu latih 8 jam
    # 1. Inisialisasi Lingkungan dan Logger
    env = PalmEnv()
    logger = Logger()

    # 2. Inisialisasi Agen (Sesuai Dimensi Observasi di palm_env.py)
    # Obs_dim disesuaikan dengan jumlah array yang dikembalikan get_observations()
    scheduler = SchedulerAgent(obs_dim=3, hidden_dim=64, action_dim=100) # 10x10 grid = 100 actions
    dispatcher = DispatcherAgent(obs_dim=3, action_dim=10) # Asumsi 10 truk
    plant = PlantAgent(obs_dim=3, action_dim=5)           # Asumsi 5 level prioritas

    # Centralized Critic: Melihat gabungan observasi ketiga agen (3+3+3 = 9)
    critic = CentralizedCritic(joint_obs_dim=9)

    optimizer = torch.optim.Adam(
        list(scheduler.parameters()) +
        list(dispatcher.parameters()) +
        list(plant.parameters()) +
        list(critic.parameters()),
        lr=1e-3
    )

    print("🚀 Memulai Training MAPPO - Palm Oil Logistics...")

    for ep in range(episodes):
        obs_dict = env.reset()
        # Jika Scheduler menggunakan LSTM/GRU, reset hidden state di awal episode
        if hasattr(scheduler, 'reset_hidden'):
            scheduler.reset_hidden()
            
        done = False
        ep_reward = 0
        policy_loss_sum = 0
        value_loss_sum = 0
        entropy_sum = 0
        
        # Variabel untuk menampung info metrik terakhir di episode ini
        last_info = {}

        while not done:
            # Ambil observasi terbaru untuk tiap agen
            obs = env.get_observations()

            # Konversi ke Tensor
            obs_s = torch.tensor(obs["scheduler"], dtype=torch.float32).unsqueeze(0)
            obs_d = torch.tensor(obs["dispatcher"], dtype=torch.float32).unsqueeze(0)
            obs_p = torch.tensor(obs["plant"], dtype=torch.float32).unsqueeze(0)

            # Generate Distribusi Aksi (Policy)
            dist_s = Categorical(logits=scheduler(obs_s))
            dist_d = Categorical(logits=dispatcher(obs_d))
            dist_p = Categorical(logits=plant(obs_p))

            # Sampling Aksi
            action_s = dist_s.sample()
            action_d = dist_d.sample()
            action_p = dist_p.sample()

            # Hitung Log Probability untuk update policy
            log_prob_s = dist_s.log_prob(action_s)
            log_prob_d = dist_d.log_prob(action_d)
            log_prob_p = dist_p.log_prob(action_p)

            # Entropy untuk mendorong eksplorasi
            entropy = (dist_s.entropy() + dist_d.entropy() + dist_p.entropy()).mean()

            # Centralized Critic: Evaluasi kondisi global (Joint Observation)
            joint_obs = torch.cat([obs_s, obs_d, obs_p], dim=-1)
            value = critic(joint_obs)

            # 3. Eksekusi Aksi di Lingkungan
            # Menangkap 'info' yang berisi metrik bisnis
            next_obs, reward, done, info = env.step({
                "scheduler": action_s.item(),
                "dispatcher": action_d.item(),
                "plant": action_p.item()
            })
            
            last_info = info # Simpan info terbaru (akumulasi episode)

            # 4. Hitung Loss (MAPPO logic sederhana)
            reward_tensor = torch.tensor(reward, dtype=torch.float32)
            advantage = reward_tensor - value.squeeze().detach()

            # Policy Loss (Actor)
            policy_loss = -(log_prob_s + log_prob_d + log_prob_p) * advantage

            # Value Loss (Critic)
            value_loss = (reward_tensor - value.squeeze()).pow(2)

            # Total Loss dengan Entropy Regularization
            loss = policy_loss.mean() + 0.5 * value_loss.mean() - 0.01 * entropy

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Akumulasi statistik untuk log
            ep_reward += reward
            policy_loss_sum += policy_loss.item()
            value_loss_sum += value_loss.item()
            entropy_sum += entropy.item()

        # 5. INTEGRASI LOGGER: Simpan hasil satu episode ke CSV
        # Mengirimkan ep_reward, loss, dan metrik bisnis dari 'last_info'
        logger.log(
            episode=ep, 
            reward=ep_reward, 
            policy_loss=policy_loss_sum, 
            value_loss=value_loss_sum, 
            entropy=entropy_sum, 
            metrics_dict=last_info
        )

        # Print progress secara berkala
        if ep % 5 == 0:
            print(f"--- Episode {ep} Summary ---")
            print(f"Delay: {last_info.get('total_delay_minutes', 0):.1f} min | Throughput: {last_info.get('total_throughput_tons', 0):.1f} ton")

    print(f"✅ Training Selesai. Log tersimpan di: {logger.filepath}")

if __name__ == "__main__":
    train_mappo()
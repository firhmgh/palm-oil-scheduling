import torch
from torch.distributions import Categorical

from env.palm_env import PalmEnv
from agents.mappo_agent import CentralizedCritic
from agents.scheduler import SchedulerAgent
from agents.dispatcher import DispatcherAgent
from agents.plant_controller import PlantAgent
from utils.logger import Logger


def train_mappo(episodes=50):
    env = PalmEnv()
    logger = Logger()

    scheduler = SchedulerAgent(obs_dim=3, hidden_dim=64, action_dim=10)
    dispatcher = DispatcherAgent(obs_dim=3, action_dim=5)
    plant = PlantAgent(obs_dim=3, action_dim=3)

    critic = CentralizedCritic(joint_obs_dim=9)

    optimizer = torch.optim.Adam(
        list(scheduler.parameters()) +
        list(dispatcher.parameters()) +
        list(plant.parameters()) +
        list(critic.parameters()),
        lr=1e-3
    )

    for ep in range(episodes):
        env.reset()
        scheduler.reset_hidden()
        done = False

        ep_reward = 0
        policy_loss_sum = 0
        value_loss_sum = 0
        entropy_sum = 0

        while not done:
            obs = env.get_observations()

            obs_s = torch.tensor(obs["scheduler"], dtype=torch.float32).unsqueeze(0)
            obs_d = torch.tensor(obs["dispatcher"], dtype=torch.float32).unsqueeze(0)
            obs_p = torch.tensor(obs["plant"], dtype=torch.float32).unsqueeze(0)

            dist_s = Categorical(logits=scheduler(obs_s))
            dist_d = Categorical(logits=dispatcher(obs_d))
            dist_p = Categorical(logits=plant(obs_p))

            action_s = dist_s.sample()
            action_d = dist_d.sample()
            action_p = dist_p.sample()

            log_prob_s = dist_s.log_prob(action_s)
            log_prob_d = dist_d.log_prob(action_d)
            log_prob_p = dist_p.log_prob(action_p)

            entropy = (dist_s.entropy() + dist_d.entropy() + dist_p.entropy()).mean()

            joint_obs = torch.cat([obs_s, obs_d, obs_p], dim=-1)
            value = critic(joint_obs)

            _, reward, done, _ = env.step({
                "scheduler": action_s.item(),
                "dispatcher": action_d.item(),
                "plant": action_p.item()
            })

            reward_tensor = torch.tensor(reward, dtype=torch.float32)
            advantage = reward_tensor - value.squeeze()

            policy_loss = -(
                log_prob_s +
                log_prob_d +
                log_prob_p
            ).mean() * advantage.detach()

            value_loss = advantage.pow(2).mean()

            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ep_reward += reward
            policy_loss_sum += policy_loss.item()
            value_loss_sum += value_loss.item()
            entropy_sum += entropy.item()

        logger.log(ep, ep_reward, policy_loss_sum, value_loss_sum, entropy_sum)

        print(
            f"Episode {ep} | "
            f"Reward: {ep_reward:.2f} | "
            f"Policy Loss: {policy_loss_sum:.3f} | "
            f"Value Loss: {value_loss_sum:.3f} | "
            f"Entropy: {entropy_sum:.3f}"
        )


if __name__ == "__main__":
    train_mappo()

from env.palm_env import PalmEnv
from baseline.heuristic import fifo_policy, oldest_first
import numpy as np

def evaluate_baseline(policy='fifo', episodes=10):
    rewards = []
    for ep in range(episodes):
        env = PalmEnv()
        state = env.reset()
        done = False
        ep_reward = 0
        while not done:
            if policy=='fifo':
                block_idx = fifo_policy(state['blocks'])
            else:
                block_idx = oldest_first(state['blocks'])
            action_dict = {'scheduler': {'block': block_idx, 'teams': 1}}
            state, reward, done, _ = env.step(action_dict)
            ep_reward += reward
        rewards.append(ep_reward)
        print(f"Episode {ep}, baseline {policy}, reward: {ep_reward}")
    np.save(f"logs/baseline_{policy}_reward.npy", rewards)
    print(f"Baseline {policy} selesai, reward disimpan di logs/")
    return rewards

if __name__ == "__main__":
    evaluate_baseline('fifo')
    evaluate_baseline('oldest')

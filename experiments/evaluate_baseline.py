from env.palm_env import PalmEnv
from baseline.heuristic import fifo_policy, oldest_first

def run_heuristic(policy_fn, episodes=10):
    env = PalmEnv(domain_random=False)
    total_rewards = []

    for ep in range(episodes):
        state = env.reset()
        done = False
        ep_reward = 0

        while not done:
            # heuristic hanya mengatur scheduler
            action = policy_fn(state["blocks"])

            state, reward, done, _ = env.step({
                "scheduler": action
            })

            ep_reward += reward

        total_rewards.append(ep_reward)

    avg_reward = sum(total_rewards) / episodes
    return avg_reward


if __name__ == "__main__":
    fifo_reward = run_heuristic(fifo_policy)
    oldest_reward = run_heuristic(oldest_first)

    print("=== BASELINE HEURISTIC ===")
    print(f"FIFO Average Reward        : {fifo_reward:.2f}")
    print(f"Oldest First Avg Reward   : {oldest_reward:.2f}")

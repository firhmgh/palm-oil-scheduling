from env.palm_env import PalmEnv

env = PalmEnv()
state = env.reset()
done = False
while not done:
    # random scheduler action
    import random
    block_idx = random.randint(0, 99)
    action_dict = {'scheduler': {'block': block_idx, 'teams': 1}}
    state, reward, done, _ = env.step(action_dict)
print("Simulator selesai")

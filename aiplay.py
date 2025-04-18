import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from game import Environment
from dqn import DQN, ReplayMemory


test_env = Environment(renderGame=True)
state = test_env.reset()
steps = 0

n_observations = len(state)
n_actions = 2



model = DQN(n_observations, n_actions)
model.load_state_dict(torch.load("model2.pth"))

done = False

while not done:
    steps += 1
    action = model.action(state)
    next_state, reward, done = test_env.step(action)
    state = next_state

    if not done:
        test_env.render()

        
print(f"Steps taken: {steps}")

test_env.quit()
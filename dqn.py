import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class DQN(nn.Module):

    def __init__(self, 
                 n_observations, 
                 n_actions,
                 epsilon_decay=0.9999):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

        self.epsilon = 1.0
        self.epsilon_decay = epsilon_decay
        self.n_actions = n_actions
        self.n_observations = n_observations

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
    
    def nextAction(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        else:
            with torch.no_grad():
                q_values = self(state)
                return q_values.argmax().item()
    
    def decayEpsilon(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, 0.01)
    

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = [None for _ in range(capacity)]
        self.position = 0

    def push(self, transition):
        self.memory[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
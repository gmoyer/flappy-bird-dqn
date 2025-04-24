from dqn import DQN, ReplayMemory
import random
import torch

class Agent:
    def __init__(self, n_observations, n_actions,
                 batch_size=32,
                 epsilon_decay=0.995,
                 lr=0.001,
                 gamma=0.99,
                 tau=0.01):

        self.n_observations = n_observations
        self.n_actions = n_actions

        self.batch_size = batch_size
        self.epsilon = 1.0
        self.gamma = gamma
        self.lr = lr
        self.tau = tau
        self.epsilon_decay = epsilon_decay

        self.policy_net = DQN(n_observations, n_actions)
        self.target_net = DQN(n_observations, n_actions)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.memory = ReplayMemory(10000)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        self.criterion = torch.nn.MSELoss()

    def nextAction(self, state):
        if random.random() < self.epsilon:
            return 1 if random.random() < 0.25 else 0
        else:
            return self.policy_net.action(state)
        
    def storeTransition(self, state, action, reward, next_state, done):
        self.memory.push((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)

        state_batch = torch.cat([t[0].unsqueeze(0) for t in transitions])
        action_batch = torch.tensor([t[1] for t in transitions])
        reward_batch = torch.tensor([t[2] for t in transitions])
        next_state_batch = torch.cat([t[3].unsqueeze(0) for t in transitions])
        done_batch = torch.tensor([t[4] for t in transitions])

        non_final_mask = torch.tensor([not done for done in done_batch], dtype=torch.bool)
        non_final_next_states = next_state_batch[non_final_mask]

        # Compute Q values for the current states
        state_action_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze()

        # Compute the expected Q values for the next states
        next_state_values = torch.zeros(self.batch_size)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute the loss
        loss = self.criterion(state_action_values, expected_state_action_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def updateTargetNetwork(self):
        for target_param, local_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_((1-self.tau) * target_param.data + self.tau * local_param.data)
    
    def decayEpsilon(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, 0.01)
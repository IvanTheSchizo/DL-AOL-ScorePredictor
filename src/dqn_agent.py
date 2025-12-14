import torch
import torch.nn as nn
import numpy as np
import random

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        return self.net(x)

class Agent:
    def __init__(self, state_dim, action_dim):
        self.dqn = DQN(state_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.dqn.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.gamma = 0.95

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_dim)
        
        with torch.no_grad():
            q_values = self.dqn(state)
            return torch.argmax(q_values).item()

    def train_step(self, state, action, reward, next_state, done):
        state = state.detach()
        
        q_values = self.dqn(state)
        q_value = q_values[action]
        
        with torch.no_grad():
            next_q = self.dqn(next_state).max()
            target_q = reward + (self.gamma * next_q * (1 - int(done)))
        
        loss = self.loss_fn(q_value, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if done and self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
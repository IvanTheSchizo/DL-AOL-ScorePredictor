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
        self.epsilon = 1.0
        self.gamma = 0.95

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_dim) # Correction: self.action_dim needs to be stored
        with torch.no_grad():
            return torch.argmax(self.dqn(state)).item()
import numpy as np
import torch
import random
import os

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

class OU_Noise:
    def __init__(self, action_dim, mu=0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dim) * self.mu
        
    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu
        
    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state

def make_dirs(paths):
    for p in paths:
        if not os.path.exists(p):
            os.makedirs(p)
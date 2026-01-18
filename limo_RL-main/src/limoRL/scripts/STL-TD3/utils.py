import numpy as np
import torch
import random
import os
import matplotlib.pyplot as plt 

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

class OU_Noise:
    """Ornstein-Uhlenbeck Process for exploration noise"""
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

def plot_learning_curve(scores, filename, window=50):
    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(scores[max(0, t-window):(t+1)])
    
    plt.figure(figsize=(10, 5))
    plt.plot(scores, alpha=0.3, color='gray', label='Raw')
    plt.plot(running_avg, color='blue', label='Average')
    plt.title('Training Learning Curve')
    plt.xlabel('Episodes')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()
import numpy as np
import torch
import params

class Buffer:
    def __init__(self, max_size=1000000):
        self.max_size = int(max_size) 
        self.ptr = 0
        self.size = 0
        self.device = params.DEVICE
        
        self.state = torch.empty((self.max_size, params.STATE_DIM), device=self.device)
        self.action = torch.empty((self.max_size, params.ACTION_DIM), device=self.device)
        self.next_state = torch.empty((self.max_size, params.STATE_DIM), device=self.device)
        
        # Dual Critic 专用
        self.r_stl = torch.empty((self.max_size, 1), device=self.device)
        self.r_aux = torch.empty((self.max_size, 1), device=self.device)
        
        self.not_done = torch.empty((self.max_size, 1), device=self.device)

    def add(self, state, action, next_state, r_s, r_a, done):
        self.state[self.ptr] = torch.FloatTensor(state).to(self.device)
        self.action[self.ptr] = torch.FloatTensor(action).to(self.device)
        self.next_state[self.ptr] = torch.FloatTensor(next_state).to(self.device)
        self.r_stl[self.ptr] = r_s
        self.r_aux[self.ptr] = r_a
        self.not_done[self.ptr] = 1. - done
        self.ptr = (self.ptr + 1) % self.max_size 
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        return (
            self.state[ind], self.action[ind], self.next_state[ind],
            self.r_stl[ind], self.r_aux[ind], self.not_done[ind]
        )
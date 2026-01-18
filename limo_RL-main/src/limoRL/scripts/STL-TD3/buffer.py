import numpy as np
import torch
import params

class ReplayBuffer:
    def __init__(self, max_size, state_dim, action_dim, batch_size):
        """
        :param max_size: 经验池最大容量 (例如 400000)
        :param state_dim: 状态空间维度 
        :param action_dim: 动作空间维度 (2)
        :param batch_size: 默认采样大小
        """
        self.max_size = int(max_size)
        self.ptr = 0
        self.size = 0
        self.batch_size = batch_size
        self.device = params.DEVICE
        # 如果显存不足 (OOM)，可以将 device 改为 'cpu'
        self.state = torch.empty((self.max_size, state_dim), device=self.device)
        self.action = torch.empty((self.max_size, action_dim), device=self.device)
        self.next_state = torch.empty((self.max_size, state_dim), device=self.device)
        
        # === Dual Critic 专用 ===
        # 分开存储任务奖励 (STL) 和辅助奖励 (Aux)
        self.r_stl = torch.empty((self.max_size, 1), device=self.device)
        self.r_aux = torch.empty((self.max_size, 1), device=self.device)
        
        self.not_done = torch.empty((self.max_size, 1), device=self.device)

    def add(self, state, action, next_state, r_s, r_a, done):
        """
        存入一条经验
        """
        # 使用 as_tensor 转换 numpy -> tensor，效率更高
        self.state[self.ptr] = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        self.action[self.ptr] = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        self.next_state[self.ptr] = torch.as_tensor(next_state, dtype=torch.float32, device=self.device)
        
        # 存入标量
        self.r_stl[self.ptr] = r_s
        self.r_aux[self.ptr] = r_a
        self.not_done[self.ptr] = 1.0 - float(done)
        
        # 指针循环移动
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size=None):
        """
        随机采样一个 Batch
        """
        if batch_size is None:
            batch_size = self.batch_size
            
        # 随机索引
        ind = np.random.randint(0, self.size, size=batch_size)
        
        return (
            self.state[ind], 
            self.action[ind], 
            self.next_state[ind], 
            self.r_stl[ind], 
            self.r_aux[ind], 
            self.not_done[ind]
        )
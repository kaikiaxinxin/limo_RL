import numpy as np
 
 
class ReplayBuffer:
    def __init__(self, max_size, state_dim, action_dim, batch_size):
        """
        初始化经验回放缓冲区。

        :param max_size: 缓冲区的最大容量
        :param state_dim: 状态空间的维度
        :param action_dim: 动作空间的维度
        :param batch_size: 每次采样的样本数量
        """
        self.mem_size = max_size  # 缓冲区的最大存储容量
        self.batch_size = batch_size  # 每次从缓冲区中采样的样本数量
        self.mem_cnt = 0  # 当前缓冲区中存储的样本数量

        # 初始化状态记忆数组，用于存储状态信息
        self.state_memory = np.zeros((max_size, state_dim))
        # 初始化动作记忆数组，用于存储动作信息
        self.action_memory = np.zeros((max_size, action_dim))
        # 初始化奖励记忆数组，用于存储奖励信息
        self.reward_memory = np.zeros((max_size, ))
        # 初始化下一个状态记忆数组，用于存储下一个状态信息
        self.next_state_memory = np.zeros((max_size, state_dim))
        # 初始化终止标志记忆数组，用于存储每个转换是否终止的标志
        self.terminal_memory = np.zeros((max_size, ), dtype=bool)

    def store_transition(self, state, action, reward, state_, done):
        """
        将一个经验转换存储到缓冲区中。

        :param state: 当前状态
        :param action: 执行的动作
        :param reward: 获得的奖励
        :param state_: 下一个状态
        :param done: 是否达到终止状态
        """
        # 计算当前存储位置的索引，使用取模运算实现循环存储
        mem_idx = self.mem_cnt % self.mem_size

        # 将当前状态存储到状态记忆数组中
        self.state_memory[mem_idx] = state
        # 将执行的动作存储到动作记忆数组中
        self.action_memory[mem_idx] = action
        # 将获得的奖励存储到奖励记忆数组中
        self.reward_memory[mem_idx] = reward
        # 将下一个状态存储到下一个状态记忆数组中
        self.next_state_memory[mem_idx] = state_
        # 将终止标志存储到终止标志记忆数组中
        self.terminal_memory[mem_idx] = done

        # 增加当前存储的样本数量
        self.mem_cnt += 1

    def sample_buffer(self):
        """
        从缓冲区中随机采样一批经验转换。

        :return: 采样的状态、动作、奖励、下一个状态和终止标志
        """
        # 计算当前缓冲区中实际可用的样本数量
        mem_len = min(self.mem_cnt, self.mem_size)
        # 从可用样本中随机选择一批样本的索引，不允许重复选择
        batch = np.random.choice(mem_len, self.batch_size, replace=False)

        # 根据采样的索引获取对应的状态
        states = self.state_memory[batch]
        # 根据采样的索引获取对应的动作
        actions = self.action_memory[batch]
        # 根据采样的索引获取对应的奖励
        rewards = self.reward_memory[batch]
        # 根据采样的索引获取对应的下一个状态
        states_ = self.next_state_memory[batch]
        # 根据采样的索引获取对应的终止标志
        terminals = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminals

    def ready(self):
        """
        检查缓冲区是否有足够的样本可以进行采样。

        :return: 如果缓冲区中的样本数量达到或超过批量大小，则返回 True，否则返回 False
        """
        return self.mem_cnt >= self.batch_size

# 导入 PyTorch 库，别名为 T，用于深度学习相关操作
import torch as T
# 导入 PyTorch 的函数式接口，用于常用的函数操作
import torch.nn.functional as F
# 导入 PyTorch 的神经网络模块
import torch.nn as nn
# 导入 PyTorch 的优化器模块
import torch.optim as optim

# 导入 NumPy 库，用于数值计算
import numpy as np
# 从当前目录下的 buffer 模块中导入 ReplayBuffer 类，用于经验回放
from .buffer import ReplayBuffer
# 导入 random 模块，用于生成随机数
import random

# 检查是否有可用的 GPU，如果有则使用 GPU 进行计算，否则使用 CPU
device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
# 打印当前使用的计算设备
print("device: ",device)

class ActorNetwork(nn.Module):
    def __init__(self, alpha, state_dim, action_dim, fc1_dim, fc2_dim, ACTION_limit_v, ACTION_limit_w):
        """
        初始化 Actor 网络。

        :param alpha: 学习率
        :param state_dim: 状态空间的维度
        :param action_dim: 动作空间的维度
        :param fc1_dim: 第一个全连接层的维度
        :param fc2_dim: 第二个全连接层的维度
        :param ACTION_limit_v: 线速度的最大限制
        :param ACTION_limit_w: 角速度的最大限制
        """
        # 调用父类的构造函数
        super(ActorNetwork, self).__init__()

        # 存储线速度的最大限制
        self.action_limit_v = ACTION_limit_v
        # 存储角速度的最大限制
        self.action_limit_w = ACTION_limit_w

        # 定义第一个全连接层，输入维度为状态空间维度，输出维度为 fc1_dim
        self.fc1 = nn.Linear(state_dim, fc1_dim)
        # 定义第二个全连接层，输入维度为 fc1_dim，输出维度为 fc2_dim
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        # 定义第三个全连接层，输入维度为 fc2_dim，输出维度为动作空间维度
        self.fc3 = nn.Linear(fc2_dim, action_dim)

        # 定义 Adam 优化器，用于更新网络参数
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)

        # 将网络移动到指定的计算设备上
        self.to(device)

    def forward(self, state):
        """
        前向传播函数。

        :param state: 输入的状态
        :return: 输出的动作
        """
        # 通过第一个全连接层并使用 ReLU 激活函数
        x = T.relu(self.fc1(state))
        # 通过第二个全连接层并使用 ReLU 激活函数
        x = T.relu(self.fc2(x))
        # 通过第三个全连接层得到动作
        action = self.fc3(x)
        # 对动作的第一个维度（线速度）使用 sigmoid 函数并乘以速度限制
        action[:, 0] = T.sigmoid(action[:, 0]) * self.action_limit_v
        # 对动作的第二个维度（角速度）使用 tanh 函数并乘以速度限制
        action[:, 1] = T.tanh(action[:, 1]) * self.action_limit_w

        return action

    def save_checkpoint(self, checkpoint_file):
        """
        保存网络的参数到指定文件。

        :param checkpoint_file: 保存参数的文件路径
        """
        T.save(self.state_dict(), checkpoint_file, _use_new_zipfile_serialization=False)

    def load_checkpoint(self, checkpoint_file):
        """
        从指定文件加载网络的参数。

        :param checkpoint_file: 加载参数的文件路径
        """
        self.load_state_dict(T.load(checkpoint_file))


class CriticNetwork(nn.Module):
    def __init__(self, beta, state_dim, action_dim, fc1_dim, fc2_dim):
        """
        初始化 Critic 网络。

        :param beta: 学习率
        :param state_dim: 状态空间的维度
        :param action_dim: 动作空间的维度
        :param fc1_dim: 第一个全连接层的维度
        :param fc2_dim: 第二个全连接层的维度
        """
        # 调用父类的构造函数
        super(CriticNetwork, self).__init__()
        # 定义第一个全连接层，输入维度为状态空间维度和动作空间维度之和，输出维度为 fc1_dim
        self.fc1 = nn.Linear(state_dim + action_dim, fc1_dim)
        # 定义第二个全连接层，输入维度为 fc1_dim，输出维度为 fc2_dim
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        # 定义输出层，输入维度为 fc2_dim，输出维度为 1
        self.q = nn.Linear(fc2_dim, 1)

        # 定义 Adam 优化器，用于更新网络参数
        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        # 将网络移动到指定的计算设备上
        self.to(device)

    def forward(self, state, action):
        """
        前向传播函数。

        :param state: 输入的状态
        :param action: 输入的动作
        :return: 输出的 Q 值
        """
        # 将状态和动作在最后一个维度拼接
        x = T.cat([state, action], dim=-1)
        # 通过第一个全连接层并使用 ReLU 激活函数
        x = T.relu(self.fc1(x))
        # 通过第二个全连接层并使用 ReLU 激活函数
        x = T.relu(self.fc2(x))
        # 通过输出层得到 Q 值
        q = self.q(x)
        return q

    def save_checkpoint(self, checkpoint_file):
        """
        保存网络的参数到指定文件。

        :param checkpoint_file: 保存参数的文件路径
        """
        T.save(self.state_dict(), checkpoint_file, _use_new_zipfile_serialization=False)

    def load_checkpoint(self, checkpoint_file):
        """
        从指定文件加载网络的参数。

        :param checkpoint_file: 加载参数的文件路径
        """
        self.load_state_dict(T.load(checkpoint_file))


class TD3:
    def __init__(self, alpha, beta, state_dim, action_dim, actor_fc1_dim, actor_fc2_dim,
                 critic_fc1_dim, critic_fc2_dim, ckpt_dir,action_limit_v,action_limit_w, gamma=0.99, tau=0.005, action_noise=0.1,
                 policy_noise=0.2, policy_noise_clip=0.5, delay_time=2, max_size=1000000,
                 batch_size=512,):
        """
        初始化 TD3 算法类。

        :param alpha: Actor 网络的学习率
        :param beta: Critic 网络的学习率
        :param state_dim: 状态空间的维度
        :param action_dim: 动作空间的维度
        :param actor_fc1_dim: Actor 网络第一个全连接层的维度
        :param actor_fc2_dim: Actor 网络第二个全连接层的维度
        :param critic_fc1_dim: Critic 网络第一个全连接层的维度
        :param critic_fc2_dim: Critic 网络第二个全连接层的维度
        :param ckpt_dir: 模型参数保存的目录
        :param action_limit_v: 线速度的最大限制
        :param action_limit_w: 角速度的最大限制
        :param gamma: 折扣因子
        :param tau: 软更新的参数
        :param action_noise: 动作噪声的标准差
        :param policy_noise: 策略噪声的标准差
        :param policy_noise_clip: 策略噪声的裁剪范围
        :param delay_time: 策略更新的延迟步数
        :param max_size: 经验回放缓冲区的最大容量
        :param batch_size: 每次训练的批次大小
        """
        # 存储折扣因子
        self.gamma = gamma
        # 存储软更新的参数
        self.tau = tau
        # 存储动作噪声的标准差
        self.action_noise = action_noise
        # 存储策略噪声的标准差
        self.policy_noise = policy_noise
        # 存储策略噪声的裁剪范围
        self.policy_noise_clip = policy_noise_clip
        # 存储策略更新的延迟步数
        self.delay_time = delay_time
        # 初始化更新次数
        self.update_time = 0
        # 存储模型参数保存的目录
        self.checkpoint_dir = ckpt_dir
        # 初始化开始的训练轮数
        self.start_epoch = 0
        # 存储每次训练的批次大小
        self.bath_size = batch_size

        # 初始化 Actor 网络
        self.actor = ActorNetwork(alpha=alpha, state_dim=state_dim, action_dim=action_dim,
                                  fc1_dim=actor_fc1_dim, fc2_dim=actor_fc2_dim,ACTION_limit_v =action_limit_v,ACTION_limit_w=action_limit_w)
        # 初始化第一个 Critic 网络
        self.critic1 = CriticNetwork(beta=beta, state_dim=state_dim, action_dim=action_dim,
                                     fc1_dim=critic_fc1_dim, fc2_dim=critic_fc2_dim)
        # 初始化第二个 Critic 网络
        self.critic2 = CriticNetwork(beta=beta, state_dim=state_dim, action_dim=action_dim,
                                     fc1_dim=critic_fc1_dim, fc2_dim=critic_fc2_dim)

        # 初始化目标 Actor 网络
        self.target_actor = ActorNetwork(alpha=alpha, state_dim=state_dim, action_dim=action_dim,
                                         fc1_dim=actor_fc1_dim, fc2_dim=actor_fc2_dim,ACTION_limit_v=action_limit_v,ACTION_limit_w=action_limit_w)
        # 初始化第一个目标 Critic 网络
        self.target_critic1 = CriticNetwork(beta=beta, state_dim=state_dim, action_dim=action_dim,
                                            fc1_dim=critic_fc1_dim, fc2_dim=critic_fc2_dim)
        # 初始化第二个目标 Critic 网络
        self.target_critic2 = CriticNetwork(beta=beta, state_dim=state_dim, action_dim=action_dim,
                                            fc1_dim=critic_fc1_dim, fc2_dim=critic_fc2_dim)

        # 初始化经验回放缓冲区
        self.memory = ReplayBuffer(max_size=max_size, state_dim=state_dim, action_dim=action_dim,
                                   batch_size=batch_size)
        # 初始化网络参数，第一次进行硬更新
        self.update_network_parameters(tau=1.0)

    def update_network_parameters(self, tau=None):
        """
        更新目标网络的参数。

        :param tau: 软更新的参数，如果为 None 则使用类中存储的 tau 值
        """
        if tau is None:
            tau = self.tau

        # 软更新目标 Actor 网络的参数
        for actor_params, target_actor_params in zip(self.actor.parameters(),
                                                     self.target_actor.parameters()):
            target_actor_params.data.copy_(tau * actor_params + (1 - tau) * target_actor_params)

        # 软更新第一个目标 Critic 网络的参数
        for critic1_params, target_critic1_params in zip(self.critic1.parameters(),
                                                         self.target_critic1.parameters()):
            target_critic1_params.data.copy_(tau * critic1_params + (1 - tau) * target_critic1_params)

        # 软更新第二个目标 Critic 网络的参数
        for critic2_params, target_critic2_params in zip(self.critic2.parameters(),
                                                         self.target_critic2.parameters()):
            target_critic2_params.data.copy_(tau * critic2_params + (1 - tau) * target_critic2_params)

    def remember(self, state, action, reward, state_, done):
        """
        将经验转换存储到经验回放缓冲区中。

        :param state: 当前状态
        :param action: 执行的动作
        :param reward: 获得的奖励
        :param state_: 下一个状态
        :param done: 是否达到终止状态
        """
        self.memory.store_transition(state, action, reward, state_, done)

    # 模型推理
    def choose_action(self, observation, train=True):
        """
        根据观测值选择动作。

        :param observation: 输入的观测值
        :param train: 是否处于训练模式，默认为 True
        :return: 选择的动作
        """
        # 将 Actor 网络设置为评估模式
        self.actor.eval()
        # 将观测值转换为 PyTorch 张量并移动到指定设备上
        state = T.tensor([observation], dtype=T.float).to(device)
        # 通过 Actor 网络得到动作
        action = self.actor.forward(state)

        if train:
            # 探索噪声
            noise = T.tensor(np.random.normal(loc=0.0, scale=self.action_noise),
                             dtype=T.float).to(device)
            # 对线速度添加噪声并进行裁剪
            action[0][0] = T.clamp(action[0][0] +noise, 0.1, 0.8)
            # 对角速度添加噪声并进行裁剪
            action[0][1] = T.clamp(action[0][1] +noise, -1.8, 1.8)

            # 以 0.05 的概率反转角速度
            if random.random() < 0.05:
                action[0][1] = -action[0][1]
        # 将 Actor 网络设置为训练模式
        self.actor.train()
        # 将动作从张量转换为 NumPy 数组并返回
        return action.squeeze().detach().cpu().numpy()

    def learn(self):
        """
        从经验回放缓冲区中采样数据进行训练。
        """
        # 如果经验回放缓冲区中的数据不足，直接返回
        if not self.memory.ready():
            return

        # 从经验回放缓冲区中采样一批数据
        states, actions, rewards, states_, terminals = self.memory.sample_buffer()
        # 将状态数据转换为 PyTorch 张量并移动到指定设备上
        states_tensor = T.tensor(states, dtype=T.float).to(device)
        # 将动作数据转换为 PyTorch 张量并移动到指定设备上
        actions_tensor = T.tensor(actions, dtype=T.float).to(device)
        # 将奖励数据转换为 PyTorch 张量并移动到指定设备上
        rewards_tensor = T.tensor(rewards, dtype=T.float).to(device)
        # 将下一个状态数据转换为 PyTorch 张量并移动到指定设备上
        next_states_tensor = T.tensor(states_, dtype=T.float).to(device)
        # 将终止标志数据转换为 PyTorch 张量并移动到指定设备上
        terminals_tensor = T.tensor(terminals).to(device)

        # 计算目标 Q 值，不需要计算梯度
        with T.no_grad():
            # 通过目标 Actor 网络得到下一个动作
            next_actions_tensor = self.target_actor.forward(next_states_tensor)
            # 生成策略噪声
            action_noise = T.tensor(np.random.normal(loc=0.0, scale=self.policy_noise),
                                    dtype=T.float).to(device)
            # 对策略噪声进行裁剪
            action_noise = T.clamp(action_noise, -self.policy_noise_clip, self.policy_noise_clip)
            # 对下一个动作添加噪声并进行裁剪
            next_actions_tensor = T.clamp(next_actions_tensor+action_noise, -1, 1)
            # 通过第一个目标 Critic 网络得到下一个状态的 Q 值
            q1_ = self.target_critic1.forward(next_states_tensor, next_actions_tensor).view(-1)
            # 通过第二个目标 Critic 网络得到下一个状态的 Q 值
            q2_ = self.target_critic2.forward(next_states_tensor, next_actions_tensor).view(-1)
            # 将终止状态的 Q 值设为 0
            q1_[terminals_tensor] = 0.0
            q2_[terminals_tensor] = 0.0
            # 取两个 Q 值中的最小值
            critic_val = T.min(q1_, q2_)
            # 计算目标 Q 值
            target = rewards_tensor + self.gamma * critic_val
        # 通过第一个 Critic 网络得到当前状态和动作的 Q 值
        q1 = self.critic1.forward(states_tensor, actions_tensor).view(-1)
        # 通过第二个 Critic 网络得到当前状态和动作的 Q 值
        q2 = self.critic2.forward(states_tensor, actions_tensor).view(-1)

        # 计算第一个 Critic 网络的损失，使用 Huber 损失
        critic1_loss = F.huber_loss(q1, target.detach())
        # 计算第二个 Critic 网络的损失，使用 Huber 损失
        critic2_loss = F.huber_loss(q2, target.detach())

        # 清空第一个 Critic 网络的梯度
        self.critic1.optimizer.zero_grad()
        # 反向传播计算第一个 Critic 网络的梯度
        critic1_loss.backward()
        # 更新第一个 Critic 网络的参数
        self.critic1.optimizer.step()

        # 清空第二个 Critic 网络的梯度
        self.critic2.optimizer.zero_grad()
        # 反向传播计算第二个 Critic 网络的梯度
        critic2_loss.backward()
        # 更新第二个 Critic 网络的参数
        self.critic2.optimizer.step()

        # 更新次数加 1
        self.update_time += 1
        # 如果更新次数不是延迟步数的整数倍，直接返回
        if self.update_time % self.delay_time != 0:
            return

        # 如果更新次数是延迟步数的整数倍，更新 Actor 网络
        if self.update_time % self.delay_time == 0:
            # 通过 Actor 网络得到新的动作
            new_actions_tensor = self.actor.forward(states_tensor)
            # 通过第一个 Critic 网络得到新动作的 Q 值
            q1 = self.critic1.forward(states_tensor, new_actions_tensor)
            # 计算 Actor 网络的损失，取负的 Q 值的均值
            actor_loss = -T.mean(q1)
            # 清空 Actor 网络的梯度
            self.actor.optimizer.zero_grad()
            # 反向传播计算 Actor 网络的梯度
            actor_loss.backward()
            # 更新 Actor 网络的参数
            self.actor.optimizer.step()
            # 更新目标网络的参数
            self.update_network_parameters()

    def save_models(self, episode):
        """
        保存所有网络的参数。

        :param episode: 当前的训练轮数
        """
        # 保存 Actor 网络的参数
        self.actor.save_checkpoint(self.checkpoint_dir + '/TD3_actor_{}.pth'.format(episode))
        print('Saving actor network successfully!')
        # 保存目标 Actor 网络的参数
        self.target_actor.save_checkpoint(self.checkpoint_dir +
                                          '/TD3_target_actor_{}.pth'.format(episode))
        print('Saving target_actor network successfully!')
        # 保存第一个 Critic 网络的参数
        self.critic1.save_checkpoint(self.checkpoint_dir + '/TD3_critic1_{}.pth'.format(episode))
        print('Saving critic1 network successfully!')
        # 保存第一个目标 Critic 网络的参数
        self.target_critic1.save_checkpoint(self.checkpoint_dir +
                                            '/TD3_target_critic1_{}.pth'.format(episode))
        print('Saving target critic1 network successfully!')
        # 保存第二个 Critic 网络的参数
        self.critic2.save_checkpoint(self.checkpoint_dir + '/TD3_critic2_{}.pth'.format(episode))
        print('Saving critic2 network successfully!')
        # 保存第二个目标 Critic 网络的参数
        self.target_critic2.save_checkpoint(self.checkpoint_dir +
                                            '/TD3_target_critic2_{}.pth'.format(episode))
        print('Saving target critic2 network successfully!')

    def load_models(self, episode):
        """
        加载所有网络的参数。

        :param episode: 要加载的训练轮数对应的模型参数
        """
        # 加载 Actor 网络的参数
        self.actor.load_checkpoint(self.checkpoint_dir + '/TD3_actor_{}.pth'.format(episode))
        print('Loading actor network successfully!')
        # 加载目标 Actor 网络的参数
        self.target_actor.load_checkpoint(self.checkpoint_dir +
                                          '/TD3_target_actor_{}.pth'.format(episode))
        print('Loading target_actor network successfully!')
        # 加载第一个 Critic 网络的参数
        self.critic1.load_checkpoint(self.checkpoint_dir + '/TD3_critic1_{}.pth'.format(episode))
        print('Loading critic1 network successfully!')
        # 加载第一个目标 Critic 网络的参数
        self.target_critic1.load_checkpoint(self.checkpoint_dir +
                                            '/TD3_target_critic1_{}.pth'.format(episode))
        print('Loading target critic1 network successfully!')
        # 加载第二个 Critic 网络的参数
        self.critic2.load_checkpoint(self.checkpoint_dir + '/TD3_critic2_{}.pth'.format(episode))
        print('Loading critic2 network successfully!')
        # 加载第二个目标 Critic 网络的参数
        self.target_critic2.load_checkpoint(self.checkpoint_dir +
                                            '/TD3_target_critic2_{}.pth'.format(episode))
        print('Loading target critic2 network successfully!')



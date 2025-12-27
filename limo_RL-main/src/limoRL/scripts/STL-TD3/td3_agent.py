import torch
import torch.nn.functional as F
import numpy as np
import copy
from td3_networks import Actor, Critic
import params

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
    def __init__(self, max_size=1e6):
        self.ptr = 0
        self.size = 0
        self.max_size = int(max_size)

        self.state = np.zeros((self.max_size, params.STATE_DIM))
        self.action = np.zeros((self.max_size, params.ACTION_DIM))
        self.next_state = np.zeros((self.max_size, params.STATE_DIM))
        # 重点：存储两个分离的奖励
        self.reward_stl = np.zeros((self.max_size, 1))
        self.reward_aux = np.zeros((self.max_size, 1))
        self.not_done = np.zeros((self.max_size, 1))

    def add(self, state, action, next_state, r_stl, r_aux, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward_stl[self.ptr] = r_stl
        self.reward_aux[self.ptr] = r_aux
        self.not_done[self.ptr] = 1. - done
        
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self):
        ind = np.random.randint(0, self.size, size=params.BATCH_SIZE)
        return (
            torch.FloatTensor(self.state[ind]).to(device),
            torch.FloatTensor(self.action[ind]).to(device),
            torch.FloatTensor(self.next_state[ind]).to(device),
            torch.FloatTensor(self.reward_stl[ind]).to(device), # STL 奖励
            torch.FloatTensor(self.reward_aux[ind]).to(device), # 辅助奖励
            torch.FloatTensor(self.not_done[ind]).to(device)
        )

class TD3_Dual_Critic:
    def __init__(self):
        # 1. Actor
        self.actor = Actor(params.STATE_DIM, params.ACTION_DIM).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=params.LR_ACTOR)

        # 2. STL Critic (只评估任务奖励)
        self.critic_stl = Critic(params.STATE_DIM, params.ACTION_DIM).to(device)
        self.critic_stl_target = copy.deepcopy(self.critic_stl)
        self.critic_stl_optimizer = torch.optim.Adam(self.critic_stl.parameters(), lr=params.LR_CRITIC)

        # 3. Aux Critic (只评估辅助奖励 - 安全与效率)
        self.critic_aux = Critic(params.STATE_DIM, params.ACTION_DIM).to(device)
        self.critic_aux_target = copy.deepcopy(self.critic_aux)
        self.critic_aux_optimizer = torch.optim.Adam(self.critic_aux.parameters(), lr=params.LR_CRITIC)

        self.total_it = 0

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer):
        self.total_it += 1

        # 从缓冲区采样 (包含分离的奖励)
        state, action, next_state, r_stl, r_aux, not_done = replay_buffer.sample()

        with torch.no_grad():
            # 目标策略平滑
            noise = (torch.randn_like(action) * params.POLICY_NOISE).clamp(-params.NOISE_CLIP, params.NOISE_CLIP)
            next_action = (self.actor_target(next_state) + noise).clamp(-1, 1)

            # --- 计算 STL Critic 的目标值 ---
            target_Q1_stl, target_Q2_stl = self.critic_stl_target(next_state, next_action)
            target_Q_stl = torch.min(target_Q1_stl, target_Q2_stl)
            target_Q_stl = r_stl + (not_done * params.GAMMA * target_Q_stl)

            # --- 计算 Aux Critic 的目标值 ---
            target_Q1_aux, target_Q2_aux = self.critic_aux_target(next_state, next_action)
            target_Q_aux = torch.min(target_Q1_aux, target_Q2_aux)
            target_Q_aux = r_aux + (not_done * params.GAMMA * target_Q_aux)

        # --- 更新 STL Critic ---
        current_Q1_stl, current_Q2_stl = self.critic_stl(state, action)
        loss_stl = F.mse_loss(current_Q1_stl, target_Q_stl) + F.mse_loss(current_Q2_stl, target_Q_stl)
        self.critic_stl_optimizer.zero_grad()
        loss_stl.backward()
        self.critic_stl_optimizer.step()

        # --- 更新 Aux Critic ---
        current_Q1_aux, current_Q2_aux = self.critic_aux(state, action)
        loss_aux = F.mse_loss(current_Q1_aux, target_Q_aux) + F.mse_loss(current_Q2_aux, target_Q_aux)
        self.critic_aux_optimizer.zero_grad()
        loss_aux.backward()
        self.critic_aux_optimizer.step()

        # --- 延迟更新 Actor ---
        if self.total_it % params.POLICY_FREQ == 0:
            # 论文A核心公式：Actor 最大化 Q_STL + Q_Aux
            # 使用 Q1 进行梯度更新
            q_stl_val = self.critic_stl.Q1(state, self.actor(state))
            q_aux_val = self.critic_aux.Q1(state, self.actor(state))
            
            # 两个 Q 值的和作为 Actor 的 loss
            actor_loss = -(q_stl_val + q_aux_val).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # 软更新目标网络
            for param, target_param in zip(self.critic_stl.parameters(), self.critic_stl_target.parameters()):
                target_param.data.copy_(params.TAU * param.data + (1 - params.TAU) * target_param.data)
            
            for param, target_param in zip(self.critic_aux.parameters(), self.critic_aux_target.parameters()):
                target_param.data.copy_(params.TAU * param.data + (1 - params.TAU) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(params.TAU * param.data + (1 - params.TAU) * target_param.data)
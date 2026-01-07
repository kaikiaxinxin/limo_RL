import torch
import torch.nn.functional as F
import numpy as np
import copy
import params

# [修正] 这里导入 SingleCritic，而不是 Critic
from networks import Actor, SingleCritic 

class TD3_Dual_Critic(object):
    def __init__(self):
        self.device = params.DEVICE
        
        # 自动获取维度
        state_dim = params.STATE_DIM
        action_dim = params.ACTION_DIM
        max_v = params.MAX_V
        max_w = params.MAX_W
        
        # === Actor ===
        self.actor = Actor(state_dim, action_dim, max_v, max_w).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=params.LR_ACTOR)
        
        # === Dual Critic (STL & Aux) ===
        # 1. STL Critic (只负责任务奖励)
        self.critic_stl = SingleCritic(state_dim, action_dim).to(self.device)
        self.critic_stl_target = copy.deepcopy(self.critic_stl)
        self.c_stl_optim = torch.optim.Adam(self.critic_stl.parameters(), lr=params.LR_CRITIC)
        
        # 2. Aux Critic (只负责安全/效率奖励)
        self.critic_aux = SingleCritic(state_dim, action_dim).to(self.device)
        self.critic_aux_target = copy.deepcopy(self.critic_aux)
        self.c_aux_optim = torch.optim.Adam(self.critic_aux.parameters(), lr=params.LR_CRITIC)
        
        self.total_it = 0

    def select_action(self, state):
        # 状态转 Tensor
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        return self.actor(state).cpu().data.numpy().flatten()

    def update(self, buffer):
        self.total_it += 1
        
        # 采样 (注意 buffer 返回 6 个值)
        state, action, next_state, r_stl, r_aux, not_done = buffer.sample(params.BATCH_SIZE)

        with torch.no_grad():
            # 目标动作平滑噪声
            noise = (torch.randn_like(action) * 0.2)
            # 注意：这里的 clamp 需要稍微宽松一点，或者针对 v/w 分别处理
            # 简单起见，这里先 clamp 到物理极限范围之外一点点也没关系，因为 Actor 输出会被截断
            noise[:, 0] = noise[:, 0].clamp(-0.2, 0.2) # v noise
            noise[:, 1] = noise[:, 1].clamp(-0.5, 0.5) # w noise
            
            next_action = self.actor_target(next_state) + noise
            # 物理限制截断
            next_action[:, 0] = next_action[:, 0].clamp(0, params.MAX_V)
            next_action[:, 1] = next_action[:, 1].clamp(-params.MAX_W, params.MAX_W)

            # --- 1. Target Q_STL ---
            target_Q1_s, target_Q2_s = self.critic_stl_target(next_state, next_action)
            target_Q_s = torch.min(target_Q1_s, target_Q2_s)
            target_Q_s = r_stl + (not_done * params.GAMMA * target_Q_s)
            
            # --- 2. Target Q_Aux ---
            target_Q1_a, target_Q2_a = self.critic_aux_target(next_state, next_action)
            target_Q_a = torch.min(target_Q1_a, target_Q2_a)
            target_Q_a = r_aux + (not_done * params.GAMMA * target_Q_a)

        # --- Update STL Critic (Huber Loss) ---
        current_Q1_s, current_Q2_s = self.critic_stl(state, action)
        loss_s = F.huber_loss(current_Q1_s, target_Q_s) + F.huber_loss(current_Q2_s, target_Q_s)
        
        self.c_stl_optim.zero_grad()
        loss_s.backward()
        self.c_stl_optim.step()
        
        # --- Update Aux Critic (Huber Loss) ---
        current_Q1_a, current_Q2_a = self.critic_aux(state, action)
        loss_a = F.huber_loss(current_Q1_a, target_Q_a) + F.huber_loss(current_Q2_a, target_Q_a)
        
        self.c_aux_optim.zero_grad()
        loss_a.backward()
        self.c_aux_optim.step()

        # --- Update Actor ---
        if self.total_it % params.POLICY_FREQ == 0:
            new_action = self.actor(state)
            
            # Actor 目标是最大化 (Q_STL + Q_Aux)
            q1_s, _ = self.critic_stl(state, new_action)
            q1_a, _ = self.critic_aux(state, new_action)
            
            actor_loss = -(q1_s + q1_a).mean()
            
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()
            
            # 软更新
            for param, target_param in zip(self.critic_stl.parameters(), self.critic_stl_target.parameters()):
                target_param.data.copy_(params.TAU * param.data + (1 - params.TAU) * target_param.data)
            
            for param, target_param in zip(self.critic_aux.parameters(), self.critic_aux_target.parameters()):
                target_param.data.copy_(params.TAU * param.data + (1 - params.TAU) * target_param.data)
            
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(params.TAU * param.data + (1 - params.TAU) * target_param.data)

    def save(self, filename):
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.critic_stl.state_dict(), filename + "_critic_stl")
        torch.save(self.critic_aux.state_dict(), filename + "_critic_aux")

    def load(self, filename):
        self.actor.load_state_dict(torch.load(filename + "_actor", map_location=self.device))
        self.critic_stl.load_state_dict(torch.load(filename + "_critic_stl", map_location=self.device))
        self.critic_aux.load_state_dict(torch.load(filename + "_critic_aux", map_location=self.device))
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_stl_target = copy.deepcopy(self.critic_stl)
        self.critic_aux_target = copy.deepcopy(self.critic_aux)
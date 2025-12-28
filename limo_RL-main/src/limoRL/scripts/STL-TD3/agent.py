import torch
import torch.nn.functional as F
import copy
import params
from networks import Actor, Critic

class TD3_Dual_Critic:
    def __init__(self):
        self.device = params.DEVICE
        
        # Actor
        self.actor = Actor(params.STATE_DIM, params.ACTION_DIM, params.MAX_ACTION).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=params.LR_ACTOR)
        
        # Critic 1: STL (任务)
        self.critic_stl = Critic(params.STATE_DIM, params.ACTION_DIM).to(self.device)
        self.critic_stl_target = copy.deepcopy(self.critic_stl)
        self.c_stl_optim = torch.optim.Adam(self.critic_stl.parameters(), lr=params.LR_CRITIC)
        
        # Critic 2: Aux (安全)
        self.critic_aux = Critic(params.STATE_DIM, params.ACTION_DIM).to(self.device)
        self.critic_aux_target = copy.deepcopy(self.critic_aux)
        self.c_aux_optim = torch.optim.Adam(self.critic_aux.parameters(), lr=params.LR_CRITIC)
        
        self.total_it = 0

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        return self.actor(state).cpu().data.numpy().flatten()

    def update(self, buffer):
        self.total_it += 1
        # 采样: r_stl 和 r_aux 是分开的！
        state, action, next_state, r_stl, r_aux, not_done = buffer.sample(params.BATCH_SIZE)

        with torch.no_grad():
            noise = (torch.randn_like(action) * 0.2).clamp(-0.5, 0.5)
            next_action = (self.actor_target(next_state) + noise).clamp(-params.MAX_ACTION, params.MAX_ACTION)
            
            # Target Q_STL
            target_Q1_s, target_Q2_s = self.critic_stl_target(next_state, next_action)
            target_Q_s = torch.min(target_Q1_s, target_Q2_s)
            target_Q_s = r_stl + (not_done * params.GAMMA * target_Q_s)
            
            # Target Q_Aux
            target_Q1_a, target_Q2_a = self.critic_aux_target(next_state, next_action)
            target_Q_a = torch.min(target_Q1_a, target_Q2_a)
            target_Q_a = r_aux + (not_done * params.GAMMA * target_Q_a)

        # Update STL Critic
        current_Q1_s, current_Q2_s = self.critic_stl(state, action)
        loss_s = F.mse_loss(current_Q1_s, target_Q_s) + F.mse_loss(current_Q2_s, target_Q_s)
        self.c_stl_optim.zero_grad()
        loss_s.backward()
        self.c_stl_optim.step()
        
        # Update Aux Critic
        current_Q1_a, current_Q2_a = self.critic_aux(state, action)
        loss_a = F.mse_loss(current_Q1_a, target_Q_a) + F.mse_loss(current_Q2_a, target_Q_a)
        self.c_aux_optim.zero_grad()
        loss_a.backward()
        self.c_aux_optim.step()

        # Update Actor
        if self.total_it % params.POLICY_FREQ == 0:
            q_s, _ = self.critic_stl(state, self.actor(state))
            q_a, _ = self.critic_aux(state, self.actor(state))
            
            actor_loss = -(q_s + q_a).mean()
            
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()
            
            # Soft update
            for param, target_param in zip(self.critic_stl.parameters(), self.critic_stl_target.parameters()):
                target_param.data.copy_(params.TAU * param.data + (1 - params.TAU) * target_param.data)
            for param, target_param in zip(self.critic_aux.parameters(), self.critic_aux_target.parameters()):
                target_param.data.copy_(params.TAU * param.data + (1 - params.TAU) * target_param.data)
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(params.TAU * param.data + (1 - params.TAU) * target_param.data)
                
    def save(self, filename):
        torch.save(self.actor.state_dict(), filename + "_actor.pth")
        torch.save(self.critic_stl.state_dict(), filename + "_critic_stl.pth")
        torch.save(self.critic_aux.state_dict(), filename + "_critic_aux.pth")
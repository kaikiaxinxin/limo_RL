import numpy as np
import pandas as pd
import os
import params
import time
import torch

class Trainer:
    def __init__(self, env, agent, buffer, noise):
        self.env = env
        self.agent = agent
        self.buffer = buffer
        self.noise = noise
        
        self.logs = {'step': [], 'reward': [], 'success': [], 'mode': []}

    def train(self):
        print("=== Start Training (F-MDP-S + Dual Critic) ===")
        state = self.env.reset(is_training=True)
        ep_reward = 0
        ep_steps = 0
        
        for t in range(params.TOTAL_STEPS):
            ep_steps += 1
            
            # 1. 动作选择 (Warmup + Noise)
            if t < params.START_STEPS:
                action = np.random.uniform(-1, 1, params.ACTION_DIM)
            else:
                action = self.agent.select_action(state)
                action = (action + self.noise.sample()).clip(-1, 1) # OU Noise

            # 2. 环境步进
            # r_tuple = (r_stl, r_aux)
            next_state, r_tuple, done, is_success = self.env.step(action)
            r_total = r_tuple[0] + r_tuple[1]
            
            # 3. 存储
            done_bool = float(done) if ep_steps < params.MAX_STEPS else 0
            self.buffer.add(state, action, next_state, r_tuple[0], r_tuple[1], done_bool)
            
            state = next_state
            ep_reward += r_total
            
            # 4. 更新
            if t >= params.START_STEPS:
                self.agent.update(self.buffer)
                
            # 5. 评估循环 (借鉴 trainer.py)
            if t > 0 and t % params.EVAL_INTERVAL == 0:
                self.evaluate(t)

            # 6. 回合结束
            if done or ep_steps >= params.MAX_STEPS:
                print(f"Train | Step: {t} | Reward: {ep_reward:.2f} | Success: {is_success}")
                self.log_data(t, ep_reward, is_success, 'train')
                
                state = self.env.reset(is_training=True)
                self.noise.reset()
                ep_reward = 0
                ep_steps = 0
                
                # 定期保存
                if t % 10000 == 0:
                    self.agent.save(os.path.join(params.MODEL_DIR, f"td3_{t}"))

    def evaluate(self, step):
        print(f"--- Evaluating at step {step} ---")
        success_count = 0
        avg_reward = 0
        
        for _ in range(params.EVAL_EPISODES):
            state = self.env.reset(is_training=False) # 无噪声 Reset
            done = False
            ep_r = 0
            while not done:
                action = self.agent.select_action(state) # 无噪声 Action
                state, r_tuple, done, succ = self.env.step(action)
                ep_r += (r_tuple[0] + r_tuple[1])
                if succ: success_count += 1
            avg_reward += ep_r
            
        success_rate = success_count / params.EVAL_EPISODES
        avg_reward /= params.EVAL_EPISODES
        print(f"Eval | Success Rate: {success_rate:.2f} | Avg Reward: {avg_reward:.2f}")
        self.log_data(step, avg_reward, success_rate, 'eval')

    def log_data(self, step, reward, success, mode):
        self.logs['step'].append(step)
        self.logs['reward'].append(reward)
        self.logs['success'].append(int(success))
        self.logs['mode'].append(mode)
        pd.DataFrame(self.logs).to_csv(os.path.join(params.LOG_DIR, "training_log.csv"), index=False)
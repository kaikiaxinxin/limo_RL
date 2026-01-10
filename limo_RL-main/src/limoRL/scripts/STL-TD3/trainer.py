import numpy as np
import pandas as pd
import os
import params
import time
import torch

# [新增] 导入绘图函数
# 请确保 plot_results.py 位于同一目录下
from plot_results import plot_training_curves 

class Trainer:
    def __init__(self, env, agent, buffer, noise):
        self.env = env
        self.agent = agent
        self.buffer = buffer
        self.noise = noise
        # 日志结构
        self.logs = {'step': [], 'reward': [], 'success': [], 'max_task': [], 'mode': []}
        
        # [新增] 用于控制绘图频率的计数器
        self.episode_count = 0 

    def train(self):
        print(f"=== Start Training (Generic Tasks: {params.NUM_TASKS} stages) ===")
        
        state = self.env.reset(is_training=True)
        ep_reward = 0
        ep_steps = 0
        
        for t in range(int(params.TOTAL_STEPS)):
            ep_steps += 1
            
            # 1. Select Action (适配物理限制)
            if t < params.START_STEPS:
                # [修改] 随机探索阶段：分别采样 v 和 w
                # v: [0, MAX_V], w: [-MAX_W, MAX_W]
                action_v = np.random.uniform(0.2, params.MAX_V)
                action_w = np.random.uniform(-params.MAX_W, params.MAX_W)
                action = np.array([action_v, action_w])
            else:
                # [修改] 网络决策阶段：增加噪声并分别裁剪
                raw_action = self.agent.select_action(state)
                noise = self.noise.sample()
                action = raw_action + noise
                
                # 分别裁剪 v 和 w
                action[0] = np.clip(action[0], 0.0, params.MAX_V)       # 线速度 >= 0
                action[1] = np.clip(action[1], -params.MAX_W, params.MAX_W) # 角速度对称

            # 2. Step
            next_state, r_tuple, done, is_success = self.env.step(action)
            r_total = r_tuple[0] + r_tuple[1]
            
            # 3. Add to Buffer
            done_bool = float(done) if ep_steps < params.MAX_STEPS else 0
            self.buffer.add(state, action, next_state, r_tuple[0], r_tuple[1], done_bool)
            
            state = next_state
            ep_reward += r_total
            
            # 4. Train
            if t >= params.START_STEPS:
                for _ in range(params.UPDATE_ITERATION):
                    self.agent.update(self.buffer)
                
            # 5. Evaluate
            if t > 0 and t % params.EVAL_INTERVAL == 0:
                self.evaluate(t)

            # 6. Save Model (建议每 5000 或 10000 步保存一次，太频繁会卡顿)
            if t > 0 and t % 5000 == 0:
                print(f"--- Saving checkpoint at step {t} ---") 
                self.agent.save(os.path.join(params.MODEL_DIR, f"td3_{t}"))

            # 7. Episode End
            if done or ep_steps >= params.MAX_STEPS:
                self.episode_count += 1
                
                # 获取当前进度
                progress = self.env.current_target_idx
                if is_success: progress = params.NUM_TASKS 
                
                print(f"Train | Step: {t} | Reward: {ep_reward:.2f} | Success: {is_success} | Task Progress: {progress}/{params.NUM_TASKS}")
                self.log_data(t, ep_reward, is_success, progress, 'train')
                
                # === [新增] 实时绘图逻辑 ===
                # 每 10 个回合更新一次图片 (避免太频繁拖慢训练)
                if self.episode_count % 10 == 0:
                    try:
                        log_file = os.path.join(params.LOG_DIR, "training_log.csv")
                        if os.path.exists(log_file):
                            plot_training_curves(log_file, window_size=50)
                    except Exception as e:
                        print(f"Warning: Plotting failed: {e}")

                state = self.env.reset(is_training=True)
                self.noise.reset()
                ep_reward = 0
                ep_steps = 0

    def evaluate(self, step):
        print(f"--- Evaluating at step {step} ---")
        
        # === [关键修复 1] 必须在这里初始化统计变量 ===
        success_count = 0
        avg_reward = 0
        avg_progress = 0  # <--- 之前报错就是因为缺了这一行！
        
        for _ in range(params.EVAL_EPISODES):
            state = self.env.reset(is_training=False)
            done = False
            ep_r = 0
            
            # === [关键修复 2] 初始化 succ 和计数器 ===
            succ = False      # 防止循环未执行导致 succ 未定义
            eval_step_count = 0 
            
            # [死循环保护] 增加步数限制
            while not done and eval_step_count < params.MAX_STEPS:
                eval_step_count += 1
                
                action = self.agent.select_action(state)
                # 动作裁剪
                action[0] = np.clip(action[0], 0.0, params.MAX_V)
                action[1] = np.clip(action[1], -params.MAX_W, params.MAX_W)
                
                state, r_tuple, done, succ = self.env.step(action)
                ep_r += (r_tuple[0] + r_tuple[1])
                
                if succ: 
                    success_count += 1
                    # 如果成功了，立即退出当前 Episode 循环（可选，加速评估）
                    # done = True 
            
            # 统计进度
            prog = self.env.current_target_idx
            if succ: prog = params.NUM_TASKS
            
            avg_progress += prog  # <--- 这里就是报错的地方 line 124
            avg_reward += ep_r
            
        success_rate = success_count / params.EVAL_EPISODES
        avg_reward /= params.EVAL_EPISODES
        avg_progress /= params.EVAL_EPISODES
        
        print(f"Eval | Success Rate: {success_rate:.2f} | Avg Reward: {avg_reward:.2f} | Avg Tasks: {avg_progress:.1f}/{params.NUM_TASKS}")
        self.log_data(step, avg_reward, success_rate, avg_progress, 'eval')
        
        if success_rate >= 0.8:
            print(f"*** Best Model Found (Success: {success_rate}) ***")
            self.agent.save(os.path.join(params.MODEL_DIR, f"best_model_{step}"))
            
        # 绘图
        try:
            plot_training_curves(os.path.join(params.LOG_DIR, "training_log.csv"))
        except Exception as e:
            print(f"Warning: Evaluation plotting failed: {e}")
    def log_data(self, step, reward, success, max_task, mode):
        self.logs['step'].append(step)
        self.logs['reward'].append(reward)
        self.logs['success'].append(int(success))
        self.logs['max_task'].append(max_task)
        self.logs['mode'].append(mode)
        pd.DataFrame(self.logs).to_csv(os.path.join(params.LOG_DIR, "training_log.csv"), index=False)
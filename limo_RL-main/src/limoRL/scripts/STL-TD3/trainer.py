import numpy as np
import pandas as pd
import os
import params
import time
import torch
# 导入绘图函数
from plot_results import plot_training_curves 

class Trainer:
    def __init__(self, env, agent, buffer, noise):
        self.env = env
        self.agent = agent
        self.buffer = buffer
        self.noise = noise
        # 日志结构
        self.logs = {'step': [], 'reward': [], 'success': [], 'max_task': [], 'mode': []}
        
        # 用于控制绘图频率的计数器
        self.episode_count = 0 

    # 增加 start_step 参数，默认为 0
    def train(self, start_step=0):
        print(f"=== Start Training (Generic Tasks: {params.NUM_TASKS} stages) ===")
        print(f"⏩ Resuming from step {start_step}...")
        
        state = self.env.reset(is_training=True)
        ep_reward = 0
        ep_steps = 0
        
        #循环从 start_step 开始
        for t in range(start_step, int(params.TOTAL_STEPS)):
            ep_steps += 1
            
            # 1. Select Action (适配物理限制)
            #只有当前步数确实小于随机探索阶段时，才使用随机动作
            # 如果是断点续训 (比如 t=15000)，这里条件为 False，直接进入网络决策
            if t < params.START_STEPS:
                # 随机探索阶段：分别采样 v 和 w
                # v: [0.2, MAX_V] (强制移动), w: [-MAX_W, MAX_W]
                action_v = np.random.uniform(0.2, params.MAX_V)
                action_w = np.random.uniform(-params.MAX_W, params.MAX_W)
                action = np.array([action_v, action_w])
            else:
                raw_action = self.agent.select_action(state)
                noise = self.noise.sample()
                action = raw_action + noise
                
                #动态最小速度限制 (Linear Decay)
                # 设定 20000 步的缓冲期，让 min_v 从 0.2 慢慢降到 0.0
                decay_steps = 20000.0
                steps_past = t - params.START_STEPS
                
                # 只有在缓冲期内才应用最小速度
                if 0 <= steps_past < decay_steps:
                    # 线性插值：(1.0 -> 0.0) * 0.2
                    current_min_v = 0.2 * (1.0 - (steps_past / decay_steps))
                else:
                    current_min_v = 0.0
                
                # 使用动态的 current_min_v 进行裁剪
                action[0] = np.clip(action[0], current_min_v, params.MAX_V)
                action[1] = np.clip(action[1], -params.MAX_W, params.MAX_W)

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

            # 6. Save Model (每 5000 步保存一次)
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
                
                # 实时绘图逻辑
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
        success_count = 0
        avg_reward = 0
        avg_progress = 0
        
        for _ in range(params.EVAL_EPISODES):
            state = self.env.reset(is_training=False)
            done = False
            ep_r = 0
            
            # 初始化 succ 和计数器
            succ = False      
            eval_step_count = 0 
            
            while not done and eval_step_count < params.MAX_STEPS:
                eval_step_count += 1
                
                action = self.agent.select_action(state)
                # 评估时严格截断，不需要最小速度限制，只要不越界即可
                action[0] = np.clip(action[0], 0.0, params.MAX_V)
                action[1] = np.clip(action[1], -params.MAX_W, params.MAX_W)
                
                state, r_tuple, done, succ = self.env.step(action)
                ep_r += (r_tuple[0] + r_tuple[1])
                
                if succ: 
                    success_count += 1
            
            # 统计进度
            prog = self.env.current_target_idx
            if succ: prog = params.NUM_TASKS
            
            avg_progress += prog
            avg_reward += ep_r
            
        success_rate = success_count / params.EVAL_EPISODES
        avg_reward /= params.EVAL_EPISODES
        avg_progress /= params.EVAL_EPISODES
        
        print(f"Eval | Success Rate: {success_rate:.2f} | Avg Reward: {avg_reward:.2f} | Avg Tasks: {avg_progress:.1f}/{params.NUM_TASKS}")
        self.log_data(step, avg_reward, success_rate, avg_progress, 'eval')
        
        if success_rate >= 0.8:
            print(f"*** Best Model Found (Success: {success_rate}) ***")
            self.agent.save(os.path.join(params.MODEL_DIR, f"best_model_{step}"))
            
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
        
        # 如果文件不存在则写入 header，否则追加模式 (防止覆盖)
        log_path = os.path.join(params.LOG_DIR, "training_log.csv")
        df = pd.DataFrame(self.logs)
        df.to_csv(log_path, index=False)
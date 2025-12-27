import os
import numpy as np
import params
from stl_env import STL_Gazebo_Env
from td3_agent import TD3_Dual_Critic, ReplayBuffer

def main():
    # 创建模型保存目录
    if not os.path.exists("./models"):
        os.makedirs("./models")

    # 初始化环境
    env = STL_Gazebo_Env()
    
    # 初始化智能体
    agent = TD3_Dual_Critic()
    replay_buffer = ReplayBuffer()
    
    total_steps = 0
    max_episodes = 2000
    
    print("=== Start Training STL-TD3 (Paper A) in Office World ===")
    
    for episode in range(max_episodes):
        obs = env.reset()
        episode_reward = 0
        episode_timesteps = 0
        done = False
        
        while not done:
            total_steps += 1
            episode_timesteps += 1
            
            # 选择动作 (加入噪声探索)
            if total_steps < 10000: # 预热
                action = np.random.uniform(-1, 1, params.ACTION_DIM)
            else:
                action = agent.select_action(np.array(obs))
                noise = np.random.normal(0, params.EXPLORE_NOISE, size=params.ACTION_DIM)
                action = (action + noise).clip(-1, 1)
            
            # 环境步进
            next_obs, total_reward, done, info = env.step(action)
            
            # 获取分解的奖励 (r_stl, r_aux)
            r_stl = info['r_stl']
            r_aux = info['r_aux']
            
            # 处理超时
            if episode_timesteps >= params.MAX_STEPS:
                done = True
                
            # 存入 buffer (mask=0 if done, else 1)
            done_bool = float(done) if episode_timesteps < params.MAX_STEPS else 0
            
            replay_buffer.add(obs, action, next_obs, r_stl, r_aux, done_bool)
            
            obs = next_obs
            episode_reward += total_reward
            
            # 训练
            if total_steps > 1000:
                agent.train(replay_buffer)
        
        print(f"Episode: {episode} | Reward: {episode_reward:.2f} | Stage: {env.c_t} | Steps: {total_steps}")
        
        # 定期保存
        if (episode + 1) % 50 == 0:
            torch.save(agent.actor.state_dict(), f"./models/actor_{episode}.pth")
            torch.save(agent.critic_stl.state_dict(), f"./models/critic_stl_{episode}.pth")
            torch.save(agent.critic_aux.state_dict(), f"./models/critic_aux_{episode}.pth")

if __name__ == "__main__":
    main()
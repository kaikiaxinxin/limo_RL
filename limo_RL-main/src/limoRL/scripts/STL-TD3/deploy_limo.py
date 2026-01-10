import rospy
import torch
import numpy as np
import os
import params
from agent import TD3_Dual_Critic
from stl_real_env_pro import STL_Real_Env

def main():
    rospy.init_node('stl_td3_deploy')
    
    # 1. 环境初始化
    try:
        env = STL_Real_Env()
    except Exception as e:
        print(f"Env Error: {e}")
        return

    # 2. Agent 初始化
    # agent.py 会读取 params.STATE_DIM，确保 params.py 里的维度配置正确
    agent = TD3_Dual_Critic()
    
    # 3. 模型加载
    # agent.load() 会自动补全 _actor 后缀，所以这里不要加后缀
    # 请确保你的模型文件名是 td3_5000_actor, td3_5000_critic_stl 等
    model_name = "best_model_5000" # 或者 "td3_5000"
    model_path = os.path.join(params.MODEL_DIR, model_name)
    
    print(f"Loading model: {model_path}...")
    if not os.path.exists(model_path + "_actor"):
        print(f"❌ Model file not found: {model_path}_actor")
        return
        
    agent.load(model_path)
    print("✅ Model loaded.")

    # 4. 主循环
    rate = rospy.Rate(10)
    print("🚀 Starting Autonomous Navigation...")
    
    try:
        while not rospy.is_shutdown():
            # 获取状态 (Dimension = 20 + 6 + Flags)
            state = env._get_obs()
            
            # 推理动作
            action = agent.select_action(state)
            
            # 执行
            env.step(action)
            
            # 打印调试
            dist = np.linalg.norm(np.array(env.pose_odom[:2]) - env.get_current_goal_pos())
            print(f"Task: {env.current_target_idx} | Dist: {dist:.2f}m | Act: [{action[0]:.2f}, {action[1]:.2f}]")
            
            rate.sleep()
            
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        env.stop()

if __name__ == '__main__':
    main()
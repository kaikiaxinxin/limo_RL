import rospy
import torch
import numpy as np
import os
import params
from agent import TD3_Dual_Critic
from stl_real_env import STL_Real_Env 

def main():
    rospy.init_node('stl_td3_deploy')
    try:
        env = STL_Real_Env()
        print("✅ Environment initialized.")
    except Exception as e:
        print(f"❌ Env Error: {e}")
        return

    # 2. Agent 初始化
    agent = TD3_Dual_Critic()
    
    # 3. 模型加载
    model_name = "td3_370000" 
    model_path = os.path.join(params.MODEL_DIR, model_name)
    
    print(f"🔄 Loading model from: {model_path} ...")
    if not os.path.exists(model_path + "_actor"):
        print(f"❌ Model file not found: {model_path}_actor")
        return
        
    agent.load(model_path)
    print("✅ Model loaded successfully.")

    # 4. 安全启动确认
    print("\n" + "="*40)
    print("⚠️  WARNING: Robot is about to move!")
    # 读取 env 中的 offset 用于提示
    print(f"   - Alignment Offset: {env.world_offset}")
    print("   - Please ensure the robot is facing the correct direction (Sim X+).")
    input("👉 Press Enter to START autonomous navigation...")
    print("="*40 + "\n")

    # 5. 主循环
    rate = rospy.Rate(5) 
    
    try:
        while not rospy.is_shutdown():
            # env._get_obs() 会自动调用 get_world_pose() 加上 offset
            state = env._get_obs()
            
            # 推理动作
            action = agent.select_action(state)
            
            # 安全截断
            safe_v = np.clip(action[0], 0.0, 0.4) 
            safe_w = np.clip(action[1], -1.0, 1.0)
            
            # 执行
            env.step(np.array([safe_v, safe_w]))
            
            # 打印任务进度 (用于调试)
            # 获取转换后的世界坐标
            curr_world_pos = env.get_world_pose()[:2]
            curr_goal = env.get_current_goal_pos()
            dist_to_goal = np.linalg.norm(curr_world_pos - curr_goal)
            
            print(f"Task: {env.current_target_idx} | WorldPos: ({curr_world_pos[0]:.1f}, {curr_world_pos[1]:.1f}) | Dist: {dist_to_goal:.2f}m")
            
            rate.sleep()
            
    except KeyboardInterrupt:
        print("\n🛑 Stopping by user request.")
    except Exception as e:
        print(f"\n❌ Runtime Error: {e}")
    finally:
        env.stop()
        print("👋 Robot Stopped.")

if __name__ == '__main__':
    main()
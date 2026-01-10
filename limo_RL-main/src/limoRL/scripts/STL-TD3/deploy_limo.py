import rospy
import torch
import numpy as np
import os
import params
from agent import TD3_Dual_Critic
# 假设您的实车环境文件名叫 stl_real_env_pro.py
from stl_real_env_pro import STL_Real_Env 

# === Sim-to-Real 坐标对齐配置 ===
# 实车开机时的原点 (0,0) 对应仿真世界中的哪个坐标？
# 仿真中 Robot 出生在 (-7.0, 0.0)，所以偏移量为 X=-7.0, Y=0.0
INITIAL_OFFSET_X = -7.0
INITIAL_OFFSET_Y = 0.0

def main():
    rospy.init_node('stl_td3_deploy')
    
    # 1. 环境初始化
    try:
        env = STL_Real_Env()
        print("✅ Environment initialized.")
    except Exception as e:
        print(f"❌ Env Error: {e}")
        return

    # 2. Agent 初始化
    # 确保 params.STATE_DIM 与训练时一致
    agent = TD3_Dual_Critic()
    
    # 3. 模型加载
    # 请修改为您效果最好的模型名称 (不要加后缀)
    model_name = "best_model_5000" 
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
    print(f"   - Alignment Offset: X={INITIAL_OFFSET_X}, Y={INITIAL_OFFSET_Y}")
    print("   - Please ensure the robot is facing the correct direction (Sim X+).")
    input("👉 Press Enter to START autonomous navigation...")
    print("="*40 + "\n")

    # 5. 主循环
    # [关键] 频率必须与训练时的 (1 / (DT * ACTION_REPEAT)) 一致
    # 假设 params.DT=0.1, ACTION_REPEAT=2 -> 5Hz
    rate = rospy.Rate(5) 
    
    try:
        while not rospy.is_shutdown():
            # === [核心修正] 坐标系注入 ===
            # 获取实车原始里程计数据 (相对于开机点)
            raw_x, raw_y = env.pose_odom[0], env.pose_odom[1]
            
            # 加上偏移量，转换为仿真世界坐标
            # 注意：这里直接修改 env 内部变量，以便 _get_obs() 计算相对目标距离时使用正确的世界坐标
            env.pose_odom[0] = raw_x + INITIAL_OFFSET_X
            env.pose_odom[1] = raw_y + INITIAL_OFFSET_Y
            
            # 获取状态 (网络输入)
            # 此时 _get_obs 内部计算的 distance 已经是基于世界坐标的了
            state = env._get_obs()
            
            # --- 调试打印 ---
            # 打印看一下转换后的坐标是否符合预期 (应该接近 -7.0, 0.0)
            # print(f"Odom Raw: ({raw_x:.2f}, {raw_y:.2f}) -> World: ({env.pose_odom[0]:.2f}, {env.pose_odom[1]:.2f})")
            
            # 推理动作
            action = agent.select_action(state)
            
            # [安全] 实车速度再次截断 (双重保险)
            # 即使训练时 Max_V 是 0.5，这里也可以限制得更死一点
            safe_v = np.clip(action[0], 0.0, 0.4) 
            safe_w = np.clip(action[1], -1.0, 1.0)
            
            # 执行动作 (传递给环境)
            # 注意：这里我们传原始 action 或 safe_action 都可以，建议传 safe
            env.step(np.array([safe_v, safe_w]))
            
            # 打印任务进度
            # 计算当前位置到当前目标的距离
            curr_goal = env.get_current_goal_pos()
            dist_to_goal = np.linalg.norm(np.array(env.pose_odom[:2]) - curr_goal)
            
            print(f"Task: {env.current_target_idx} | Dist: {dist_to_goal:.2f}m | V: {safe_v:.2f}, W: {safe_w:.2f}")
            
            rate.sleep()
            
    except KeyboardInterrupt:
        print("\n🛑 Stopping by user request.")
    except Exception as e:
        print(f"\n❌ Runtime Error: {e}")
    finally:
        # 退出时强制停车
        env.stop()
        print("👋 Robot Stopped.")

if __name__ == '__main__':
    main()
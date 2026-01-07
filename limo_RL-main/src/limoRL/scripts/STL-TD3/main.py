import argparse
import torch
import numpy as np
import os
import sys

# 导入自定义模块
import params
from stl_env import STL_Gazebo_Env
from agent import TD3_Dual_Critic  # 确保 agent.py 里类名一致
from buffer import ReplayBuffer     
from trainer import Trainer
from utils import OU_Noise

# === 1. 硬件与环境检测 ===
def check_environment():
    print("\n" + "="*40)
    print(f"🚀 STL-TD3-Dual-Critic Navigation Training")
    print("="*40)
    
    # 检查 CUDA
    if torch.cuda.is_available():
        print(f"✅ Hardware: CUDA Available")
        print(f"   - GPU: {torch.cuda.get_device_name(0)}")
        print(f"   - Index: {params.DEVICE}")
    else:
        print(f"❌ Hardware: CUDA NOT available. Using CPU (Slow!)")
    
    # 检查任务配置
    print("-" * 40)
    print(f"📋 Task Configuration (N={params.NUM_TASKS}):")
    for i, task in enumerate(params.TASK_CONFIG):
        print(f"   - Task {i}: Type={task['type']}, Pos={task['pos']}, Radius={task['radius']}m, Time={task['time']}s")
    
    # 检查状态维度
    print("-" * 40)
    print(f"🧠 State Space Dimensions:")
    print(f"   - Lidar: {params.LIDAR_DIM}")
    print(f"   - Robot: {params.ROBOT_STATE_DIM}")
    print(f"   - Flags: {params.FLAG_DIM} (2 * {params.NUM_TASKS} tasks)")
    print(f"   = TOTAL: {params.STATE_DIM}")
    print("="*40 + "\n")

def main():
    # === 2. 命令行参数 (支持断点续训) ===
    parser = argparse.ArgumentParser(description="TD3 STL Navigation")
    parser.add_argument("--seed", default=0, type=int, help="Random seed")
    parser.add_argument("--load_model", default="", type=str, help="Model step to load (e.g. '10000' or 'best_5000')")
    args = parser.parse_args()

    # 设置随机种子
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # 打印环境信息
    check_environment()
    
    # 创建目录
    if not os.path.exists(params.MODEL_DIR): os.makedirs(params.MODEL_DIR)
    if not os.path.exists(params.LOG_DIR): os.makedirs(params.LOG_DIR)

    # === 3. 模块实例化 ===
    print("🛠️  Initializing modules...")
    
    # 环境
    env = STL_Gazebo_Env()
    
    # 智能体
    # 注意：agent 内部会自动读取 params.STATE_DIM，所以这里不需要传参，或者根据您的 agent __init__ 修改
    agent = TD3_Dual_Critic() 
    
    # 如果指定了加载模型
    if args.load_model:
        model_path = os.path.join(params.MODEL_DIR, args.load_model)
        print(f"🔄 Loading checkpoint from: {model_path} ...")
        # 需要在 agent.py 中实现 load 函数，或者手动加载
        try:
            agent.load(model_path) # 假设您在 agent.py 里写了 load 方法
            print("✅ Model loaded successfully!")
        except Exception as e:
            print(f"⚠️  Failed to load model: {e}")
            print("   -> Starting from scratch.")

    # 经验回放池 
    # [修正] 根据您提供的 buffer.py，类名是 ReplayBuffer，且需要传参
    buffer = ReplayBuffer(
        max_size=int(params.TOTAL_STEPS), # 或者设个固定大值如 1e6
        state_dim=params.STATE_DIM,
        action_dim=params.ACTION_DIM,
        batch_size=params.BATCH_SIZE
    )
    
    # 噪声
    noise = OU_Noise(params.ACTION_DIM)
    
    # === 4. 训练托管 ===
    print("🟢 Starting Training Loop...")
    trainer = Trainer(env, agent, buffer, noise)
    
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n🛑 Training interrupted by user.")
    finally:
        # 这里可以加一些清理工作，比如保存当前未保存的模型
        print("👋 Exiting.")

if __name__ == "__main__":
    main()
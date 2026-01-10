import argparse
import torch
import numpy as np
import os
import sys

# 导入自定义模块
import params
from stl_env import STL_Gazebo_Env
from agent import TD3_Dual_Critic
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
    parser.add_argument("--load_model", default="", type=str, help="Model name to load (e.g. 'td3_15000')")
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
    agent = TD3_Dual_Critic() 
    
    # [新增] 起始步数变量
    start_step = 0

    # 如果指定了加载模型
    if args.load_model:
        model_path = os.path.join(params.MODEL_DIR, args.load_model)
        print(f"🔄 Loading checkpoint from: {model_path} ...")
        
        try:
            agent.load(model_path)
            print("✅ Model loaded successfully!")
            
            # [新增] 智能解析步数
            try:
                # 尝试从文件名 "td3_15000" 中提取 "15000"
                # 如果是 "best_model_5000"，也能提取出 "5000"
                if "best_model" in args.load_model:
                     # 最佳模型通常用于评估或微调，我们假设它已经过了随机阶段
                     # 这里给一个大于 START_STEPS 的值，或者解析后缀
                     parsed_step = int(args.load_model.split('_')[-1])
                     start_step = max(parsed_step, params.START_STEPS + 1)
                else:
                    # 标准 checkpoint
                    start_step = int(args.load_model.split('_')[-1])
                
                print(f"⏱️  Resuming training from step: {start_step}")
                
            except Exception as parse_err:
                print(f"⚠️  Could not parse step from filename ({parse_err}).")
                print(f"   -> Defaulting to params.START_STEPS + 1 ({params.START_STEPS + 1}) to skip random phase.")
                start_step = params.START_STEPS + 1
                
        except Exception as e:
            print(f"⚠️  Failed to load model: {e}")
            print("   -> Starting from scratch.")

    # 经验回放池 
    buffer = ReplayBuffer(
        max_size=int(params.TOTAL_STEPS), 
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
        # [修改] 将 start_step 传入 train 函数
        trainer.train(start_step=start_step)
    except KeyboardInterrupt:
        print("\n🛑 Training interrupted by user.")
    finally:
        print("👋 Exiting.")

if __name__ == "__main__":
    main()
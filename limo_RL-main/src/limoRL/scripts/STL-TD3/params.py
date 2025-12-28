import os
import torch

# === 路径配置 ===
CURR_PATH = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(CURR_PATH, "models")
LOG_DIR = os.path.join(CURR_PATH, "logs")

# === 任务区域 (Paper A: 序贯目标) ===
# 区域 A (必须先到达)
GOAL_A_POS = [6.0, -6.0]
# 区域 B (到达A后去这里)
GOAL_B_POS = [3.0, 9.0]
# 初始区域 (加入随机噪声后的中心)
INIT_POS = [-2.0, 0.0]
# 判定半径
AREA_RADIUS = 0.5

# === 状态空间 ===
LIDAR_DIM = 20            
ROBOT_STATE_DIM = 4       
FLAG_DIM = 4              
STATE_DIM = LIDAR_DIM + ROBOT_STATE_DIM + FLAG_DIM

# === 动作空间 ===
ACTION_DIM = 2
MAX_ACTION = 0.8          

# === 训练超参数 ===
MAX_STEPS = 1000           # 单回合最大步数
TOTAL_STEPS = 400000      
BATCH_SIZE = 512
GAMMA = 0.99
TAU = 0.005
LR_ACTOR = 3e-4
LR_CRITIC = 3e-4
START_STEPS = 5000       
POLICY_FREQ = 2
EVAL_INTERVAL = 5000      
EVAL_EPISODES = 5         

# === 奖励权重 ===
W_STAGE = 40.0       # 适当提高阶段奖励，鼓励完成任务
W_PROG = 2.0         # 稍微降低进度权重，防止它为了赶路而忽略安全
W_COLL = 100.0        # 大幅提高碰撞惩罚 (建议 50.0 或 100.0)
W_EFF = 0.05         # 稍微增加一点时间惩罚，鼓励快速到达

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
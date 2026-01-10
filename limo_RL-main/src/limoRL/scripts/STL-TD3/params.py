import os
import torch

# === 路径配置 ===
CURR_PATH = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(CURR_PATH, "models")
LOG_DIR = os.path.join(CURR_PATH, "logs")

# === [核心] 泛化任务配置 ===
# 定义序贯任务列表 (按执行顺序排列)
# type: 'F' (Finally/Reach), 'G' (Global/Stay - 预留接口)
# pos: [x, y]
# radius: 判定半径
# time: 时间窗口 (秒)
TASK_CONFIG = [
    {'type': 'F', 'pos': [6.0, -6.0], 'radius': 1.5, 'time': 15.0}, # 任务 0
    {'type': 'F', 'pos': [3.0, 9.0],  'radius': 1.5, 'time': 10.0}, # 任务 1
    # {'type': 'F', 'pos': [0.0, 0.0],  'radius': 0.5, 'time': 10.0}, # 示例: 扩展任务 2
]

NUM_TASKS = len(TASK_CONFIG)
DT = 0.1  # 控制周期 0.1s

INIT_POS = [-7.0, 0.0]  
# === 状态空间 ===
LIDAR_DIM = 20            
ROBOT_STATE_DIM = 6       # [x, y, cos, sin, v, w]

#标志位维度 = 任务数 * 2 (每个任务有 1个c_t 和 1个f_t)
FLAG_DIM = NUM_TASKS * 2              
STATE_DIM = LIDAR_DIM + ROBOT_STATE_DIM + FLAG_DIM

# === 动作空间 ===
# 分别定义线速度和角速度限制
MAX_V = 0.8  # m/s
MAX_W = 1.5  # rad/s (适当增大转弯能力)
ACTION_DIM = 2
# === 训练超参数 ===
MAX_STEPS = 1000          
TOTAL_STEPS = 1000000      
BATCH_SIZE = 256        
GAMMA = 0.99
TAU = 0.005
LR_ACTOR = 3e-4
LR_CRITIC = 3e-4
START_STEPS = 10000        
POLICY_FREQ = 2
EVAL_INTERVAL = 10000      
EVAL_EPISODES = 5         
UPDATE_ITERATION =1     
ACTION_REPEAT = 2         

# === 奖励权重 ===
# 在 # === 奖励权重 === 部分增加一行
W_STAGE = 40.0       
W_PROG = 10.0         
W_COLL = 50.0        
W_EFF = 0.08         
W_MOVE = 0.2  

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
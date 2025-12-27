import numpy as np

# ================= 任务区域定义 (根据您的world文件) =================
# 初始区域 (Init Area)
INIT_POS = [-7.0, 0.0] 
# 任务区域 A (Phi1 Area) - 必须先到达这里
GOAL_A_POS = [6.0, -6.0]
# 任务区域 B (Phi2 Area) - 到达A之后再去这里
GOAL_B_POS = [3.0, 9.0]

# 区域判定阈值 (米) - 考虑到box size是2x2，半径设为1.0
AREA_RADIUS = 1.0 

# ================= 状态与动作空间 =================
# 激光雷达特征数 (论文提到分组降采样，例如24维)
LIDAR_DIM = 24 
# 机器人物理状态 (x, y, yaw, v, w) 或简化为 (rel_dist, rel_angle, yaw)
ROBOT_STATE_DIM = 4 # [rel_dist_x, rel_dist_y, cos(yaw), sin(yaw)]
# 任务标志位 c_t (2个阶段: [Flag_A_done, Flag_B_done])
STAGE_FLAG_DIM = 2 
# STL时序标志位 f_t (2个任务的鲁棒度历史)
STL_FLAG_DIM = 2

# 总状态维度 z_t
STATE_DIM = LIDAR_DIM + ROBOT_STATE_DIM + STAGE_FLAG_DIM + STL_FLAG_DIM
# 动作维度 (v, w)
ACTION_DIM = 2
MAX_ACTION = 1.0 # 归一化动作

# ================= 训练超参数 (Dual Critic TD3) =================
MAX_STEPS = 500       # 单回合最大步数 (论文中有时间限制)
BATCH_SIZE = 256
GAMMA = 0.99
TAU = 0.005
LR_ACTOR = 3e-4
LR_CRITIC = 3e-4
EXPLORE_NOISE = 0.1
POLICY_NOISE = 0.2
NOISE_CLIP = 0.5
POLICY_FREQ = 2

# ================= 结构化奖励权重 (论文A核心) =================
W_STAGE = 50.0      # 阶段完成的大奖励 (r_stage)
W_PROG = 2.0        # 进度引导奖励权重 (r_progress)
W_COLL = 20.0       # 碰撞惩罚 (r_collision)
W_EFF = 0.05        # 效率/时间惩罚 (r_efficiency)
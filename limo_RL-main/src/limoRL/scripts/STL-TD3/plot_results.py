import matplotlib
# [关键修复] 必须在导入 pyplot 之前设置 'Agg' 后端
# 'Agg' 表示 Anti-Grain Geometry，是专门用于生成图像文件的非交互式后端
# 这行代码能完美解决 "main thread is not in main loop" 错误
matplotlib.use('Agg')

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

# === 1. 配置风格与配色 ===
COLORS = ["#C848b9", "#F962A7", "#FD836D", "#FFBA69"]

# 风格兼容性设置
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except OSError:
    try:
        plt.style.use('seaborn-whitegrid')
    except:
        plt.style.use('ggplot')

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.linewidth'] = 1.5

def smooth_data(data, window_size):
    """
    计算滑动平均和标准差范围
    """
    # 确保输入是 numpy array，防止 Pandas 版本兼容性问题
    data = np.array(data)
    
    if len(data) < window_size:
        return data, data, data
    
    # 使用 Pandas 的 rolling 方法进行平滑
    series = pd.Series(data)
    mean = series.rolling(window=window_size, min_periods=1).mean().values
    std = series.rolling(window=window_size, min_periods=1).std().fillna(0).values
    
    lower = mean - std
    upper = mean + std
    return mean, lower, upper

def plot_training_curves(log_path, window_size=50):
    if not os.path.exists(log_path):
        return

    # === 2. 读取数据 ===
    try:
        df = pd.read_csv(log_path)
    except pd.errors.EmptyDataError:
        return

    if df.empty:
        return
    
    # 分离 训练数据(train) 和 评估数据(eval)
    train_df = df[df['mode'] == 'train'].reset_index(drop=True)
    eval_df = df[df['mode'] == 'eval'].reset_index(drop=True)

    # 创建画布
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=False)
    
    # ==========================
    # 子图 1: Training Reward
    # ==========================
    if not train_df.empty:
        steps = train_df['step'].values
        rewards = train_df['reward'].values
        
        # 数据平滑
        smooth_r, lower_r, upper_r = smooth_data(rewards, window_size)
        
        # 绘图
        color = COLORS[0]
        ax1.plot(steps, smooth_r, color=color, linewidth=2, label='Smoothed Reward')
        ax1.fill_between(steps, lower_r, upper_r, facecolor=color, alpha=0.2)
        ax1.plot(steps, rewards, color=color, alpha=0.15, linewidth=0.5)
        
        ax1.set_title('Training Reward Curve', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Total Reward', fontsize=12)
        ax1.legend(loc='upper left')
        ax1.grid(True, linestyle='--', alpha=0.6)
    else:
        ax1.text(0.5, 0.5, 'No Training Data Yet', ha='center')

    # ==========================
    # 子图 2: Success Rate
    # ==========================
    if not train_df.empty:
        steps_t = train_df['step'].values
        succ_t = train_df['success'].values
        
        smooth_s_t, _, _ = smooth_data(succ_t, window_size * 2) 
        
        ax2.plot(steps_t, smooth_s_t, color=COLORS[1], linewidth=2, label='Train Success Rate')
        ax2.fill_between(steps_t, 0, smooth_s_t, facecolor=COLORS[1], alpha=0.1)

    if not eval_df.empty:
        steps_e = eval_df['step'].values
        succ_e = eval_df['success'].values
        
        ax2.plot(steps_e, succ_e, color=COLORS[2], marker='o', linewidth=2, label='Eval Success Rate')

    ax2.set_title('Success Rate (Train vs Eval)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Training Steps', fontsize=12)
    ax2.set_ylabel('Success Rate (0-1)', fontsize=12)
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend(loc='upper left')
    ax2.grid(True, linestyle='--', alpha=0.6)

    # === 3. 保存与显示 ===
    plt.tight_layout()
    save_path = os.path.join(os.path.dirname(log_path), 'training_curves.png')
    plt.savefig(save_path, dpi=300)
    
    # 关闭画布释放内存
    plt.close(fig)

if __name__ == "__main__":
    CURR_PATH = os.path.dirname(os.path.abspath(__file__))
    LOG_FILE = os.path.join(CURR_PATH, "logs", "training_log.csv")
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--window', type=int, default=50, help='Smoothing window size')
    args = parser.parse_args()

    plot_training_curves(LOG_FILE, window_size=args.window)
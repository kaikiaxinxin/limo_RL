```markdown
# 🚗 Limo 机器人 TD3-STL 算法实物部署指南

本文档详细说明了如何将训练好的 **TD3-STL 强化学习导航模型** 部署到 **松灵 Limo 实车** 上。内容涵盖硬件环境配置、核心代码实现、坐标采集方法、实验操作步骤及常见故障排查。

---

## 📋 1. 硬件与网络准备

### 1.1 设备清单
* **主机 (PC)**: 运行深度强化学习模型 (TD3-STL)，充当 ROS Master。
* **从机 (Limo)**: 运行底层驱动 (底盘 + 雷达)，接收控制指令。
* **网络环境**: 两者需连接同一局域网 (建议使用手机 5G 热点或独立路由器，避免校园网/公司内网的防火墙干扰)。

### 1.2 网络配置 (关键步骤)
假设 IP 分配如下（请根据实际终端输入 `ifconfig` 的结果修改）：
* **PC IP**: `172.20.10.5`
* **Limo IP**: `172.20.10.6`

#### PC 端配置
在 PC 终端执行 (或写入 `~/.bashrc`):
```bash
export ROS_MASTER_URI=[http://172.20.10.5:11311](http://172.20.10.5:11311)
export ROS_IP=172.20.10.5

```

#### Limo 端配置

SSH 登录 Limo 后执行:

```bash
export ROS_MASTER_URI=[http://172.20.10.5:11311](http://172.20.10.5:11311)
export ROS_IP=172.20.10.6

```

#### 验证连接

1. **PC 端**: 启动 `roscore`。
2. **Limo 端**: 运行 `rostopic list`。如果能看到话题列表，说明通信成功。

---

## 🛠️ 2. 核心程序文件

请在 PC 端的工作空间 `src/limoRL/scripts/STL-TD3/` 目录下创建或更新以下 3 个脚本。

### 2.1 配置文件 `params.py`

**用途**: 定义实物场景中的任务点坐标。需根据第 3 节“场地坐标采集”的结果进行修改。

```python
# 修改 TASK_CONFIG 部分，替换为你实际测量的坐标
TASK_CONFIG = [
    # 任务 0: 例如门口 (x, y)，半径建议放大到 0.5m 以适应里程计漂移
    {'type': 'F', 'pos': [2.5, -1.2], 'radius': 0.5, 'time': 20.0}, 
    
    # 任务 1: 例如走廊尽头
    {'type': 'F', 'pos': [5.0, 0.5],  'radius': 0.5, 'time': 20.0},
]

# 其他参数保持不变
# LIDAR_DIM = 20
# STATE_DIM = ...

```

### 2.2 实车环境接口 `stl_real_env_pro.py`

**用途**: 负责处理 `/limo/scan` 雷达数据和 `/odom` 里程计，并进行 Sim-to-Real 的对齐（归一化、降采样）。

```python
import rospy
import numpy as np
import math
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
import params 

class STL_Real_Env:
    def __init__(self):
        # 话题配置
        self.pub_cmd = rospy.Publisher('/cmd_vel', Twist, queue_size=5)
        self.sub_odom = rospy.Subscriber('/odom', Odometry, self._odom_cb)
        
        # 状态初始化
        self.scan_data = np.zeros(params.LIDAR_DIM)
        self.pose_odom = [0.0, 0.0, 0.0] 
        self.robot_vel = [0.0, 0.0]
        
        # 任务标志位
        self.num_tasks = params.NUM_TASKS
        self.c_t = np.zeros(self.num_tasks)
        self.f_t = np.full(self.num_tasks, -0.5)
        self.current_target_idx = 0
        
        print("Waiting for Limo connection...")
        try:
            # 兼容性检查：确保雷达和里程计都有数据
            rospy.wait_for_message('/limo/scan', LaserScan, timeout=5)
            rospy.wait_for_message('/odom', Odometry, timeout=5)
            print("✅ Connected to Limo!")
        except:
            print("❌ Connection failed! Check 'roslaunch limo_bringup limo_start.launch'")
            raise

    def _odom_cb(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny, cosy)
        self.pose_odom = [p.x, p.y, yaw]
        self.robot_vel = [msg.twist.twist.linear.x, msg.twist.twist.angular.z]

    def _process_scan(self, msg):
        raw = np.array(msg.ranges)
        # 数据清洗：将 inf 和 >5.0 的值截断为 5.0 (与训练归一化系数保持一致)
        raw[np.isinf(raw)] = 5.0
        raw[np.isnan(raw)] = 5.0
        raw[raw > 5.0] = 5.0
        
        # 降维 (例如 720 -> 20)
        chunk = len(raw) // params.LIDAR_DIM
        scan = []
        for i in range(params.LIDAR_DIM):
            scan.append(np.min(raw[i*chunk : (i+1)*chunk]))
        self.scan_data = np.array(scan)

    def get_current_goal_pos(self):
        idx = min(self.current_target_idx, self.num_tasks - 1)
        return np.array(params.TASK_CONFIG[idx]['pos'])

    def step(self, action):
        # 1. 动作执行 (安全限速 0.4 m/s)
        vel = Twist()
        real_v = np.clip(action[0], 0, 0.4) 
        real_w = np.clip(action[1], -params.MAX_W, params.MAX_W)
        vel.linear.x = real_v
        vel.angular.z = real_w
        self.pub_cmd.publish(vel)
        
        # 2. 同步观测 (阻塞等待新一帧雷达，确保决策实时性)
        try:
            scan_msg = rospy.wait_for_message('/limo/scan', LaserScan, timeout=0.5)
            self._process_scan(scan_msg)
        except:
            pass 
            
        # 3. 逻辑更新
        self._check_task_status()
        return self._get_obs()

    def _check_task_status(self):
        curr = np.array(self.pose_odom[:2])
        goal = self.get_current_goal_pos()
        dist = np.linalg.norm(curr - goal)
        if dist < params.TASK_CONFIG[self.current_target_idx]['radius']:
            print(f"🌟 Task {self.current_target_idx} Reached!")
            if self.current_target_idx < self.num_tasks - 1:
                self.current_target_idx += 1
                self.c_t[self.current_target_idx - 1] = 1.0 

    def _get_obs(self):
        # 归一化系数与训练保持一致 (5.0)
        scan = np.clip(self.scan_data / 5.0, 0, 1)
        rx, ry, ryaw = self.pose_odom
        goal = self.get_current_goal_pos()
        dx = goal[0] - rx
        dy = goal[1] - ry
        lx = dx * math.cos(ryaw) + dy * math.sin(ryaw)
        ly = -dx * math.sin(ryaw) + dy * math.cos(ryaw)
        robot = np.array([lx, ly, math.cos(ryaw), math.sin(ryaw), self.robot_vel[0], self.robot_vel[1]])
        flags = np.concatenate((self.c_t, self.f_t))
        return np.concatenate((scan, robot, flags))

    def stop(self):
        self.pub_cmd_vel.publish(Twist())

```

### 2.3 部署主程序 `deploy_limo_pro.py`

**用途**: 加载 PyTorch 模型并进行推理循环。

```python
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

    # 2. 加载模型
    agent = TD3_Dual_Critic()
    # 修改为你的最佳模型名 (不带 _actor 后缀)
    model_name = "best_model_5000" 
    model_path = os.path.join(params.MODEL_DIR, model_name)
    
    print(f"Loading model: {model_path}...")
    if not os.path.exists(model_path + "_actor"):
        print(f"❌ Model file not found: {model_path}_actor")
        return
        
    agent.load(model_path)
    print("✅ Model loaded.")

    # 3. 主循环
    rate = rospy.Rate(10) # 10Hz
    print("🚀 Starting Autonomous Navigation...")
    
    try:
        while not rospy.is_shutdown():
            state = env._get_obs()
            action = agent.select_action(state)
            env.step(action)
            
            dist = np.linalg.norm(np.array(env.pose_odom[:2]) - env.get_current_goal_pos())
            print(f"Task: {env.current_target_idx} | Dist: {dist:.2f}m | Act: [{action[0]:.2f}, {action[1]:.2f}]")
            rate.sleep()
            
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        env.stop()

if __name__ == '__main__':
    main()

```

---

## 📍 3. 场地坐标采集 (Calibration)

实车导航基于**里程计 (Odom)**，坐标原点 `(0,0)` 是**小车上电启动底盘驱动的位置**。因此，必须先手动采集目标点相对于起点的坐标。

**操作步骤：**

1. **定义原点**：用胶带在地上标记一个“出发点”，并规定车头朝向（X轴正方向）。
2. **启动 Limo**：将车摆好，SSH 运行:
```bash
roslaunch limo_bringup limo_start.launch pub_odom_tf:=false

```


3. **启动遥控**：PC 端运行:
```bash
roslaunch limo_bringup limo_teletop_keyboard.launch

```


4. **采集坐标**：
* 遥控小车开到任务 A 点。
* PC 终端查看坐标：`rostopic echo /odom/pose/pose/position -n 1`
* 记录 x, y 值。
* 继续开到任务 B 点，记录 x, y 值。


5. **更新配置**：将记录的值填入 `params.py` 的 `TASK_CONFIG` 中。

---

## 🚀 4. 实验操作流程

### 步骤 1: 物理就位

* 将 Limo 搬回胶带标记的 **原点 (0,0)**。
* 确保车头朝向正确（与采集坐标时一致）。

### 步骤 2: 启动底层 (Limo 端)

* 如果之前运行过，**必须重启** `limo_start.launch` 以清零里程计。
```bash
roslaunch limo_bringup limo_start.launch pub_odom_tf:=false

```



### 步骤 3: 启动导航 (PC 端)

* 确保 `roscore` 已由 Limo 或 PC 启动。
```bash
cd ~/your_ws/src/limoRL/scripts/STL-TD3
python3 deploy_limo_pro.py

```



### 步骤 4: 观察与急停

* 观察终端打印的距离信息。
* **急停**：若车失控，在运行 python 的终端狂按 `Ctrl+C`，程序会自动发送 0 速度。

---

## ❓ 5. 常见问题排查 (Troubleshooting)

| 现象 | 可能原因 | 解决方案 |
| --- | --- | --- |
| **卡在 "Waiting for Limo..."** | 网络不通或话题名错误 | 1. 互 `ping` 对方 IP。<br>

<br>2. 检查 `ROS_MASTER_URI`。<br>

<br>3. `rostopic list` 确认是否有 `/limo/scan`。 |
| **车原地打转/倒车** | 坐标系/电机方向反了 | 1. 检查 `stl_real_env_pro.py` 中是否需要给 `action[1]` 加负号。<br>

<br>2. 检查雷达是否装反 (RViz 查看)。 |
| **雷达数据全是 5.0** | 雷达被遮挡/驱动TF问题 | 1. PC 运行 `rviz`，Fixed Frame 选 `odom`，Add LaserScan，看点云是否显示在车身内。 |
| **未到终点就显示 Reached** | 里程计漂移过大 | 1. 缩短任务距离。<br>

<br>2. 在 `params.py` 中适当增大 `radius` (如 0.6m)。<br>

<br>3. 检查地面摩擦力。 |
| **报错 Model not found** | 模型路径/文件名不对 | 检查 `models` 文件夹，确认文件名是否为 `best_model_5000_actor`。修改脚本中的 `model_name`。 |

---

*Generated for Limo Robot RL Deployment.*

```

```
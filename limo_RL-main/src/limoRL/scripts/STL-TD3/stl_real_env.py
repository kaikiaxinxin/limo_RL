import rospy
import numpy as np
import math
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
import params # 完美调用 params.py

class STL_Real_Env:
    def __init__(self):
        # 话题配置 (保持 DDPG 的成功经验)
        self.pub_cmd = rospy.Publisher('/cmd_vel', Twist, queue_size=5)
        self.sub_odom = rospy.Subscriber('/odom', Odometry, self._odom_cb)
        
        # 状态初始化
        self.scan_data = np.zeros(params.LIDAR_DIM)
        self.pose_odom = [0.0, 0.0, 0.0] 
        self.robot_vel = [0.0, 0.0]
        
        # 任务标志位初始化 (完美适配 F-MDP-S)
        self.num_tasks = params.NUM_TASKS
        self.c_t = np.zeros(self.num_tasks)
        self.f_t = np.full(self.num_tasks, -0.5) # 保持与 stl_env.py 一致的默认值
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
        # 获取实车速度用于状态输入
        self.robot_vel = [msg.twist.twist.linear.x, msg.twist.twist.angular.z]

    def _process_scan(self, msg):
        raw = np.array(msg.ranges)
        # [关键修正] 保持与 stl_env.py 一致的数据清洗逻辑
        # 仿真中 inf 被设为 7.0，归一化分母是 5.0。
        # 这里我们把 > 5.0 的都截断为 5.0，保证输入网络的数据在 [0, 1] 范围内
        raw[np.isinf(raw)] = 5.0
        raw[np.isnan(raw)] = 5.0
        raw[raw > 5.0] = 5.0
        
        # 降维
        chunk = len(raw) // params.LIDAR_DIM
        scan = []
        for i in range(params.LIDAR_DIM):
            scan.append(np.min(raw[i*chunk : (i+1)*chunk]))
        self.scan_data = np.array(scan)

    def get_current_goal_pos(self):
        idx = min(self.current_target_idx, self.num_tasks - 1)
        return np.array(params.TASK_CONFIG[idx]['pos'])

    def step(self, action):
        # 1. 动作执行
        vel = Twist()
        # 实车安全限速 (0.4 m/s 比较稳妥，训练时是 0.8)
        # 注意：这里仅仅是物理限速，不改变输入网络的 action 值
        real_v = np.clip(action[0], 0, 0.4) 
        real_w = np.clip(action[1], -params.MAX_W, params.MAX_W)
        
        vel.linear.x = real_v
        vel.angular.z = real_w
        self.pub_cmd.publish(vel)
        
        # 2. 同步观测 (Block until new scan)
        try:
            scan_msg = rospy.wait_for_message('/limo/scan', LaserScan, timeout=0.5)
            self._process_scan(scan_msg)
        except:
            pass # 超时则沿用上一帧，防止卡死
            
        # 3. 逻辑更新
        self._check_task_status()
        
        return self._get_obs()

    def _check_task_status(self):
        # 简化的任务完成判定，仅用于切换目标
        curr = np.array(self.pose_odom[:2])
        goal = self.get_current_goal_pos()
        dist = np.linalg.norm(curr - goal)
        
        if dist < params.TASK_CONFIG[self.current_target_idx]['radius']:
            print(f"🌟 Task {self.current_target_idx} Reached!")
            if self.current_target_idx < self.num_tasks - 1:
                self.current_target_idx += 1
                self.c_t[self.current_target_idx - 1] = 1.0 # 更新状态向量里的 c_t

    def _get_obs(self):
        # [关键修正] 归一化系数必须是 5.0，与 stl_env.py 保持一致！
        scan = np.clip(self.scan_data / 5.0, 0, 1)
        
        rx, ry, ryaw = self.pose_odom
        goal = self.get_current_goal_pos()
        
        # 坐标变换 (Global -> Robot Frame)
        dx = goal[0] - rx
        dy = goal[1] - ry
        lx = dx * math.cos(ryaw) + dy * math.sin(ryaw)
        ly = -dx * math.sin(ryaw) + dy * math.cos(ryaw)
        
        # 拼装 Robot 状态 (6维)
        robot = np.array([lx, ly, math.cos(ryaw), math.sin(ryaw), self.robot_vel[0], self.robot_vel[1]])
        
        # 拼装 Flags (F-MDP-S 特有)
        flags = np.concatenate((self.c_t, self.f_t))
        
        return np.concatenate((scan, robot, flags))

    def stop(self):
        self.pub_cmd_vel.publish(Twist())
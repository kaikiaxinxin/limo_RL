import rospy
import numpy as np
import math
import params
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from collections import deque
from tf.transformations import euler_from_quaternion

class STL_Real_Env:
    def __init__(self):
        # 1. 订阅与发布
        self.pub_cmd = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.sub_scan = rospy.Subscriber('/scan', LaserScan, self._scan_cb) # 实车通常直接叫 /scan
        self.sub_odom = rospy.Subscriber('/odom', Odometry, self._odom_cb)
        
        # 2. 状态变量
        self.pose_odom = [0.0, 0.0, 0.0] # [x, y, yaw]
        self.robot_vel = [0.0, 0.0]      # [v, w]
        self.scan_data = np.ones(params.LIDAR_DIM) * 5.0
        
        self.has_odom = False
        self.latest_scan_msg = None
        
        # 3. STL 逻辑状态
        self.num_tasks = params.NUM_TASKS
        self.c_t = np.zeros(self.num_tasks)
        self.f_t = np.full(self.num_tasks, -0.5)
        
        # 历史窗口
        self.histories = []
        for task in params.TASK_CONFIG:
            win_len = int(task['time'] / params.DT)
            self.histories.append(deque(maxlen=win_len))
            
        self.current_target_idx = 0
        
        # 等待硬件就绪
        print("Waiting for Real Limo sensors...")
        while not self.has_odom or self.latest_scan_msg is None:
            rospy.sleep(0.1)
        print("✅ Sensors Ready.")

    def _scan_cb(self, msg):
        self.latest_scan_msg = msg

    def _odom_cb(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        
        orientation_list = [q.x, q.y, q.z, q.w]
        (roll, pitch, yaw) = euler_from_quaternion(orientation_list)
        
        self.pose_odom = [p.x, p.y, yaw]
        self.robot_vel = [msg.twist.twist.linear.x, msg.twist.twist.angular.z]
        self.has_odom = True

    def get_current_goal_pos(self):
        idx = min(self.current_target_idx, self.num_tasks - 1)
        return np.array(params.TASK_CONFIG[idx]['pos'])

    def step(self, action):
        # 1. 发送动作
        vel = Twist()
        vel.linear.x = action[0]
        vel.angular.z = action[1]
        self.pub_cmd.publish(vel)
        
        # 2. STL 逻辑更新 (在实车上只做判定，不训练)
        # 获取“世界坐标” (注意：deploy_limo.py 会在外部修改 self.pose_odom 加上 offset)
        curr_pos = np.array(self.pose_odom[:2]) 
        
        for k in range(self.num_tasks):
            task = params.TASK_CONFIG[k]
            target_pos = np.array(task['pos'])
            radius = task['radius']
            
            dist = np.linalg.norm(curr_pos - target_pos)
            rho = radius - dist
            is_satisfied = (rho >= 0)
            
            self.histories[k].append(is_satisfied)
            self.f_t[k] = self._calculate_f_score(self.histories[k], task['type'])
            
            # 状态机跳转
            if self.c_t[k] == 0:
                prev_done = (k == 0) or (self.c_t[k-1] == 1)
                if prev_done and is_satisfied:
                    self.c_t[k] = 1.0
                    print(f"🎉 Task {k} Reached! Moving to next.")
                    if k < self.num_tasks - 1:
                        self.current_target_idx = k + 1
                    else:
                        print("🏆 ALL TASKS COMPLETED.")

    def _calculate_f_score(self, history, task_type):
        hist_list = list(history)
        window_size = len(hist_list)
        if task_type == 'F':
            best_l = -1
            for l in range(window_size):
                if hist_list[l]: best_l = l
            if best_l == -1: return -0.5
            return ((best_l + 1) / window_size) - 0.5
        return -0.5

    def _get_obs(self):
        # === [核心修复] 雷达数据处理 ===
        if self.latest_scan_msg:
            raw = np.array(self.latest_scan_msg.ranges)
            
            # 1. 数据清洗
            raw[np.isinf(raw)] = 6.0
            raw[np.isnan(raw)] = 6.0
            
            # 2. [关键] 强制截断到 5.0m (Sim-to-Real Range Match)
            # 即使硬件能测到 6.0m，我们也只给网络看 5.0m
            raw = np.clip(raw, 0.0, 5.0)
            
            # 3. 降采样
            # 实车雷达已经是 180度 FOV (-90~90)，所以直接降采样即可
            chunk = len(raw) // params.LIDAR_DIM
            scan_list = []
            for i in range(params.LIDAR_DIM):
                seg = raw[i*chunk : (i+1)*chunk]
                if len(seg) > 0:
                    scan_list.append(np.min(seg))
                else:
                    scan_list.append(5.0)
            self.scan_data = np.array(scan_list)
        
        # 归一化 (匹配训练时的 /5.0)
        scan = np.clip(self.scan_data / 5.0, 0, 1)
        
        # 机器人状态
        rx, ry, ryaw = self.pose_odom
        goal = self.get_current_goal_pos()
        
        dx = goal[0] - rx
        dy = goal[1] - ry
        
        lx = dx * math.cos(ryaw) + dy * math.sin(ryaw)
        ly = -dx * math.sin(ryaw) + dy * math.cos(ryaw)
        
        v = self.robot_vel[0]
        w = self.robot_vel[1]
        
        robot = np.array([lx, ly, math.cos(ryaw), math.sin(ryaw), v, w])
        
        flags = np.concatenate((self.c_t, self.f_t))
        
        return np.concatenate((scan, robot, flags))

    def stop(self):
        self.pub_cmd.publish(Twist())
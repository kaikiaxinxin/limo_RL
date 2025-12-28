import rospy
import numpy as np
import math
import time
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
import params

class STL_Gazebo_Env:
    def __init__(self):
        try:
            rospy.init_node('stl_td3_node', anonymous=True)
        except:
            pass
        
        self.pub_cmd = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        # 回归最稳健的 reset_world
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_world', Empty)
        
        self.robot_pose = [0.0, 0.0, 0.0]
        self.robot_orientation = [0.0, 0.0] # [roll, pitch] 用于翻车检测
        self.scan_data = np.ones(params.LIDAR_DIM) * 10.0
        
        # F-MDP-S 状态
        self.c_t = np.array([0.0, 0.0]) 
        self.f_t = np.array([-0.5, -0.5]) 
        self.last_dist = 0.0
        
        rospy.sleep(1.0)

    def get_scan(self):
        try:
            # [修复1] 话题名称改为 /limo/scan，解决超时警告
            msg = rospy.wait_for_message('/limo/scan', LaserScan, timeout=1.0)
            raw = np.array(msg.ranges)
            raw[np.isinf(raw)] = 10.0
            raw[np.isnan(raw)] = 10.0
            
            chunk = len(raw) // params.LIDAR_DIM
            scan = []
            for i in range(params.LIDAR_DIM):
                scan.append(np.min(raw[i*chunk:(i+1)*chunk]))
            self.scan_data = np.array(scan)
            return True # 成功获取
        except Exception as e:
            # 仅在调试时打印，避免刷屏
            # rospy.logwarn(f"Get Scan failed: {e}") 
            return False

    def get_odom(self):
        try:
            msg = rospy.wait_for_message('/odom', Odometry, timeout=1.0)
            self.robot_pose[0] = msg.pose.pose.position.x
            self.robot_pose[1] = msg.pose.pose.position.y
            
            q = msg.pose.pose.orientation
            # Yaw
            siny = 2.0 * (q.w * q.z + q.x * q.y)
            cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            self.robot_pose[2] = math.atan2(siny, cosy)

            # [翻车检测数据] Roll & Pitch
            # Roll
            sinr = 2.0 * (q.w * q.x + q.y * q.z)
            cosr = 1.0 - 2.0 * (q.x * q.x + q.y * q.y)
            self.robot_orientation[0] = math.atan2(sinr, cosr)
            # Pitch
            sinp = 2.0 * (q.w * q.y - q.z * q.x)
            if abs(sinp) >= 1:
                self.robot_orientation[1] = math.copysign(math.pi / 2, sinp)
            else:
                self.robot_orientation[1] = math.asin(sinp)

            return True
        except:
            return False

    def get_current_goal(self):
        return np.array(params.GOAL_A_POS) if self.c_t[0] == 0 else np.array(params.GOAL_B_POS)

    def reset(self, is_training=True):
        # 1. 重置物理世界 (最稳健的方式)
        rospy.wait_for_service('/gazebo/reset_world')
        try:
            self.reset_proxy()
        except:
            pass
            
        # 2. 强制停车
        self.pub_cmd.publish(Twist())
        rospy.sleep(0.5) 
        
        # 3. 变量重置
        self.c_t = np.array([0.0, 0.0])
        self.f_t = np.array([-0.5, -0.5])
        
        # 4. 获取初始观测
        self.get_odom()
        self.get_scan()
        self.last_dist = np.linalg.norm(np.array(self.robot_pose[:2]) - self.get_current_goal())
        
        return self._get_obs()

    def step(self, action):
        # 动作执行
        vel = Twist()
        vel.linear.x = (action[0] + 1.0) / 2.0 * params.MAX_ACTION
        vel.angular.z = action[1] * 1.5
        self.pub_cmd.publish(vel)
        rospy.sleep(0.1)
        
        # 更新传感器数据
        self.get_odom()
        self.get_scan()
        
        # === 翻车检测 ===
        # 如果倾斜超过 0.6 弧度 (~35度)，强制结束
        if abs(self.robot_orientation[0]) > 0.6 or abs(self.robot_orientation[1]) > 0.6:
            print(">>> Robot Flipped! Resetting...")
            done = True
            is_success = False
            # 给予惩罚并返回
            r_aux = -50.0
            r_stl = 0.0
            # [修复2] 返回元组 (r_stl, r_aux) 以匹配 trainer.py
            return self._get_obs(), (r_stl, r_aux), done, is_success

        # === F-MDP-S 逻辑 ===
        curr_pos = np.array(self.robot_pose[:2])
        dist_A = np.linalg.norm(curr_pos - np.array(params.GOAL_A_POS))
        dist_B = np.linalg.norm(curr_pos - np.array(params.GOAL_B_POS))
        
        # 更新 STL 鲁棒度 f_t
        self.f_t[0] = 0.5 * np.tanh(params.AREA_RADIUS - dist_A)
        self.f_t[1] = 0.5 * np.tanh(params.AREA_RADIUS - dist_B)
        
        r_stl = 0.0
        r_aux = 0.0
        done = False
        is_success = False
        
        # 阶段跳转
        if self.c_t[0] == 0: # 阶段 1
            if dist_A < params.AREA_RADIUS:
                self.c_t[0] = 1.0
                r_stl += params.W_STAGE
                print(f"[{time.time():.2f}] Stage A Reached!")
                self.last_dist = dist_B 
        elif self.c_t[0] == 1 and self.c_t[1] == 0: # 阶段 2
            if dist_B < params.AREA_RADIUS:
                self.c_t[1] = 1.0
                r_stl += params.W_STAGE
                print(f"[{time.time():.2f}] Stage B Reached! Mission Complete!")
                done = True
                is_success = True
        
        # 进度奖励
        curr_target_dist = np.linalg.norm(curr_pos - self.get_current_goal())
        r_stl += params.W_PROG * (self.last_dist - curr_target_dist)
        self.last_dist = curr_target_dist
        
        # 碰撞检测
        if np.min(self.scan_data) < 0.25:
            r_aux -= params.W_COLL
            done = True
            print(">>> Collision!")
            
        r_aux -= params.W_EFF
        
        # [修复2] 关键修改：
        # trainer.py 期望: next_state, r_tuple, done, is_success = env.step()
        # 所以这里必须返回: obs, (r_stl, r_aux), done, is_success
        return self._get_obs(), (r_stl, r_aux), done, is_success

    def _get_obs(self):
        scan = np.clip(self.scan_data / 5.0, 0, 1)
        goal = self.get_current_goal()
        dx = goal[0] - self.robot_pose[0]
        dy = goal[1] - self.robot_pose[1]
        yaw = self.robot_pose[2]
        lx = dx * math.cos(yaw) + dy * math.sin(yaw)
        ly = -dx * math.sin(yaw) + dy * math.cos(yaw)
        
        robot = np.array([lx, ly, math.cos(yaw), math.sin(yaw)])
        return np.concatenate((scan, robot, self.c_t, self.f_t))
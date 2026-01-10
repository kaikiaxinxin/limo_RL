import rospy
import numpy as np
import math
import time
from collections import deque
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from gazebo_msgs.msg import ModelStates, ModelState
from gazebo_msgs.srv import SetModelState
from std_srvs.srv import Empty
from tf.transformations import quaternion_from_euler, euler_from_quaternion
import params

class STL_Gazebo_Env:
    def __init__(self):
        try:
            rospy.init_node('stl_td3_node', anonymous=True)
        except:
            pass
        
        self.pub_cmd = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_world', Empty)
        
        # 1. 裁判员数据 (Ground Truth)
        self.pose_gt = [0.0, 0.0, 0.0, 0.0, 0.0] # [x, y, yaw, roll, pitch]
        
        # 2. 运动员数据 (Odom)
        self.pose_odom = [0.0, 0.0, 0.0]
        self.robot_vel = [0.0, 0.0] 
        
        self.has_odom = False
        self.latest_scan_msg = None
        self.scan_data = np.ones(params.LIDAR_DIM) * 5.0 
        
        # STL 状态
        self.num_tasks = params.NUM_TASKS
        self.c_t = np.zeros(self.num_tasks)
        self.f_t = np.full(self.num_tasks, -0.5)
        
        self.histories = []
        for task in params.TASK_CONFIG:
            win_len = int(task['time'] / params.DT)
            self.histories.append(deque(maxlen=win_len))
        
        self.current_target_idx = 0
        self.last_dist = 0.0
        
        # 订阅
        self.sub_scan = rospy.Subscriber('/limo/scan', LaserScan, self._scan_cb)
        self.sub_gt = rospy.Subscriber('/gazebo/model_states', ModelStates, self._gt_cb)
        self.sub_odom = rospy.Subscriber('/odom', Odometry, self._odom_cb)
        
        print("Waiting for Gazebo data...")
        while self.latest_scan_msg is None or not self.has_odom:
            rospy.sleep(0.1)
        print("✅ Gazebo ready (Aligned to 180 deg FOV).")

    def _scan_cb(self, msg): 
        self.latest_scan_msg = msg

    def _gt_cb(self, msg):
        try:
            if "limo" in msg.name:
                idx = msg.name.index("limo")
                p = msg.pose[idx].position
                q = msg.pose[idx].orientation
                orientation_list = [q.x, q.y, q.z, q.w]
                (roll, pitch, yaw) = euler_from_quaternion(orientation_list)
                self.pose_gt = [p.x, p.y, yaw, roll, pitch]
        except ValueError: pass

    def _odom_cb(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny, cosy)
        self.pose_odom = [p.x, p.y, yaw]
        self.robot_vel = [msg.twist.twist.linear.x, msg.twist.twist.angular.z]
        self.has_odom = True

    def get_scan(self):
        if self.latest_scan_msg is None: return False
        try:
            msg = self.latest_scan_msg
            raw = np.array(msg.ranges)
            
            # === [核心修复] FOV 对齐逻辑 ===
            # 仿真雷达可能是 -120~120 (240度)，实车是 -90~90 (180度)
            # 我们强制只取中间的 -90 ~ +90 度
            min_angle = -np.pi / 2  # -90 deg
            max_angle = np.pi / 2   # +90 deg
            
            # 计算对应的索引范围
            # 索引 = (目标角度 - 起始角度) / 增量
            # 注意 msg.angle_min 可能是 -2.09 (仿真)
            start_idx = int((min_angle - msg.angle_min) / msg.angle_increment)
            end_idx = int((max_angle - msg.angle_min) / msg.angle_increment)
            
            # 边界保护
            start_idx = max(0, start_idx)
            end_idx = min(len(raw), end_idx)
            
            # 裁剪数据 (只保留中间 180 度)
            cropped_scan = raw[start_idx:end_idx]
            
            # === 数据清洗 ===
            cropped_scan[np.isinf(cropped_scan)] = 5.0
            cropped_scan[np.isnan(cropped_scan)] = 5.0
            # 实车最大6.0，仿真最大8.0，统一截断为 5.0
            cropped_scan = np.clip(cropped_scan, 0.0, 5.0)
            
            # === 降采样 ===
            # 确保即使数据点变少了，依然映射到 LIDAR_DIM
            if len(cropped_scan) == 0: return False # 异常保护
            
            chunk = len(cropped_scan) // params.LIDAR_DIM
            scan = []
            for i in range(params.LIDAR_DIM):
                segment = cropped_scan[i*chunk:(i+1)*chunk]
                if len(segment) > 0:
                    scan.append(np.min(segment))
                else:
                    scan.append(5.0)
            
            self.scan_data = np.array(scan)
            return True
        except Exception as e:
            print(f"Scan Error: {e}")
            return False

    def get_current_goal_pos(self):
        idx = min(self.current_target_idx, self.num_tasks - 1)
        return np.array(params.TASK_CONFIG[idx]['pos'])

    def reset(self, is_training=True):
        rospy.wait_for_service('/gazebo/reset_world')
        try: self.reset_proxy()
        except: pass
        
        self.pub_cmd.publish(Twist())
        rospy.sleep(0.5)
        
        if is_training:
            rand_x = -7.0 + np.random.uniform(-0.5, 0.5)
            rand_y = 0.0 + np.random.uniform(-0.5, 0.5)
            rand_yaw = np.random.uniform(-0.5, 0.5)
            self._set_model_state(rand_x, rand_y, rand_yaw)
        else:
            self._set_model_state(-7.0, 0.0, 0.0)

        self.c_t = np.zeros(self.num_tasks)
        self.f_t = np.full(self.num_tasks, -0.5)
        self.current_target_idx = 0
        
        for h in self.histories:
            h.clear()
            for _ in range(h.maxlen): h.append(False)
        
        self.get_scan()
        
        curr_pos_gt = np.array(self.pose_gt[:2])
        self.last_dist = np.linalg.norm(curr_pos_gt - self.get_current_goal_pos())
        
        return self._get_obs()

    def _set_model_state(self, x, y, yaw):
        try:
            set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            state_msg = ModelState()
            state_msg.model_name = 'limo'
            state_msg.pose.position.x = x
            state_msg.pose.position.y = y
            state_msg.pose.position.z = 0.15
            q = quaternion_from_euler(0, 0, yaw)
            state_msg.pose.orientation.x = q[0]
            state_msg.pose.orientation.y = q[1]
            state_msg.pose.orientation.z = q[2]
            state_msg.pose.orientation.w = q[3]
            set_state(state_msg)
        except: pass

    def step(self, action):
        total_r_stl = 0.0
        total_r_aux = 0.0
        done = False
        is_success = False
        
        for _ in range(params.ACTION_REPEAT):
            vel = Twist()
            vel.linear.x = action[0] 
            vel.angular.z = action[1]
            self.pub_cmd.publish(vel)
            
            rospy.sleep(params.DT)
            self.get_scan()
            
            r_s, r_a, d, succ = self._compute_reward_general()
            total_r_stl += r_s
            total_r_aux += r_a
            
            if d:
                done = True
                is_success = succ
                break
        
        return self._get_obs(), (total_r_stl, total_r_aux), done, is_success

    def _compute_reward_general(self):
        curr_pos = np.array(self.pose_gt[:2]) 
        r_stage = 0.0
        r_progress = 0.0
        r_collision = 0.0
        r_efficiency = -params.W_EFF
        r_move = 0.0
        done = False
        succ = False
        
        for k in range(self.num_tasks):
            task = params.TASK_CONFIG[k]
            target_pos = np.array(task['pos'])
            radius = task['radius']
            dist = np.linalg.norm(curr_pos - target_pos)
            rho = radius - dist
            is_satisfied = (rho >= 0)
            self.histories[k].append(is_satisfied)
            self.f_t[k] = self._calculate_f_score(self.histories[k], task['type'])
            
            if self.c_t[k] == 0:
                prev_done = (k == 0) or (self.c_t[k-1] == 1)
                if prev_done and is_satisfied:
                    self.c_t[k] = 1.0
                    r_stage += params.W_STAGE * (k + 1)
                    print(f"[{time.time():.2f}] >>> Task {k} Completed! <<<")
                    if k < self.num_tasks - 1:
                        self.current_target_idx = k + 1
                        new_target = np.array(params.TASK_CONFIG[self.current_target_idx]['pos'])
                        self.last_dist = np.linalg.norm(curr_pos - new_target)
                    else:
                        self.current_target_idx = k 
                        done = True
                        succ = True
                        print(f"[{time.time():.2f}] >>> MISSION COMPLETE! <<<")

        if not succ:
            target_pos = self.get_current_goal_pos()
            curr_dist = np.linalg.norm(curr_pos - target_pos)
            r_progress = params.W_PROG * (self.last_dist - curr_dist)
            self.last_dist = curr_dist

        # [修复] 正确的翻车检测
        if abs(self.pose_gt[3]) > 0.5 or abs(self.pose_gt[4]) > 0.5:
            print("💀 Robot Flipped!")
            r_collision = -params.W_COLL
            done = True

        if np.min(self.scan_data) < 0.25:
            r_collision = -params.W_COLL
            done = True
        
        v_current = self.robot_vel[0]
        if v_current > 0.05:
            r_move = params.W_MOVE * v_current
        
        return (r_stage + r_progress), (r_collision + r_efficiency + r_move), done, succ

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
        scan = np.clip(self.scan_data / 5.0, 0, 1)
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
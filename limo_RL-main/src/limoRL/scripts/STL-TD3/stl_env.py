import rospy
import numpy as np
import math
import time
from collections import deque
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from gazebo_msgs.msg import ModelStates # [新增] 真值消息
from std_srvs.srv import Empty
import params
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
from tf.transformations import quaternion_from_euler
class STL_Gazebo_Env:
    def __init__(self):
        try:
            rospy.init_node('stl_td3_node', anonymous=True)
        except:
            pass
        
        self.pub_cmd = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_world', Empty)
        
        # === [核心] 双数据源定义 ===
        # 1. 裁判员数据 (Ground Truth) -> 用于计算 Reward
        self.pose_gt = [0.0, 0.0, 0.0] # [x, y, yaw]
        
        # 2. 运动员数据 (Odom Sensor) -> 用于生成 State
        self.pose_odom = [0.0, 0.0, 0.0]
        self.robot_vel = [0.0, 0.0] # [v, w] 只有 odom 有速度
        
        self.scan_data = np.ones(params.LIDAR_DIM) * 10.0
        
        # F-MDP-S 状态初始化
        self.num_tasks = params.NUM_TASKS
        self.c_t = np.zeros(self.num_tasks)
        self.f_t = np.full(self.num_tasks, -0.5)
        
        self.histories = []
        for task in params.TASK_CONFIG:
            win_len = int(task['time'] / params.DT)
            self.histories.append(deque(maxlen=win_len))
        
        self.current_target_idx = 0
        self.last_dist = 0.0
        
        # 异步数据订阅
        self.latest_scan_msg = None
        self.sub_scan = rospy.Subscriber('/limo/scan', LaserScan, self._scan_cb)
        
        # [修改] 同时订阅真值和里程计
        self.sub_gt = rospy.Subscriber('/gazebo/model_states', ModelStates, self._gt_cb)
        self.sub_odom = rospy.Subscriber('/odom', Odometry, self._odom_cb)
        
        print(f"Environment initialized with {self.num_tasks} sequential tasks.")
        # 等待数据 (只要 Odom 和 Scan 到了就能跑，GT 可能会慢一点点)
        while self.latest_scan_msg is None or self.pose_odom[0] == 0.0:
            rospy.sleep(0.1)
        print("Gazebo ready.")

    # === 回调函数 ===
    def _scan_cb(self, msg): 
        self.latest_scan_msg = msg

    def _gt_cb(self, msg):
        try:
            target_idx = -1
            for i, name in enumerate(msg.name):
                # 只要名字里包含 limo 就算找到 (适应 limo_ackerman, limo_diff 等)
                if "limo" in name: 
                    target_idx = i
                    break
            if target_idx != -1:
                p = msg.pose[target_idx].position
                q = msg.pose[target_idx].orientation
                # 四元数转欧拉角
                siny = 2.0 * (q.w * q.z + q.x * q.y)
                cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
                yaw = math.atan2(siny, cosy)
                self.pose_gt = [p.x, p.y, yaw]
        except ValueError: pass

    def _odom_cb(self, msg):
        # 提取里程计 (带噪声/漂移)
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny, cosy)
        self.pose_odom = [p.x, p.y, yaw]
        self.robot_vel = [msg.twist.twist.linear.x, msg.twist.twist.angular.z]

    def get_scan(self):
        if self.latest_scan_msg is None: return False
        try:
            raw = np.array(self.latest_scan_msg.ranges)
            raw[np.isinf(raw)] = 5.0
            raw[np.isnan(raw)] = 5.0
            raw = np.clip(raw, 0.0, 5.0)
            
            chunk = len(raw) // params.LIDAR_DIM
            scan = []
            for i in range(params.LIDAR_DIM):
                scan.append(np.min(raw[i*chunk:(i+1)*chunk]))
            
            self.scan_data = np.array(scan)
            return True
        except: return False

    def get_current_goal_pos(self):
        idx = min(self.current_target_idx, self.num_tasks - 1)
        return np.array(params.TASK_CONFIG[idx]['pos'])

    def reset(self, is_training=True):
        # 1. 重置整个物理世界 (清除所有残留状态)
        rospy.wait_for_service('/gazebo/reset_world')
        try:
            self.reset_proxy()
        except rospy.ServiceException as e:
            print(f"Reset service failed: {e}")
        
        # 2. 强制停止机器人 (清除之前的速度指令)
        self.pub_cmd.publish(Twist())
        rospy.sleep(0.5) # 等待物理引擎稳定
        
        # 3. [优化] 随机设置机器人初始位置 (仅在训练时)
        if is_training:
            # 在 (-7, 0) 附近随机
            # x: [-7.5, -6.5], y: [-0.5, 0.5]
            rand_x = -7.0 + np.random.uniform(-0.5, 0.5)
            rand_y = 0.0 + np.random.uniform(-0.5, 0.5)
            # yaw: 随机偏转 +/- 0.5 弧度 (约 30 度)
            rand_yaw = np.random.uniform(-0.5, 0.5)
            
            self._set_model_state(rand_x, rand_y, rand_yaw)
        else:
            # 评估模式：固定在标准起点，保证公平对比
            self._set_model_state(-7.0, 0.0, 0.0)

        # 4. 重置内部逻辑状态
        self.c_t = np.zeros(self.num_tasks)
        self.f_t = np.full(self.num_tasks, -0.5)
        self.current_target_idx = 0
        
        # 清空历史轨迹 buffer
        for h in self.histories:
            h.clear()
            for _ in range(h.maxlen): h.append(False)
        
        # 5. 获取初始观测
        self.get_scan()
        
        # 使用 Ground Truth (pose_gt) 计算初始距离，确保 Reward 计算准确
        # 注意：Training 时用 GT 计算 Reward，但 Obs 用 Odom
        curr_pos_gt = np.array(self.pose_gt[:2])
        self.last_dist = np.linalg.norm(curr_pos_gt - self.get_current_goal_pos())
        
        return self._get_obs()

    # [新增] 辅助函数：设置 Gazebo 模型状态
    def _set_model_state(self, x, y, yaw):
        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            state_msg = ModelState()
            state_msg.model_name = 'limo'  # 确保这里名字和 launch 文件里 spawn 的名字一致
            state_msg.pose.position.x = x
            state_msg.pose.position.y = y
            state_msg.pose.position.z = 0.1 # 稍微抬高一点防止卡进地里
            
            q = quaternion_from_euler(0, 0, yaw)
            state_msg.pose.orientation.x = q[0]
            state_msg.pose.orientation.y = q[1]
            state_msg.pose.orientation.z = q[2]
            state_msg.pose.orientation.w = q[3]
            
            set_state(state_msg)
        except rospy.ServiceException as e:
            print(f"Set model state failed: {e}")

    def step(self, action):
        total_r_stl = 0.0
        total_r_aux = 0.0
        done = False
        is_success = False
        
        for _ in range(params.ACTION_REPEAT):
            vel = Twist()
            # [适配] 使用 params.MAX_V/W
            vel.linear.x = action[0] 
            vel.angular.z = action[1]
            self.pub_cmd.publish(vel)
            rospy.sleep(0.1)
            
            self.get_scan() # 刷新雷达
            # pose_gt 和 pose_odom 会由回调自动刷新，不需要显式 get
            
            # 计算奖励 (使用 GT)
            r_s, r_a, d, succ = self._compute_reward_general()
            total_r_stl += r_s
            total_r_aux += r_a
            
            if d:
                done = True
                is_success = succ
                break
        
        return self._get_obs(), (total_r_stl, total_r_aux), done, is_success

    def _compute_reward_general(self):
        # === [核心修改] 使用真值 (GT) 计算奖励 ===
        # 只有上帝视角才能公正评判
        curr_pos = np.array(self.pose_gt[:2]) 
        
        r_stage = 0.0
        r_progress = 0.0
        r_collision = 0.0
        r_efficiency = -params.W_EFF
        
        done = False
        succ = False
        
        # 1. 遍历任务 (逻辑不变，但数据源变了)
        for k in range(self.num_tasks):
            task = params.TASK_CONFIG[k]
            target_pos = np.array(task['pos'])
            radius = task['radius']
            
            # 计算鲁棒度 rho (基于真值)
            dist = np.linalg.norm(curr_pos - target_pos)
            rho = radius - dist
            is_satisfied = (rho >= 0)
            
            self.histories[k].append(is_satisfied)
            self.f_t[k] = self._calculate_f_score(self.histories[k], task['type'])
            
            # 更新阶段锁
            if self.c_t[k] == 0:
                prev_done = (k == 0) or (self.c_t[k-1] == 1)
                if prev_done and is_satisfied:
                    self.c_t[k] = 1.0
                    r_stage += params.W_STAGE * (k + 1)
                    print(f"[{time.time():.2f}] >>> Task {k} Completed! (GT Checked) <<<", flush=True)
                    
                    if k < self.num_tasks - 1:
                        self.current_target_idx = k + 1
                        new_target = np.array(params.TASK_CONFIG[self.current_target_idx]['pos'])
                        # 重置距离基准
                        self.last_dist = np.linalg.norm(curr_pos - new_target)
                    else:
                        self.current_target_idx = k 
                        done = True
                        succ = True
                        print(f"[{time.time():.2f}] >>> MISSION COMPLETE! <<<", flush=True)

        # 2. 进度奖励 (基于真值)
        if succ:
            r_progress = 0.0
        else:
            target_pos = self.get_current_goal_pos()
            curr_dist = np.linalg.norm(curr_pos - target_pos)
            r_progress = params.W_PROG * (self.last_dist - curr_dist)
            self.last_dist = curr_dist

        # 3. 辅助奖励
        # 翻车检测 (用真值比较准)
        if abs(self.pose_gt[2]) > 0.6: # 这里的 pose_gt[2] 实际上是 yaw，翻车应该看 roll/pitch，这里简化处理或需要扩展gt
             # 如果 gt 只存了 yaw，这里暂时用 odom 或扩展 gt
             pass 

        if np.min(self.scan_data) < 0.25:
            r_collision = -params.W_COLL
            done = True
        
        # [优化] 计算速度奖励 (Velocity Reward)
        # 只有当前进速度 v > 0 时才有正向奖励
        v_current = self.robot_vel[0]
        r_move = params.W_MOVE * v_current
        
        # 将 r_move 加入到 Aux Reward (第二项) 中
        # 这样 Actor 为了最大化总分，会倾向于输出更大的 v
        return (r_stage + r_progress), (r_collision + r_efficiency + r_move), done, succ

    def _calculate_f_score(self, history, task_type):
        # ... (保持原来的逻辑不变) ...
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
        # === [核心修改] 使用里程计 (Odom) 生成观测 ===
        # 模拟实车环境，网络只能看到传感器数据
        scan = np.clip(self.scan_data / 5.0, 0, 1)
        
        # 获取里程计读数
        rx, ry, ryaw = self.pose_odom
        
        # 计算相对坐标 (基于里程计)
        goal = self.get_current_goal_pos()
        dx = goal[0] - rx
        dy = goal[1] - ry
        
        # 坐标变换 (世界系 -> 机器人系)
        lx = dx * math.cos(ryaw) + dy * math.sin(ryaw)
        ly = -dx * math.sin(ryaw) + dy * math.cos(ryaw)
        
        v = self.robot_vel[0]
        w = self.robot_vel[1]
        
        robot = np.array([lx, ly, math.cos(ryaw), math.sin(ryaw), v, w])
        flags = np.concatenate((self.c_t, self.f_t))
        
        return np.concatenate((scan, robot, flags))
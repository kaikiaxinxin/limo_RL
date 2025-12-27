import rospy
import numpy as np
import math
from geometry_msgs.msg import Twist, Pose
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState
import params

class STL_Gazebo_Env:
    def __init__(self):
        # 初始化ROS节点
        rospy.init_node('stl_td3_gym', anonymous=True)
        
        # 话题接口
        self.pub_cmd_vel = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.sub_odom = rospy.Subscriber('/odom', Odometry, self.odom_callback)
        self.sub_scan = rospy.Subscriber('/scan', LaserScan, self.scan_callback)
        
        # 内部状态变量
        self.robot_pose = [0.0, 0.0, 0.0] # x, y, yaw
        self.scan_data = np.ones(params.LIDAR_DIM) * 10.0
        
        # === F-MDP-S 核心状态变量 ===
        # c_t: 任务完成阶段标志位 [Flag_A, Flag_B]
        # [0, 0] -> 未开始
        # [1, 0] -> 完成A, 正在去B
        # [1, 1] -> 完成B (任务结束)
        self.c_t = np.array([0.0, 0.0])
        
        # f_t: STL鲁棒度标志位 (简化实现，用于网络输入)
        self.f_t = np.array([-0.5, -0.5]) 
        
        # 上一时刻距离 (用于计算 r_progress)
        self.last_dist_to_current_goal = 0.0
        
        rospy.sleep(1.0) # 等待连接

    def odom_callback(self, msg):
        self.robot_pose[0] = msg.pose.pose.position.x
        self.robot_pose[1] = msg.pose.pose.position.y
        # 四元数转欧拉角
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.robot_pose[2] = math.atan2(siny_cosp, cosy_cosp)

    def scan_callback(self, msg):
        # 激光雷达降采样处理: 将原始数据分为 LIDAR_DIM 组，取每组最小值
        raw_data = np.array(msg.ranges)
        raw_data[np.isinf(raw_data)] = 10.0
        raw_data[np.isnan(raw_data)] = 10.0
        
        # 假设激光雷达是360度或270度，均匀切片
        chunk_size = len(raw_data) // params.LIDAR_DIM
        obs_scan = []
        for i in range(params.LIDAR_DIM):
            chunk = raw_data[i*chunk_size : (i+1)*chunk_size]
            obs_scan.append(np.min(chunk))
        self.scan_data = np.array(obs_scan)

    def get_current_goal(self):
        # 根据 c_t 判断当前目标
        if self.c_t[0] == 0:
            return np.array(params.GOAL_A_POS) # 阶段1：去A
        else:
            return np.array(params.GOAL_B_POS) # 阶段2：去B

    def get_distance(self, pos1, pos2):
        return np.linalg.norm(np.array(pos1) - np.array(pos2))

    def reset(self):
        # 1. 重置机器人位置到 Init Area
        self.set_robot_state(params.INIT_POS[0], params.INIT_POS[1], 0.0)
        
        # 2. 重置 F-MDP-S 标志位
        self.c_t = np.array([0.0, 0.0])
        self.f_t = np.array([-0.5, -0.5])
        
        # 3. 初始化距离
        current_goal = self.get_current_goal()
        self.last_dist_to_current_goal = self.get_distance(self.robot_pose[:2], current_goal)
        
        rospy.sleep(0.5) # 等待物理稳定
        return self._get_observation()

    def step(self, action):
        # action: [v, w] 归一化到 [-1, 1] -> 映射到实车速度
        linear_vel = (action[0] + 1.0) / 2.0 * 1.0 # [0, 1.0] m/s
        angular_vel = action[1] * 1.0              # [-1.0, 1.0] rad/s
        
        cmd = Twist()
        cmd.linear.x = linear_vel
        cmd.angular.z = angular_vel
        self.pub_cmd_vel.publish(cmd)
        
        rospy.sleep(0.1) # 执行动作 0.1s (10Hz)
        
        # === 核心逻辑：状态迁移与奖励计算 ===
        reward_stl = 0.0
        reward_aux = 0.0
        done = False
        
        curr_pos = np.array(self.robot_pose[:2])
        
        # 1. 检查阶段完成 (Stage Check)
        dist_to_A = self.get_distance(curr_pos, params.GOAL_A_POS)
        dist_to_B = self.get_distance(curr_pos, params.GOAL_B_POS)
        
        # 逻辑锁：必须先完成A，才能触发B
        if self.c_t[0] == 0:
            # 当前目标是 A
            if dist_to_A < params.AREA_RADIUS:
                # === 触发阶段转换：A 完成 ===
                self.c_t[0] = 1.0
                self.f_t[0] = 0.5 # 标记A已满足
                reward_stl += params.W_STAGE # 给予巨大的阶段奖励
                print(">>> Stage A Reached! Switching to B.")
                # 重置对B的距离记录
                self.last_dist_to_current_goal = dist_to_B 
        elif self.c_t[0] == 1 and self.c_t[1] == 0:
            # 当前目标是 B (且A已完成)
            if dist_to_B < params.AREA_RADIUS:
                # === 触发阶段转换：B 完成 ===
                self.c_t[1] = 1.0
                self.f_t[1] = 0.5
                reward_stl += params.W_STAGE
                print(">>> Stage B Reached! Mission Complete.")
                done = True
        
        # 2. 进度奖励 (Progress Reward)
        # 始终引导向"当前未完成的阶段目标"移动
        curr_goal = self.get_current_goal()
        dist_to_curr_goal = self.get_distance(curr_pos, curr_goal)
        progress = self.last_dist_to_current_goal - dist_to_curr_goal
        reward_stl += params.W_PROG * progress
        self.last_dist_to_current_goal = dist_to_curr_goal
        
        # 3. 碰撞检测与效率 (Auxiliary Reward)
        if np.min(self.scan_data) < 0.3: # 碰撞阈值
            reward_aux -= params.W_COLL
            done = True
            print(">>> Collision!")
            
        reward_aux -= params.W_EFF # 时间惩罚
        
        # 4. 返回分解后的奖励 (用于Dual Critic)
        # 注意：这里step返回总奖励，但在info中携带分解奖励，或者直接返回 tuple
        total_reward = reward_stl + reward_aux
        
        obs = self._get_observation()
        
        # 将分解奖励放入 info
        info = {'r_stl': reward_stl, 'r_aux': reward_aux}
        
        return obs, total_reward, done, info

    def _get_observation(self):
        # 构造扩展状态 z_t = [Scan, Rel_Pose, c_t, f_t]
        
        # 归一化雷达
        scan_norm = self.scan_data / 10.0 
        
        # 计算相对当前目标的坐标 (简单的本体感知)
        target_pos = self.get_current_goal()
        rel_x = target_pos[0] - self.robot_pose[0]
        rel_y = target_pos[1] - self.robot_pose[1]
        # 旋转到机器人坐标系
        yaw = self.robot_pose[2]
        local_x = rel_x * math.cos(-yaw) - rel_y * math.sin(-yaw)
        local_y = rel_x * math.sin(-yaw) + rel_y * math.cos(-yaw)
        
        robot_state = np.array([local_x, local_y, math.cos(yaw), math.sin(yaw)])
        
        # 拼接所有特征 (严格按照 F-MDP-S)
        z_t = np.concatenate((scan_norm, robot_state, self.c_t, self.f_t))
        return z_t

    def set_robot_state(self, x, y, yaw):
        # 调用Gazebo服务重置位置
        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            sms = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            msg = ModelState()
            msg.model_name = 'limo' # 确保模型名称正确
            msg.pose.position.x = x
            msg.pose.position.y = y
            msg.pose.position.z = 0.1
            # 欧拉角转四元数 (略简化)
            msg.pose.orientation.z = math.sin(yaw/2)
            msg.pose.orientation.w = math.cos(yaw/2)
            sms(msg)
        except rospy.ServiceException as e:
            print("Set Model State failed: %s" % e)
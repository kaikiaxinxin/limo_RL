#!/usr/bin/python3
# 导入 ROS 的 Python 客户端库，用于与 ROS 系统进行交互
import rospy
# 导入 NumPy 库，用于高效的数值计算
import numpy as np
# 导入 Python 内置的数学库，提供基本的数学函数
import math
# 导入 copy 模块，用于对象的复制操作
import copy
# 导入 time 模块，用于处理时间相关的操作
import time
# 从 math 库中导入圆周率常量 pi
from math import pi
# 从当前目录下的 respawnGoal 模块中导入 Respawn 类
from .respawnGoal import Respawn
# 从 geometry_msgs.msg 模块中导入 Twist、Point 和 Pose 消息类型
# Twist 用于表示线速度和角速度，Point 用于表示三维空间中的点，Pose 用于表示位置和姿态
from geometry_msgs.msg import Twist, Point, Pose
# 从 sensor_msgs.msg 模块中导入 LaserScan 消息类型，用于接收激光雷达扫描数据
from sensor_msgs.msg import LaserScan
# 从 nav_msgs.msg 模块中导入 Odometry 消息类型，用于接收机器人的里程计信息
from nav_msgs.msg import Odometry
# 从 std_srvs.srv 模块中导入 Empty 服务类型，用于调用无参数的服务
from std_srvs.srv import Empty
# 从 tf.transformations 模块中导入 euler_from_quaternion 和 quaternion_from_euler 函数
# 用于四元数和欧拉角之间的相互转换
from tf.transformations import euler_from_quaternion, quaternion_from_euler



class Env():
    def __init__(self,action_dim = 2):
        """
        初始化环境类。

        :param action_dim: 动作空间的维度，默认为 2。
        """
        self.goal_x = 0  # 目标点的 x 坐标
        self.goal_y = 0  # 目标点的 y 坐标
        self.heading = 0  # 机器人朝向目标的角度
        self.initGoal = True  # 是否首次初始化目标点的标志
        self.get_goalbox = False  # 是否到达目标点的标志
        
        self.position = Point()  # 机器人的位置，改为Point类型，便于直接访问x、y
        # 发布速度指令的话题，消息类型为 Twist，队列大小为 5
        self.pub_cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size=5)
        # 订阅里程计消息的话题，回调函数为 getOdometry
        self.sub_odom = rospy.Subscriber('odom', Odometry, self.getOdometry)
        # 重置 Gazebo 仿真世界的服务代理
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_world', Empty)
        # 恢复 Gazebo 物理引擎的服务代理
        self.unpause_proxy = rospy.ServiceProxy('gazebo/unpause_physics', Empty)
        # 暂停 Gazebo 物理引擎的服务代理
        self.pause_proxy = rospy.ServiceProxy('gazebo/pause_physics', Empty)
        self.respawn_goal = Respawn()  # 用于重新生成目标点的对象
        
        self.last_distance = 0  # 上一时刻到目标点的距离
        self.past_distance = 0.  # 过去时刻到目标点的距离
        self.initial_diatance = 0.  # 初始时刻到目标点的距离
        self.stopped = 0  # 停止标志
        self.action_dim = action_dim  # 动作空间的维度

        # self.x_gap_last = 0
        # self.y_gap_last = 0
        
        # 设置 ROS 关闭时的回调函数
        rospy.on_shutdown(self.shutdown)
        
    def shutdown(self):
        """
        ROS 关闭时的回调函数，用于停止机器人。
        """
        rospy.loginfo("Stopping TurtleBot")
        self.pub_cmd_vel.publish(Twist())  # 发布速度为 0 的指令
        rospy.sleep(1)  # 休眠 1 秒
    
    def getGoalDistace(self):
        """
        计算机器人当前位置到目标点的距离。

        :return: 机器人到目标点的距离，保留两位小数。
        """
        goal_distance = round(math.hypot(self.goal_x - self.position.x, self.goal_y - self.position.y), 2)
        self.past_distance = goal_distance
        self.initial_diatance = goal_distance
        return goal_distance
    
    def getOdometry(self, odom):
        """
        里程计消息的回调函数，用于更新机器人的位置和朝向目标的角度。

        :param odom: 里程计消息，包含机器人的位置和姿态信息。
        """
        # 获得机器人的具体位置
        self.past_position = copy.deepcopy(self.position)
        self.position = odom.pose.pose.position
        orientation = odom.pose.pose.orientation
        orientation_list = [orientation.x, orientation.y, orientation.z, orientation.w]
        _, _, yaw = euler_from_quaternion(orientation_list)  # 将四元数转换为欧拉角
        
        goal_angle = math.atan2(self.goal_y - self.position.y, self.goal_x - self.position.x)  # 计算目标点的角度
        heading = goal_angle - yaw  # 计算机器人朝向目标的角度
        
        if heading > pi:
            heading -= 2 * pi

        elif heading < -pi:
            heading += 2 * pi

        self.heading = round(heading, 3)  # 保留三位小数
        
    def getState(self, scan,past_action):
        """
        根据激光雷达数据和上一时刻的动作获取当前环境状态。

        :param scan: 激光雷达扫描数据。
        :param past_action: 上一时刻的动作。
        :return: 当前环境状态和是否结束的标志。
        """
        # state 20个激光雷达数据 + heading + current_disctance + obstacle_min_range, obstacle_angle 
        
        scan_range = []  # 存储处理后的激光雷达数据
        heading = self.heading  # 机器人朝向目标的角度
        min_range = 0.20  # 最小安全距离
        done = False  # 是否结束的标志

        for i in range(len(scan.ranges)):
            if scan.ranges[i] == float('Inf') or scan.ranges[i] >7.0:
                scan_range.append(7.0)  # 处理无穷大或超过 7.0 的数据
            elif np.isnan(scan.ranges[i]):
                scan_range.append(0)  # 处理 NaN 数据
            else:
                scan_range.append(scan.ranges[i])  # 正常数据直接添加
        
        # min state 
        obstacle_min_range = round(min(scan_range), 2)  # 障碍物的最小距离，保留两位小数

        # obstacle_angle = np.argmin(scan_range)
        # x_gap  = self.goal_x - self.position.x
        # y_gap = self.goal_y - self.position.y
        
        if min_range > min(scan_range) > 0:
            print("scan_range",scan_range)
            print("min_range",min_range)
            print("min(scan_range)",min(scan_range))
            done = True  # 检测到碰撞，设置结束标志

            
        current_distance = round(math.hypot(self.goal_x - self.position.x, self.goal_y - self.position.y),2)  # 当前到目标点的距离
        if current_distance < 0.3:
            self.get_goalbox = True  # 到达目标点

        # return scan_range + [heading, current_distance, obstacle_min_range, obstacle_angle], done
        return scan_range + [heading, current_distance, obstacle_min_range,past_action[0],past_action[1]], done
    
    def setReward(self, state, action,done):
        """
        根据当前状态、动作和是否结束的标志设置奖励。

        :param state: 当前环境状态。
        :param action: 当前动作。
        :param done: 是否结束的标志。
        :return: 奖励值和是否结束的标志。
        """

        current_distance = state[-4]  # 当前到目标点的距离
        heading = state[-5]  # 机器人朝向目标的角度
        obstacle_min_range = state[-3]  # 障碍物的最小距离

        # 距离奖励 
        distance_reward = -current_distance  # 距离越远，奖励越低
        
        # 方向奖励
        turn_reward = -abs(heading)  # 偏离目标方向越大，奖励越低

        # 躲避障碍物体 Reward
        if obstacle_min_range < 0.8:
            ob_reward = -2 ** (0.6/obstacle_min_range)  # 距离障碍物越近，惩罚越大
        else:
            ob_reward = 0  # 距离障碍物足够远，无惩罚
        reward = distance_reward + turn_reward + ob_reward  # 总奖励

        if done:
            rospy.loginfo("Collision!!")
            reward = -200.  # 发生碰撞，给予较大惩罚
            self.pub_cmd_vel.publish(Twist())  # 停止机器人
            self.respawn_goal.index = 0  # 重置目标点索引

        if self.get_goalbox:
            rospy.loginfo("Goal!!")
            reward = 1600.  # 到达目标点，给予较大奖励
            self.pub_cmd_vel.publish(Twist())  # 停止机器人
            self.goal_x, self.goal_y = self.respawn_goal.getPosition(True, delete=True)  # 重新生成目标点
            self.goal_distance = self.getGoalDistace()  # 重新计算到目标点的距离
            self.get_goalbox = False  # 重置到达目标点的标志
            

        return reward, done
     
    def step(self, action,past_action):
        """
        执行一个动作步，更新环境状态并返回新状态、奖励和是否结束的标志。

        :param action: 当前动作。
        :param past_action: 上一时刻的动作。
        :return: 新的环境状态、奖励值和是否结束的标志。
        """
        linear_vel = action[0]  # 线速度
        ang_vel = action[1]  # 角速度
        vel_cmd = Twist()  # 速度指令消息
        vel_cmd.linear.x = linear_vel
        vel_cmd.angular.z = ang_vel
        self.pub_cmd_vel.publish(vel_cmd)  # 发布速度指令

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('limo/scan', LaserScan, timeout=5)  # 等待激光雷达数据
            except:
                pass

        state, done = self.getState(data, past_action)  # 获取当前环境状态     
        reward, done = self.setReward(state, action,done)  # 设置奖励
        
        return np.asarray(state), reward, done

    
    
    def reset(self):
        """
        重置环境，包括重置仿真世界、重新生成目标点等。

        :return: 重置后的环境状态。
        """
        rospy.wait_for_service('gazebo/reset_world')
        try:
            self.reset_proxy()  # 调用重置仿真世界的服务
        except (rospy.ServiceException) as e:
            print("gazebo/reset_simulation service call failed")

        data = None
        
        while data is None:
            try:
                data = rospy.wait_for_message('/limo/scan', LaserScan, timeout=5)  # 等待激光雷达数据
            except:
                pass
        
        if self.initGoal:
            self.goal_x, self.goal_y = self.respawn_goal.getPosition()  # 首次初始化目标点
            self.initGoal = False
            
        ## mabe debug
        # else:
        #     self.goal_x, self.goal_y = self.respawn_goal.getPosition(True, delete=True)
           

        print("reset successfully")
        self.goal_distance = self.getGoalDistace()  # 计算到目标点的距离
        state, _ = self.getState(data, [0]*self.action_dim)  # 获取重置后的环境状态
        return np.asarray(state)
    
        

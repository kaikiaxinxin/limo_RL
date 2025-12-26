#!/usr/bin/env python
#################################################################################
# Copyright 2018 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#################################################################################

# Authors: Gilbert #

# 导入 ROS 的 Python 客户端库，用于与 ROS 系统进行交互
import rospy
# 导入 random 模块，用于生成随机数
import random
# 导入 time 模块，用于处理时间相关操作
import time
# 导入 os 模块，用于与操作系统进行交互，如文件路径操作
import os
# 导入 math 模块，用于数学计算
import math
# 从 gazebo_msgs.srv 模块中导入 SpawnModel 和 DeleteModel 服务类型
# 分别用于在 Gazebo 中生成和删除模型
from gazebo_msgs.srv import SpawnModel, DeleteModel
# 从 gazebo_msgs.msg 模块中导入 ModelStates 消息类型，用于获取 Gazebo 中所有模型的状态
from gazebo_msgs.msg import ModelStates
# 从 geometry_msgs.msg 模块中导入 Pose 消息类型，用于表示位置和姿态
from geometry_msgs.msg import Pose

class Respawn():
    def __init__(self):
        """
        初始化 Respawn 类，设置目标模型的路径、初始位置，订阅 Gazebo 模型状态话题等。
        """
        # 获取当前脚本文件的绝对路径
        self.modelPath = os.path.dirname(os.path.realpath(__file__))
        # 将路径中的 'scripts/TD3' 替换为 'models/turtlebot3_square/goal_box/model.sdf'
        # 得到目标模型的 SDF 文件路径
        self.modelPath = self.modelPath.replace('scripts/TD3',
                                                'models/turtlebot3_square/goal_box/model.sdf')
        # 以只读模式打开目标模型的 SDF 文件
        self.f = open(self.modelPath, 'r')
        # 读取文件内容
        self.model = self.f.read()
        # 从 ROS 参数服务器获取阶段编号
        self.stage = rospy.get_param('/stage_number', 4)
        # 创建一个 Pose 对象，用于存储目标的位置和姿态
        self.goal_position = Pose()
        # 初始化目标的初始 x 坐标
        self.init_goal_x = 4.0
        # 初始化目标的初始 y 坐标
        self.init_goal_y = 4.0

        # self.init_goal_x = 1.2
        # self.init_goal_y = 1.8

        # 设置目标位置的 x 坐标
        self.goal_position.position.x = self.init_goal_x
        # 设置目标位置的 y 坐标
        self.goal_position.position.y = self.init_goal_y
        # 定义目标模型的名称
        self.modelName = 'goal'
        
        # 定义障碍物的位置，这里使用元组表示二维坐标
        self.obstacle_1 = 0.3, 0.3
        self.obstacle_2 = 0.3, -0.3
        self.obstacle_3 = -0.3, 0.3
        self.obstacle_4 = -0.3, -0.3
        # 记录上一个目标的 x 坐标
        self.last_goal_x = self.init_goal_x
        # 记录上一个目标的 y 坐标
        self.last_goal_y = self.init_goal_y
        # 记录上一个目标的索引
        self.last_index = 0
        # 订阅 Gazebo 模型状态话题，回调函数为 self.checkModel
        self.sub_model = rospy.Subscriber('gazebo/model_states', ModelStates, self.checkModel)
        # 标记是否检测到目标模型
        self.check_model = False
        # 当前目标的索引
        self.index = 0
        # 索引计数
        self.index_num = 0
        # 一个用于计算的数值
        self.R_num = 1

    def checkModel(self, model):
        """
        检查 Gazebo 中是否存在目标模型。

        :param model: ModelStates 消息，包含 Gazebo 中所有模型的状态信息
        """
        # 初始标记为未检测到目标模型
        self.check_model = False
        # 遍历所有模型名称
        for i in range(len(model.name)):
            # 如果找到名为 'goal' 的模型
            if model.name[i] == "goal":
                # 标记为检测到目标模型
                self.check_model = True
   
    def respawnModel(self):
        """
        在 Gazebo 中重新生成目标模型。
        """
        # print("self.check_model: ",self.check_model)
        while True:
            # 如果未检测到目标模型
            if not self.check_model:
                # 等待 Gazebo 的生成模型服务可用
                rospy.wait_for_service('gazebo/spawn_sdf_model')
                # 创建一个服务代理，用于调用生成模型服务
                spawn_model_prox = rospy.ServiceProxy('gazebo/spawn_sdf_model', SpawnModel)
                # 调用服务生成目标模型
                spawn_model_prox(self.modelName, self.model, 'robotos_name_space', self.goal_position, "world")
                # 记录目标模型的位置信息
                rospy.loginfo("Goal position : %.1f, %.1f", self.goal_position.position.x,
                              self.goal_position.position.y)
                break
            else:
                # 如果检测到目标模型，提示重新加载 Gazebo 与训练脚本
                print("重新加载 gazebo 与 训练脚本")
                pass

    # def deleteModel(self):
    #     """
    #     在 Gazebo 中删除目标模型。
    #     """
    #     while True:
    #         # 如果检测到目标模型
    #         if self.check_model:
    #             # 等待 Gazebo 的删除模型服务可用
    #             rospy.wait_for_service('gazebo/delete_model')
    #             # 创建一个服务代理，用于调用删除模型服务
    #             del_model_prox = rospy.ServiceProxy('gazebo/delete_model', DeleteModel)
    #             # 调用服务删除目标模型
    #             del_model_prox(self.modelName)
    #             break
    #         else:
    #             pass
    def deleteModel(self):
        """
        在 Gazebo 中删除目标模型。
        """
        while True:
            # 如果检测到目标模型
            if self.check_model:
                # 等待 Gazebo 的删除模型服务可用
                rospy.wait_for_service('gazebo/delete_model')
                # 创建一个服务代理，用于调用删除模型服务
                del_model_prox = rospy.ServiceProxy('gazebo/delete_model', DeleteModel)
                # 调用服务删除目标模型
                del_model_prox(self.modelName)
                # 增加一个小的延迟
                time.sleep(0.1)
                break
            else:
                pass

    def getPosition(self, position_check=False, delete=False):
        """
        获取目标的位置，并根据条件删除和重新生成目标模型。

        :param position_check: 是否进行位置检查，默认为 False
        :param delete: 是否删除现有目标模型，默认为 False
        :return: 目标的 x 坐标和 y 坐标
        """
        if delete:
            self.deleteModel()

        # 固定目标点，无论 stage
        self.goal_position.position.x = 4.0  # 你想要的固定x
        self.goal_position.position.y = 4.0  # 你想要的固定y

        time.sleep(0.5)
        self.respawnModel()
        self.last_goal_x = self.goal_position.position.x
        self.last_goal_y = self.goal_position.position.y

        return self.goal_position.position.x, self.goal_position.position.y

#!/usr/bin/env python3
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
# 导入 time 模块，用于处理时间相关操作，如休眠
import time
# 从 geometry_msgs.msg 模块中导入 Twist 消息类型，用于表示线速度和角速度
from geometry_msgs.msg import Twist
# 从 gazebo_msgs.msg 模块中导入 ModelState 和 ModelStates 消息类型
# ModelState 用于设置模型的状态，ModelStates 用于获取所有模型的状态
from gazebo_msgs.msg import ModelState, ModelStates

class Moving():
    def __init__(self):
        """
        初始化 Moving 类，创建一个发布者并调用 moving 方法。
        """
        # 创建一个发布者，发布消息到 'gazebo/set_model_state' 话题，消息类型为 ModelState，队列大小为 1
        self.pub_model = rospy.Publisher('gazebo/set_model_state', ModelState, queue_size=1)
        
        # 调用 moving 方法，开始移动障碍物
        self.moving()

    def moving(self):
        """
        循环移动名为 'obstacle' 的模型，为其设置角速度。
        """
        # 当 ROS 节点未关闭时，持续执行循环
        while not rospy.is_shutdown():
            # 创建一个 ModelState 对象，用于设置障碍物的状态
            obstacle = ModelState()
            # 等待获取 'gazebo/model_states' 话题的消息，该消息包含所有模型的状态
            model = rospy.wait_for_message('gazebo/model_states', ModelStates)
            
            # 遍历所有模型的名称
            for i in range(len(model.name)):
                # 找到名为 'obstacle' 的模型
                if model.name[i] == 'obstacle':
                    # 设置要操作的模型名称为 'obstacle'
                    obstacle.model_name = 'obstacle'
                    # 设置障碍物的位姿为当前获取到的位姿
                    obstacle.pose = model.pose[i]
                    # 创建一个 Twist 对象，用于设置障碍物的速度
                    obstacle.twist = Twist()
                    # 设置障碍物的角速度为 0.75 rad/s
                    obstacle.twist.angular.z = 0.75
                    # 发布障碍物的状态信息，使其在 Gazebo 中移动
                    self.pub_model.publish(obstacle)
                    # 休眠 0.1 秒，控制更新频率
                    time.sleep(0.1)

def main():
    """
    主函数，初始化 ROS 节点并创建 Moving 类的实例。
    """
    # 初始化 ROS 节点，节点名为 'moving_1'
    rospy.init_node('moving_1')
    # 创建 Moving 类的实例，开始移动障碍物
    moving = Moving()

if __name__ == '__main__':
    # 当脚本作为主程序运行时，调用 main 函数
    main()

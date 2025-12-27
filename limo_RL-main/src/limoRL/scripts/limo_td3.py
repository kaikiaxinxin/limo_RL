#!/usr/bin/env python3

import numpy as np                                                               
import math
import time
import rospy
from std_msgs.msg import Float32MultiArray
import os
import sys
import copy
# 从 TD3 模块的 TD3Net 文件中导入 TD3 类，用于实现 TD3 强化学习算法
from TD3.TD3Net import TD3
# 从 TD3 模块的 Environment 文件中导入 Env 类，用于创建强化学习环境
from TD3.Environment import Env
# 从 utils 模块中导入 plotLearning 函数，用于绘制学习曲线
from utils import plotLearning
# 将当前文件所在目录的上一级目录添加到系统路径中，方便导入模块
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import os


if __name__=='__main__':
    
    PI = math.pi
    # 定义动作空间的维度
    ACTION_DIMENSION = 2
    # 定义线速度的最大值，单位为 m/s
    ACTION_V_MAX = 0.8 
    # 定义角速度的最大值，单位为 rad/s
    ACTION_W_MAX = 2.0 
    # 动态获取当前脚本的绝对路径，这样以后你把项目搬到哪里都能跑
    curr_path = os.path.dirname(os.path.abspath(__file__))
    # 设置模型保存路径 (注意这里用了 ../ 回退到上一级)
    CKPT_DIR = curr_path + "/../train/TD3/model"
    save_figure_path = curr_path + "/../train/TD3/png"
    
    # 初始化 TD3 智能体
    agent = TD3(alpha=0.0003, beta=0.0003, state_dim=25,action_dim=2, actor_fc1_dim=400, actor_fc2_dim=300,
                action_limit_v=ACTION_V_MAX,action_limit_w=ACTION_W_MAX,critic_fc1_dim=400, critic_fc2_dim=300,
                ckpt_dir=CKPT_DIR, gamma=0.99,tau=0.005, action_noise=0.1, policy_noise=0.2, policy_noise_clip=0.5,
                delay_time=2, max_size=10000000, batch_size=2048)

    # 初始化 ROS 节点，节点名为 limo_td3
    rospy.init_node('limo_td3')
    # 创建一个发布者，用于发布训练结果，话题名为 result，消息类型为 Float32MultiArray，队列大小为 5
    pub_result = rospy.Publisher('result', Float32MultiArray, queue_size=5)
    # 创建一个发布者，用于发布智能体选择的动作，话题名为 get_action，消息类型为 Float32MultiArray，队列大小为 5
    pub_get_action = rospy.Publisher('get_action', Float32MultiArray, queue_size=5)
    # 初始化训练结果消息
    result = Float32MultiArray()
    # 初始化动作消息
    get_action = Float32MultiArray()
    # 记录训练开始的时间
    start_time =time.time()
    
    # 初始化强化学习环境
    env = Env(action_dim=ACTION_DIMENSION)
    # 初始化上一时刻的动作，初始值为全 0
    past_action = np.zeros(ACTION_DIMENSION)
    
    # 用于记录每个回合的总奖励
    score_history = []

    
    # 加载模型 
    # agent.load_models(episode=1150)
    # 开始训练，总共进行 50000 个回合
    for e in range(1,50000):
        
        # 重置环境，获取初始状态
        state = env.reset()      
        # 初始化当前回合的总奖励
        episode_reward_sum = 0   
        # 标记当前回合是否结束
        done = False                       
        # 定义每个回合的最大步数
        episode_step=6000        
        for t in range(episode_step):
            # 智能体根据当前状态选择一个动作，处于训练模式
            action = agent.choose_action(state,train=True)          
            # 执行动作，获取新的状态、奖励以及回合是否结束的标志
            new_state,reward,done = env.step(action,past_action)    
            # 将当前的状态、动作、奖励、新状态和回合结束标志存入经验回放缓冲区
            agent.remember(state, action, reward, new_state, done)  
            # 智能体从经验回放缓冲区中采样数据进行学习
            agent.learn()                                           
            # 累加当前回合的总奖励
            episode_reward_sum +=reward
            # 更新上一时刻的动作
            past_action = copy.deepcopy(action)
            # 更新当前状态
            state = copy.deepcopy(new_state)
            
            # 打印当前回合数、是否结束、奖励、动作、状态、总奖励、线速度和角速度
            print("ep ",e)
            print("done ",done)
            print("reward ",reward)
            print("action ",action)
            print("state ",state)
            print("reward_sum ",episode_reward_sum)
            print("v = ",action[0])
            print("w = ",action[1])
            print("\n")        

            # 如果当前步数达到 1200，认为超时，结束当前回合
            if t >=1200:
                rospy.loginfo("time out!")
                done = True
            
            # 如果当前回合的总奖励小于等于 -2400，认为奖励不佳，结束当前回合
            if episode_reward_sum <= -2400:
                rospy.loginfo("reward fail")
                done = True

            # 如果回合结束，跳出当前步数循环
            if done:
                print("after done ")
                break

        
        # 保存模型和奖励曲线
        # 将当前回合的总奖励添加到历史记录中
        score_history.append(episode_reward_sum)
        # 每 5 个回合保存一次模型和学习曲线
        if e % 5 == 0:
            # 定义学习曲线图片的文件名
            filename_score = save_figure_path + "score_history_" + str(e) + ".png"
            # 绘制学习曲线并保存为图片
            plotLearning(score_history, filename_score, window=100)
            # 保存智能体的模型参数
            agent.save_models(episode=e)
            # 将每个回合的总奖励保存到文本文件中
            with open(save_figure_path +'score_history.txt', 'w') as f:
                for score in score_history:
                    f.write(str(score) + '\n')

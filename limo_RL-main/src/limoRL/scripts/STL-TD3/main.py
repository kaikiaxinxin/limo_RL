import params
from stl_env import STL_Gazebo_Env
from agent import TD3_Dual_Critic
from buffer import Buffer
from trainer import Trainer
from utils import set_seed, OU_Noise, make_dirs

def main():
    # 1. 初始化
    set_seed(0)
    make_dirs([params.MODEL_DIR, params.LOG_DIR])
    
    # 2. 模块实例化
    env = STL_Gazebo_Env()
    agent = TD3_Dual_Critic()
    buffer = Buffer(max_size=params.TOTAL_STEPS)
    noise = OU_Noise(params.ACTION_DIM)
    
    # 3. 训练托管
    trainer = Trainer(env, agent, buffer, noise)
    trainer.train()

if __name__ == "__main__":
    main()
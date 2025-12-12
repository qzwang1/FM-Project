import os
import time
import numpy as np
import gymnasium as gym
import scenic
from stable_baselines3 import PPO
from scenic.gym import ScenicGymEnv
from franka_simulator import WebotsFrankaSimulator
from controller import Supervisor

def main():
    print("\n" + "="*50)
    print("       >>> 进入演示模式 (Enjoy Mode) <<<        ")
    print("       正在加载 best_model.zip ...")
    print("="*50 + "\n")

    # 1. 路径设置 (使用你之前验证过的绝对路径)
    SCENIC_FILE_PATH = "/Users/liuyanqing/research/ACL-experiments/Webots/scenarios/final_project.scenic"
    # 模型文件必须在当前目录下
    MODEL_PATH = "best_model" 

    # 2. 初始化 Webots
    supervisor = Supervisor()
    simulator = WebotsFrankaSimulator(supervisor)

    # 3. 定义空间 (必须与训练时完全一致: 7动作, 13观测)
    action_space = gym.spaces.Box(
        low=np.full(shape=(7,), fill_value=-1.0), 
        high=np.full(shape=(7,), fill_value=1.0),
        dtype=np.float32
    )
    
    observation_space = gym.spaces.Box( 
        low=-np.inf,
        high=np.inf,
        shape=(13,), 
        dtype=np.float64                                  
    )

    # 4. 加载场景
    scenario = scenic.scenarioFromFile(SCENIC_FILE_PATH,
                                    model="scenic.simulators.webots.model",
                                    mode2D=False)

    # 5. 创建环境
    # 注意：feedback_fn 防止报错
    env = ScenicGymEnv(
        scenario, 
        simulator, 
        render_mode=None, 
        max_steps=2000, 
        action_space=action_space,
        observation_space=observation_space,
        feedback_fn=lambda result: None 
    )

    # 6. 加载模型
    # 检查文件是否存在
    if not os.path.exists(MODEL_PATH + ".zip"):
        print(f"!!! 错误: 找不到模型文件 {MODEL_PATH}.zip !!!")
        print("请确保你把 zip 文件复制到了 controllers/scenic_supervisor/ 目录下")
        return

    model = PPO.load(MODEL_PATH, env=env)
    print(">>> 模型加载成功！开始演示... <<<")

    # 7. 循环演示
    obs, _ = env.reset()
    
    try:
        while True:
            # 【关键】deterministic=True 
            # 让机器人展示它学到的“最优解”，不再随机乱动
            action, _ = model.predict(obs, deterministic=True)
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            # 如果回合结束，自动重置
            if terminated or truncated:
                print("--- 成功抓取/回合结束，重置场景 ---")
                obs, _ = env.reset()
                
    except KeyboardInterrupt:
        print("\n演示结束。")

if __name__ == "__main__":
    main()
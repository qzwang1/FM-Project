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
    print("="*50)
    print("       加载模型并开始演示 (Enjoy Mode)        ")
    print("="*50)

    # 1. 路径设置 (确保指向你正确的 Scenic 文件)
    # 建议使用绝对路径，或者确保相对路径正确
    SCENIC_FILE_PATH = "/Users/liuyanqing/research/ACL-experiments/Webots/scenarios/final_project.scenic"
    MODEL_PATH = "ppo_panda_final" # 不需要加 .zip，SB3 会自动找

    # 2. 初始化 Webots Supervisor
    supervisor = Supervisor()

    # 3. 初始化模拟器
    simulator = WebotsFrankaSimulator(supervisor)

    # 4. 定义 Action 和 Observation Space (必须与训练时完全一致!)
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

    # 5. 加载 Scenic 场景
    print(f"Loading Scenic file: {SCENIC_FILE_PATH}")
    scenario = scenic.scenarioFromFile(SCENIC_FILE_PATH,
                                    model="scenic.simulators.webots.model",
                                    mode2D=False)

    # 6. 创建 Gym 环境
    env = ScenicGymEnv(
        scenario, 
        simulator, 
        render_mode=None, 
        max_steps=2000, 
        action_space=action_space,
        observation_space=observation_space,
        feedback_fn=lambda result: None # 防止报错
    )

    # 7. 加载训练好的模型
    if not os.path.exists(MODEL_PATH + ".zip"):
        print(f"Error: Model file '{MODEL_PATH}.zip' not found!")
        return

    print(f"Loading model from: {MODEL_PATH}")
    model = PPO.load(MODEL_PATH, env=env)

    # 8. 开始循环演示
    obs, _ = env.reset()
    print("\n>>> Demo Started! Press 'Real-time' in Webots to watch. <<<\n")

    try:
        while True:
            # 关键点：deterministic=True
            # 这意味着让机器人输出它认为“最好”的动作，而不是随机探索
            action, _ = model.predict(obs, deterministic=True)
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                print("--- Episode Finished. Resetting... ---")
                obs, _ = env.reset()
                
    except KeyboardInterrupt:
        print("\nDemo stopped by user.")
    finally:
        # 虽然 Webots 控制器通常不手动 close，但在脚本中是个好习惯
        pass

if __name__ == "__main__":
    main()
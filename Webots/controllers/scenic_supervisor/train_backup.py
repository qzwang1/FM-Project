from scenic.gym import ScenicGymEnv
import scenic
from franka_simulator import WebotsFrankaSimulator

import gymnasium as gym
import numpy as np
from controller import Supervisor

# 引入 PPO 算法
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
import os

# --- 1. 初始化 Webots Supervisor ---
supervisor = Supervisor() 

# --- 2. 初始化模拟器 ---
simulator = WebotsFrankaSimulator(supervisor)

# --- 3. 定义 Action Space (7维) ---
action_space = gym.spaces.Box(
    low=np.full(shape=(7,), fill_value=-1.0), 
    high=np.full(shape=(7,), fill_value=1.0),
    dtype=np.float32
)

# --- 4. 定义 Observation Space (13维) ---
observation_space = gym.spaces.Box( 
    low=-np.inf,
    high=np.inf,
    shape=(13,), 
    dtype=np.float64                                  
)

max_steps = 2000 
total_timesteps = 500000 # 从 100,000 改为 500,000

# --- 5. 加载 Scenic 场景 ---
scenario = scenic.scenarioFromFile("/Users/liuyanqing/research/ACL-experiments/Webots/scenarios/final_project.scenic",
                                model="scenic.simulators.webots.model",
                                mode2D=False)

# --- 6. 创建 Gym 环境 ---
env = Monitor(ScenicGymEnv(
    scenario, 
    simulator, 
    render_mode=None, 
    max_steps=max_steps, 
    action_space=action_space,
    observation_space=observation_space,
    # 【关键修改】防止 'NoneType is not callable' 错误
    feedback_fn=lambda result: None
)) 

eval_callback = EvalCallback(env, best_model_save_path='./logs/',
                             log_path='./logs/', eval_freq=1000,
                             deterministic=True, render=False)

# --- 7. 设置 PPO 模型并开始训练 ---
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_tensorboard/")

print("---------------------------------------")
print("       Start PPO Training...           ")
print("       (按 Ctrl+C 可以随时停止并保存)     ")
print("---------------------------------------")

try:
    # 尝试一直训练到结束
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)
except KeyboardInterrupt:
    # 如果你按了 Ctrl+C，会跳到这里
    print("\n\n!!! 检测到人工停止 (User Interrupted) !!!")
    print("正在紧急保存模型，请稍候...")
finally:
    # 无论正常结束还是由 Ctrl+C 结束，都会执行这里
    model.save("ppo_panda_final")
    print(">>> 模型已保存为 ppo_panda_final.zip <<<")
    print("Training Finished.")
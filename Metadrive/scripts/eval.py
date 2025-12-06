import os
import numpy as np
import pandas as pd
import torch
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO

import scenic
from custom.custom_gym import CustomMetaDriveEnv
from custom.custom_simulator import CustomMetaDriveSimulator


SCENIC_EVAL_FILE = "./scenarios/driver_eval.scenic"  
SUMO_MAP = "./CARLA/Town01.net.xml"

max_steps = 200


RANDOM_MODEL_PATH = "./baseline-random-1/ppo_metadrive_model.zip"
CE_MODEL_PATH = "./verifai-ce/ppo_metadrive_verifai.zip"
BO_MODEL_PATH = "./verifai-bo/ppo_metadrive_verifai.zip"

N_EVAL_EPISODES = 50


RESULTS_DIR = "./eval_results"
os.makedirs(RESULTS_DIR, exist_ok=True)


class AutoBoxObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs, _info = env.reset()
        obs = np.asarray(obs, dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=obs.shape,
            dtype=np.float32,
        )

    def observation(self, observation):
        return np.asarray(observation, dtype=np.float32)


def make_eval_env():
    scenario = scenic.scenarioFromFile(
        SCENIC_EVAL_FILE,
        model="scenic.simulators.metadrive.model",
        mode2D=True,
    )

    action_space = spaces.Box(
        low=np.array([-1.0, -1.0], dtype=np.float32),
        high=np.array([1.0, 1.0], dtype=np.float32),
        shape=(2,),
        dtype=np.float32,
    )

    env = CustomMetaDriveEnv(
        scenario=scenario,
        simulator=CustomMetaDriveSimulator(
            sumo_map=SUMO_MAP,
            max_steps=max_steps,
        ),
        max_steps=max_steps,
        action_space=action_space,
        file=SCENIC_EVAL_FILE,
    )

    # eval 时统一关掉 VerifAI（只用 Scenic random）
    if hasattr(env, "enable_verifai"):
        env.enable_verifai(False)
    # 不再调用 set_verifai_feedback(None)，避免 float(None) 报错

    env = AutoBoxObsWrapper(env)
    return env


def eval_model(model_path: str, n_episodes: int, name: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=== Evaluating {name} ({model_path}) on device {device} ===")

    env = make_eval_env()
    model = PPO.load(model_path, device=device)

    ep_ids = []
    rewards = []
    collisions = []
    coverages = []
    lengths = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        truncated = False

        ep_reward = 0.0
        ep_len = 0
        ep_collision = 0
        ep_cov_sum = 0.0

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)

            ep_reward += float(reward)
            ep_len += 1

            if isinstance(info, dict):
                coll = int(info.get("collision", 0))
                if coll == 1:
                    ep_collision = 1
                step_cov = float(info.get("coverage_step", 0.0))
                ep_cov_sum += step_cov

        ep_cov = ep_cov_sum / float(ep_len) if ep_len > 0 else 0.0

        ep_ids.append(ep)
        rewards.append(ep_reward)
        collisions.append(ep_collision)
        coverages.append(ep_cov)
        lengths.append(ep_len)

        print(
            f"[{name}] Ep {ep:03d} | "
            f"R={ep_reward:8.2f}, len={ep_len:3d}, "
            f"coll={ep_collision}, cov={ep_cov:.4f}"
        )

    env.close()

    rewards = np.array(rewards, dtype=float)
    collisions = np.array(collisions, dtype=float)
    coverages = np.array(coverages, dtype=float)
    lengths = np.array(lengths, dtype=float)

    print(f"\n--- Summary for {name} ---")
    print(f"Episodes            : {n_episodes}")
    print(f"Mean reward         : {rewards.mean():.3f} ± {rewards.std():.3f}")
    print(f"Mean episode length : {lengths.mean():.3f}")
    print(f"Collision rate      : {collisions.mean():.3f}")
    print(f"Mean coverage (↓)   : {coverages.mean():.5f}")

 
    df = pd.DataFrame(
        {
            "model": [name] * n_episodes,
            "episode": ep_ids,
            "reward": rewards,
            "length": lengths,
            "collision": collisions,
            "coverage": coverages,
        }
    )
    csv_path = os.path.join(RESULTS_DIR, f"eval_{name}.csv")
    df.to_csv(csv_path, index=False)
    print(f"Per-episode results saved to: {csv_path}")


if __name__ == "__main__":
    eval_model(BO_MODEL_PATH, N_EVAL_EPISODES, name="BO")

import gc
import os
import time

import gymnasium as gym
import numpy as np
import pandas as pd
import scenic
import torch
from gymnasium import ObservationWrapper, spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from custom.custom_gym import CustomMetaDriveEnv
from custom.custom_simulator import CustomMetaDriveSimulator


SCENIC_FILE = "./scenarios/driver.scenic"
SUMO_MAP = "./CARLA/Town01.net.xml"

max_steps = 200         
episodes = 250         
total_timesteps = max_steps * episodes

LOG_DIR = "./verifai-bo"
os.makedirs(LOG_DIR, exist_ok=True)


WARMUP_EPISODES = 50


def compute_verifai_feedback(
    ep_reward: float,
    ep_len: int,
    max_steps: int,
    collisions: int,
    coverage: float,
) -> float:

    w_coll = 3.0       
    w_early = 2.0      
    w_negR = 1.0      
    w_cov = 0.5        
    R_norm = 100.0    
    cov_target = 0.7   


    diff_coll = w_coll * (1.0 if collisions > 0 else 0.0)


    diff_early = w_early * max(0.0, 1.0 - ep_len / float(max_steps))


    diff_negR = w_negR * max(0.0, -ep_reward / R_norm)


    diff_cov = w_cov * max(0.0, cov_target - coverage)

    difficulty = diff_coll + diff_early + diff_negR + diff_cov


    return -difficulty


class AutoBoxObsWrapper(ObservationWrapper):

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


class MetricsCallback(BaseCallback):
    def __init__(self, csv_path: str, verbose: int = 0):
        super().__init__(verbose)

        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_collisions = []
        self.episode_coverages = []

        self.cumulative_collisions = []
        self.cumulative_coverages = []

        self.total_collision = 0.0  
        self.total_coverage = 0.0

        self._ep_reward = 0.0
        self._ep_length = 0
        self._ep_collision = 0
        self._ep_coverage_sum = 0.0

        self.csv_path = csv_path
        self._csv_initialized = os.path.exists(csv_path)

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards", None)
        dones = self.locals.get("dones", None)
        infos = self.locals.get("infos", None)

        if rewards is None or dones is None or infos is None:
            return True

        r = float(rewards[0])
        done = bool(dones[0])
        info = infos[0] if isinstance(infos, (list, tuple)) else infos

        self._ep_reward += r
        self._ep_length += 1

        if isinstance(info, dict):
            # step 级 info
            coll = int(info.get("collision", 0))
            if coll == 1:
                self._ep_collision = 1

            step_cov = float(info.get("coverage_step", 0.0))
            self._ep_coverage_sum += step_cov

        if done:
            if self._ep_length > 0:
                coverage_rate = self._ep_coverage_sum / float(self._ep_length)
            else:
                coverage_rate = 0.0

            self.episode_rewards.append(self._ep_reward)
            self.episode_lengths.append(self._ep_length)
            self.episode_collisions.append(self._ep_collision)
            self.episode_coverages.append(coverage_rate)

          
            self.total_coverage += coverage_rate
            self.cumulative_coverages.append(self.total_coverage)

            self.total_collision += self._ep_collision
            self.cumulative_collisions.append(self.total_collision)


            ep_idx = len(self.episode_rewards) - 1

            row = {
                "episode": ep_idx,
                "reward": self.episode_rewards[ep_idx],
                "length": self.episode_lengths[ep_idx],
                "collision": self.episode_collisions[ep_idx],
                "coverage": self.episode_coverages[ep_idx],
                "total_collision": self.cumulative_collisions[ep_idx],
                "total_coverage": self.cumulative_coverages[ep_idx],
            }
            df = pd.DataFrame([row])


            if not self._csv_initialized:
                df.to_csv(self.csv_path, index=False, mode="w", header=True)
                self._csv_initialized = True
            else:
                df.to_csv(self.csv_path, index=False, mode="a", header=False)

            if self.verbose > 0:
                print(
                    f"[Metrics] ep {ep_idx} | "
                    f"R={row['reward']:.2f}, "
                    f"coll={row['collision']}, "
                    f"cov={row['coverage']:.3f}, "
                    f"len={row['length']}"
                )


            try:
                feedback = compute_verifai_feedback(
                    ep_reward=row["reward"],
                    ep_len=row["length"],
                    max_steps=max_steps,
                    collisions=row["collision"],
                    coverage=row["coverage"],
                )


                base_env = self.training_env.envs[0]
                raw_env = base_env.unwrapped  

                if hasattr(raw_env, "set_verifai_feedback"):
                    raw_env.set_verifai_feedback(feedback)

           
                if hasattr(raw_env, "enable_verifai") and (ep_idx + 1 >= WARMUP_EPISODES):
                    raw_env.enable_verifai(True)

            except Exception as e:
                if self.verbose > 0:
                    print(f"[MetricsCallback] VerifAI feedback update failed: {e}")

            self._ep_reward = 0.0
            self._ep_length = 0
            self._ep_collision = 0
            self._ep_coverage_sum = 0.0

        return True

    def save_csv(self, filename: str):
        n = min(
            len(self.episode_rewards),
            len(self.episode_lengths),
            len(self.episode_collisions),
            len(self.episode_coverages),
            len(self.cumulative_collisions),
            len(self.cumulative_coverages),
        )

        if n == 0:
            print("No finished episodes, skip saving metrics.")
            return

        data = {
            "episode": np.arange(n),
            "reward": self.episode_rewards[:n],
            "length": self.episode_lengths[:n],
            "collision": self.episode_collisions[:n],
            "coverage": self.episode_coverages[:n],
            "total_collision": self.cumulative_collisions[:n],
            "total_coverage": self.cumulative_coverages[:n],
        }
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)


def make_env():
    scenario = scenic.scenarioFromFile(
        SCENIC_FILE,
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
        file=SCENIC_FILE,
    )

    env = AutoBoxObsWrapper(env)

    env = Monitor(
        env,
        filename=os.path.join(LOG_DIR, "monitor.csv"),
        info_keywords=("collision", "coverage_total"),
    )

    return env


if __name__ == "__main__":
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("DEBUG total_timesteps =", total_timesteps)

    env = make_env()

    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)


    metrics_csv = os.path.join(LOG_DIR, "training_metrics_incremental.csv")
    metrics_callback = MetricsCallback(csv_path=metrics_csv, verbose=1)


    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        device=device,
        tensorboard_log=LOG_DIR,
        n_steps=max_steps,      
        batch_size=max_steps,   
        n_epochs=4,             
        gamma=0.99,
        learning_rate=3e-4,
        clip_range=0.2,
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=metrics_callback,
    )

    model_path = os.path.join(LOG_DIR, "ppo_metadrive_verifai")
    model.save(model_path)
    print(f"Model saved to: {model_path}")

    final_metrics_csv = os.path.join(LOG_DIR, "training_metrics_final.csv")
    metrics_callback.save_csv(final_metrics_csv)
    print(f"Final training metrics saved to: {final_metrics_csv}")

    env.close()
    gc.collect()
    end_time = time.time()
    print(f"Total time: {end_time - start_time:.1f} seconds")

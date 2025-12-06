from scenic.core.simulators import Simulator, Simulation
from scenic.core.scenarios import Scenario
import gymnasium as gym
from gymnasium import spaces
from typing import Callable
from metadrive.envs import MetaDriveEnv


from scenic.core.simulators import Simulator, Simulation
from scenic.core.scenarios import Scenario, Scene
from scenic.core.distributions import RejectionException 
from scenic.core.serialization import SerializationError
import gymnasium as gym
from gymnasium import spaces
from typing import Callable
import numpy as np
import random
import scenic

from scenic.core.errors import setDebuggingOptions

setDebuggingOptions(verbosity=0, fullBacktrace=False, debugExceptions=False, debugRejections=False)



#TODO make ResetException
class ResetException(Exception):
    def __init__(self):
        super().__init__("Resetting")

class CustomMetaDriveEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4} # TODO placeholder, add simulator-specific entries
    
    def __init__(self, 
                 scenario,
                 simulator : Simulator,
                 file: str, 
                 render_mode=None, 
                 max_steps = 100,
                 observation_space : spaces.Dict = spaces.Dict(),
                 action_space : spaces.Dict = spaces.Dict(),
                 record_scenic_sim_results : bool = True,
                 feedback_fn : callable = lambda x: x,
                 genetic_flag : bool = True): # empty string means just pure scenic???

        assert render_mode is None or render_mode in self.metadata["render_modes"]

        self.observation_space = observation_space
        self.action_space = action_space
        self.render_mode = render_mode
        self.max_steps = max_steps - 1 # FIXME, what was this about again?
        self.simulator = simulator
        self.scenario = scenario
        self.simulation_results = []

        self.genetic_flag = genetic_flag

        self.use_verifai = False
        self.feedback_result = None


        self.loop = None
        self.record_scenic_sim_results = record_scenic_sim_results
        self.feedback_fn = feedback_fn

        self.episode_counter = 0 # id to map instances
        self.episode_plvs = {}
        self.previous_scenes = {}
        self.previous_scenes_params = {}


        self.gae_lambda = 0.95
        self.gamma      = 0.99
        self.pvl_threshold = 0

        self.episode_rewards = []
        self.episode_values  = []

        self.scenic_file = file

    def _make_run_loop(self):
        while True:
            try:
                scene = self.get_scene()
                with self.simulator.simulateStepped(scene, maxSteps=self.max_steps) as simulation:
                    steps_taken = 0

                    observation = simulation.get_obs()
                    info = simulation.get_info()
                    actions = yield observation, info
                    simulation.actions = actions

                    while True:

                        simulation.advance()
                        steps_taken += 1


                        observation = simulation.get_obs()
                        info = simulation.get_info()
                        reward = simulation.get_reward()

                        terminated = False  
                        truncated = False  

                        if simulation.get_truncation():
                            terminated = True

                        if steps_taken >= self.max_steps:
                            truncated = True

                        if terminated or truncated:
                            yield observation, reward, terminated, truncated, info
                            break  

                       
                        actions = yield observation, reward, False, False, info
                        simulation.actions = actions

            except ResetException:
                continue



    def reset(self, seed=None, options=None): # TODO will setting seed here conflict with VerifAI's setting of seed?
        # only setting enviornment seed, not torch seed?
        if self.episode_counter > 0:
            self.compute_episode_pvl()
        super().reset(seed=seed)
        self.rewards = []
        self.values  = []
        if self.loop is None:
            self.loop = self._make_run_loop()
            observation, info = next(self.loop) # not doing self.scene.send(action) just yet
        else:
            observation, info = self.loop.throw(ResetException())
        return observation, info
        
    def step(self, action):
        assert not (self.loop is None), "self.loop is None, have you called reset()?"

        observation, reward, terminated, truncated, info = self.loop.send(action)
        return observation, reward, terminated, truncated, info

    def render(self): # TODO figure out if this function has to be implemented here or if super() has default implementation
        """
        likely just going to be something like simulation.render() or something
        """
        # FIXME for one project only...also a bit hacky...
        # self.env.render()
        pass

    def close(self):
        self.simulator.destroy()

    def log_episode_stats(self,reward,value):
        """
        Docstring for log_episode_stats
        
        :param reward: Episode rewards
        :param value: Value estimates from the model
        """
        self.episode_rewards.append(reward)
        self.episode_values.append(value)

    # def compute_episode_pvl(self):
    #     """
    #     Docstring for compute_episode_pvl
        
    #     :Compute the average postive value loss per episode 
    #     """
    #     if len(self.episode_rewards) == 0:
    #    	    return
       	    
    #     lastgaelam = 0 
    #     advantages = [0] * len(self.episode_rewards) # hold the  
    #     for t in reversed(range(len(self.episode_rewards))):
    #         if t == len(self.episode_rewards) - 1:
    #             next_v = 0
    #             nextnonterminal = 0
    #         else:
    #             next_v = self.episode_values[t+1]
    #             nextnonterminal = 1
    #         delta = self.episode_rewards[t] + self.gamma * next_v * nextnonterminal - self.episode_values[t]
    #         advantages[t] = lastgaelam = delta[0] + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
    #         advantages[t] = max(advantages[t],0)        

    #     pvl = np.sum(advantages)/len(advantages)

    #     if pvl > self.pvl_threshold or self.curr_scene_id < 10:
    #         self.episode_plvs[self.curr_scene_id] = np.sum(advantages)/len(advantages)
    #         self.curr_scene_id += 1
    def compute_episode_pvl(self):
        return

    def set_verifai_feedback(self, feedback: float):
        self.feedback_result = float(feedback)
    
    def enable_verifai(self, flag: bool = True):
        self.use_verifai = bool(flag)

    def get_scene(self):
        scenario = self.scenario
        if isinstance(scenario, list):
            scenario = random.choice(scenario)

        if self.use_verifai and (self.feedback_result is not None):
            scene, _ = scenario.generate(feedback=self.feedback_result)
        else:
            scene, _ = scenario.generate()

        if isinstance(scene, list):
            if len(scene) == 0:
                raise RuntimeError("Scenic returned an empty scene list")
            scene = scene[0]

        return scene


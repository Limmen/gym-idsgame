import gymnasium as gym
import gym_idsgame
import numpy as np
from stable_baselines3 import PPO

class SingleAgentEnvWrapper(gym.Env):

    def __init__(self, idsgame_env, defender_action: int):
        self.defender_action = defender_action
        self.idsgame_env = idsgame_env
        self.observation_space = gym.spaces.Box(low=np.array([0]*10), high=np.array([0]*10), dtype=np.int32)
        self.action_space = idsgame_env.action_space

    def step(self, a: int):
        action = (a, self.defender_action)
        obs, rewards, done, _, info = self.idsgame_env.step(action)
        return obs[0], rewards[0], done, _, info

    def reset(self, seed: int = 0):
        o, _ = self.idsgame_env.reset()
        return o[0], {}

    def render(self, mode: str ='human'):
        self.idsgame_env.render()


if __name__ == '__main__':
    idsgame_env = gym.make("idsgame-minimal_defense-v19")
    env = SingleAgentEnvWrapper(idsgame_env=idsgame_env, defender_action=0)
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=25000)
    obs, _ = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, _, info = env.step(action)
        env.render("human")

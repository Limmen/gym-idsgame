import gym
import numpy as np
from gym_idsgame.envs.dao.idsgame_config import IdsGameConfig

class BaselineEnvWrapper(gym.Env):

  def __init__(self, env_name : str, idsgame_config: IdsGameConfig = None, save_dir: str = None, initial_state_path: str = None):
    super(BaselineEnvWrapper, self).__init__()
    self.idsgame_env = gym.make(env_name, idsgame_config=idsgame_config,
                   save_dir=save_dir,
                   initial_state_path=initial_state_path)

    self.action_space = self.idsgame_env.attacker_action_space
    self.observation_space = gym.spaces.Box(low=0, high=self.idsgame_env.idsgame_config.game_config.max_value,
                   shape=(self.idsgame_env.idsgame_config.game_config.num_nodes*((self.idsgame_env.idsgame_config.game_config.num_attack_types+1)*2),),
                   dtype=np.float32)
    self.prev_episode_hacked = False
    self.prev_episode_detected = False
    self.metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50  # Video rendering speed
    }
    #self.observation_space = self.idsgame_env.observation_space

  def step(self, action):
    joint_action = (action, None)
    obs_prime, reward, done, info = self.idsgame_env.step(joint_action)
    attacker_reward, defender_reward = reward
    obs_prime_attacker, obs_prime_defender = obs_prime
    complete_obs = np.concatenate([obs_prime_attacker.flatten(), obs_prime_defender.flatten()])
    # if not self.idsgame_env.is_attack_legal(action):
    #   attacker_reward = -100
    return complete_obs, attacker_reward, done, info

  def reset(self):
    self.prev_episode_hacked = self.idsgame_env.state.hacked
    self.prev_episode_detected = self.idsgame_env.state.detected
    obs = self.idsgame_env.reset()
    obs_attacker, obs_defender = obs
    complete_obs = np.concatenate([obs_attacker.flatten(), obs_defender.flatten()])
    return complete_obs

  def render(self, mode='human'):
    return self.idsgame_env.render(mode=mode)

  def close (self):
    return self.idsgame_env.close()


  def is_attack_legal(self, attack_action: int) -> bool:
    """
    Check if a given attack is legal or not.

    :param attack_action: the attack to verify
    :return: True if legal otherwise False
    """
    return self.idsgame_env.is_attack_legal(attack_action)


  def is_defense_legal(self, defense_action: int) -> bool:
    """
    Check if a given defense is legal or not.

    :param defense_action: the defense action to verify
    :return: True if legal otherwise False
    """
    return self.idsgame_env.is_defense_legal(defense_action)

  def num_attack_actions(self):
    return self.idsgame_env.num_attack_actions

  def num_defense_actions(self):
    return self.idsgame_env.num_defense_actions

  def hack_probability(self):
    if self.num_games > 0:
      return self.num_hacks/self.num_games
    else:
      return 0.0

  def games(self):
    return self.num_games
    #return self.idsgame_env.hack_probability()
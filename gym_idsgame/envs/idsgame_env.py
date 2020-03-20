from typing import Union
import gym
import numpy as np
from gym_idsgame.envs.constants import constants
import os
from gym_idsgame.envs.dao.game_state import GameState
from gym_idsgame.envs.dao.node_type import NodeType
from gym_idsgame.envs.dao.idsgame_config import IdsGameConfig
import gym_idsgame.envs.util.idsgame_util as util
from gym_idsgame.envs.rendering.agents.defender import Defender

class IdsGameEnv(gym.Env):
    """
    TODO
    """

    def __init__(self, idsgame_config: IdsGameConfig = None):
        """
        TODO
        """
        if idsgame_config is None:
            idsgame_config = IdsGameConfig()
        util.validate_config(idsgame_config)
        self.idsgame_config: IdsGameConfig = idsgame_config
        self.state: GameState = self.idsgame_config.game_config.initial_state.copy()
        self.observation_space = GameState.get_attacker_observation_space(
            self.idsgame_config.game_config.max_value, self.idsgame_config.game_config.num_attack_types,
            self.idsgame_config.game_config.num_nodes)
        self.defender = Defender(self.idsgame_config.defender_policy)
        self.action_space = GameState.get_attacker_action_space(self.idsgame_config.game_config.num_actions)
        self.viewer = None
        self.steps_beyond_done = None
        self.metadata = {
         'render.modes': ['human', 'rgb_array'],
         'video.frames_per_second' : 50 # Video rendering speed
        }
        self.reward_range = (float(constants.GAME_CONFIG.NEGATIVE_REWARD), float(constants.GAME_CONFIG.POSITIVE_REWARD))

    def __get_server_under_attack(self, action:int) -> Union[int, Union[int,int], int]:
        server_id = action // self.idsgame_config.game_config.num_attack_types
        server_pos = self.idsgame_config.game_config.network_config.get_node_pos(server_id)
        attack_type = self.__get_attack_type(action)
        return server_id, server_pos, attack_type

    def __get_attack_type(self, action):
        attack_type = action % self.idsgame_config.game_config.num_attack_types
        return attack_type

    def step(self, action):
        # Initialization
        reward = 0
        info = {}
        self.state.attack_events = []
        self.state.defense_events = []

        # 1. Interpret attacker action
        attacker_pos = self.state.attacker_pos
        target_node_id, target_pos, attack_type = self.__get_server_under_attack(action)

        # 2. Sample defense action
        defense_pos, defense_type, defense_node_id = self.__defense_action()

        # 3. Defend
        self.state.defend(defense_node_id, defense_type, self.idsgame_config.game_config.max_value)
        self.state.add_defense_event(defense_pos, defense_type)

        if util.is_attack_legal(attacker_pos, target_pos, self.idsgame_config.game_config.num_cols,
                                self.idsgame_config.game_config.network_config.adjacency_matrix):
            # 4. Attack
            self.state.attack(target_node_id, attack_type, self.idsgame_config.game_config.max_value)
            self.state.add_attack_event(target_pos, attack_type)

            # 5. Simulate attack outcome
            attack_successful = self.state.simulate_attack(target_node_id, attack_type)

            # 6. Update state based on attack outcome
            if attack_successful:
                self.state.attacker_pos = target_pos
                if target_pos == self.idsgame_config.game_config.network_config.data_pos:
                    self.state.done = True
                    self.state.hacked = True
                    reward = constants.GAME_CONFIG.POSITIVE_REWARD
                else:
                    detected = self.state.simulate_detection(target_node_id)
                    if detected:
                        self.state.done = True
                        self.state.detected = True
                        reward = -constants.GAME_CONFIG.POSITIVE_REWARD
        if self.state.done:
            if self.steps_beyond_done is None:
                self.steps_beyond_done = 0
            else:
                gym.logger.warn("You are calling 'step()' even though this environment has already returned done = True. "
                                "You should always call 'reset()' once you receive 'done = True' -- "
                                "any further steps are undefined behavior.")
                self.steps_beyond_done += 1
        self.state.game_step += 1
        observation = self.state.get_attacker_observation(self.idsgame_config.game_config.max_value,
                                                          self.idsgame_config.game_config.num_attack_types,
                                                          self.idsgame_config.game_config.num_nodes)
        if self.viewer is not None:
            self.viewer.gameframe.set_state(self.state)
        return observation, reward, self.state.done, info

    def reset(self):
        """
        Resets the environment and returns the initial state

        :return: the initial state
        """
        self.steps_beyond_done = None
        self.state.new_game(self.idsgame_config.game_config.initial_state)
        if self.viewer is not None:
            self.viewer.gameframe.reset()
        observation = self.state.get_attacker_observation(self.idsgame_config.game_config.max_value,
                                                          self.idsgame_config.game_config.num_attack_types,
                                                          self.idsgame_config.game_config.num_nodes)
        return observation

    def render(self, mode='human'):
        """
        Renders the environment

        Supported rendering modes:

        - human: render to the current display or terminal and
          return nothing. Usually for human consumption.
        - rgb_array: Return an numpy.ndarray with shape (x, y, 3),
          representing RGB values for an x-by-y pixel image, suitable
          for turning into a video.

        :param mode: the rendering mode
        :return: True (if human mode) otherwise an rgb array
        """
        if mode not in self.metadata["render.modes"]:
            raise NotImplemented("mode: {} is not supported".format(mode))
        if self.viewer is None:
            self.__setup_viewer()
        arr = self.viewer.render(return_rgb_array = mode=='rgb_array')
        self.state.attack_events = []
        self.state.defense_events = []
        return arr

    def __setup_viewer(self):
        """
        Setup for the viewer to use for rendering
        :return: None
        """
        from gym_idsgame.envs.rendering.viewer import Viewer
        script_dir = os.path.dirname(__file__)
        resource_path = os.path.join(script_dir, './rendering/', constants.RENDERING.RESOURCES_DIR)
        self.idsgame_config.render_config.resources_dir = resource_path
        self.viewer = Viewer(idsgame_config=self.idsgame_config)
        self.viewer.agent_start()

    def close(self):
        """
        Closes the viewer (cleanup)

        :return: None
        """
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def __defense_action(self) -> Union[Union[int, int], int, int]:
        defense_row, defense_col, defense_type = self.defender.policy.action(self.state)
        defense_pos = (defense_row, defense_col)
        defense_node_id = self.idsgame_config.game_config.network_config.get_node_id(defense_pos)
        return defense_pos, defense_type, defense_node_id
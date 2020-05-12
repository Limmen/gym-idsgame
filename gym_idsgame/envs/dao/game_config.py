"""
Game-specific configuration for the gym-idsgame environment
"""
import gym
import numpy as np
from gym_idsgame.envs.dao.game_state import GameState
from gym_idsgame.envs.dao.network_config import NetworkConfig

class GameConfig():
    """
    DTO with game configuration parameters
    """
    def __init__(self, network_config: NetworkConfig = None, manual_attacker: bool = True, num_layers: int = 1,
                 num_servers_per_layer: int = 2, num_attack_types: int = 10, max_value: int = 9,
                 initial_state: GameState = None, manual_defender: bool = False, initial_state_path :str = None,
                 dense_rewards = False, min_random_a_val :int = 0, min_random_d_val :int = 0,
                 min_random_det_val :int = 0):
        """
        Class constructor, initializes the DTO

        :param network_config: the network configuration of the game (e.g. number of nodes and their connectivity)
        :param manual_attacker: whether the attacker is controlled manually or by an agent
        :param manual_attacker: whether the defender is controlled manually or by an agent
        :param num_layers: the number of layers in the network
        :param num_servers_per_layer: the number of servers per layer in the network
        :param num_attack_types: the number of attack types
        :param max_value: max value for a defense/attack attribute
        :param initial_state: the initial state
        :param initial_state_path: path to the initial state saved on disk
        :param dense_rewards: if true, give hacker dense rewards (reward for each intermediate server hacked)
        :param min_random_a_val: minimum attack value when randomizing the state
        :param min_random_d_val: minimum defense value when randomizing the state
        :param min_random_det_val: minimum detection value when randomizing the state
        """
        self.manual_attacker = manual_attacker
        self.manual_defender = manual_defender
        self.num_layers = num_layers
        self.num_servers_per_layer = num_servers_per_layer
        self.num_attack_types = num_attack_types
        self.max_value = max_value
        self.min_random_a_val = min_random_a_val
        self.min_random_d_val = min_random_d_val
        self.min_random_det_val = min_random_det_val
        self.num_rows = self.num_layers + 2
        self.num_nodes = self.num_layers * self.num_servers_per_layer + 2  # +2 for Start and Data Nodes
        self.num_cols = self.num_servers_per_layer
        self.num_attack_actions = self.num_attack_types * self.num_nodes
        self.num_defense_actions = (self.num_attack_types+1) * self.num_nodes
        self.num_states = self.num_nodes
        self.network_config = network_config
        self.initial_state_path = initial_state_path
        self.defense_val = 2
        self.attack_val = 0
        self.num_vulnerabilities_per_node = 1
        self.det_val = 2
        self.vulnerabilitiy_val = 0
        self.num_vulnerabilities_per_layer = None
        if network_config is None:
            self.network_config = NetworkConfig(self.num_rows, self.num_cols, connected_layers=False)
        self.initial_state = initial_state
        if self.initial_state is None and self.initial_state_path is not None:
            self.initial_state = GameState.load(self.initial_state)
        if self.initial_state is None and self.initial_state_path is None:
            self.initial_state = GameState(min_random_a_val=min_random_a_val, min_random_det_val=min_random_det_val,
                                           min_random_d_val=min_random_d_val)
            self.initial_state.default_state(self.network_config.node_list, self.network_config.start_pos,
                                             self.num_attack_types, network_config=self.network_config)
        self.dense_rewards = dense_rewards

    def set_load_initial_state(self, initial_state_path: str) -> None:
        """
        Sets the initial state by loading it from disk

        :param initial_state_path:
        :return: None
        """
        self.initial_state = GameState.load(initial_state_path)

    def set_initial_state(self, defense_val=2, attack_val=0,
                  num_vulnerabilities_per_node=1, det_val=2, vulnerability_val=0,
                          num_vulnerabilities_per_layer=None):
        """
        Utility function for setting the initial game state

        :param defense_val: defense value for defense types that are not vulnerable
        :param attack_val: attack value for attack types
        :param num_vulnerabilities_per_node: number of vulnerabilities per node
        :param det_val: detection value per node
        :param vulnerability_val: defense value for defense types that are vulnerable
        :param num_vulnerabilities_per_layer: number of vulnerabilities per layer
        :param min_random_val: minimum val when randomizing the state
        :return:
        """
        if num_vulnerabilities_per_layer is None:
            num_vulnerabilities_per_layer = self.num_servers_per_layer
        self.defense_val = defense_val
        self.attack_val = attack_val
        self.num_vulnerabilities_per_layer = num_vulnerabilities_per_layer
        self.det_val = det_val
        self.vulnerabilitiy_val = vulnerability_val
        self.num_vulnerabilities_per_node = num_vulnerabilities_per_node
        self.initial_state.set_state(self.network_config.node_list, self.num_attack_types, defense_val=defense_val,
                                     attack_val=attack_val, num_vulnerabilities_per_node=num_vulnerabilities_per_node,
                                     det_val=det_val, vulnerability_val=vulnerability_val,
                                     network_config=self.network_config,
                                     num_vulnerabilities_per_layer=num_vulnerabilities_per_layer)

    def get_attacker_observation_space(self) -> gym.spaces.Box:
        """
        Creates an OpenAI-Gym Space for the game observation

        :return: observation space
        """
        high_row = np.array([self.max_value] * (self.num_attack_types + 1))
        high = np.array([high_row] * self.num_nodes)
        low = np.zeros((self.num_nodes, self.num_attack_types + 1))
        observation_space = gym.spaces.Box(low=low, high=high, dtype=np.int32)
        return observation_space

    def get_defender_observation_space(self) -> gym.spaces.Box:
        """
        Creates an OpenAI-Gym Space for the game observation

        :return: observation space
        """
        high_row = np.array([self.max_value] * (self.num_attack_types + 1))
        high = np.array([high_row] * 1)
        low = np.zeros((1, self.num_attack_types + 1))
        observation_space = gym.spaces.Box(low=low, high=high, dtype=np.int32)
        return observation_space

    def get_action_space(self, defender :bool = False) -> gym.spaces.Discrete:
        """
        Creates an OpenAi-Gym space for the actions in the environment

        :param defender: boolean flag if defender or not
        :return: action space
        """
        if defender:
            return gym.spaces.Discrete(self.num_defense_actions)
        else:
            return gym.spaces.Discrete(self.num_attack_actions)

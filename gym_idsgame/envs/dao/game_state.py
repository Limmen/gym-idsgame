"""
Stateful data of the gym-idsgame environment
"""
from typing import Union, List
import numpy as np
from gym_idsgame.envs.dao.node_type import NodeType
from gym_idsgame.envs.constants import constants
from gym_idsgame.envs.dao.attack_defense_event import AttackDefenseEvent
from gym_idsgame.envs.dao.network_config import NetworkConfig

class GameState():
    """
    DTO representing the state of the game
    """

    def __init__(self, attack_values: np.ndarray = None, defense_values: np.ndarray = None,
                 defense_det: np.ndarray = None,
                 attacker_pos: Union[int, int] = (0, 0), game_step: int = 0, attacker_cumulative_reward: int = 0,
                 defender_cumulative_reward: int = 0,
                 num_games: int = 0, attack_events: List[AttackDefenseEvent] = [],
                 defense_events: List[AttackDefenseEvent] = [],
                 done: bool = False, detected: bool = False, attack_type: int = 0, num_hacks: int = 0,
                 hacked: bool = False):
        """
        Constructor, initializes the DTO

        :param attack_values: the attack values for resource nodes in the network
        :param defense_values: the defense values for resource nodes in the network
        :param defense_det: detection values for resource nodes in the network
        :param attacker_pos: position of attacker in the network
        :param game_step: the number of steps of the current game
        :param attacker_cumulative_reward: the cumulative reward over all games of the attacker
        :param defender_cumulative_reward: the cumulative reward over all games of the defender
        :param num_games: the number of games played
        :param attack_events: attack events that are in queue to be simulated
        :param defense_events: defense events that are in queue to be simulated
        :param done: True if the game is over and otherwise False
        :param detected: True if the attacker is in a detected state, otherwise False
        :param attack_type: the type of the last attack
        :param num_hacks: number of wins for the attacker
        :param hacked: True if the attacker hacked the data node otherwise False
        """
        self.attack_values = attack_values
        self.defense_values = defense_values
        self.defense_det = defense_det
        self.attacker_pos = attacker_pos
        self.game_step = game_step
        self.attacker_cumulative_reward = attacker_cumulative_reward
        self.defender_cumulative_reward = defender_cumulative_reward
        self.num_games = num_games
        self.attack_events = attack_events
        self.defense_events = defense_events
        self.done = done
        self.detected = detected
        self.attack_type = attack_type
        self.num_hacks = num_hacks
        self.hacked = hacked
        self.action_descriptors = ["Injection", "Authentication", "CrossSite", "References", "Misssconfiguration",
                                   "Exposure", "Access", "Forgery", "Vulnerabilities", "Redirects"]

    def default_state(self, node_list: List[int], attacker_pos: Union[int, int], num_attack_types: int) -> None:
        """
        Creates a default state

        :param graph_layout: the layout of the network to create a default state for
        :param num_rows: the number of rows in the grid network
        :param num_cols: the number of columns in the grid network
        :param num_attack_types: the number of attack types
        :return: None
        """
        num_nodes = len(node_list)
        attack_values = np.zeros((num_nodes, num_attack_types))
        defense_values = np.zeros((num_nodes, num_attack_types))
        det_values = np.zeros(num_nodes)
        for node_id in range(num_nodes):
            # if node_list[node_id] == NodeType.START.value:
            #     det_values[node_id] = 2
            if node_list[node_id] == NodeType.DATA.value or node_list[node_id] == NodeType.SERVER.value:
                defense_values[node_id] = [2] * num_attack_types
                defense_values[node_id][0] = 0 # vulnerability
                det_values[node_id] = 2
        self.attack_values = attack_values.astype(np.int32)
        self.defense_values = defense_values.astype(np.int32)
        self.defense_det = det_values.astype(np.int32)
        self.attacker_pos = attacker_pos
        self.game_step = 0
        self.attacker_cumulative_reward = 0
        self.defender_cumulative_reward = 0
        self.num_games = 0
        self.attack_events = []
        self.defense_events = []
        self.done = False
        self.detected = False
        self.attack_type = 0
        self.num_hacks = 0
        self.hacked = False

    def new_game(self, init_state: "GameState") -> None:
        """
        Updates the current state for a new game

        :param init_state: the initial state of the first game
        :return: None
        """
        self.game_step = 0
        self.done = False
        self.attack_type = 0
        self.num_games += 1
        self.attack_events = []
        self.defense_events = []
        self.attacker_pos = init_state.attacker_pos
        self.attack_values = np.copy(init_state.attack_values)
        self.defense_values = np.copy(init_state.defense_values)
        self.defense_det = np.copy(init_state.defense_det)
        if self.detected:
            self.attacker_cumulative_reward -= constants.GAME_CONFIG.POSITIVE_REWARD
            self.defender_cumulative_reward += constants.GAME_CONFIG.POSITIVE_REWARD
        self.detected = False
        if self.hacked:
            self.attacker_cumulative_reward += constants.GAME_CONFIG.POSITIVE_REWARD
            self.defender_cumulative_reward -= constants.GAME_CONFIG.POSITIVE_REWARD
            self.num_hacks += 1
        self.hacked = False

    def copy(self) -> "GameState":
        """
        Creates a copy of the state

        :return: a copy of the current state
        """
        new_state = GameState()
        new_state.attack_values = np.copy(self.attack_values)
        new_state.defense_values = np.copy(self.defense_values)
        new_state.defense_det = np.copy(self.defense_det)
        new_state.attacker_pos = self.attacker_pos
        new_state.game_step = self.game_step
        new_state.attacker_cumulative_reward = self.attacker_cumulative_reward
        new_state.defender_cumulative_reward = self.defender_cumulative_reward
        new_state.num_games = self.num_games
        new_state.attack_events = self.attack_events
        new_state.defense_events = self.defense_events
        new_state.done = self.done
        new_state.detected = self.detected
        new_state.attack_type = self.attack_type
        new_state.num_hacks = self.num_hacks
        new_state.hacked = self.hacked
        return new_state

    def attack(self, node_id: int, attack_type: int, max_value: int, network_config: NetworkConfig) -> None:
        """
        Increments the attack value of the specified node and attack type

        :param node_id: id of the node to defend
        :param attack_type: the type of attack attribute to increment
        :param max_value: the maximum defense value
        :param network_config: NetworkConfig
        :return: None
        """
        if network_config.node_list[node_id] != NodeType.START and self.attack_values[node_id][attack_type] < max_value:
            self.attack_values[node_id][attack_type] += 1

    def defend(self, node_id: int, defense_type: int, max_value: int, network_config: NetworkConfig) -> None:
        """
        Increments the defense value of the specified node and defense type

        :param node_id: id of the node to defend
        :param defense_type: the type of defense attribute to increment
        :param max_value: the maximum defense value
        :param network_config: NetworkConfig
        :return: None
        """
        if network_config.node_list[node_id] != NodeType.START and \
                self.defense_values[node_id][defense_type] < max_value:
            self.defense_values[node_id][defense_type] += 1

    def simulate_attack(self, attacked_node_id: int, attack_type: int, network_config: NetworkConfig) -> bool:
        """
        Simulates an attack operation

        :param attacked_node_id: the id of the node that is attacked
        :param attack_type: the type of the attack
        :param network_config: NetworkConfig
        :return: True if the attack was successful otherwise False
        """
        if network_config.node_list[attacked_node_id] == NodeType.START:
            return True
        return self.attack_values[attacked_node_id][attack_type] > self.defense_values[attacked_node_id][attack_type]

    def simulate_detection(self, node_id: int) -> bool:
        """
        Simulates detection for a unsuccessful attack

        :param node_id: the id of the node to simulate deteciton of
        :return: True if the node was detected, otherwise False
        """
        return np.random.rand() < self.defense_det[node_id] / 10

    def get_attacker_observation(self, network_config: NetworkConfig) -> np.ndarray:
        """
        Converts the state of the dynamical system into an observation for the attacker. As the environment
        is a partially observed markov decision process, the attacker observation is only a subset of the game state

        :param network_config: the network configuration of the game
        :return: An observation of the environment
        """
        # +1 to have an extra feature that indicates if this is the node that the attacker is currently in
        attack_observation = np.zeros((len(network_config.node_list), self.attack_values.shape[1]+1))
        current_pos = self.attacker_pos
        current_node_id = network_config.get_node_id(current_pos)
        current_row, current_col = current_pos
        current_adjacency_matrix_id = network_config.get_adjacency_matrix_id(current_row, current_col)
        for node_id in range(len(network_config.node_list)):
            pos = network_config.get_node_pos(node_id)
            node_row, node_col = pos
            node_adjacency_matrix_id = network_config.get_adjacency_matrix_id(node_row, node_col)
            if node_id == current_node_id:
                attack_observation[node_id] = np.append(self.attack_values[node_id], 1)
            if network_config.adjacency_matrix[current_adjacency_matrix_id][node_adjacency_matrix_id]:
                attack_observation[node_id] = np.append(self.attack_values[node_id], 0)
        return attack_observation

    def get_attacker_node_from_observation(self, observation: np.ndarray) -> int:
        for node_id in range(len(observation)):
            if observation[node_id][-1] == 1:
                return node_id
        raise AssertionError("Could not find the node that the attacker is in")

    def add_attack_event(self, target_pos: Union[int, int], attack_type: int) -> None:
        """
        Adds an attack event to the state

        :param target_pos: position in the grid of the target node
        :param attack_type: the type of the attack
        :return: None
        """
        attack_event = AttackDefenseEvent(target_pos, attack_type)
        self.attack_events.append(attack_event)

    def add_defense_event(self, target_pos: Union[int, int], defense_type: int) -> None:
        """
        Adds a defense event to the state

        :param target_pos: the position in the grid of the target node
        :param defense_type: the type of the defense
        :return: None
        """
        defense_event = AttackDefenseEvent(target_pos, defense_type)
        self.defense_events.append(defense_event)

    def get_defender_observation(self):
        pass

    def restart(self):
        self.num_games = 0
        self.num_hacks = 0
        self.defender_cumulative_reward = 0
        self.attacker_cumulative_reward = 0

import numpy as np
from typing import Union, List
import gym
from gym_idsgame.envs.dao.node_type import NodeType
from gym_idsgame.envs.constants import constants
from gym_idsgame.envs.dao.attack_defense_event import AttackDefenseEvent

class GameState():
    """
    DTO representing the state of the game
    """

    def __init__(self, attack_values:np.ndarray = np.array([0]), defense_values:np.ndarray = np.array([0]),
                 defense_det:np.ndarray = np.array([0]),
                 attacker_pos:Union[int, int] = (0,0), game_step:int = 0, attacker_cumulative_reward:int = 0,
                 defender_cumulative_reward :int =0,
                 num_games=0, attack_events:List[AttackDefenseEvent] = [], defense_events:List[AttackDefenseEvent] = [],
                 done:bool=False, detected:bool = False, attack_type:int=0, num_hacks:int = 0, hacked:bool=False):
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
        self.attack_values=attack_values
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

    def default_state(self, node_list: List[int], attacker_pos: Union[int, int], num_attack_types:int) -> None:
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
            if node_list[node_id] == NodeType.DATA.value or node_list[node_id] == NodeType.SERVER.value:
                defense_values[node_id] = [2] * num_attack_types
                defense_values[node_id][0] = 0 # vulnerability
                det_values[node_id] = 2
        self.attack_values =  attack_values.astype(np.int32)
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
        self.num_games +=1
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
            self.num_hacks +=1
        self.hacked = False
        return

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

    def attack(self, node_id:int, attack_type:int, max_value:int) -> None:
        """
        Increments the attack value of the specified node and attack type

        :param node_id: id of the node to defend
        :param attack_type: the type of attack attribute to increment
        :param max_value: the maximum defense value
        :return: None
        """
        if self.attack_values[node_id][attack_type] < max_value:
            self.attack_values[node_id][attack_type] += 1

    def defend(self, node_id:int, defense_type:int, max_value:int) -> None:
        """
        Increments the defense value of the specified node and defense type

        :param node_id: id of the node to defend
        :param defense_type: the type of defense attribute to increment
        :param max_value: the maximum defense value
        :return: None
        """
        if self.defense_values[node_id][defense_type] < max_value:
            self.defense_values[node_id][defense_type] += 1

    def simulate_attack(self, attacked_node_id:int, attack_type:int) -> bool:
        """
        Simulates an attack operation

        :param attacked_node_id: the id of the node that is attacked
        :param attack_type: the type of the attack
        :return: True if the attack was successful otherwise False
        """
        if self.attack_values[attacked_node_id][attack_type] > self.defense_values[attacked_node_id][attack_type]:
            return True
        else:
            return False

    def simulate_detection(self, node_id) -> bool:
        """
        Simulates detection for a unsuccessful attack

        :return: True if the node was detected, otherwise False
        """
        if np.random.rand() < self.defense_det[node_id] / 10:
            return True
        else:
            return False

    @staticmethod
    def get_attacker_observation_space(max_value, num_attack_types, num_nodes):
        high_row = np.array([max_value] * (num_attack_types + 1))
        high = np.array([high_row] * num_nodes)
        low = np.zeros((num_nodes, num_attack_types + 1))
        observation_space = gym.spaces.Box(low=low, high=high, dtype=np.int32)
        return observation_space

    @staticmethod
    def get_attacker_action_space(num_actions):
        return gym.spaces.Discrete(num_actions)

    def get_attacker_observation(self, num_rows, num_cols, num_attack_types):
        attack_observation = np.zeros((num_rows, num_cols, num_attack_types))
        return attack_observation

    def add_attack_event(self, target_pos, attack_type):
        attack_event = AttackDefenseEvent(target_pos, attack_type)
        self.attack_events.append(attack_event)

    def add_defense_event(self, target_pos, defense_type):
        defense_event = AttackDefenseEvent(target_pos, defense_type)
        self.defense_events.append(defense_event)

    def get_defender_observation(self):
        pass

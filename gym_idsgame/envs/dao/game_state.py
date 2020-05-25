"""
Stateful data of the gym-idsgame environment
"""
from typing import Union, List
import numpy as np
import pickle
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
                 num_games: int = 0, attack_events: List[AttackDefenseEvent] = None,
                 defense_events: List[AttackDefenseEvent] = None,
                 done: bool = False, detected: bool = False, attack_type: int = 0, num_hacks: int = 0,
                 hacked: bool = False, min_random_a_val :int = 0, min_random_d_val :int = 0,
                 min_random_det_val :int = 0,
                 max_value : int = 9, reconnaissace_state : np.ndarray = None):
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
        :param min_random_a_val: minimum attack value when randomizing the state
        :param min_random_d_val: minimum defense value when randomizing the state
        :param min_random_det_val: minimum detection value when randomizing the state
        :param max_value: the maximum value of attack/defense attributes
        :param reconnaissace_state: the state of the reconnaissance activities by the attacker
        """
        self.attack_values = attack_values
        self.defense_values = defense_values
        self.defense_det = defense_det
        self.attacker_pos = attacker_pos
        self.reconnaissance_state = reconnaissace_state
        self.game_step = game_step
        self.attacker_cumulative_reward = attacker_cumulative_reward
        self.defender_cumulative_reward = defender_cumulative_reward
        self.num_games = num_games
        self.attack_events = attack_events
        self.defense_events = defense_events
        self.min_random_a_val = min_random_a_val
        self.min_random_d_val = min_random_d_val
        self.min_random_det_val = min_random_det_val
        self.max_value = max_value
        if self.attack_events is None:
            self.attack_events = []
        if self.defense_events is None:
            self.defense_events = []
        self.done = done
        self.detected = detected
        self.attack_defense_type = attack_type
        self.num_hacks = num_hacks
        self.hacked = hacked
        self.action_descriptors = ["Injection", "Authentication", "CrossSite", "References", "Misssconfiguration",
                                   "Exposure", "Access", "Forgery", "Vulnerabilities", "Redirects"]

    def default_state(self, node_list: List[int], attacker_pos: Union[int, int], num_attack_types: int,
                      network_config: NetworkConfig, randomize_state : bool = False) -> None:
        """
        Creates a default state

        :param graph_layout: the layout of the network to create a default state for
        :param num_rows: the number of rows in the grid network
        :param num_cols: the number of columns in the grid network
        :param num_attack_types: the number of attack types
        :param network_config: network config
        :param randomize_state: boolean flag whether to create the state randomly
        :return: None
        """
        self.set_state(node_list, num_attack_types, network_config=network_config,
                       num_vulnerabilities_per_layer=network_config.num_cols, randomize_state=randomize_state)
        self.attacker_pos = attacker_pos
        self.game_step = 0
        self.attacker_cumulative_reward = 0
        self.defender_cumulative_reward = 0
        self.num_games = 0
        self.attack_events = []
        self.defense_events = []
        self.done = False
        self.detected = False
        self.attack_defense_type = 0
        self.num_hacks = 0
        self.hacked = False


    def set_state(self, node_list : List, num_attack_types : int, defense_val :int = 2, attack_val :int = 0,
                  num_vulnerabilities_per_node : int = 1, det_val : int = 2, vulnerability_val : int = 0,
                  num_vulnerabilities_per_layer : int = 1,
                  network_config : NetworkConfig = None, randomize_state : bool = False):
        """
        Sets the state

        :param node_list: list of nodes
        :param num_attack_types:  number of attack types
        :param defense_val: defense value for defense types that are not vulnerable
        :param attack_val: attack value for attack types
        :param num_vulnerabilities_per_node: number of vulnerabilities per node
        :param det_val: detection value per node
        :param vulnerability_val: defense value for defense types that are vulnerable
        :param num_vulnerabilities_per_layer: number of vulnerabilities per layer
        :param network_config: network configuration
        :param randomize_state: boolean flag whether to create the state randomly
        :return: None
        """
        num_nodes = len(node_list)
        attack_values = np.zeros((num_nodes, num_attack_types))
        defense_values = np.zeros((num_nodes, num_attack_types))
        det_values = np.zeros(num_nodes)
        reconnaissance_state = np.full((num_nodes, num_attack_types), constants.GAME_CONFIG.INITIAL_RECONNAISSANCE_STATE)

        d_val = defense_val
        a_val = attack_val
        de_val = det_val

        vulnerabilities_per_layer = np.zeros((network_config.num_rows, network_config.num_cols))
        for row in range(1, network_config.num_rows-1):
            vulnerabilities = np.random.choice(network_config.num_cols, size=num_vulnerabilities_per_layer,
                                               replace=False)
            vulnerabilities_per_layer[row][vulnerabilities] = 1

        for node_id in range(num_nodes):
            row, col = network_config.get_node_pos(node_id)
            num_vuln = min(num_vulnerabilities_per_node, num_attack_types)
            vulnerabilities = []
            if vulnerabilities_per_layer[row][col] == 1 or node_list[node_id] == NodeType.DATA.value:
                vulnerabilities = np.random.choice(num_attack_types, size=num_vuln) # random vulnerability per node
            if node_list[node_id] == NodeType.DATA.value or node_list[node_id] == NodeType.SERVER.value:
                d_vals = []
                a_vals = []
                for at in range(num_attack_types):
                    if randomize_state:
                        d_val = max(self.min_random_d_val,
                                    np.random.choice(list(range(self.min_random_d_val, defense_val + 1))))
                        a_val = max(self.min_random_a_val,
                                    np.random.choice(list(range(self.min_random_a_val, attack_val + 1))))
                        de_val = max(self.min_random_det_val,
                                     np.random.choice(list(range(self.min_random_det_val, det_val + 1))))
                    d_vals.append(d_val)
                    a_vals.append(a_val)
                defense_values[node_id] = d_vals
                det_values[node_id] = de_val
                attack_values[node_id] = a_vals
                for vuln_id in vulnerabilities:
                    defense_values[node_id][vuln_id] = vulnerability_val  # vulnerability (lower defense)
        self.attack_values = attack_values.astype(np.int32)
        self.defense_values = defense_values.astype(np.int32)
        self.defense_det = det_values.astype(np.int32)
        self.reconnaissance_state = reconnaissance_state.astype(np.int32)


    def new_game(self, init_state: "GameState", a_reward : int = 0, d_reward : int = 0,
                 update_stats = True, randomize_state : bool = False, network_config : NetworkConfig = None,
                 num_attack_types : int = None, defense_val : int = None, attack_val : int = None,
                 det_val : int = None, vulnerability_val : int = None,
                 num_vulnerabilities_per_layer : int = None, num_vulnerabilities_per_node : int = None) -> None:
        """
        Updates the current state for a new game

        :param init_state: the initial state of the first game
        :param a_reward: the reward delta to increment or decrement the attacker cumulative reward with
        :param d_reward: the reward delta to increment or decrement the defender cumulative reward with
        :param randomize_state: boolean flag whether to create the state randomly
        :param network_config: network config (necessary if randomizing the state every game)
        :param defense_val: defense value for defense types that are not vulnerable
        :param attack_val: attack value for attack types
        :param num_vulnerabilities_per_node: number of vulnerabilities per node
        :param det_val: detection value per node
        :param vulnerability_val: defense value for defense types that are vulnerable
        :param num_vulnerabilities_per_layer: number of vulnerabilities per layer
        :return: None
        """
        if update_stats:
            self.num_games += 1
            if self.hacked:
                self.attacker_cumulative_reward += a_reward
                self.defender_cumulative_reward += d_reward
                self.num_hacks += 1
            if self.detected:
                self.attacker_cumulative_reward += a_reward
                self.defender_cumulative_reward += d_reward
        self.done = False
        self.attack_defense_type = 0
        self.game_step = 0
        self.attack_events = []
        self.defense_events = []
        self.attacker_pos = init_state.attacker_pos
        if not randomize_state:
            self.attack_values = np.copy(init_state.attack_values)
            self.defense_values = np.copy(init_state.defense_values)
            self.defense_det = np.copy(init_state.defense_det)
            self.reconnaissance_state = np.copy(init_state.reconnaissance_state)
        else:
            self.set_state(network_config.node_list, num_attack_types, network_config=network_config,
                           num_vulnerabilities_per_layer=num_vulnerabilities_per_layer, randomize_state=randomize_state,
                           defense_val=defense_val,
                           attack_val=attack_val, num_vulnerabilities_per_node=num_vulnerabilities_per_node,
                           det_val=det_val, vulnerability_val=vulnerability_val)
        self.detected = False
        self.hacked = False

    def copy(self) -> "GameState":
        """
        Creates a copy of the state

        :return: a copy of the current state
        """
        new_state = GameState(min_random_a_val=self.min_random_a_val, min_random_d_val=self.min_random_d_val,
                              min_random_det_val=self.min_random_det_val, max_value=self.max_value)
        new_state.attack_values = np.copy(self.attack_values)
        new_state.defense_values = np.copy(self.defense_values)
        new_state.defense_det = np.copy(self.defense_det)
        new_state.reconnaissance_state = np.copy(self.reconnaissance_state)
        new_state.attacker_pos = self.attacker_pos
        new_state.game_step = self.game_step
        new_state.attacker_cumulative_reward = self.attacker_cumulative_reward
        new_state.defender_cumulative_reward = self.defender_cumulative_reward
        new_state.num_games = self.num_games
        new_state.attack_events = self.attack_events
        new_state.defense_events = self.defense_events
        new_state.done = self.done
        new_state.detected = self.detected
        new_state.attack_defense_type = self.attack_defense_type
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

    def reconnaissance(self, node_id: int, attack_type: int) -> int:
        """
        Performs a reconnaissance activity for the attacker

        :param node_id: id of the node to defend
        :param attack_type: the type of attack attribute to increment
        :param max_value: the maximum defense value
        :param network_config: NetworkConfig
        :return: reward
        """
        reward = 0*constants.GAME_CONFIG.POSITIVE_REWARD \
            if self.reconnaissance_state[node_id][attack_type] == self.defense_values[node_id][attack_type] \
            else 1*constants.GAME_CONFIG.POSITIVE_REWARD
        self.reconnaissance_state[node_id][attack_type] = self.defense_values[node_id][attack_type]
        return reward

    def defend(self, node_id: int, defense_type: int, max_value: int, network_config: NetworkConfig,
               detect : bool = False) -> bool:
        """
        Increments the defense value of the specified node and defense type

        :param node_id: id of the node to defend
        :param defense_type: the type of defense attribute to increment
        :param max_value: the maximum defense value
        :param network_config: NetworkConfig
        :param detect: True if it is a detect action otherwise False
        :return: True if update had effect, otherwise False
        """
        if detect or defense_type >= self.defense_values.shape[1]:
            if network_config.node_list[node_id] != NodeType.START and self.defense_det[node_id] < max_value:
                self.defense_det[node_id] += 1
                return True
        else:
            if network_config.node_list[node_id] != NodeType.START and \
                    self.defense_values[node_id][defense_type] < max_value:
                self.defense_values[node_id][defense_type] += 1
                return True
        return False

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

    def simulate_detection(self, node_id: int, reconnaissance: bool) -> bool:
        """
        Simulates detection for a unsuccessful attack

        :param node_id: the id of the node to simulate deteciton of
        :param reconnaissance: boolean flag, if true simulate detection of reconnaissance activity
        :return: True if the node was detected, otherwise False
        """
        if not reconnaissance:
            return np.random.rand() < self.defense_det[node_id] / 10
        else:
            det_prob = self.defense_det[node_id] / 10
            return np.random.rand() < det_prob/100

    def get_attacker_observation(self, network_config: NetworkConfig, local_view=False, reconnaissance = False) -> np.ndarray:
        """
        Converts the state of the dynamical system into an observation for the attacker. As the environment
        is a partially observed markov decision process, the attacker observation is only a subset of the game state

        :param network_config: the network configuration of the game
        :param local_view: boolean flag indicating whether observations are provided in a local view or not
        :param reconnaissance: boolean flag indicating whether reconnaissance states should be included
        :return: An observation of the environment
        """
        if not reconnaissance:
            # +1 to have an extra feature that indicates if this is the node that the attacker is currently in
            attack_observation = np.zeros((len(network_config.node_list), self.attack_values.shape[1] + 1))
        else:
            # +1 to have an extra feature that indicates if this is the node that the attacker is currently in
            attack_observation = np.zeros((len(network_config.node_list), (self.attack_values.shape[1]*2 + 1)))

        current_pos = self.attacker_pos
        current_node_id = network_config.get_node_id(current_pos)
        current_row, current_col = current_pos
        current_adjacency_matrix_id = network_config.get_adjacency_matrix_id(current_row, current_col)

        if local_view:
            neighbors = []

        for node_id in range(len(network_config.node_list)):
            pos = network_config.get_node_pos(node_id)
            node_row, node_col = pos
            node_adjacency_matrix_id = network_config.get_adjacency_matrix_id(node_row, node_col)
            if local_view:
                if network_config.adjacency_matrix[current_adjacency_matrix_id][node_adjacency_matrix_id] \
                        and node_id != current_node_id:
                    neighbor_data = np.append(self.attack_values[node_id], node_id)
                    neighbor_row, neighbor_col = network_config.get_node_pos(node_id)
                    neighbors.append((neighbor_row, neighbor_col, neighbor_data))
            else:
                if node_id == current_node_id:
                    if not reconnaissance:
                        attack_observation[node_id] = np.append(self.attack_values[node_id], 1)
                    else:
                        attack_observation[node_id] = np.append(np.append(self.attack_values[node_id], 1),
                                                                self.reconnaissance_state[node_id])
                elif network_config.fully_observed:
                    attack_observation[node_id] = np.append(self.attack_values[node_id], 0)
                elif network_config.adjacency_matrix[current_adjacency_matrix_id][node_adjacency_matrix_id]:
                    if not reconnaissance:
                        attack_observation[node_id] = np.append(self.attack_values[node_id], 0)
                    else:
                        attack_observation[node_id] = np.append(np.append(self.attack_values[node_id], 0),
                                                                self.reconnaissance_state[node_id])

        if local_view:
            # sort by row then col
            sorted_neighbors = sorted(neighbors, key=lambda x: (x[0], x[1]))
            neighbor_data = np.array(list(map(lambda x: x[2], sorted_neighbors)))
            neighbor_ids = neighbor_data[:,-1]
            local_view_obs = np.zeros((network_config.max_neighbors, self.attack_values.shape[1] + 2))
            for n in range(network_config.max_neighbors):
                rel_neighbor_pos = network_config.relative_neighbor_positions[n]
                neighbor_pos = (current_row + rel_neighbor_pos[0], current_col + rel_neighbor_pos[1])
                for i in range(len(neighbor_ids)):
                    node_id = neighbor_ids[i]
                    node_pos = network_config.get_node_pos(node_id)
                    if node_pos == neighbor_pos and node_pos[0] <= current_row:
                        local_view_obs[n] = np.append(neighbor_data[i], 1)
            attack_observation = np.array(local_view_obs)
        return attack_observation

    def get_attacker_node_from_observation(self, observation: np.ndarray, reconnaissance : bool = False) -> int:
        """
        Extracts which node the attacker is currently at from the observation representation

        :param observation: the observation representation emitted from the environment
        :param reconnaissance: boolean flag indicating whether the observation is from an env with reconnaissance state
        :return: the id of the node that the attacker is in
        """

        for node_id in range(len(observation)):
            if not reconnaissance:
                if observation[node_id][-1] == 1:
                    return node_id
            else:
                if observation[node_id][self.attack_values.shape[1]] == 1:
                    return node_id
        raise AssertionError("Could not find the node that the attacker is in")

    def add_attack_event(self, target_pos: Union[int, int], attack_type: int, attacker_pos: Union[int, int],
                         reconnaissance : bool = False) -> None:
        """
        Adds an attack event to the state

        :param target_pos: position in the grid of the target node
        :param attack_type: the type of the attack
        :param attacker_pos: position of the attacker
        :param reconnaissance: boolean flag indicating whether it is a reconnaissance event
        :return: None
        """
        attack_event = AttackDefenseEvent(target_pos, attack_type, attacker_pos=attacker_pos,
                                          reconnaissance=reconnaissance)
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

    def get_defender_observation(self, network_config: NetworkConfig):
        """
        Converts the state of the dynamical system into an observation for the defender. As the environment
        is a partially observed markov decision process, the defender observation is only a subset of the game state

        :param network_config: the network configuration of the game
        :return: An observation of the environment
        """
        # +1 for the detection value
        defense_observation = np.zeros((len(network_config.node_list), self.defense_values.shape[1] + 1))
        for node_id in range(len(network_config.node_list)):
            defense_observation[node_id] = np.append(self.defense_values[node_id], self.defense_det[node_id])
        return defense_observation

    def restart(self) -> None:
        """
        Resets the game state, clears up all the history
        :return: Noen
        """
        self.num_games = 0
        self.num_hacks = 0
        self.defender_cumulative_reward = 0
        self.attacker_cumulative_reward = 0

    @staticmethod
    def load(path):
        filehandler = open(path, 'rb')
        return pickle.load(filehandler)

    @staticmethod
    def save(path, state):
        filehandler = open(path + "/initial_state.pkl", 'wb')
        pickle.dump(state, filehandler)

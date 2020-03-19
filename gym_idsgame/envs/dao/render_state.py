import numpy as np
from typing import Union
from gym_idsgame.envs.dao.node_type import NodeType

class RenderState():
    """
    DTO representing the state of a current rendering
    """

    def __init__(self, attack_values:np.ndarray = np.array([0]), defense_values:np.ndarray = np.array([0]),
                 defense_det:np.ndarray = np.array([0]),
                 attacker_pos:Union[int, int] = (0,0), game_step:int = 0, attacker_cumulative_reward:int = 0,
                 defender_cumulative_reward :int =0,
                 num_games=0, attack_events:list = [], defense_events:list = [], done:bool=False, detected:bool = False,
                 attack_type:int=0):
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

    def default_state(self, graph_layout: np.ndarray, num_rows:int, num_cols:int, num_attack_types:int) -> None:
        """
        Creates a default state

        :param graph_layout: the layout of the network to create a default state for
        :param num_rows: the number of rows in the grid network
        :param num_cols: the number of columns in the grid network
        :param num_attack_types: the number of attack types
        :return: None
        """
        attack_values = np.zeros((num_rows, num_cols, num_attack_types))
        defense_values = np.zeros((num_rows, num_cols, num_attack_types))
        det_values = np.zeros((num_rows, num_cols))
        for i in range(num_rows):
            for j in range(num_cols):
                if (graph_layout[i][j] == NodeType.DATA.value
                        or graph_layout[i][j] == NodeType.SERVER.value):
                    defense_values[i][j] = [2] * num_attack_types
                    defense_values[i][j][0] = 0  # vulnerability
                    det_values[i][j] = 2
        self.attack_values =  attack_values.astype(np.int32)
        self.defense_values = defense_values.astype(np.int32)
        self.defense_det = det_values.astype(np.int32)
        self.attacker_pos = (num_rows - 1, num_cols // 2)
        self.game_step = 0
        self.attacker_cumulative_reward = 0
        self.defender_cumulative_reward = 0
        self.num_games = 0
        self.attack_events = []
        self.defense_events = []
        self.done = False
        self.detected = False
        self.attack_type = 0

    def new_game(self, init_state: "RenderState") -> None:
        """
        Updates the current state for a new game

        :param init_state: the initial state of the first game
        :return: None
        """
        self.game_step = 0
        self.done = False
        self.detected = False
        self.attack_type = 0
        self.num_games +=1
        self.attack_events = []
        self.defense_events = []
        self.attacker_pos = init_state.attacker_pos
        self.attack_values = np.copy(init_state.attack_values)
        self.defense_values = np.copy(init_state.defense_values)
        self.defense_det = np.copy(init_state.defense_det)

    def copy(self) -> "RenderState":
        """
        Creates a copy of the state

        :return: a copy of the current state
        """
        new_state = RenderState()
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
        return new_state

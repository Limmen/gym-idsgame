import numpy as np
from gym_idsgame.envs.dao.node_type import NodeType
class RenderState():

    def __init__(self, attack_values = np.array([0]), defense_values = np.array([0]), defense_det = [0],
                 attacker_pos = (0,0), game_step = 0, attacker_cumulative_reward = 0, defender_cumulative_reward=0,
                 num_games=0, attack_events = [], defense_events = [], done=False, detected = False, attack_type=0):
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

    def default_state(self, graph_layout, num_rows, num_cols, num_attack_types):
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

    def new_game(self, init_state):
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




    def copy(self):
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

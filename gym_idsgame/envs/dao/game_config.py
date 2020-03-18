from gym_idsgame.envs.rendering.constants import constants
from gym_idsgame.envs.dao.render_state import RenderState
import numpy as np

class GameConfig():

    def __init__(self, adjacency_matrix= None, graph_layout = None, manual = True, num_layers = 1,
                 num_servers_per_layer = 2, num_attack_types = 10, max_value = 10,
                 defense_policy=constants.BASELINE_POLICIES.NAIVE_DETERMINISTIC,
                 initial_state = None):
        self.adjacency_matrix = adjacency_matrix
        self.graph_layout = graph_layout
        self.manual = manual
        self.num_layers = num_layers
        self.num_servers_per_layer = num_servers_per_layer
        self.num_attack_types = num_attack_types
        self.max_value = max_value
        self.defense_policy = defense_policy
        self.initial_state = initial_state
        self.initial_state = initial_state
        self.num_rows = self.num_layers + 2
        self.num_cols = self.num_servers_per_layer
        if self.adjacency_matrix is None or self.graph_layout is None:
            self.graph_layout, self.adjacency_matrix = self.initialize_graph_config()
        if self.initial_state is None:
            self.initial_state = RenderState()
            self.initial_state.default_state(self.graph_layout, self.num_rows, self.num_cols, self.num_attack_types)

    def initialize_graph_config(self):
        graph_layout = np.zeros((self.num_rows, self.num_cols))
        for i in range(self.num_rows):
            for j in range(self.num_cols):
                if i == self.num_rows - 1:
                    if j == self.num_cols // 2:
                        graph_layout[i][j] = constants.NODE_TYPES.START
                    else:
                        graph_layout[i][j] = constants.NODE_TYPES.NONE
                elif i == 0:
                    if j == self.num_cols // 2:
                        graph_layout[i][j] = constants.NODE_TYPES.DATA
                    else:
                        graph_layout[i][j] = constants.NODE_TYPES.NONE
                else:
                    graph_layout[i][j] = constants.NODE_TYPES.SERVER

        adjacency_matrix = np.zeros((self.num_rows * self.num_cols,
                                     self.num_cols * self.num_rows))
        for i in range(self.num_rows * self.num_cols):
            row_1 = i // self.num_cols
            col_1 = i % self.num_cols
            for j in range(self.num_rows * self.num_cols):
                row_2 = j // self.num_cols
                col_2 = j % self.num_cols
                if row_1 == 0:
                    if row_2 == 1 and col_1 == self.num_cols // 2:
                        adjacency_matrix[i][j] = 1
                        adjacency_matrix[j][i] = 1
                elif row_1 == self.num_rows - 1:
                    if (row_2 == self.num_rows - 2
                            and col_1 == self.num_cols // 2):
                        adjacency_matrix[i][j] = 1
                        adjacency_matrix[j][i] = 1
                elif (row_2 == row_1 + 1 and col_1 == col_2):
                    adjacency_matrix[i][j] = 1
                    adjacency_matrix[j][i] = 1
        return graph_layout, adjacency_matrix
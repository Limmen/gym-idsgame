import numpy as np
from gym_idsgame.envs.dao.node_type import NodeType

class NetworkConfig:

    def __init__(self, num_rows:int, num_cols:int):
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.graph_layout = self.__default_graph_layout()
        self.adjacency_matrix = self.__default_adjacency_matrix()

    def __default_graph_layout(self) -> np.ndarray:
        graph_layout = np.zeros((self.num_rows, self.num_cols))
        for i in range(self.num_rows):
            for j in range(self.num_cols):
                if i == self.num_rows - 1:
                    if j == self.num_cols // 2:
                        graph_layout[i][j] = NodeType.START.value
                    else:
                        graph_layout[i][j] = NodeType.EMPTY.value
                elif i == 0:
                    if j == self.num_cols // 2:
                        graph_layout[i][j] = NodeType.DATA.value
                    else:
                        graph_layout[i][j] = NodeType.EMPTY.value
                else:
                    graph_layout[i][j] = NodeType.SERVER.value
        return graph_layout

    def __default_adjacency_matrix(self) -> np.ndarray:
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
        return adjacency_matrix

    @property
    def start_row(self):
        start_row, _ = self.start_pos
        return start_row

    @property
    def data_row(self):
        data_row, _ = self.data_pos
        return data_row

    @property
    def start_pos(self):
        for i in range(self.num_rows):
            for j in range(self.num_cols):
                if self.graph_layout[i][j] == NodeType.START.value:
                    return i,j
        raise AssertionError("Could not find start node in graph layout")

    @property
    def data_pos(self):
        for i in range(self.num_rows):
            for j in range(self.num_cols):
                if self.graph_layout[i][j] == NodeType.DATA.value:
                    return i, j
        raise AssertionError("Could not find data node in graph layout")
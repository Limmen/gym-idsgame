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
                if row_1 == self.data_row:
                    if row_2 == self.data_row+1 and col_1 == self.data_col:
                        adjacency_matrix[i][j] = 1
                        adjacency_matrix[j][i] = 1
                elif row_1 == self.start_row:
                    if row_2 == self.start_row-1 and col_1 == self.start_col:
                        adjacency_matrix[i][j] = 1
                        adjacency_matrix[j][i] = 1
                elif row_2 == (row_1 + 1) and col_1 == col_2 and row_1 != self.start_row and row_2 != self.start_row:
                    adjacency_matrix[i][j] = 1
                    adjacency_matrix[j][i] = 1
        return adjacency_matrix


    def get_coords(self, adjacency_matrix_id):
        row = adjacency_matrix_id // self.num_cols
        col = adjacency_matrix_id % self.num_cols
        return row, col

    @property
    def start_row(self):
        start_row, _ = self.start_pos
        return start_row

    @property
    def data_row(self):
        data_row, _ = self.data_pos
        return data_row

    @property
    def start_col(self):
        _, start_col = self.start_pos
        return start_col

    @property
    def data_col(self):
        _, data_col = self.data_pos
        return data_col

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
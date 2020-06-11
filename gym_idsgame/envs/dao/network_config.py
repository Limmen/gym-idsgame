"""
Network configuration for the gym-idsgame environment. E.g. number of servers, connectivity, etc.
"""
from typing import Union, List
import numpy as np
from gym_idsgame.envs.dao.node_type import NodeType

class NetworkConfig:
    """
    DTO with configuration of the network for the game, i.e. the servers and their connectivity
    """
    def __init__(self, num_rows: int, num_cols: int, connected_layers : bool = False, fully_observed = False,
                 relative_neighbor_positions : List = None):
        """
        Constructor

        :param num_rows: the number of rows in the network layout (think like a grid)
        :param num_cols: the number of columns in the network layout
        :param connected_layers: whether layers are connected with horizontal links or not
        :param fully_observed: boolean flag that indicates whether the environment
                              is fully observed or not (by default the environment is partially observed)
        :param relative_neighbor_positions: max num neighbors and their positions
        """
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.connected_layers = connected_layers
        self.graph_layout = self.__default_graph_layout()
        self.adjacency_matrix = self.__default_adjacency_matrix()
        self.fully_observed = fully_observed
        self.relative_neighbor_positions = relative_neighbor_positions
        self.max_neighbors = self.__max_num_neighbors()

    def __max_num_neighbors(self):
        max_neighbors = 0
        for node in range(len(self.node_list)):
            num_neighbors = self.adjacency_matrix[node].sum()
            max_neighbors = max(max_neighbors, num_neighbors)
        return max_neighbors

    def __default_graph_layout(self) -> np.ndarray:
        """
        Creates a default graph layout with a specific set of rows

        :return: Numpy array with a grid and in each position in the grid there is a node-type:
                START, EMPTY, SERVER, or DATA
        """
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
        """
        Creates a default adjacency matrix for a given graph layout

        :return: a numpy matrix representing the adjacency matrix with dimension (num_rows*num_cols, num_rows*num_cols)
        """
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
                elif self.connected_layers:
                    if row_2 == row_1 and col_1 == col_2+1 and row_1 != self.start_row and row_1 != self.data_row:
                        adjacency_matrix[i][j] = 1
                        adjacency_matrix[j][i] = 1
        return adjacency_matrix.astype(np.int32)


    def get_coords_of_adjacency_matrix_id(self, node_id: int) -> Union[int, int]:
        """
        Gets the grid-coordinates of a node id

        :param node_id: the id of the node in the adjacency matrix
        :return: (row,col)
        """
        row = node_id // self.num_cols
        col = node_id % self.num_cols
        return row, col

    def get_adjacency_matrix_id(self, row: int, col:int) -> Union[int, int]:
        """
        Get the adjacency matrix id from a set of coordinates in the grid

        :param row: the row
        :param col: the col
        :return: (row,col)
        """
        return row*self.num_cols + col

    @property
    def start_row(self) -> int:
        """
        :return: the starting row of the attacker
        """
        start_row, _ = self.start_pos
        return start_row

    @property
    def data_row(self) -> int:
        """
        :return: the row of the data node
        """
        data_row, _ = self.data_pos
        return data_row

    @property
    def start_col(self) -> int:
        """
        :return: the starting col of the attacker
        """
        _, start_col = self.start_pos
        return start_col

    @property
    def data_col(self) -> int:
        """
        :return: the column of the data node
        """
        _, data_col = self.data_pos
        return data_col

    @property
    def start_pos(self) -> int:
        """
        :return: the starting position of the attacker
        """
        for i in range(self.num_rows):
            for j in range(self.num_cols):
                if self.graph_layout[i][j] == NodeType.START.value:
                    return i, j
        raise AssertionError("Could not find start node in graph layout")

    @property
    def data_pos(self) -> int:
        """
        :return: the position of the data node in the graph
        """
        for i in range(self.num_rows):
            for j in range(self.num_cols):
                if self.graph_layout[i][j] == NodeType.DATA.value:
                    return i, j
        raise AssertionError("Could not find data node in graph layout")


    @property
    def node_list(self) -> List[int]:
        """
        :return: a list of node-types where the index in the list corresponds to the node id.
        """
        node_list = []
        for i in range(self.num_rows):
            for j in range(self.num_cols):
                if self.graph_layout[i][j] != NodeType.EMPTY.value:
                    node_list.append(self.graph_layout[i][j])
        return node_list

    def get_node_pos(self, node_id: int) -> Union[int, int]:
        """
        Utility function for getting the position in the network of a node id.

        :param node_id: the id of the node
        :return: the row,col position in the network

        :raises ValueError when the node id cannot be recognized
        """
        count = 0
        for i in range(self.num_rows):
            for j in range(self.num_cols):
                if node_id == count and self.graph_layout[i][j] != NodeType.EMPTY.value:
                    return (i, j)
                if self.graph_layout[i][j] != NodeType.EMPTY.value:
                    count += 1
        raise ValueError("Invalid node id: {}".format(node_id))

    def get_node_id(self, pos: Union[int, int]) -> int:
        """
        Gets the node-id of a position in the grid

        :param pos: row,col in the grid
        :return: the id of the node in the grid, -1 if the node is empty

        :raises ValueError when the position could not be found
        """
        row, col = pos
        count = 0
        for i in range(self.num_rows):
            for j in range(self.num_cols):
                if row == i and col == j:
                    if self.graph_layout[i][j] == NodeType.EMPTY.value:
                        return -1
                    return count
                if self.graph_layout[i][j] != NodeType.EMPTY.value:
                    count += 1
        raise ValueError("Invalid node position")


    def get_row_ids(self, row):
        ids = []
        count = 0
        for j in range(self.num_cols):
            if self.graph_layout[row][j] != NodeType.EMPTY.value:
                ids.append(count)
                count += 1
        return ids

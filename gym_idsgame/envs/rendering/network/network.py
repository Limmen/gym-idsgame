from typing import Union
from gym_idsgame.envs.rendering.util.render_util import batch_line
from gym_idsgame.envs.dao.idsgame_config import IdsGameConfig
from gym_idsgame.envs.dao.game_state import GameState
from gym_idsgame.envs.dao.node_type import NodeType
from gym_idsgame.envs.rendering.network.nodes.data_node import DataNode
from gym_idsgame.envs.rendering.network.nodes.server_node import ServerNode
from gym_idsgame.envs.rendering.network.nodes.start_node import StartNode
from gym_idsgame.envs.rendering.network.nodes.empty_node import EmptyNode
from gym_idsgame.envs.rendering.network.nodes.node import Node
from gym_idsgame.envs.constants import constants

class Network:
    """
    Class representing the resource network for rendering
    """
    def __init__(self, idsgame_config: IdsGameConfig):
        """
        Class constructor, initializes the network

        :param idsgame_config: config for the IdsGameEnv
        """
        self.idsgame_config = idsgame_config
        self.grid = [[self.__create_node(i, j) for j in range(self.idsgame_config.game_config.num_cols)] for i in
                     range(self.idsgame_config.game_config.num_rows)]

    def create_links(self) -> None:
        """
        Creates links between the nodes in the network according to the game configuration
        :return: None
        """
        root_row, root_col = self.idsgame_config.game_config.network_config.start_pos
        root_edge = self.__root_edge(self.grid[root_row][root_col])
        for i in range(self.idsgame_config.game_config.network_config.adjacency_matrix.shape[0]-1, -1, -1):
            for j in range(i-1, -1,-1):
                if self.idsgame_config.game_config.network_config.adjacency_matrix[i][j] == int(1):
                    row_1, col_1 = self.idsgame_config.game_config.network_config.get_coords(i)
                    n1 = self.grid[row_1][col_1]
                    row_2, col_2 = self.idsgame_config.game_config.network_config.get_coords(j)
                    n2 = self.grid[row_2][col_2]
                    self.__create_link(n1, n2, root_edge)


    def set_node_states(self, game_state: GameState) -> None:
        """
        Updates the node states

        :param game_state: the render state to update the nodes with
        :return: None
        """
        for i in range(self.idsgame_config.game_config.num_rows):
            for j in range(self.idsgame_config.game_config.num_cols):
                node = self.grid[i][j]
                if node.node_type != NodeType.EMPTY:
                    node.set_state(game_state.attack_values[node.id], game_state.defense_values[node.id],
                                              game_state.defense_det[node.id])

    def __create_node(self, row:int, col:int) -> Node:
        """
        Creates a node in the network. Based on the network config it creates either a DATA node, a START node,
        a SERVER node, or an EMPTY node.

        :param row: row in the grid
        :param col: column in the grid
        :return: the created node
        """
        node_id = self.idsgame_config.game_config.network_config.get_node_id((row, col))
        if self.idsgame_config.game_config.network_config.graph_layout[row][col] == NodeType.DATA.value:
            return DataNode(self.idsgame_config, row, col, node_id) # Data node
        elif self.idsgame_config.game_config.network_config.graph_layout[row][col] == NodeType.START.value:
            return StartNode(self.idsgame_config, row, col, node_id) # Start node
        elif self.idsgame_config.game_config.network_config.graph_layout[row][col] == NodeType.SERVER.value:
            return ServerNode(self.idsgame_config, row, col, node_id) # Server node
        else:
            return EmptyNode(self.idsgame_config, row, col, node_id) # Empty node

    def __create_link(self, n1: Node, n2: Node, root_edge) -> None:
        """
        Creates a link in the network between two nodes

        :param n1: node1
        :param n2: node2
        :param root_edge: root edge
        :return: None
        """
        if n1.node_type == NodeType.START:
            assert n2.node_type == NodeType.SERVER
            edges = self.__connect_start_and_server_nodes(n1, n2)
            edges.append(root_edge)
            n1.add_out_edges(edges)
            n2.add_in_edges(edges)
        elif n1.node_type == NodeType.SERVER and n2.node_type == NodeType.SERVER:
            edges = self.__connect_server_and_server_nodes(n1, n2)
            n1.add_out_edges(edges)
            n2.add_in_edges(edges)
        elif n1.node_type == NodeType.SERVER and n2.node_type == NodeType.DATA:
            edges = self.__connect_server_and_data_nodes(n1, n2)
            if n1.col == n2.col:
                pass
            n1.add_out_edges(edges)
            n2.add_in_edges(edges)
        else:
            raise AssertionError("Linktype not recognized")

    def get(self, pos: Union[int, int]) -> Node:
        """
        Gets a node at a given position in the network

        :param pos: the position to get the node from
        :return: the node
        """
        row, col = pos
        return self.grid[row][col]

    def __root_edge(self, n1:Node):
        """
        Creates the "root edge", the edge between the START node and all immediate child nodes.
        This edge is created in a special method because it should be blinking when visualizing all attacks on
        the servers in the layer below the start node

        :param n1: node1
        :return: the created edge (openGL vertex list)
        """
        x1, y1, col1, row1 = n1.get_link_coords(lower=True, upper=False)
        return batch_line(x1, y1 + self.idsgame_config.render_config.rect_size / 6,
                          x1, y1-self.idsgame_config.render_config.rect_size / 6,
        constants.RENDERING.BLACK, self.idsgame_config.render_config.batch, self.idsgame_config.render_config.background,
                          self.idsgame_config.render_config.line_width)

    def __connect_start_and_server_nodes(self, n1:Node, n2:Node) -> list:
        """
        Creates a link between the start node and server nodes on the layer below

        :param n1: node1
        :param n2: node2
        :return: a list of the created links
        """
        x1, y1, col1, row1 = n1.get_link_coords(lower=True, upper=False)
        x2, y2, col2, row2 = n2.get_link_coords(upper=True, lower=False)
        edges = []
        e1 = batch_line(x1, y1, x2, y1, constants.RENDERING.BLACK, self.idsgame_config.render_config.batch,
                        self.idsgame_config.render_config.background, self.idsgame_config.render_config.line_width)
        e2 = batch_line(x2, y1, x2, y2, constants.RENDERING.BLACK, self.idsgame_config.render_config.batch,
                        self.idsgame_config.render_config.background, self.idsgame_config.render_config.line_width)
        edges.append(e1)
        edges.append(e2)
        return edges

    def __connect_server_and_server_nodes(self, n1:Node, n2:Node) -> list:
        """
        Creates a link between two server nodes
        :param n1: node1
        :param n2: node2
        :return: the created link
        """
        x1, y1, col1, row1 = n1.get_link_coords(lower=True, upper=False)
        x2, y2, col2, row2 = n2.get_link_coords(upper=True, lower=False)
        e1 = batch_line(x2, y1, x2, y2, constants.RENDERING.BLACK, self.idsgame_config.render_config.batch,
                        self.idsgame_config.render_config.background, self.idsgame_config.render_config.line_width)
        return [e1]

    def __connect_server_and_data_nodes(self, n1:Node, n2:Node) -> list:
        """
        Creates a link between a server node and the data node

        :param n1: node1
        :param n2: node2
        :return: a list of the created links
        """
        x1, y1, col1, row1 = n1.get_link_coords(upper=False, lower=True)
        x2, y2, col2, row2 = n2.get_link_coords(upper=True, lower=False)
        edges = []
        e1 = batch_line(x1, y1, x1, y2, constants.RENDERING.BLACK, self.idsgame_config.render_config.batch,
                        self.idsgame_config.render_config.background, self.idsgame_config.render_config.line_width)
        e2 = batch_line(x1, y2, x2, y2, constants.RENDERING.BLACK, self.idsgame_config.render_config.batch,
                        self.idsgame_config.render_config.background, self.idsgame_config.render_config.line_width)
        edges.append(e1)
        edges.append(e2)
        if col1 == col2:
            e3 = batch_line(x2, y2, x2, y2-self.idsgame_config.render_config.rect_size/3, constants.RENDERING.BLACK,
                            self.idsgame_config.render_config.batch, self.idsgame_config.render_config.background,
                            self.idsgame_config.render_config.line_width)
            edges.append(e3)
        return edges
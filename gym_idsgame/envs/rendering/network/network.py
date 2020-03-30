"""
Represents the network that the attacker must penetrate and defender should defend in the gym-idsgame environment
"""
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
        self.horizontal_links = []

    def create_links(self) -> None:
        """
        Creates links between the nodes in the network according to the game configuration
        :return: None
        """

        # Create Root Link
        root_row, root_col = self.idsgame_config.game_config.network_config.start_pos
        root_edge = self.__root_edge(self.grid[root_row][root_col])

        # Create Leaf Link
        leaf_edge = self.__leaf_edge(self.grid[0][root_col])

        # Create horizontal links (shared links)
        _, root_y, _, _ = self.grid[root_row][root_col].get_link_coords(lower=True, upper=False)
        top_layer_links = self.__create_horizontal_links(root_y,  self.idsgame_config.game_config.num_rows-2)
        _, data_y, _, _ = self.grid[0][root_col].get_link_coords(lower=False, upper=True)
        bottom_layer_links = self.__create_horizontal_links(data_y, 1)
        self.horizontal_links.append(bottom_layer_links)

        if self.idsgame_config.game_config.network_config.link_between_layers:
            for row in range(1, self.idsgame_config.game_config.num_rows-1):
                y = self.grid[row][0].y + self.idsgame_config.render_config.rect_size/8.5
                links = self.__create_horizontal_links(y, row)
                self.horizontal_links.append(links)

        self.horizontal_links.append(top_layer_links)

        # Add horizontal links to each node (for dynamic visualization later on)

        for col in range(0, self.idsgame_config.game_config.num_cols):
            if col < root_col:
                for i in range(col, root_col):
                    self.grid[self.idsgame_config.game_config.num_rows - 2][col].add_in_edge(top_layer_links[i])
                    self.grid[1][col].add_out_edge(bottom_layer_links[i])
            if col > root_col:
                for i in range(root_col, col):
                    self.grid[self.idsgame_config.game_config.num_rows - 2][col].add_in_edge(top_layer_links[i])
                    self.grid[1][col].add_out_edge(bottom_layer_links[i])


        if self.idsgame_config.game_config.network_config.link_between_layers:
            for row in range(1, self.idsgame_config.game_config.num_rows - 1):
                for col in range(0, self.idsgame_config.game_config.num_cols):
                    if col < root_col:
                        for i in range(col, root_col):
                            self.grid[row][col].add_in_edge(top_layer_links[i])
                            self.grid[row][col].add_out_edge(top_layer_links[i])
                    if col > root_col:
                        for i in range(root_col, col):
                            self.grid[row][col].add_in_edge(top_layer_links[i])
                            self.grid[row][col].add_out_edge(top_layer_links[i])

        # Create Leaf Link and Vertical Links between servers
        for i in range(self.idsgame_config.game_config.network_config.adjacency_matrix.shape[0]-1, -1, -1):
            for j in range(i-1, -1,-1):
                if self.idsgame_config.game_config.network_config.adjacency_matrix[i][j] == int(1):
                    row_1, col_1 = self.idsgame_config.game_config.network_config.get_coords_of_adjacency_matrix_id(i)
                    n1 = self.grid[row_1][col_1]
                    row_2, col_2 = self.idsgame_config.game_config.network_config.get_coords_of_adjacency_matrix_id(j)
                    n2 = self.grid[row_2][col_2]
                    self.__create_link(n1, n2)

        # add leaf edge to servers on last layer
        for j in range(self.idsgame_config.game_config.num_servers_per_layer):
            row = 1
            col = j
            self.grid[row][col].add_out_edge(leaf_edge)

        # add root edge to servers on first layer
        for j in range(self.idsgame_config.game_config.num_servers_per_layer):
            row = self.idsgame_config.game_config.num_rows-2
            col = j
            self.grid[row][col].add_in_edge(root_edge)
        self.grid[root_row][root_col].add_out_edge(root_edge)


    def __create_horizontal_links(self, y_coord: float, row: int) -> list:
        # create horizontal links of first layer that must be shared
        y_coords = []
        x_coords = []
        horizontal_edges = []
        for col in range(self.idsgame_config.game_config.num_servers_per_layer):
            x1, y1, _, _ = self.grid[row][col].get_link_coords(lower=True, upper=False)
            y_coords.append(y_coord)
            x_coords.append(x1)

        for i in range(0, len(x_coords) - 1):
            if i < len(x_coords) - 1:
                horizontal_edge = batch_line(x_coords[i], y_coords[i], x_coords[i + 1], y_coords[i + 1],
                                             constants.RENDERING.BLACK, self.idsgame_config.render_config.batch,
                                             self.idsgame_config.render_config.background,
                                             self.idsgame_config.render_config.line_width)
                horizontal_edges.append(horizontal_edge)
        return horizontal_edges


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

    def __create_node(self, row: int, col: int) -> Node:
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

    def __create_link(self, n1: Node, n2: Node) -> None:
        """
        Creates a link in the network between two nodes

        :param n1: node1
        :param n2: node2
        :return: None
        """
        if n1.node_type == NodeType.START:
            assert n2.node_type == NodeType.SERVER
            edge = self.__connect_start_and_server_nodes(n1, n2)
            n1.add_out_edge(edge)
            n2.add_in_edge(edge)
        elif n1.node_type == NodeType.SERVER and n2.node_type == NodeType.SERVER:
            edge = self.__connect_server_and_server_nodes(n1, n2)
            n1.add_out_edge(edge)
            n2.add_in_edge(edge)
        elif n1.node_type == NodeType.SERVER and n2.node_type == NodeType.DATA:
            edge = self.__connect_server_and_data_nodes(n1, n2)
            n1.add_out_edge(edge)
            n2.add_in_edge(edge)
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

    def __root_edge(self, n1: Node):
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

    def __leaf_edge(self, n1: Node):
        """
        Creates the "leaf edge", the edge between the DATA node and the nodes on the layer above
        This edge is created in a special method because it should be blinking when visualizing all attacks on
        the data node

        :param n1: node1
        :return: the created edge (openGL vertex list)
        """
        leaf_edge = None
        x2, y2, col2, row2 = n1.get_link_coords(upper=True, lower=False)
        leaf_edge = batch_line(x2, y2, x2, y2 - self.idsgame_config.render_config.rect_size / 3,
                               constants.RENDERING.BLACK,
                               self.idsgame_config.render_config.batch, self.idsgame_config.render_config.background,
                               self.idsgame_config.render_config.line_width)
        return leaf_edge

    def __connect_start_and_server_nodes(self, n1: Node, n2: Node):
        """
        Creates a vertical link between the start node and server nodes on the layer below

        :param n1: node1
        :param n2: node2
        :return: a list of the created links
        """
        x1, y1, col1, row1 = n1.get_link_coords(lower=True, upper=False)
        x2, y2, col2, row2 = n2.get_link_coords(upper=True, lower=False)
        e1 = batch_line(x2, y1, x2, y2, constants.RENDERING.BLACK, self.idsgame_config.render_config.batch,
                        self.idsgame_config.render_config.background, self.idsgame_config.render_config.line_width)
        return e1

    def __connect_server_and_server_nodes(self, n1: Node, n2: Node):
        """
        Creates a vertical link between two server nodes
        :param n1: node1
        :param n2: node2
        :return: the created link
        """
        _, y1, _, _ = n1.get_link_coords(lower=True, upper=False)
        x2, y2, col2, row2 = n2.get_link_coords(upper=True, lower=False)
        e1 = batch_line(x2, y1, x2, y2, constants.RENDERING.BLACK, self.idsgame_config.render_config.batch,
                        self.idsgame_config.render_config.background, self.idsgame_config.render_config.line_width)
        return e1

    def __connect_server_and_data_nodes(self, n1: Node, n2: Node):
        """
        Creates a vertical link between a server node and the data node

        :param n1: node1
        :param n2: node2
        :return: the created link
        """
        x1, y1, col1, row1 = n1.get_link_coords(upper=False, lower=True)
        x2, y2, col2, row2 = n2.get_link_coords(upper=True, lower=False)
        e1 = batch_line(x1, y1, x1, y2, constants.RENDERING.BLACK, self.idsgame_config.render_config.batch,
                        self.idsgame_config.render_config.background, self.idsgame_config.render_config.line_width)
        return e1
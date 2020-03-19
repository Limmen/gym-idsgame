from gym_idsgame.envs.rendering.util.render_util import batch_line
from gym_idsgame.envs.dao.render_config import RenderConfig
from gym_idsgame.envs.dao.node_type import NodeType
from gym_idsgame.envs.rendering.network.nodes.data_node import DataNode
from gym_idsgame.envs.rendering.network.nodes.server_node import ServerNode
from gym_idsgame.envs.rendering.network.nodes.start_node import StartNode
from gym_idsgame.envs.rendering.network.nodes.empty_node import EmptyNode
from gym_idsgame.envs.rendering.network.nodes.node import Node
from gym_idsgame.envs.constants import constants

class Network:
    """
    Class representing the resource network in the rendering
    """
    def __init__(self, render_config: RenderConfig):
        self.render_config = render_config
        self.grid = [[self.__create_node(i, j) for j in range(self.render_config.game_config.num_cols)] for i in
                     range(self.render_config.game_config.num_rows)]

    def create_links(self):
        for i in range(self.render_config.game_config.network_config.adjacency_matrix.shape[0]-1, -1, -1):
            for j in range(i-1, -1,-1):
                if self.render_config.game_config.network_config.adjacency_matrix[i][j] == int(1):
                    row_1, col_1 = self.render_config.game_config.network_config.get_coords(i)
                    n1 = self.grid[row_1][col_1]
                    row_2, col_2 = self.render_config.game_config.network_config.get_coords(j)
                    n2 = self.grid[row_2][col_2]
                    self.__create_link(n1, n2)

    def set_node_states(self, render_state):
        for i in range(self.render_config.game_config.num_rows):
            for j in range(self.render_config.game_config.num_cols):
                self.grid[i][j].set_state(render_state.attack_values[i][j], render_state.defense_values[i][j],
                    render_state.defense_det[i][j])

    def is_attack_legal(self, attacker_pos, target_pos):
        attacker_row, attacker_col = attacker_pos
        target_row, target_col = target_pos
        attacker_id = attacker_row*self.render_config.game_config.num_cols + attacker_col
        target_id = target_row* self.render_config.game_config.num_cols + target_col
        return self.render_config.game_config.network_config.adjacency_matrix[attacker_id][target_id] == int(1)

    def __create_node(self, i, j) -> Node:
        if self.render_config.game_config.network_config.graph_layout[i][j] == NodeType.DATA.value:
            return DataNode(self.render_config, i, j) # Data node
        elif self.render_config.game_config.network_config.graph_layout[i][j] == NodeType.START.value:
            return StartNode(self.render_config, i, j) # Start node
        elif self.render_config.game_config.network_config.graph_layout[i][j] == NodeType.SERVER.value:
            return ServerNode(self.render_config, i, j) # Server node
        else:
            return EmptyNode(self.render_config, i, j) # Empty node

    def __create_link(self, n1: Node, n2: Node):
        if n1.node_type == NodeType.START:
            assert n2.node_type == NodeType.SERVER
            edges = self.__connect_start_and_server_nodes(n1, n2)
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

    def get(self, pos) -> Node:
        row, col = pos
        return self.grid[row][col]

    def __root_edge(self, n1, n2):
        x1, y1, col1, row1 = n1.get_link_coords(lower=True, upper=False)
        x2, y2, col2, row2 = n2.get_link_coords(upper=True, lower=False)
        return batch_line(x1, y1 + self.render_config.rect_size / 6, x2, y2, constants.RENDERING.BLACK,
                          self.render_config.batch, self.render_config.background, self.render_config.line_width)

    def __connect_start_and_server_nodes(self, n1, n2):
        x1, y1, col1, row1 = n1.get_link_coords(lower=True, upper=False)
        x2, y2, col2, row2 = n2.get_link_coords(upper=True, lower=False)
        edges = []
        e1 = batch_line(x1, y1, x2, y1, constants.RENDERING.BLACK, self.render_config.batch,
                        self.render_config.background, self.render_config.line_width)
        e2 = batch_line(x2, y1, x2, y2, constants.RENDERING.BLACK, self.render_config.batch,
                        self.render_config.background, self.render_config.line_width)
        edges.append(e1)
        edges.append(e2)
        return edges

    def __connect_server_and_server_nodes(self, n1, n2):
        x1, y1, col1, row1 = n1.get_link_coords(lower=True, upper=False)
        x2, y2, col2, row2 = n2.get_link_coords(upper=True, lower=False)
        e1 = batch_line(x2, y1, x2, y2, constants.RENDERING.BLACK, self.render_config.batch,
                        self.render_config.background, self.render_config.line_width)
        return [e1]

    def __connect_server_and_data_nodes(self, n1, n2):
        x1, y1, col1, row1 = n1.get_link_coords(upper=False, lower=True)
        x2, y2, col2, row2 = n2.get_link_coords(upper=True, lower=False)
        edges = []
        e1 = batch_line(x1, y1, x1, y2, constants.RENDERING.BLACK, self.render_config.batch,
                        self.render_config.background, self.render_config.line_width)
        e2 = batch_line(x1, y2, x2, y2, constants.RENDERING.BLACK, self.render_config.batch,
                        self.render_config.background, self.render_config.line_width)
        edges.append(e1)
        edges.append(e2)
        if col1 == col2:
            e3 = batch_line(x2, y2, x2, y2-self.render_config.rect_size/3, constants.RENDERING.BLACK,
                            self.render_config.batch, self.render_config.background, self.render_config.line_width)
            edges.append(e3)
        return edges
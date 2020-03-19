from gym_idsgame.envs.rendering.network.nodes.node import Node
from gym_idsgame.envs.dao.render_config import RenderConfig
from gym_idsgame.envs.dao.node_type import NodeType

class EmptyNode(Node):
    """
    Represents an empty node in the grid-network
    """

    def __init__(self, render_config: RenderConfig, row: int, col: int):
        """
        Class constructor, initializes the node

        :param render_config: the configuration for rendering, e.g. the font-size etc.
        :param row: the row in the grid
        :param col: the column in the grid
        """
        super(EmptyNode, self).__init__()
        self.render_config = render_config
        self.row = row
        self.col = col

    @property
    def node_type(self) -> NodeType:
        """
        :return: the node type
        """
        return NodeType.EMPTY

    # --- Inherited methods----
    # empty nodes cannot be attacked or defended so simply do nothing when they are called

    def manual_blink_defense(self, i):
        pass

    def manual_blink_attack(self, i, edges=None):
        pass

    def set_state(self, attack_values, defense_values, det_value):
        pass

    def defend(self, defense_type):
        pass

    def reset(self):
        pass

    def add_in_edges(self, edges):
        pass

    def add_out_edges(self, edges):
        pass

    def get_link_coords(self, upper=True, lower=False):
        pass

    def get_coords(self):
        pass

    def get_node(self):
        pass

    def unschedule(self):
        pass

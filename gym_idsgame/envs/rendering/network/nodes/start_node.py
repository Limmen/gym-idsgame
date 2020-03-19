from typing import Union
from gym_idsgame.envs.rendering.network.nodes.node import Node
from gym_idsgame.envs.rendering.util.render_util import create_circle
from gym_idsgame.envs.dao.render_config import RenderConfig
from gym_idsgame.envs.constants import constants
from gym_idsgame.envs.dao.node_type import NodeType

class StartNode(Node):
    """
    Represents the start node the grid network
    """
    def __init__(self, render_config: RenderConfig, row: int, col: int):
        """
        Initializes the node

        :param render_config: the render config, e.g scale of the node, color, etc.
        :param row: the row in the grid network
        :param col: the column in the grid network
        """
        super(StartNode, self).__init__()
        self.render_config = render_config
        self.row = row
        self.col = col
        self.x = self.col * self.render_config.rect_size + self.render_config.rect_size / 2
        self.y = self.row * int(self.render_config.rect_size / 1.5) + (self.render_config.rect_size / 1.5) / 2
        self.radius = self.render_config.rect_size / 7
        self.__draw()

    @property
    def node_type(self) -> NodeType:
        """
        :return: the node type (START)
        """
        return NodeType.START

    def __draw(self) -> None:
        """
        Draws the node (a black circle)
        :return: None
        """
        create_circle(self.x, self.y, self.radius, self.render_config.batch, self.render_config.first_foreground,
                      constants.RENDERING.BLACK)

    def get_link_coords(self, upper: bool = True, lower: bool = False) -> Union[float, float, int, int]:
        """
        Gets the coordinates of link endpoints of the node

        :param upper: if True, returns the upper endpoint
        :param lower: if False, returns the lower endpoint
        :return: (x-coordinate, y-coordinate, grid-column, grid-row)
        """
        x = self.col * self.render_config.rect_size + self.render_config.rect_size / 2
        y = (self.row + 1) * (self.render_config.rect_size / 1.5) - self.render_config.rect_size / 1.75
        return x, y, self.col, self.row

    # --- Inherited methods----
    # the start node cannot be attacked or defended so simply do nothing when they are called

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

    def unschedule(self):
        pass

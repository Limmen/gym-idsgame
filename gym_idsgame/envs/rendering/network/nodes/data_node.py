"""
Represents a data node in the network of the gym-idsgame environment
"""
from typing import Union
import pyglet
from pyglet import clock
from gym_idsgame.envs.constants import constants
from gym_idsgame.envs.rendering.network.nodes.resource_node import ResourceNode
from gym_idsgame.envs.dao.idsgame_config import IdsGameConfig
from gym_idsgame.envs.dao.node_type import NodeType

class DataNode(ResourceNode):
    """
    Class representing a Data node in the game network
    """
    def __init__(self, idsgame_config: IdsGameConfig, row: int, col: int, id: int):
        """
        Constructor, Initializes the node

        :param idsgame_config: config for IdsGameEnv
        :param row: the row in the grid of the node
        :param col: the column in the grid of the node
        :param id: the id of the node
        """
        avatar = pyglet.resource.image(idsgame_config.render_config.data_filename)
        super(DataNode, self).__init__(avatar, idsgame_config, idsgame_config.render_config.background)
        self.col = col
        self.row = row
        self._id = id
        self.scale = idsgame_config.render_config.data_scale
        self.reset()

    @property
    def node_type(self) -> NodeType:
        """
        The node type
        """
        return NodeType.DATA

    @property
    def id(self) -> int:
        """
        :return: the id of the node
        """
        return self._id

    def init_labels(self) -> None:
        """
        Initializes labels of the node

        :return: Noneg
        """
        attack_label_x = self.col * self.idsgame_config.render_config.rect_size + \
                         self.idsgame_config.render_config.rect_size / 2
        attack_label_y = self.row * int((self.idsgame_config.render_config.rect_size) / 1.5) + \
                         self.idsgame_config.render_config.rect_size / 4
        defense_label_x = self.col * self.idsgame_config.render_config.rect_size + \
                          self.idsgame_config.render_config.rect_size / 2
        defense_label_y = self.row * int((self.idsgame_config.render_config.rect_size) / 1.5) + \
                          self.idsgame_config.render_config.rect_size / 7
        det_label_x = self.col * self.idsgame_config.render_config.rect_size + \
                      self.idsgame_config.render_config.rect_size / 3.5
        det_label_y = self.row * int((self.idsgame_config.render_config.rect_size) / 1.5) + \
                      self.idsgame_config.render_config.rect_size / 3
        self.create_labels(attack_label_x=attack_label_x, attack_label_y=attack_label_y,
                           defense_label_x=defense_label_x, defense_label_y=defense_label_y,
                           det_label_x=det_label_x, det_label_y=det_label_y)

    def visualize_attack(self, attack_type: int, edges_list: list = None) -> None:
        """
        Simulates an attack against the node.

        :param attack_type: the type of the attack
        :param edges_list: edges list for visualization (blinking)
        :return: None
        """
        for i in range(0, self.idsgame_config.render_config.num_blinks):
            if i % 2 == 0:
                clock.schedule_once(self.blink_red_attack,
                                    self.idsgame_config.render_config.blink_interval * i, edges_list)
            else:
                clock.schedule_once(self.blink_black_attack,
                                    self.idsgame_config.render_config.blink_interval * i, edges_list)

    def blink_red_attack(self, dt, edges_list: list) -> None:
        """
        Makes the node and its links blink red to visualize an attack

        :param dt: the time since the last scheduled blink
        :param edges_list: list of edges to blink
        :return: None
        """
        color = constants.RENDERING.RED
        color_list = list(color) + list(color)
        for edges in edges_list:
            for e1 in edges:
                e1.colors = color_list
        lbl_color = constants.RENDERING.RED_ALPHA
        self.attack_label.color = lbl_color
        self.color = constants.RENDERING.RED

    def blink_black_attack(self, dt, edges_list: list) -> None:
        """
        Makes the node and its links blink black to visualize an attack

        :param dt: the time since the last scheduled blink
        :param edges_list: list of edges to blink
        :return: None
        """
        color = constants.RENDERING.BLACK
        color_list = list(color) + list(color)
        for edges in edges_list:
            for e1 in edges:
                e1.colors = color_list
        lbl_color = constants.RENDERING.BLACK_ALPHA
        self.attack_label.color = lbl_color
        self.color = constants.RENDERING.WHITE

    def center_avatar(self) -> None:
        """
        Utility function for centering the avatar inside a cell

        :return: None
        """
        self.x = self.col*self.idsgame_config.render_config.rect_size + \
                 self.idsgame_config.render_config.rect_size/2.5
        self.y = int((self.idsgame_config.render_config.rect_size/1.5))*self.row + \
                 self.idsgame_config.render_config.rect_size/3.5

    def get_link_coords(self, upper: bool = True, lower: bool = False) -> Union[float, float, int, int]:
        """
        Gets the coordinates of link endpoints of the node

        :param upper: if True, returns the upper endpoint
        :param lower: if False, returns the lower endpoint
        :return: (x-coordinate, y-coordinate, grid-column, grid-row)
        """
        assert not (upper and lower)
        x = self.col * self.idsgame_config.render_config.rect_size + \
            self.idsgame_config.render_config.rect_size / 2
        y = (self.row + 1) * (self.idsgame_config.render_config.rect_size / 1.5)
        return x, y, self.col, self.row


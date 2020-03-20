import pyglet
from abc import ABC, abstractmethod
import numpy as np
from pyglet import clock
from gym_idsgame.envs.rendering.network.nodes.node import Node
from gym_idsgame.envs.constants import constants
from gym_idsgame.envs.dao.idsgame_config import IdsGameConfig
from gym_idsgame.envs.rendering.util.render_util import batch_label

class ResourceNode(pyglet.sprite.Sprite, Node, ABC):
    """
    Abstract resource in the grid network, representing either a server or a data node

    Subclasses pyglet.sprite.Sprite to be able to override draw() and update() methods
    and define state of the sprite
    """

    def __init__(self, avatar, idsgame_config: IdsGameConfig, group):
        """
        Class constructor, initializes the resource node

        :param avatar: the avatar of the node
        :param idsgame_config: configuration for the IdsGameEnv
        :param group: the group to render the resource in (background or foreground)
        """
        self.idsgame_config = idsgame_config
        super(ResourceNode, self).__init__(avatar, batch=idsgame_config.render_config.batch, group=group)
        self.outgoing_edges = []
        self.incoming_edges = []
        self.initialize_state()

    def visualize_defense(self, defend_type: int) -> None:
        """
        Simulates defense of the node

        :param defend_type: the type of defense
        :return: None
        """
        for i in range(0, self.idsgame_config.render_config.num_blinks):
            if i % 2 == 0:
                clock.schedule_once(self.blink_green_defense, self.idsgame_config.render_config.blink_interval * i)
            else:
                clock.schedule_once(self.blink_black_defense, self.idsgame_config.render_config.blink_interval * i)

    def set_labels(self) -> None:
        """
        Updates the labels of the node

        :return: None
        """
        self.attack_label.text = self.attack_text
        self.defense_label.text = self.defense_text
        self.det_label.text = self.det_text

    @property
    def attack_text(self) -> str:
        """
        :return: the attack text of the node
        """
        return "A=" + ",".join(map(lambda x: str(x), self.attack_values))

    @property
    def defense_text(self) -> str:
        """
        :return: the defense text of the node
        """
        return "D=" + ",".join(map(lambda x: str(x), self.defense_values))

    @property
    def det_text(self) -> str:
        """
        :return: the detection text of the node
        """
        return "Det=" + str(self.det)

    def initialize_state(self) -> None:
        """
        initializes the state of the node
        :return: None
        """
        self.attack_values = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.defense_values = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
        self.det = 2

    def blink_green_defense(self, dt) -> None:
        """
        Visualizes a defense action by blinking green

        :param dt: the time since the last blink
        :return: None
        """
        color = constants.RENDERING.GREEN_ALPHA
        self.defense_label.color = color
        self.color = constants.RENDERING.GREEN

    def blink_black_defense(self, dt) -> None:
        """
        Visualizes a defense action by blinking black

        :param dt: time since the last blink
        :return: None
        """
        color = constants.RENDERING.BLACK_ALPHA
        self.defense_label.color = color
        self.color = constants.RENDERING.WHITE

    def unschedule(self) -> None:
        """
        Unschedules all event of this node (e.g blink events)
        :return:
        """
        clock.unschedule(self.blink_green_defense)
        clock.unschedule(self.blink_red_attack)

    def set_state(self, attack_values, defense_values, det_value) -> None:
        """
        Sets the state of the node

        :param attack_values: attack values
        :param defense_values: defense values
        :param det_value: detection probabilities
        :return: None
        """
        self.attack_values = attack_values
        self.defense_values = defense_values
        self.det = det_value
        self.set_labels()

    def manual_blink_defense(self, i: int) -> None:
        """
        Manual defense blink, when not using the clock to schedule blinks but rather ticking the clock manually.
        Used when the agent plays the game and not a human.

        :param i: the blink number
        :return: None
        """
        if i % 2 == 0:
            self.blink_green_defense(0)
        else:
            self.blink_black_defense(0)

    def manual_blink_attack(self, i:int, edges:list=None) -> None:
        """
        Manual attack blink, when not using the clock to schedule blinks but rather ticking the clock manually.
        Used when the agent plays the game and not a human.

        :param i: the blink number
        :return: None
       """
        if i % 2 == 0:
            self.blink_red_attack(0, edges_list=edges)
        else:
            self.blink_black_attack(0, edges_list=edges)

    def create_labels(self, attack_label_x: int, attack_label_y: int, defense_label_x: int, defense_label_y: int,
                      det_label_x: int, det_label_y: int) -> None:
        """
        Creates the labels of the node

        :param attack_label_x: the x coordinate of the attack label
        :param attack_label_y: the y coordinate of the attack label
        :param defense_label_x: the x coordinate of the defense label
        :param defense_label_y: the y coordinate of the defense label
        :param det_label_x: the x coordinate of the detection label
        :param det_label_y: the y coordinate of the detection label
        :return: None
        """
        self.attack_label = batch_label(self.attack_text, attack_label_x, attack_label_y,
                                        constants.RENDERING.NODE_STATE_FONT_SIZE, constants.RENDERING.BLACK_ALPHA,
                                        self.idsgame_config.render_config.batch,
                                        self.idsgame_config.render_config.background, multiline=False,
                                        width=self.idsgame_config.render_config.rect_size)
        self.defense_label = batch_label(self.defense_text, defense_label_x, defense_label_y,
                                         constants.RENDERING.NODE_STATE_FONT_SIZE, constants.RENDERING.BLACK_ALPHA,
                                         self.idsgame_config.render_config.batch,
                                         self.idsgame_config.render_config.background, multiline=False,
                                         width=self.idsgame_config.render_config.rect_size)
        self.det_label = batch_label(self.det_text, det_label_x, det_label_y,
                                     constants.RENDERING.NODE_STATE_FONT_SIZE, constants.RENDERING.BLACK_ALPHA,
                                     self.idsgame_config.render_config.batch,
                                     self.idsgame_config.render_config.background, multiline=False,
                                     width=self.idsgame_config.render_config.rect_size)

    def add_out_edges(self, edges: list) -> None:
        """
        Adds a list of outgoing edge from the node

        :param edges: edges to add
        :return: None
        """
        self.outgoing_edges.append(edges)

    def add_in_edges(self, edges: list) -> None:
        """
        Adds a list of ingoing edges to the node

        :param edges: edges to add
        :return: None
        """
        self.incoming_edges.append(edges)

    def reset(self) -> None:
        """
        Resets the node, centers the image, resets the state and the labels.
        :return: None
        """
        self.center_avatar()
        self.initialize_state()
        self.init_labels()


    # Abstract methods to be implemented by sub-classes
    @abstractmethod
    def center_avatar(self):
        pass

    @abstractmethod
    def visualize_attack(self, attack_type, edges_list=None):
        pass

    @abstractmethod
    def blink_red_attack(self, dt, edges_list=None):
        pass

    @abstractmethod
    def blink_black_attack(self, dt, edges_list=None):
        pass
from gym_idsgame.envs.rendering.network.nodes.node import Node
import pyglet
from gym_idsgame.envs.constants import constants
from gym_idsgame.envs.dao.render_config import RenderConfig
from gym_idsgame.envs.rendering.util.render_util import batch_label
from abc import ABC, abstractmethod
import numpy as np
from pyglet import clock

class ResourceNode(pyglet.sprite.Sprite, Node, ABC):
    """

    """

    def __init__(self, avatar, render_config: RenderConfig, group):
        self.render_config = render_config
        super(ResourceNode, self).__init__(avatar, batch=render_config.batch, group=group)

    @abstractmethod
    def center_avatar(self):
        pass

    @abstractmethod
    def create_labels(self):
        pass

    @abstractmethod
    def simulate_attack(self, attack_type, edges_list=None):
        pass

    @abstractmethod
    def attack_red(self, dt, edges_list=None):
        pass

    @abstractmethod
    def attack_black(self, dt, edges_list=None):
        pass

    def reset(self):
        self.initialize_state()
        self.set_labels()

    def defend(self, defend_type):
        if self.defense_values[defend_type] < self.max_value - 1:
            self.defense_values[defend_type] += 1
        self.defense_label.text = self.get_defense_text()
        for i in range(0, self.num_blinks):
            if i % 2 == 0:
                clock.schedule_once(self.defense_green, self.blink_interval * i)
            else:
                clock.schedule_once(self.defense_black, self.blink_interval * i)

    def simulate_detection(self):
        if np.random.rand() < self.det / 10:
            return True
        else:
            return False

    def set_labels(self):
        self.attack_label.text = self.get_attack_text()
        self.defense_label.text = self.get_defense_text()
        self.det_label.text = self.get_det_text()

    def get_attack_text(self):
        return "A=" + ",".join(map(lambda x: str(x), self.attack_values))

    def get_defense_text(self):
        return "D=" + ",".join(map(lambda x: str(x), self.defense_values))

    def get_det_text(self):
        return "Det=" + str(self.det)

    def initialize_state(self):
        self.attack_values = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.defense_values = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
        self.det = 2

    def defense_green(self, dt):
        color = constants.GAMEFRAME.GREEN_ALPHA
        self.defense_label.color = color
        self.color = constants.GAMEFRAME.GREEN

    def defense_black(self, dt):
        color = constants.GAMEFRAME.BLACK_ALPHA
        self.defense_label.color = color
        self.color = constants.GAMEFRAME.WHITE

    def unschedule(self):
        clock.unschedule(self.defense_green)
        clock.unschedule(self.attack_red)

    def set_state(self, attack_values, defense_values, det_value):
        self.attack_values = attack_values
        self.defense_values = defense_values
        self.det = det_value
        self.set_labels()

    def manual_blink_defense(self, i):
        if i % 2 == 0:
            self.defense_green(0)
        else:
            self.defense_black(0)

    def manual_blink_attack(self, i):
        if i % 2 == 0:
            self.attack_red(0)
        else:
            self.attack_black(0)

    def create_labels(self, attack_label_x: int, attack_label_y: int, defense_label_x: int, defense_label_y: int,
                      det_label_x: int, det_label_y: int) -> None:
        self.attack_label = batch_label(self.get_attack_text(), attack_label_x, attack_label_y,
                                        constants.GAMEFRAME.NODE_STATE_FONT_SIZE, constants.GAMEFRAME.BLACK_ALPHA,
                                        self.render_config.batch,
                                        self.render_config.background, multiline=False,
                                        width=self.render_config.rect_size)
        self.defense_label = batch_label(self.get_defense_text(), defense_label_x, defense_label_y,
                                         constants.GAMEFRAME.NODE_STATE_FONT_SIZE, constants.GAMEFRAME.BLACK_ALPHA,
                                         self.render_config.batch,
                                         self.render_config.background, multiline=False,
                                         width=self.render_config.rect_size)
        self.det_label = batch_label(self.get_det_text(), det_label_x, det_label_y,
                                     constants.GAMEFRAME.NODE_STATE_FONT_SIZE, constants.GAMEFRAME.BLACK_ALPHA,
                                     self.render_config.batch, self.render_config.background, multiline=False,
                                     width=self.render_config.rect_size)
from gym_idsgame.envs.rendering.network.nodes.node import Node
from gym_idsgame.envs.rendering.util.render_util import create_circle
from gym_idsgame.envs.dao.render_config import RenderConfig
from gym_idsgame.envs.constants import constants

class StartNode(Node):

    def __init__(self, render_config: RenderConfig, row: int, col: int):
        super(StartNode, self).__init__()
        self.render_config = render_config
        self.row = row
        self.col = col
        self.__draw()

    def __draw(self):
        create_circle(self.col * self.render_config.rect_size + self.render_config.rect_size / 2,
                      self.row * int(self.render_config.rect_size / 1.5) + (self.render_config.rect_size / 1.5) / 2,
                      self.render_config.rect_size / 7, self.render_config.batch, self.render_config.first_foreground,
                      constants.GAMEFRAME.BLACK)

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

    def add_in_edge(self, edges):
        pass

    def add_out_edge(self, edges):
        pass

    def get_link_coords(self, upper=True, lower=False):
        pass

    def get_coords(self):
        pass

    def get_node(self):
        pass

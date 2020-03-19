from gym_idsgame.envs.rendering.network.nodes.node import Node
from gym_idsgame.envs.rendering.util.render_util import create_circle
from gym_idsgame.envs.dao.render_config import RenderConfig
from gym_idsgame.envs.constants import constants
from gym_idsgame.envs.dao.node_type import NodeType

class StartNode(Node):

    def __init__(self, render_config: RenderConfig, row: int, col: int):
        super(StartNode, self).__init__()
        self.render_config = render_config
        self.row = row
        self.col = col
        self.x = self.col * self.render_config.rect_size + self.render_config.rect_size / 2
        self.y = self.row * int(self.render_config.rect_size / 1.5) + (self.render_config.rect_size / 1.5) / 2
        self.radius = self.render_config.rect_size / 7
        self.__draw()

    @property
    def node_type(self):
        return NodeType.START

    def __draw(self):
        create_circle(self.x, self.y, self.radius, self.render_config.batch, self.render_config.first_foreground,
                      constants.RENDERING.BLACK)

    def manual_blink_defense(self, i):
        raise NotImplementedError("Cannot defend the start node")

    def manual_blink_attack(self, i, edges=None):
        raise NotImplementedError("Cannot attack the start node")

    def set_state(self, attack_values, defense_values, det_value):
        return

    def defend(self, defense_type):
        raise NotImplementedError("Cannot defend the start node")

    def reset(self):
        return

    def add_in_edges(self, edges):
        return

    def add_out_edges(self, edges):
        return

    def unschedule(self):
        pass

    def get_link_coords(self, upper=True, lower=False):
        x = self.col * self.render_config.rect_size + self.render_config.rect_size / 2
        y = (self.row + 1) * (self.render_config.rect_size / 1.5) - self.render_config.rect_size / 1.75
        return x,y,self.col,self.row

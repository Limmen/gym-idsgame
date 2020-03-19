import pyglet
from gym_idsgame.envs.constants import constants
from pyglet import clock
from gym_idsgame.envs.rendering.network.nodes.resource_node import ResourceNode
from gym_idsgame.envs.dao.render_config import RenderConfig
from gym_idsgame.envs.dao.node_type import NodeType

class DataNode(ResourceNode):
    def __init__(self, render_config: RenderConfig, row: int, col: int):
        avatar = pyglet.resource.image(render_config.data_filename)
        super(DataNode, self).__init__(avatar, render_config, render_config.background)
        self.node_type = NodeType.DATA
        self.col = col
        self.row = row
        self.scale = render_config.data_scale
        self.center_avatar()
        self.initialize_state()
        self.init_labels()

    def init_labels(self):
        attack_label_x = self.col * self.render_config.rect_size + self.render_config.rect_size / 2
        attack_label_y = self.row * int((self.render_config.rect_size) / 1.5) + self.render_config.rect_size / 4
        defense_label_x = self.col * self.render_config.rect_size + self.render_config.rect_size / 2
        defense_label_y = self.row * int((self.render_config.rect_size) / 1.5) + self.render_config.rect_size / 7
        det_label_x = self.col * self.render_config.rect_size + self.render_config.rect_size / 3.5
        det_label_y = self.row * int((self.render_config.rect_size) / 1.5) + self.render_config.rect_size / 3
        self.create_labels(attack_label_x=attack_label_x, attack_label_y=attack_label_y,
                           defense_label_x=defense_label_x, defense_label_y=defense_label_y,
                           det_label_x=det_label_x, det_label_y=det_label_y)

    def simulate_attack(self, attack_type, edges_list=None):
        for i in range(0, self.num_blinks):
            if i % 2 == 0:
                clock.schedule_once(self.attack_red, self.render_config.blink_interval * i, edges_list)
            else:
                clock.schedule_once(self.attack_black, self.render_config.blink_interval * i, edges_list)
        if self.attack_values[attack_type] < self.max_value-1:
            self.attack_values[attack_type] += 1
        self.attack_label.text = self.get_attack_text()
        if self.attack_values[attack_type] > self.defense_values[attack_type]:
            return True  # successful attack
        else:
            return False

    def attack_red(self, dt, edges_list) -> None:
        color = constants.RENDERING.RED
        color_list = list(color) + list(color)
        for edges in edges_list:
            for e1 in edges:
                e1.colors = color_list
        lbl_color = constants.RENDERING.RED_ALPHA
        self.attack_label.color = lbl_color
        self.color = constants.RENDERING.RED

    def attack_black(self, dt, edges_list) -> None:
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
        :return: The centered coordinates in the grid
        """
        self.x = self.col*self.render_config.rect_size + self.render_config.rect_size/2.5
        self.y = int((self.render_config.rect_size/1.5))*self.row + self.render_config.rect_size/3.5

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


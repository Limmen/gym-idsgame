from gym_idsgame.envs.rendering.util.render_util import batch_label
from gym_idsgame.envs.rendering.constants import constants
from pyglet import clock
from gym_idsgame.envs.rendering.network.resource_node import ResourceNode
from gym_idsgame.envs.dao.render_config import RenderConfig
import pyglet

class Server(ResourceNode):

    def __init__(self, render_config: RenderConfig, row: int, col: int):
        avatar = pyglet.resource.image(render_config.server_filename)
        super(Server, self).__init__(avatar, batch=render_config.batch,
                                   group=render_config.background)
        self.col = col
        self.row = row
        self.render_config = render_config
        self.scale = render_config.server_scale
        self.center_avatar()
        self.initialize_state()
        self.create_labels()

    def create_labels(self):
        lbl_color = constants.GAMEFRAME.BLACK_ALPHA
        lbl = self.get_attack_text()
        self.attack_label = batch_label(lbl, self.col * self.render_config.rect_size + self.render_config.rect_size / 2,
                                        self.row * int((self.render_config.rect_size) / 1.5)
                                        + self.render_config.rect_size / 4,
                                        constants.GAMEFRAME.NODE_STATE_FONT_SIZE, lbl_color, self.render_config.batch,
                                        self.render_config.background, multiline=False,
                                        width=self.render_config.rect_size)
        lbl = self.get_defense_text()
        self.defense_label = batch_label(lbl, self.col * self.render_config.rect_size + self.render_config.rect_size / 2,
                                         self.row * int((self.render_config.rect_size) / 1.5)
                                         + self.render_config.rect_size / 7,
                                         constants.GAMEFRAME.NODE_STATE_FONT_SIZE, lbl_color, self.render_config.batch,
                                         self.render_config.background, multiline=False,
                                         width=self.render_config.rect_size)
        lbl = self.get_det_text()
        self.det_label = batch_label(lbl, self.col * self.render_config.rect_size + self.render_config.rect_size / 3,
                                     self.row * int((self.render_config.rect_size) / 1.5)
                                     + self.render_config.rect_size / 3,
                                     constants.GAMEFRAME.NODE_STATE_FONT_SIZE, lbl_color, self.render_config.batch,
                                     self.render_config.background, multiline=False,
                                     width=self.render_config.rect_size)

    def simulate_attack(self, attack_type, edges_list=None):
        for i in range(0, self.num_blinks):
            if i % 2 == 0:
                clock.schedule_once(self.attack_red, self.render_config.blink_interval * i)
            else:
                clock.schedule_once(self.attack_black, self.render_config.blink_interval * i)
        if self.attack_values[attack_type] < self.render_config.game_config.max_value-1:
            self.attack_values[attack_type] += 1
        self.attack_label.text = self.get_attack_text()
        if self.attack_values[attack_type] > self.defense_values[attack_type]:
            return True # successful attack
        else:
            return False

    def attack_red(self, dt, edges_list=None):
        color = constants.GAMEFRAME.RED
        color_list = list(color) + list(color)
        for edges in self.incoming_edges:
            for e1 in edges:
                e1.colors = color_list
        lbl_color = constants.GAMEFRAME.RED_ALPHA
        self.attack_label.color = lbl_color
        self.color = constants.GAMEFRAME.RED

    def attack_black(self, dt, edges_list=None):
        color = constants.GAMEFRAME.BLACK
        color_list = list(color) + list(color)
        for edges in self.incoming_edges:
            for e1 in edges:
                e1.colors = color_list
        lbl_color = constants.GAMEFRAME.BLACK_ALPHA
        self.attack_label.color = lbl_color
        self.color = constants.GAMEFRAME.WHITE

    def center_avatar(self):
        """
        Utiltiy function for centering the avatar inside a cell
        :return: The centered coordinates in the grid
        """
        self.x = self.col*self.render_config.rect_size + self.render_config.rect_size/2.3
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

    def draw(self, y, x, color, batch, background, foreground, avatar, scale, server=False, data=False,
             start=False, max_value=10, blink_interval=constants.GAMEFRAME.MANUAL_BLINK_INTERVAL,
             num_blinks=constants.GAMEFRAME.MANUAL_NUM_BLINKS):
        pass

    def get_link_coords(self, upper=True, lower=False):
        pass

    def get_coords(self):
        pass

    def get_node(self):
        pass
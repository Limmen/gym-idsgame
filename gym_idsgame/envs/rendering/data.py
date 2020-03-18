import pyglet
from gym_idsgame.envs.rendering.render_util import batch_label, batch_rect_fill, batch_rect_border, batch_circle, create_circle
from gym_idsgame.envs.rendering import constants
from pyglet import clock
import numpy as np

class Data(pyglet.sprite.Sprite):
    """

    TODO

    Subclasses pyglet.sprite.Sprite to be able to override draw() and update() methods
    and define state of the sprite
    """
    def __init__(self, avatar_path, col, row, batch, background, foreground, size, scale=0.25,
                 policy = constants.BASELINE_POLICIES.NAIVE_DETERMINISTIC, max_value = 10,
                 blink_interval = constants.GAMEFRAME.BLINK_INTERVAL, num_blinks = constants.GAMEFRAME.NUM_BLINKS):
        self.avatar = pyglet.resource.image(avatar_path)
        self.background = background
        self.foreground = foreground
        self.max_value = max_value
        super(Data, self).__init__(self.avatar, batch=batch, group=background)
        self.col = col
        self.row = row
        self.size = size
        self.__center_avatar()
        self.scale = scale
        self.batch = batch
        self.initialize_state()
        self.create_labels()
        self.incoming_edges = []
        self.outgoing_edges = []
        self.cumulative_reward = 0
        self.policy = policy
        self.blink_interval = blink_interval
        self.num_blinks = num_blinks

    def create_labels(self):
        lbl_color = constants.GAMEFRAME.BLACK_ALPHA
        lbl = self.get_attack_text()
        self.attack_label = batch_label(lbl, self.col * self.size + self.size / 2, self.row * int((self.size) / 1.5) + self.size / 4,
                                        constants.GAMEFRAME.NODE_STATE_FONT_SIZE, lbl_color, self.batch, self.background, multiline=False,
                                        width=self.size)
        lbl = self.get_defense_text()
        self.defense_label = batch_label(lbl, self.col * self.size + self.size / 2, self.row * int((self.size) / 1.5) + self.size / 7,
                                         constants.GAMEFRAME.NODE_STATE_FONT_SIZE, lbl_color, self.batch, self.background, multiline=False,
                                         width=self.size)
        lbl = self.get_det_text()
        self.det_label = batch_label(lbl, self.col * self.size + self.size / 3.5, self.row * int((self.size) / 1.5) + self.size / 3,
                                     constants.GAMEFRAME.NODE_STATE_FONT_SIZE, lbl_color, self.batch, self.background, multiline=False,
                                     width=self.size)

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

    def simulate_attack(self, attack_type, edges_list):
        for i in range(0, self.num_blinks):
            if i % 2 == 0:
                clock.schedule_once(self.attack_red, self.blink_interval * i, edges_list)
            else:
                clock.schedule_once(self.attack_black, self.blink_interval * i, edges_list)
        if self.attack_values[attack_type] < self.max_value-1:
            self.attack_values[attack_type] += 1
        self.attack_label.text = self.get_attack_text()
        if self.attack_values[attack_type] > self.defense_values[attack_type]:
            return True  # successful attack
        else:
            return False

    def simulate_detection(self):
        if np.random.rand() < self.det/10:
            return True
        else:
            return False

    def attack_red(self, dt, edges_list):
        color = constants.GAMEFRAME.RED
        color_list = list(color) + list(color)
        for edges in edges_list:
            for e1 in edges:
                e1.colors = color_list
        lbl_color = constants.GAMEFRAME.RED_ALPHA
        self.attack_label.color = lbl_color
        self.color = constants.GAMEFRAME.RED

    def attack_black(self, dt, edges_list):
        color = constants.GAMEFRAME.BLACK
        color_list = list(color) + list(color)
        for edges in edges_list:
            for e1 in edges:
                e1.colors = color_list
        lbl_color = constants.GAMEFRAME.BLACK_ALPHA
        self.attack_label.color = lbl_color
        self.color = constants.GAMEFRAME.WHITE

    def __center_avatar(self):
        """
        Utiltiy function for centering the avatar inside a cell
        :return: The centered coordinates in the grid
        """
        self.x = self.col*self.size + self.size/2.5
        self.y = int((self.size/1.5))*self.row + self.size/3.5

    def add_reward(self, reward):
        self.cumulative_reward += reward

    def set_reward(self, reward):
        self.cumulative_reward = reward

    def update(self):
        """
        TODO
        :return:
        """
        pass

    def reset(self):
        self.initialize_state()
        self.set_labels()

    def defense_action(self, network_layout):
        if self.policy == constants.BASELINE_POLICIES.RANDOM:
            defend_type = np.random.randint(len(self.defense_values))
            while True:
                random_row = np.random.randint(network_layout.shape[0])
                random_col = np.random.randint(network_layout.shape[1])
                if network_layout[random_row, random_col] == constants.NODE_TYPES.RESOURCE or network_layout[random_row, random_col] == constants.NODE_TYPES.DATA:
                    return random_row, random_col, defend_type
        elif self.policy == constants.BASELINE_POLICIES.NAIVE_DETERMINISTIC:
            defend_type = 1
            for i in range(network_layout.shape[0]):
                for j in range(network_layout.shape[1]):
                    if network_layout[i, j] == constants.NODE_TYPES.RESOURCE or network_layout[i, j] == constants.NODE_TYPES.DATA:
                        return i, j, defend_type


    def defend(self, defend_type):
        if self.defense_values[defend_type] < self.max_value-1:
            self.defense_values[defend_type] += 1
        self.defense_label.text = self.get_defense_text()
        for i in range(0, self.num_blinks):
            if i % 2 == 0:
                clock.schedule_once(self.defense_green, self.blink_interval * i)
            else:
                clock.schedule_once(self.defense_black, self.blink_interval * i)

    def manual_blink_defense(self, i):
        if i % 2 == 0:
            self.defense_green(0)
        else:
            self.defense_black(0)

    def manual_blink_attack(self, i, edges):
        if i % 2 == 0:
            self.attack_red(0, edges)
        else:
            self.attack_black(0, edges)

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

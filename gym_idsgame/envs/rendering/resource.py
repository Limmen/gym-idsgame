import pyglet
from gym_idsgame.envs.rendering.render_util import batch_label, batch_rect_fill, batch_rect_border, batch_circle, create_circle
from gym_idsgame.envs.rendering import constants
from pyglet import clock
import numpy as np

class Resource(pyglet.sprite.Sprite):
    """

    TODO

    Subclasses pyglet.sprite.Sprite to be able to override draw() and update() methods
    and define state of the sprite
    """
    def __init__(self, avatar_path, col, row, batch, foreground, background, size, scale=0.25):
        """
        Class constructor, initializes the resource sprite

        :param avatar_path: path to the avatar file to use for the agent
        :param col: the starting x column in the grid
        :param row: the starting y row in the grid
        :param scale: the scale of the avatar
        :param batch: the batch to add this element to
        :param foreground: the batch foreground group
        :param background: the batch foreground group
        :param size: size of the cell in the grid
        """
        self.avatar = pyglet.resource.image(avatar_path)
        self.foreground = foreground
        self.background = background
        super(Resource, self).__init__(self.avatar, batch=batch, group=background)
        self.col = col
        self.row = row
        self.size = size
        self.__center_avatar()
        #self.reset()
        self.scale = scale
        self.batch = batch
        self.initialize_state()
        self.create_labels()
        self.incoming_edges = []
        self.outgoing_edges = []

    def create_labels(self):
        lbl_color = constants.IDSGAME.BLACK_ALPHA
        lbl = self.get_attack_text()
        self.attack_label = batch_label(lbl, self.col * self.size + self.size / 2, self.row * int((self.size) / 1.5) + self.size / 4,
                    constants.IDSGAME.NODE_STATE_FONT_SIZE, lbl_color, self.batch, self.background, multiline=False,
                    width=self.size)
        lbl = self.get_defense_text()
        self.defense_label = batch_label(lbl, self.col * self.size + self.size / 2, self.row * int((self.size) / 1.5) + self.size / 7,
                    constants.IDSGAME.NODE_STATE_FONT_SIZE, lbl_color, self.batch, self.background, multiline=False,
                    width=self.size)
        lbl = self.get_det_text()
        self.det_label = batch_label(lbl, self.col * self.size + self.size / 3, self.row * int((self.size) / 1.5) + self.size / 3,
                    constants.IDSGAME.NODE_STATE_FONT_SIZE, lbl_color, self.batch, self.background, multiline=False,
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
        self.attack_values = [0,0,0,0,0,0,0,0,0,0]
        self.defense_values = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
        self.det = 0.2

    def simulate_attack(self, attack_type):
        for i in range(0, constants.IDSGAME.NUM_BLINKS):
            if i % 2 == 0:
                clock.schedule_once(self.attack_red, constants.IDSGAME.BLINK_INTERVAL * i)
            else:
                clock.schedule_once(self.attack_black, constants.IDSGAME.BLINK_INTERVAL * i)
        if self.attack_values[attack_type] < 10:
            self.attack_values[attack_type] += 1
        self.attack_label.text = self.get_attack_text()
        if self.attack_values[attack_type] > self.defense_values[attack_type]:
            return True # successful attack
        else:
            return False

    def simulate_detection(self):
        if np.random.rand() < self.det:
            return True
        else:
            return False

    def attack_red(self, dt):
        color = constants.IDSGAME.RED
        color_list = list(color) + list(color)
        for edges in self.incoming_edges:
            for e1 in edges:
                e1.colors = color_list
        lbl_color = constants.IDSGAME.RED_ALPHA
        self.attack_label.color = lbl_color
        self.color = constants.IDSGAME.RED

    def attack_black(self, dt):
        color = constants.IDSGAME.BLACK
        color_list = list(color) + list(color)
        for edges in self.incoming_edges:
            for e1 in edges:
                e1.colors = color_list
        lbl_color = constants.IDSGAME.BLACK_ALPHA
        self.attack_label.color = lbl_color
        self.color = constants.IDSGAME.WHITE

    def __center_avatar(self):
        """
        Utiltiy function for centering the avatar inside a cell
        :return: The centered coordinates in the grid
        """
        self.x = self.col*self.size + self.size/2.3
        self.y = int((self.size/1.5))*self.row + self.size/3.5

    def update(self):
        """
        TODO
        :return:
        """
        pass

    def reset(self):
        self.initialize_state()
        self.set_labels()

    def defend(self, defend_type):
        if self.defense_values[defend_type] < 10:
            self.defense_values[defend_type] += 1
        self.defense_label.text = self.get_defense_text()
        for i in range(0, constants.IDSGAME.NUM_BLINKS):
            if i % 2 == 0:
                clock.schedule_once(self.defense_green, constants.IDSGAME.BLINK_INTERVAL * i)
            else:
                clock.schedule_once(self.defense_black, constants.IDSGAME.BLINK_INTERVAL * i)

    def defense_green(self, dt):
        color = constants.IDSGAME.GREEN_ALPHA
        self.defense_label.color = color
        self.color = constants.IDSGAME.GREEN

    def defense_black(self, dt):
        color = constants.IDSGAME.BLACK_ALPHA
        self.defense_label.color = color
        self.color = constants.IDSGAME.WHITE

    def unschedule(self):
        clock.unschedule(self.defense_green)
        clock.unschedule(self.attack_red)

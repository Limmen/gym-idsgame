from gym_idsgame.envs.rendering.resource import Resource
from gym_idsgame.envs.rendering.data import Data
from gym_idsgame.envs.rendering.hacker import Hacker
from gym_idsgame.envs.rendering.render_util import batch_label, batch_rect_fill, batch_rect_border, batch_circle, create_circle
from gym_idsgame.envs.rendering import constants

class ResourceNode:
    """
    Represents an individual resource-node in the network
    """

    def __init__(self, size):
        self.size = size
        self.resource = None
        self.hacker = None
        self.data = None
        self.circle = False

    def draw(self, y, x, color, batch, background, foreground, avatar, scale, server = False, data = False,
             start= False):
        #batch_rect_border(x * self.size, y * int((self.size/1.5)), self.size, int(self.size/1.5), color, batch, background)
        if server:
            self.resource = Resource(avatar, x, y, batch, background, self.size, scale=scale)
            lbl = "A=00,10,20,30,40,50,60,70,80,90"
            lbl_color = constants.IDSGAME.BLACK_ALPHA
            batch_label(lbl, x * self.size + self.size / 2, y * int((self.size)/1.5) + self.size / 4,
                        constants.IDSGAME.NODE_STATE_FONT_SIZE, lbl_color, batch, background, multiline=False,
                        width=self.size)
            lbl = "D=00,10,20,30,40,50,60,70,80,90"
            lbl_color = constants.IDSGAME.BLACK_ALPHA
            batch_label(lbl, x * self.size + self.size / 2, y * int((self.size) / 1.5) + self.size / 7,
                        constants.IDSGAME.NODE_STATE_FONT_SIZE, lbl_color, batch, background, multiline=False,
                        width=self.size)
        elif start:
            create_circle(x * self.size + self.size / 2, y * int(self.size / 1.5) + (self.size / 1.5)/2, self.size / 7,
                          batch, background, color)
            self.circle = True
            self.col = x
            self.row = y
        elif data:
            self.data = Data(avatar, x, y, batch, background, self.size, scale=scale)
            lbl = "A=00,10,20,30,40,50,60,70,80,90"
            lbl_color = constants.IDSGAME.BLACK_ALPHA
            batch_label(lbl, x * self.size + self.size / 2, y * int((self.size) / 1.5) + self.size / 4,
                        constants.IDSGAME.NODE_STATE_FONT_SIZE, lbl_color, batch, background, multiline=False,
                        width=self.size)
            lbl = "D=00,10,20,30,40,50,60,70,80,90"
            lbl_color = constants.IDSGAME.BLACK_ALPHA
            batch_label(lbl, x * self.size + self.size / 2, y * int((self.size) / 1.5) + self.size / 7,
                        constants.IDSGAME.NODE_STATE_FONT_SIZE, lbl_color, batch, background, multiline=False,
                        width=self.size)

    def get_link_coords(self, upper=True, lower=False):
        if self.resource is not None:
            if upper:
                x = self.resource.col*self.resource.size + self.resource.size/2
                y = (self.resource.row+1)*(self.resource.size/1.5) - self.size/6
            elif lower:
                x = self.resource.col * self.resource.size + self.resource.size / 2
                y = (self.resource.row + 1) * (self.resource.size / 1.5) - self.size / 1.75
            return x,y,self.resource.col, self.resource.row
        elif self.circle:
            x = self.col * self.size + self.size / 2
            y = (self.row + 1) * (self.size / 1.5) - self.size / 1.75
            return x,y,self.col,self.row
        elif self.data is not None:
            x = self.data.col * self.data.size + self.data.size / 2
            y = (self.data.row + 1) * (self.data.size / 1.5) - self.size / 15
            return x, y, self.data.col, self.data.row
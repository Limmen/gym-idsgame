"""
Attacker agent-sprite to be rendered for the gym-idgame environment
"""
from typing import Union
import pyglet
from gym_idsgame.envs.dao.idsgame_config import IdsGameConfig

class Attacker(pyglet.sprite.Sprite):
    """
    Class representing the attacker the game

    Subclasses pyglet.sprite.Sprite to be able to override draw() and update() methods
    and define state of the sprite
    """

    def __init__(self, idsgame_config: IdsGameConfig, col: int, row: int):
        """
        Constructor, initializes the attacker

        :param idsgame_config: configuration for IdsGameEnv
        :param col: the column in the grid where the attacker is currently
        :param row: the row in the grid where the attacker is currently
        """
        self.idsgame_config = idsgame_config
        self.avatar = pyglet.resource.image(idsgame_config.render_config.attacker_filename)
        super(Attacker, self).__init__(self.avatar, batch=idsgame_config.render_config.batch,
                                       group=idsgame_config.render_config.first_foreground)
        self.x = 0
        self.y = 0
        self.col = col
        self.row = row
        self.starting_col = col
        self.starting_row = row
        self.scale = idsgame_config.render_config.attacker_scale
        self.cage_avatar = pyglet.resource.image(self.idsgame_config.render_config.cage_filename)
        self.cage = pyglet.sprite.Sprite(self.cage_avatar, x=self.x, y=self.y, batch=idsgame_config.render_config.batch,
                                         group=idsgame_config.render_config.second_foreground)
        self.cage.scale = self.idsgame_config.render_config.cage_scale
        self.cage.visible = False
        self.hidden = False
        self.reset()

    def __center_avatar(self) -> None:
        """
        Utiltiy function for centering the avatar inside a cell

        :return: None
        """
        if self.col < (self.idsgame_config.game_config.network_config.num_cols // 2):
            self.x = self.idsgame_config.render_config.width // 2 - \
                     (self.idsgame_config.game_config.network_config.num_cols // 2 - (self.col)) * \
                     self.idsgame_config.render_config.rect_size - self.idsgame_config.render_config.rect_size / 20
        elif self.col > (self.idsgame_config.game_config.network_config.num_cols // 2):
            self.x = self.idsgame_config.render_config.width // 2 + \
                     (self.col - (self.idsgame_config.game_config.network_config.num_cols // 2)) * \
                     self.idsgame_config.render_config.rect_size - self.idsgame_config.render_config.rect_size / 20
        else:
            self.x = self.idsgame_config.render_config.width // 2 - self.idsgame_config.render_config.rect_size / 20
        self.y = int(self.idsgame_config.render_config.rect_size / 1.5) * \
                 self.row + self.idsgame_config.render_config.rect_size / 4.5

    def move_to_pos(self, pos: Union[int, int]) -> None:
        """
        Moves the attacker to a specific position in the grid

        :param pos: the poition to move the attacker to
        :return: None
        """
        row, col = pos
        self.col = col
        self.row = row
        self.__center_avatar()
        # If moving to a server node, move a little bit to the right so it does not cover the text
        if not (self.row == self.starting_row and self.col == self.starting_col):
            self.x = self.x + self.idsgame_config.render_config.rect_size / 5
            self.y = self.y + self.idsgame_config.render_config.rect_size / 15

    def move_to_coords(self, x: float, y: float, col: int, row: int) -> None:
        """
        Moves the attacker to a specific set of coordinates

        :param x: the x coordinate
        :param y: the y coordinate
        :param col: the column in the grid
        :param row: the row in the grid
        :return: None
        """
        self.x = x + self.idsgame_config.render_config.rect_size / 5
        self.y = y
        self.col = col
        self.row = row

    def reset(self) -> None:
        """
        Resets the attacker, moves the attacker back to the start-node and removes the cage

        :return: None
        """
        self.col = self.starting_col
        self.row = self.starting_row
        if self.hidden:
            self.visible = False
        self.cage.visible = False
        self.__center_avatar()

    def detected(self) -> None:
        """
        Called when the attacker was detected, shows a cage over the attacker on the screen

        :return: None
        """
        self.cage.x = self.x
        self.cage.y = self.y
        self.cage.visible = True
        self.visible = True

    def undetect(self) -> None:
        """
        Removes the cage from the attacker

        :return: None
        """
        self.cage.visible = False
        if self.hidden:
            self.visible = False


    def hide(self):
        self.visible = False
        self.hidden = True

    def show(self):
        self.visible = True
        self.hidden = False

    @property
    def pos(self):
        """
        :return: the grid position of the attacker
        """
        return (self.row, self.col)

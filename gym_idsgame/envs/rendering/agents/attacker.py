import pyglet
from gym_idsgame.envs.dao.render_config import RenderConfig

class Attacker(pyglet.sprite.Sprite):
    """
    TODO
    """

    def __init__(self, render_config: RenderConfig, col, row):
        self.render_config = render_config
        self.avatar = pyglet.resource.image(render_config.attacker_filename)
        super(Attacker, self).__init__(self.avatar, batch=render_config.batch, group=render_config.first_foreground)
        self.col = col
        self.row = row
        self.starting_col = col
        self.starting_row = row
        self.scale = render_config.attacker_scale
        self.cage_avatar = pyglet.resource.image(self.render_config.cage_filename)
        self.cage = pyglet.sprite.Sprite(self.cage_avatar, x=self.x, y=self.y, batch=render_config.batch,
                                         group=render_config.second_foreground)
        self.cage.scale = self.render_config.cage_scale
        self.cage.visible = False
        self.reset()

    def __center_avatar(self) -> None:
        """
        Utiltiy function for centering the avatar inside a cell
        :return: The centered coordinates in the grid
        """
        self.x = self.col * self.render_config.rect_size + self.render_config.rect_size / 2.65
        self.y = int(self.render_config.rect_size / 1.5) * self.row + self.render_config.rect_size / 4.5

    def move_to(self, x, y, col, row) -> None:
        self.x = x + self.render_config.rect_size / 5
        self.y = y
        self.col = col
        self.row = row

    def reset(self) -> None:
        self.col = self.starting_col
        self.row = self.starting_row
        self.cage.visible = False
        self.__center_avatar()

    def detected(self) -> None:
        self.cage.x = self.x
        self.cage.y = self.y
        self.cage.visible = True

    def undetect(self) -> None:
        self.cage.visible = False

    @property
    def pos(self):
        return (self.row, self.col)

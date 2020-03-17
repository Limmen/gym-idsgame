import pyglet


class Hacker(pyglet.sprite.Sprite):
    """
    Represents the hacker in the ids game.

    Subclasses pyglet.sprite.Sprite to be able to override draw() and update() methods
    and define state of the sprite
    """

    def __init__(self, avatar_path, col, row, batch, first_foreground, second_foreground, size, scale=0.25):
        self.avatar = pyglet.resource.image(avatar_path)
        super(Hacker, self).__init__(self.avatar, batch=batch, group=first_foreground)
        self.col = col
        self.row = row
        self.starting_col = col
        self.starting_row = row
        self.size = size
        self.scale = scale
        self.batch = batch
        self.first_foreground = first_foreground
        self.second_foreground = second_foreground
        self.cumulative_reward = 0
        self.cage_avatar = pyglet.resource.image("cage.png")
        self.cage = pyglet.sprite.Sprite(self.cage_avatar, x=self.x, y=self.y, batch=batch, group=second_foreground)
        self.cage.scale = 0.05
        self.cage.visible = False
        self.reset()

    def __center_avatar(self):
        """
        Utiltiy function for centering the avatar inside a cell
        :return: The centered coordinates in the grid
        """
        self.x = self.col * self.size + self.size / 2.65
        self.y = int(self.size / 1.5) * self.row + self.size / 4.5

    def move_to(self, x, y, col, row):
        self.x = x + self.size / 5
        self.y = y
        self.col = col
        self.row = row

    def add_reward(self, reward):
        self.cumulative_reward += reward

    def reset(self):
        self.col = self.starting_col
        self.row = self.starting_row
        self.cage.visible = False
        self.__center_avatar()

    def detected(self):
        self.cage.x = self.x
        self.cage.y = self.y
        self.cage.visible = True

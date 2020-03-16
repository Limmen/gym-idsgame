import pyglet

class Hacker(pyglet.sprite.Sprite):
    """
    Represents the hacker in the ids game.

    Subclasses pyglet.sprite.Sprite to be able to override draw() and update() methods
    and define state of the sprite
    """
    def __init__(self, avatar_path, col, row, batch,
                 group, size, scale=0.25):
        self.avatar = pyglet.resource.image(avatar_path)
        super(Hacker, self).__init__(self.avatar, batch=batch, group=group)
        self.col = col
        self.row = row
        self.size = size
        self.__center_avatar()
        #self.reset()
        self.scale = scale
        self.batch = batch
        self.group = group

    def __center_avatar(self):
        """
        Utiltiy function for centering the avatar inside a cell
        :return: The centered coordinates in the grid
        """
        self.x = self.col*self.size + self.size / 2.65
        self.y = int(self.size/1.5)*self.row + self.size/4.5

    def update(self):
        """
        TODO

        :return:
        """
        pass

    def reset(self):
        pass
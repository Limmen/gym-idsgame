import pyglet

class Data(pyglet.sprite.Sprite):
    """

    TODO

    Subclasses pyglet.sprite.Sprite to be able to override draw() and update() methods
    and define state of the sprite
    """
    def __init__(self, avatar_path, col, row, batch, group, size, scale=0.25):
        """
        Class constructor, initializes the resource sprite

        :param avatar_path: path to the avatar file to use for the agent
        :param col: the starting x column in the grid
        :param row: the starting y row in the grid
        :param scale: the scale of the avatar
        :param batch: the batch to add this element to
        :param group: the batch group to add this element to (e.g. foreground or background)
        :param size: size of the cell in the grid
        """
        self.avatar = pyglet.resource.image(avatar_path)
        super(Data, self).__init__(self.avatar, batch=batch, group=group)
        self.col = col
        self.row = row
        self.size = size
        self.__center_avatar()
        #self.reset()
        self.scale = scale
        self.batch = batch

    def __center_avatar(self):
        """
        Utiltiy function for centering the avatar inside a cell
        :return: The centered coordinates in the grid
        """
        self.x = self.col*self.size + self.size/2.5
        self.y = int((self.size/1.5))*self.row + self.size/3.5

    def update(self):
        """
        TODO
        :return:
        """
        pass

    def reset(self):
        pass

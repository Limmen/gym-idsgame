from gym_idsgame.envs.rendering import constants
from gym_idsgame.envs.rendering.render_util import batch_label, batch_rect_fill, batch_rect_border

class GridCell:
    """
    Represents an individual cell in the rendering
    """

    def __init__(self, size, goal_state = False):
        """
        Constructor, initialize state

        :param size: the size of the cell for rendering
        :param goal_state whether this cell is a goal state or not
        """
        self.size = size
        self.goal_state = goal_state

    def draw(self, y, x, color, batch, group):
        """
        Draws itself, i.e, a cell-square in the grid

        :param x: the x coordinate of the lower-left  corner of the cell
        :param y: the y coordinate of the lower-left  corner of the cell
        :param lbl_color: the border color of the cell
        :param batch: the batch to add this element to
        :param group: the batch group to add this element to (e.g. foreground or background)
        :return: None
        """
        batch_rect_border(x * self.size, y * self.size, self.size, self.size, color, batch, group)
        lbl = str(constants.GRIDWORLD.NEGATIVE_REWARD)
        lbl_color = constants.GRIDWORLD.BLACK_ALPHA
        if self.goal_state:
            lbl = str(constants.GRIDWORLD.POSITIVE_REWARD)
            batch_rect_fill(x * self.size, y * self.size, self.size, self.size, constants.GRIDWORLD.GREY, batch, group)
            lbl_color = constants.GRIDWORLD.RED_ALPHA
        return batch_label(lbl, x * self.size + self.size / 2, y * self.size + self.size / 2,
                           constants.GRIDWORLD.REWARD_FONT_SIZE, lbl_color, batch, group)
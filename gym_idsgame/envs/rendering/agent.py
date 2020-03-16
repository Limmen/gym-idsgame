import pyglet

class Agent(pyglet.sprite.Sprite):
    """
    Represents the agent in the grid world.
    Subclasses pyglet.sprite.Sprite to be able to override draw() and update() methods
    and define state of the sprite
    """
    def __init__(self, avatar_path, col, row, rect_size, num_cols, num_rows, goal_state_y, goal_state_x, batch,
                 group, scale=0.25):
        """
        Class constructor, initializes the agent

        :param avatar_path: path to the avatar file to use for the agent
        :param col: the starting x column in the grid
        :param row: the starting y row in the grid
        :param rect_size: size of a rectangle in the grid
        :param scale: the scale of the avatar
        :param num_cols: the maximum column in the grid
        :param num_rows: the maximum row in the grid
        :param goal_state_x: the column of the goal state
        :param goal_state_y: the row of the goal state
        :param batch: the batch to add this element to
        :param group: the batch group to add this element to (e.g. foreground or background)
        """
        self.avatar = pyglet.resource.image(avatar_path)
        self.rect_size = rect_size
        self.initial_col = col
        self.initial_row = row
        self.max_row = num_rows
        self.max_col = num_cols
        super(Agent, self).__init__(self.avatar, batch=batch, group=group)
        self.reset()
        self.__center_avatar()
        self.scale = scale
        self.goal_state_y = goal_state_y
        self.goal_state_x = goal_state_x
        self.batch = batch

    def __center_avatar(self):
        """
        Utiltiy function for centering the avatar inside a cell

        :return: The centered coordinates in the grid
        """
        self.x = self.col*self.rect_size + self.rect_size / 5
        self.y = self.rect_size*self.row - self.rect_size


    def move_left(self):
        """
        Moves the agent one step to the left in the grid

        :return: None
        """
        self.col -= 1
        self.__update_step()
        self.__update_reward()

    def move_right(self):
        """
        Moves the agent one step to the right in the grid

        :return: None
        """
        self.col += 1
        self.__update_step()
        self.__update_reward()

    def move_up(self):
        """
        Moves the agent one step up in the grid

        :return: None
        """
        self.row += 1
        self.__update_step()
        self.__update_reward()

    def move_down(self):
        """
        Moves the agent one step down in the grid

        :return: None
        """
        self.row -= 1
        self.__update_step()
        self.__update_reward()

    def update(self):
        """
        Updats the agent position in the grid, (sets the x,y coordinates to the center of the current cell).
        :return:
        """
        self.__check_bounds()
        self.__center_avatar()

    def __update_step(self):
        """
        Updates the game step

        :return: None
        """
        self.step += 1

    def __update_reward(self):
        """
        Update reward of the agent

        :return: None
        """
        if self.col != self.goal_state_x or self.row != (self.goal_state_y+1):
            self.reward -=1
        else:
            self.reward +=1
            self.reset()

    def reset(self):
        self.col = self.initial_col
        self.row = self.initial_row
        self.reward = 0
        self.step = 0

    def __check_bounds(self):
        """
        Utility method for making sure that the agent does not move out of bounds in the grid

        :return: None
        """
        self.col = max(0, self.col)
        self.col = min(self.max_col-1, self.col)
        self.row = max(1, self.row)
        self.row = min(self.max_row, self.row)

    def set_state(self, state):
        """
        Setter for the current agent state

        :param state: the state
        :return: None
        """
        old_col, old_row = self.col,self.row
        self.col = state[0]
        self.row = self.max_row - state[1]  # state[1] is inverted so subtract it so that it makes sense visually.
        if self.col != old_col or self.row != old_row:
            self.__update_step()
            self.__update_reward()
import pyglet
from gym_idsgame.envs.rendering.resource_network import ResourceNetwork
from gym_idsgame.envs.rendering.hacker import Hacker
from gym_idsgame.envs.rendering import constants
from gym_idsgame.envs.rendering.render_util import batch_label, batch_rect_fill, batch_line
from gym_idsgame.envs.rendering.resource import Resource
import os

class GameFrame(pyglet.window.Window):
    """
    A class representing the OpenGL/Pyglet Game Frame
    By subclassing pyglet.window.Window, event handlers can be defined simply by overriding functions, e.g.
    event handler for on_draw is defined by overriding the on_draw function.
    """

    def __init__(self, height=1000, width = 1000, rect_size = constants.IDSGAME.RECT_SIZE,
                 bg_color = constants.IDSGAME.WHITE, border_color = constants.IDSGAME.BLACK,
                 hacker_avatar_filename = constants.IDSGAME.HACKER_AVATAR_FILENAME,
                 server_filename = constants.IDSGAME.SERVER_AVATAR_FILENAME,
                 data_filename=constants.IDSGAME.DATA_AVATAR_FILENAME,
                 resources_dir = constants.IDSGAME.RESOURCES_DIR,
                 agent_scale = 0.3, resource_scale = 0.2, data_scale = 0.2, caption="GridWorld", manual = True,
                 line_width = constants.IDSGAME.LINE_WIDTH
                 ):
        """
        Gridframe constructor, initializes frame state and creates the window.

        :param height: height of the window
        :param width: width of the window
        :param rect_size: size of each cell in the grid
        :param bg_color: the color of the background of the grid
        :param border_color: the color of the border of the grid
        :param hacker_avatar_filename: name of the file-avatar to use for the agent
        :param resources_dir: the directory where resources are put (e.g. images)
        :param agent_scale: the scale of the agent avatar
        :param caption: caption of the frame
        :param manual: whether to setup the grid for manual play with keyboard
        :param line_width: line width
        """
        super(GameFrame, self).__init__(height=height, width=width, caption=caption) # call constructor of parent class
        self.bg_color = bg_color
        self.border_color = border_color
        self.rect_size = rect_size
        if self.rect_size < 0 or self.rect_size > (self.height - constants.IDSGAME.PANEL_HEIGHT) or self.rect_size > self.width:
            raise AssertionError("Rectangle size cannot be less than 0 and not greater than {}".format(
                max((self.height - constants.IDSGAME.PANEL_HEIGHT), self.width)))
        self.num_rows = (self.height - constants.IDSGAME.PANEL_HEIGHT) // int((self.rect_size/1.5))
        self.num_cols = self.width//self.rect_size
        if self.num_rows < 3 or self.num_cols < 1:
            raise AssertionError("The frame-size is too small, the frame must be large enough to fit "
                                 "at least three rows and one column")
        self.num_cells = self.num_rows*self.num_cols
        self.resource_network = ResourceNetwork(self.rect_size, self.num_rows, self.num_cols)
        self.resources_dir = resources_dir
        self.setup_resources_path()
        self.manual = manual
        self.avatar_filename = hacker_avatar_filename
        self.server_filename = server_filename
        self.data_filename = data_filename
        self.agent_scale = agent_scale
        self.resource_scale = resource_scale
        self.data_scale = data_scale
        self.line_width = line_width
        self.create_batch()

    def create_batch(self):
        """
        Creates a batch of elements to render. By grouping elements in a batch we can utilize OpenGL batch rendering
        and reduce the cpu <–> gpu data transfers and the number of draw-calls.

        :return: None
        """
        self.batch = pyglet.graphics.Batch()
        self.background = pyglet.graphics.OrderedGroup(0)
        self.foreground = pyglet.graphics.OrderedGroup(1)

        # ---- Background ----

        # Sets the background color
        batch_rect_fill(0, 0, self.width, self.height, self.bg_color, self.batch, self.background)

        # Resource Network
        for i in range(self.resource_network.num_rows):
            for j in range(self.resource_network.num_cols):
                # Data node
                if i == 0:
                    self.resource_network.grid[i][j].draw(i, j, self.border_color, self.batch, self.background,
                                                          self.foreground,
                                                          self.data_filename, self.data_scale,
                                                          data=(j == (self.num_cols // 2)))
                # Server node
                if i > 0 and i < self.resource_network.num_rows-1:
                    self.resource_network.grid[i][j].draw(i, j, self.border_color, self.batch, self.background,
                                                          self.foreground,
                                                          self.server_filename, self.resource_scale, server=True)
                # Start node
                if i == self.resource_network.num_rows-1:
                    self.resource_network.grid[i][j].draw(i, j, self.border_color, self.batch, self.background,
                                                          self.foreground,
                                                          self.avatar_filename, self.agent_scale,
                                                          start=(j == (self.num_cols//2)))
        # Hacker starts at the start node
        self.hacker = Hacker(self.avatar_filename, self.num_cols//2,
                             self.resource_network.num_rows-1, self.batch, self.foreground, self.rect_size,
                             scale=self.agent_scale)

        # Connect start node with server nodes
        for i in range(self.num_cols):
            self.resource_network.connect_start_and_server_nodes(
                self.resource_network.grid[self.resource_network.num_rows-1][self.num_cols//2],
                self.resource_network.grid[self.resource_network.num_rows - 2][i],
                constants.IDSGAME.BLACK, self.batch, self.background, self.line_width)

        # Connect server nodes with server nodes on next layer
        for i in range(1, self.resource_network.num_rows-2):
            for j in range(self.resource_network.num_cols):
                self.resource_network.connect_server_and_server_nodes(
                    self.resource_network.grid[i][j],
                    self.resource_network.grid[i+1][j],
                    constants.IDSGAME.BLACK, self.batch, self.background, self.line_width
                )

        # Connect server nodes on final layer with data node
        for j in range(self.resource_network.num_cols):
            self.resource_network.connect_server_and_data_nodes(
                self.resource_network.grid[1][j],
                self.resource_network.grid[0][self.num_cols//2],
                constants.IDSGAME.BLACK, self.batch, self.background, self.line_width
            )

        # ---- Foreground ----

        # Agent
        #x, y = self.width / 2 - 100, self.height - constants.IDSGAME.PANEL_HEIGHT * 2
        #self.agent = Hacker(self.avatar_filename, x, y, self.batch, self.foreground, scale=self.agent_scale)

        # Resources
        #self.cpu1 = Resource(self.server_filename, 10, self.height / 2 - constants.IDSGAME.PANEL_HEIGHT, self.batch,
                             #self.foreground, scale=self.resource_scale)

        # Links
        #batch_line(self.width // 2 - 100, self.height - constants.IDSGAME.PANEL_HEIGHT * 2, 10,
        #           self.height // 2 - constants.IDSGAME.PANEL_HEIGHT, constants.IDSGAME.BLACK, self.batch, self.foreground,
        #           self.line_width)

        # Panel Labels
        batch_label("R: ", constants.IDSGAME.PANEL_LEFT_MARGIN, self.height - constants.IDSGAME.PANEL_TOP_MARGIN,
                    constants.IDSGAME.PANEL_FONT_SIZE, constants.IDSGAME.BLACK_ALPHA, self.batch,
                    self.foreground)
        batch_label("t: ", constants.IDSGAME.PANEL_LEFT_MARGIN,
                    self.height - constants.IDSGAME.PANEL_TOP_MARGIN * 2,
                    constants.IDSGAME.PANEL_FONT_SIZE, constants.IDSGAME.BLACK_ALPHA,
                    self.batch, self.foreground)
        #self.reward_label = batch_label(str(self.agent.reward), constants.GRIDWORLD.PANEL_LEFT_MARGIN * 2,
        #            self.height - constants.GRIDWORLD.PANEL_TOP_MARGIN,
        #            constants.GRIDWORLD.PANEL_FONT_SIZE, constants.GRIDWORLD.BLACK_ALPHA, self.batch,
        #            self.foreground)
        #self.step_label = batch_label(str(self.agent.step), constants.GRIDWORLD.PANEL_LEFT_MARGIN * 2,
        #            self.height - constants.GRIDWORLD.PANEL_TOP_MARGIN * 2,
        #            constants.GRIDWORLD.PANEL_FONT_SIZE, constants.GRIDWORLD.BLACK_ALPHA, self.batch,
        #            self.foreground)

    def setup_resources_path(self):
        """
        Setup path to resources (e.g. images)

        :return: None
        """
        if os.path.exists(self.resources_dir):
            pyglet.resource.path = [self.resources_dir]
        else:
            script_dir = os.path.dirname(__file__)
            resource_path = os.path.join(script_dir, './', constants.IDSGAME.RESOURCES_DIR)
            pyglet.resource.path = [resource_path]
        pyglet.resource.reindex()

    def on_draw(self):
        """
        Event handler for on_draw event. OpenGL does not remember what was rendered on the previous frame so
        we redraw each frame every time. This method is typically called many times per second.

        Draws the GridWorld Frame

        :return: None
        """
        # Clear the window
        self.clear()
        # Draw batch with the frame contents
        self.batch.draw()
        # Make this window the current OpenGL rendering context
        self.switch_to()


    def on_key_press(self, symbol, modifiers):
        """
        Event handler for on_key_press event.
        The user can move the agent with key presses.

        :param symbol: the symbol of the keypress
        :param modifiers: _
        :return: None
        """
        if self.manual:
            if symbol == pyglet.window.key.LEFT:
                self.agent.move_left()
            elif symbol == pyglet.window.key.RIGHT:
                self.agent.move_right()
            elif symbol == pyglet.window.key.UP:
                self.agent.move_up()
            elif symbol == pyglet.window.key.DOWN:
                self.agent.move_down()
            elif symbol == pyglet.window.key.SPACE:
                pass


    def update(self, dt):
        """
        Event handler for the update-event (timer-based typically), used to update the state of the grid.

        :param dt: the number of seconds since the function was last called
        :return: None
        """
        pass

    def set_state(self, state):
        """
        TODO

        :param state: the state
        :return: None
        """
        pass

    def reset(self):
        """
        Resets the agent state without closing the screen

        :return: None
        """
        self.agent.reset()

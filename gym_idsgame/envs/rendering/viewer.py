try:
    import pyglet
except ImportError as e:
    raise ImportError('''
    Cannot import pyglet.
    HINT: you can install pyglet directly via 'pip install pyglet'.
    But if you really just want to install all Gym dependencies and not have to think about it,
    'pip install -e .[all]' or 'pip install gym[all]' will do it.
    ''')

try:
    from pyglet.gl import *
except ImportError as e:
    raise ImportError('''
    Error occurred while running `from pyglet.gl import *`
    HINT: make sure you have OpenGL install. On Ubuntu, you can run 'apt-get install python-opengl'.
    If you're running on a server, you may need a virtual frame buffer; something like this should work:
    'xvfb-run -s \"-screen 0 1400x900x24\" python <your_script.py>'
    ''')

from gym_idsgame.envs.rendering.grid_frame import GridFrame
from gym_idsgame.envs.rendering import constants
import numpy as np

class Viewer():
    def __init__(self, height=1000, width=800, rect_size=constants.GRIDWORLD.RECT_SIZE,
              bg_color=constants.GRIDWORLD.WHITE, border_color=constants.GRIDWORLD.BLACK,
              avatar_filename=constants.GRIDWORLD.AVATAR_FILENAME,
              resources_dir=constants.GRIDWORLD.RESOURCES_DIR, agent_start_x=None, agent_start_y=None,
              agent_scale=0.2, caption="GridWorld", manual=True):
        """
        Creates the a viewer for the grid world

        :param height: height of the window
        :param width: width of the window
        :param rect_size: size of each cell in the grid
        :param bg_color: the color of the background of the grid
        :param border_color: the color of the border of the grid
        :param avatar_filename: name of the file-avatar to use for the agent
        :param resources_dir: the directory where resources are put (e.g. images)
        :param agent_start_x: the agent starting position (column index)
        :param agent_start_y: the agent starting position (row index)
        :param agent_scale: the scale of the agent avatar
        :param caption: caption of the frame
        :param manual: whether to setup the grid for manual play with keyboard
        """

        self.height = height
        self.width = width
        self.rect_size = rect_size
        self.bg_color = bg_color
        self.border_color = border_color
        self.avatar_filename = avatar_filename
        self.resources_dir = resources_dir
        self.agent_start_x = agent_start_x
        self.agent_start_y = agent_start_y
        self.agent_scale = agent_scale
        self.caption = caption
        self.manual = manual
        self.isopen = True

    def manual_start(self):
        """
        Starts the gridframe app in a manual mode where the agent is controlled through arrow-keys

        :return: None
        """
        self.gridframe = GridFrame(width=self.width, height=self.height, rect_size=self.rect_size, bg_color=self.bg_color,
                              border_color=self.border_color,
                              avatar_filename=self.avatar_filename, resources_dir=self.resources_dir,
                              agent_start_x=self.agent_start_x, agent_start_y=self.agent_start_y,
                              agent_scale=self.agent_scale, caption=self.caption, manual=self.manual)
        self.gridframe.on_close = self.window_closed_by_user
        self.isopen = True
        pyglet.clock.schedule_interval(self.gridframe.update, 1 / 60.)
        pyglet.app.run()

    def agent_start(self):
        """
        Creates the frame in a agent-mode, where actions are taken programmatically rather than through
        moving arrow-keys.
        """
        self.gridframe = GridFrame(width=self.width, height=self.height, rect_size=self.rect_size,
                                   bg_color=self.bg_color,
                                   border_color=self.border_color,
                                   avatar_filename=self.avatar_filename, resources_dir=self.resources_dir,
                                   agent_start_x=self.agent_start_x, agent_start_y=self.agent_start_y,
                                   agent_scale=self.agent_scale, caption=self.caption, manual=self.manual)
        self.gridframe.on_close = self.window_closed_by_user
        self.isopen = True

    def window_closed_by_user(self):
        """
        Callback when the frame is closed by the user

        :return: None
        """
        self.isopen = False
        self.gridframe.close()

    def close(self):
        """
        Closes the frame

        :return: None
        """
        self.gridframe.close()

    def render(self, return_rgb_array = False):
        """
        Renders a a frame. Using pyglet together with openAI gym means that we have to integrate OpenGL's event-loop
        with the event-loop of the RL agent and the gym framework. That's why we render things manually and dispatch
        events manually rather than just calling pyglet.app.run().

        :param return_rgb_array: boolean whether to return rgb array or not

        :return: None
        """
        self.gridframe.clear() # Clears the frame
        self.gridframe.switch_to() # Make this window the current OpenGL rendering context
        self.gridframe.dispatch_events() # Poll the OS for events and call related handlers for updating the frame
        self.gridframe.on_draw() # Draw the frame
        if return_rgb_array:
            arr = self.extract_rgb_array()
        self.gridframe.flip() # Swaps the OpenGL front and back buffers Updates the visible display with the back buffer
        return arr if return_rgb_array else self.isopen

    def extract_rgb_array(self):
        """
        Extract RGB array from pyglet, this can then be used to record video of the rendering through gym's API

        :return: RGB Array [height, width, 3]
        """
        buffer = pyglet.image.get_buffer_manager().get_color_buffer()
        image_data = buffer.get_image_data()
        arr = np.fromstring(image_data.get_data(), dtype=np.uint8, sep='')
        # In https://github.com/openai/gym-http-api/issues/2, we
        # discovered that someone using Xmonad on Arch was having
        # a window of size 598 x 398, though a 600 x 400 window
        # was requested. (Guess Xmonad was preserving a pixel for
        # the boundary.) So we use the buffer height/width rather
        # than the requested one.
        arr = arr.reshape(buffer.height, buffer.width, 4)
        arr = arr[::-1, :, 0:3]
        return arr
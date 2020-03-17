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

from gym_idsgame.envs.rendering.game_frame import GameFrame
from gym_idsgame.envs.rendering import constants
import numpy as np

class Viewer():
    def __init__(self, num_layers = 1, num_servers_per_layer = 2, num_attack_types = 10, max_value = 10,
                 defense_policy = constants.BASELINE_POLICIES.NAIVE_DETERMINISTIC,
                 resources_dir = constants.IDSGAME.RESOURCES_DIR):

        self.num_layers = num_layers
        self.num_servers_per_layer = num_servers_per_layer
        self.num_attack_types = num_attack_types
        self.max_value = max_value
        self.defense_policy = defense_policy
        self.resources_dir = resources_dir
        self.isopen = True

    def manual_start(self):
        """
        Starts the gridframe app in a manual mode where the agent is controlled through arrow-keys

        :return: None
        """
        self.gridframe = GameFrame(num_layers = self.num_layers, num_servers_per_layer = self.num_servers_per_layer,
                                   num_attack_types = self.num_attack_types, max_value = self.max_value,
                                   defense_policy = self.defense_policy,resources_dir = self.resources_dir,
                                   manual = True)
        self.gridframe.on_close = self.window_closed_by_user
        self.isopen = True
        pyglet.clock.schedule_interval(self.gridframe.update, 1 / 60.)
        pyglet.app.run()

    def agent_start(self):
        """
        Creates the frame in a agent-mode, where actions are taken programmatically rather than through
        moving arrow-keys.
        """
        self.gridframe = GameFrame(num_layers=self.num_layers, num_servers_per_layer=self.num_servers_per_layer,
                                   num_attack_types=self.num_attack_types, max_value=self.max_value,
                                   defense_policy=self.defense_policy, resources_dir=self.resources_dir,
                                   manual=False)
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
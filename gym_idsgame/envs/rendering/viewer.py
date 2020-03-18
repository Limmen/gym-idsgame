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

from gym_idsgame.envs.rendering.frames.game_frame import GameFrame
from gym_idsgame.envs.rendering.constants import constants
from gym_idsgame.envs.dao.render_config import RenderConfig
import numpy as np
import time

class Viewer():
    def __init__(self, render_config: RenderConfig):

        self.render_config = render_config
        self.isopen = True

    def manual_start(self):
        """
        Starts the gridframe app in a manual mode where the agent is controlled through arrow-keys

        :return: None
        """
        self.render_config.manual = True
        self.gameframe = GameFrame(render_config=self.render_config)
        self.gameframe.on_close = self.window_closed_by_user
        self.isopen = True
        pyglet.clock.schedule_interval(self.gameframe.update, 1 / 60.)
        pyglet.app.run()

    def agent_start(self):
        """
        Creates the frame in a agent-mode, where actions are taken programmatically rather than through
        moving arrow-keys.
        """
        self.render_config.manual = False
        self.gameframe = GameFrame(render_config=self.render_config)
        self.gameframe.on_close = self.window_closed_by_user
        self.isopen = True

    def window_closed_by_user(self):
        """
        Callback when the frame is closed by the user

        :return: None
        """
        self.isopen = False
        self.gameframe.close()

    def close(self):
        """
        Closes the frame

        :return: None
        """
        self.gameframe.close()

    def render_frame(self, return_rgb_array = False):
        self.gameframe.clear()  # Clears the frame
        self.gameframe.switch_to()  # Make this window the current OpenGL rendering context
        self.gameframe.dispatch_events()  # Poll the OS for events and call related handlers for updating the frame
        self.gameframe.on_draw()  # Draw the frame
        if return_rgb_array:
            arr = self.extract_rgb_array()
        self.gameframe.flip()  # Swaps the OpenGL front and back buffers Updates the visible display with the back buffer
        return arr if return_rgb_array else self.isopen


    def render(self, return_rgb_array = False):
        """
        Renders a a frame. Using pyglet together with openAI gym means that we have to integrate OpenGL's event-loop
        with the event-loop of the RL agent and the gym framework. That's why we render things manually and dispatch
        events manually rather than just calling pyglet.app.run().

        :param return_rgb_array: boolean whether to return rgb array or not

        :return: None
        """
        arr = self.render_frame(return_rgb_array)
        for i in range(self.gameframe.num_blinks):
            if len(self.gameframe.defense_events) > 0 or len(self.gameframe.attack_events) > 0:
                self.gameframe.simulate_events(i)
                arr = self.render_frame()
                time.sleep(self.gameframe.blink_interval)
        self.gameframe.reset_events()

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
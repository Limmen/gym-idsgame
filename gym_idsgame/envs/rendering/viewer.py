"""
The viewer for rendering the gym-idsgame environment. Supports both agent-mode and manual-mode
"""
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
from typing import Union
import numpy as np
import time
import sys
from gym_idsgame.envs.rendering.frames.game_frame import GameFrame
from gym_idsgame.envs.dao.idsgame_config import IdsGameConfig

class Viewer():
    """
    Class representing a viewer for the IDS-game. Can be used for either agent-view, or manual-view (human player)
    """
    def __init__(self, idsgame_config: IdsGameConfig):
        """

        :param idsgame_config: configuratin for the IdsGameEnv
        """
        self.idsgame_config = idsgame_config
        self.isopen = True

    def manual_start_attacker(self) -> None:
        """
        Starts the IDS-game app in a manual mode where the attacker is controlled with keyboard and mouse

        :return: None
        """
        self.idsgame_config.game_config.manual_attacker = True
        self.gameframe = GameFrame(idsgame_config=self.idsgame_config)
        self.gameframe.on_close = self.window_closed_by_user
        self.isopen = True
        pyglet.clock.schedule_interval(self.gameframe.update, 1 / 60.)
        pyglet.app.run()

    def manual_start_defender(self) -> None:
        """
        Starts the IDS-game app in a manual mode where the defender is controlled with keyboard and mouse

        :return: None
        """
        self.idsgame_config.game_config.manual_defender = True
        self.gameframe = GameFrame(idsgame_config=self.idsgame_config)
        self.gameframe.on_close = self.window_closed_by_user
        self.isopen = True
        pyglet.clock.schedule_interval(self.gameframe.update, 1 / 60.)
        pyglet.app.run()

    def agent_start(self) -> None:
        """
        Creates the IDS-game frame in agent-mode, where actions are taken programmatically rather than through
        moving mouse and keyboard.
        """
        self.idsgame_config.game_config.manual_attacker = False
        self.idsgame_config.game_config.manual_defender = False
        self.gameframe = GameFrame(idsgame_config=self.idsgame_config)
        self.gameframe.on_close = self.window_closed_by_user
        self.isopen = True

    def window_closed_by_user(self) -> None:
        """
        Callback when the frame is closed by the user

        :return: None
        """
        self.isopen = False
        self.gameframe.close()
        print("Window closed, exiting")
        sys.exit(0)

    def close(self) -> None:
        """
        Closes the frame

        :return: None
        """
        self.gameframe.close()

    def render_frame(self, return_rgb_array: bool = False):
        """
        Renders a frame manually.

        Using pyglet together with openAI gym means that we have to integrate OpenGL's event-loop
        with the event-loop of the RL agent and the gym framework. That's why we render things manually and dispatch
        events manually rather than just calling pyglet.app.run().

        :param return_rgb_array: if this is true it returns the RGB array for the rendered frame (for recording)
        :return: RGB array or bool
        """
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
        Renders a state of the IDS game. A single state might include many frames. For example if an attack or defense
        move was made, this will cause several frames to visualize the attack/defense.

        :param return_rgb_array: boolean whether to return rgb array or not

        :return: RGB array or bool
        """
        self.gameframe.unschedule_events()
        frames = []
        arr = self.render_frame(return_rgb_array)
        frames.append(arr)
        for i in range(self.gameframe.idsgame_config.render_config.num_blinks):
            if len(self.gameframe.game_state.defense_events) > 0 or len(self.gameframe.game_state.attack_events) > 0:
                self.gameframe.simulate_events(i)
                arr = self.render_frame(return_rgb_array=return_rgb_array)
                frames.append(arr)
                time.sleep(self.gameframe.idsgame_config.render_config.blink_interval)
        self.gameframe.reset_events()

        return np.array(frames) if return_rgb_array else self.isopen

    def extract_rgb_array(self) -> np.ndarray:
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
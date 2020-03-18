from gym_idsgame.envs.rendering.constants import constants
from gym_idsgame.envs.dao.game_config import GameConfig
import pyglet
class RenderConfig:

    def __init__(self, game_config:GameConfig, resources_dir=constants.GAMEFRAME.RESOURCES_DIR,
                 blink_interval = constants.GAMEFRAME.MANUAL_BLINK_INTERVAL,
                 num_blinks = constants.GAMEFRAME.MANUAL_NUM_BLINKS):
        self.game_config = game_config
        self.rect_size = constants.GAMEFRAME.RECT_SIZE
        self.bg_color = constants.GAMEFRAME.WHITE
        self.border_color = constants.GAMEFRAME.BLACK
        self.avatar_filename = constants.GAMEFRAME.HACKER_AVATAR_FILENAME
        self.server_filename = constants.GAMEFRAME.SERVER_AVATAR_FILENAME
        self.data_filename = constants.GAMEFRAME.DATA_AVATAR_FILENAME
        self.minimum_width = constants.GAMEFRAME.MIN_WIDTH
        self.agent_scale = 0.3
        self.server_scale = 0.2
        self.data_scale = 0.2
        self.line_width = constants.GAMEFRAME.LINE_WIDTH
        self.height = constants.GAMEFRAME.PANEL_HEIGHT + int((self.rect_size / 1.5)) * self.game_config.num_rows
        self.width = max(self.minimum_width, self.rect_size * self.game_config.num_cols)
        self.caption = constants.GAMEFRAME.CAPTION
        self.resources_dir = resources_dir
        self.blink_interval = blink_interval
        self.num_blinks = num_blinks
        self.batch = pyglet.graphics.Batch()
        self.background = pyglet.graphics.OrderedGroup(0)
        self.first_foreground = pyglet.graphics.OrderedGroup(1)
        self.second_foreground = pyglet.graphics.OrderedGroup(2)
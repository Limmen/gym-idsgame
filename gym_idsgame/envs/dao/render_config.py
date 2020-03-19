from gym_idsgame.envs.constants import constants
from gym_idsgame.envs.dao.game_config import GameConfig
from gym_idsgame.envs.policy_baselines.policy import Policy
import pyglet

class RenderConfig:

    def __init__(self, game_config:GameConfig, defender_policy: Policy, resources_dir=constants.RENDERING.RESOURCES_DIR,
                 blink_interval = constants.RENDERING.MANUAL_BLINK_INTERVAL,
                 num_blinks = constants.RENDERING.MANUAL_NUM_BLINKS,
                 ):
        self.game_config = game_config
        self.defender_policy = defender_policy
        self.rect_size = constants.RENDERING.RECT_SIZE
        self.bg_color = constants.RENDERING.WHITE
        self.border_color = constants.RENDERING.BLACK
        self.attacker_filename = constants.RENDERING.HACKER_AVATAR_FILENAME
        self.server_filename = constants.RENDERING.SERVER_AVATAR_FILENAME
        self.data_filename = constants.RENDERING.DATA_AVATAR_FILENAME
        self.cage_filename = constants.RENDERING.CAGE_AVATAR_FILENAME
        self.minimum_width = constants.RENDERING.MIN_WIDTH
        self.attacker_scale = constants.RENDERING.ATTACKER_AVATAR_SCALE
        self.server_scale = constants.RENDERING.SERVER_AVATAR_SCALE
        self.data_scale = constants.RENDERING.DATA_AVATAR_SCALE
        self.cage_scale = constants.RENDERING.CAGE_AVATAR_SCALE
        self.line_width = constants.RENDERING.LINE_WIDTH
        self.height = constants.RENDERING.PANEL_HEIGHT + int((self.rect_size / 1.5)) * self.game_config.num_rows
        self.width = max(self.minimum_width, self.rect_size * self.game_config.num_cols)
        self.caption = constants.RENDERING.CAPTION
        self.resources_dir = resources_dir
        self.blink_interval = blink_interval
        self.num_blinks = num_blinks
        self.batch = pyglet.graphics.Batch()
        self.background = pyglet.graphics.OrderedGroup(0)
        self.first_foreground = pyglet.graphics.OrderedGroup(1)
        self.second_foreground = pyglet.graphics.OrderedGroup(2)
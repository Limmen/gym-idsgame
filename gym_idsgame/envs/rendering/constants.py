class GAMEFRAME:
    """
    IDSGame constants
    """
    RECT_SIZE = 200
    BLACK = (0, 0, 0)
    BLUE = (0, 0, 255)
    RED = (255, 0, 0)
    RED_ALPHA = (255, 0, 0, 255)
    GREEN = (0, 128, 0)
    GREEN_ALPHA = (0, 128, 0, 255)
    LIME = (0, 255, 0)
    BLACK_ALPHA = (0, 0, 0, 255)
    WHITE_ALPHA = (255, 255, 255, 255)
    RED_ALPHA = (128, 0, 0, 255)
    WHITE = (255,255,255)
    GREY = (220, 220, 220)
    RESOURCES_DIR = "resources"
    HACKER_AVATAR_FILENAME = "hacker.png"
    SERVER_AVATAR_FILENAME = "ibm_tower.png"
    DATA_AVATAR_FILENAME = "data.png"
    PANEL_HEIGHT = 50
    PANEL_LEFT_MARGIN = 60
    PANEL_TOP_MARGIN = 30
    PANEL_FONT_SIZE = 12
    POSITIVE_REWARD = +100
    NEGATIVE_REWARD = -100
    NODE_STATE_FONT_SIZE = 10
    LINE_WIDTH = 2
    NUM_BLINKS = 14
    BLINK_INTERVAL = 0.2
    CAPTION = "IDS Game"
    MIN_WIDTH = 470

class NODE_TYPES:
    """
    TODO
    """
    NONE = 0
    START = 1
    RESOURCE = 2
    DATA = 3

class BASELINE_POLICIES:
    RANDOM = "random"
    NAIVE_DETERMINISTIC = "naive_deterministic"

class RENDER_STATE:
    ATTACK_VALUES = "attack_values"
    DEFENSE_VALUES = "defense_values"
    DEFENSE_DET = "defense_det"
    ATTACKER_POS = "attacker_pos"
    GAME_STEP = "game_step"
    ATTACKER_CUMULATIVE_REWARD = "attacker_cumulative_reward"
    DEFENDER_CUMULATIVE_REWARD = "attacker_cumulative_reward"
    NUM_GAMES = "num_games"
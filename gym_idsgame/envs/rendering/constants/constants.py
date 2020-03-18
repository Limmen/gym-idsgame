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
    MANUAL_NUM_BLINKS = 14
    MANUAL_BLINK_INTERVAL = 0.3
    AGENT_NUM_BLINKS = 6
    AGENT_BLINK_INTERVAL = 0.000001
    CAPTION = "IDS Game"
    MIN_WIDTH = 470

class NODE_TYPES:
    """
    TODO
    """
    NONE = 0
    START = 1
    SERVER = 2
    DATA = 3

class BASELINE_POLICIES:
    RANDOM = "random"
    NAIVE_DETERMINISTIC = "naive_deterministic"

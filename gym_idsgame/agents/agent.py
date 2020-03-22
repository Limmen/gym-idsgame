"""
Abstract agent for the gym-idsgame environment
"""
from abc import ABC
from gym_idsgame.envs.dao.game_config import GameConfig

class Agent(ABC):
    """
    Abstract class representing a policy
    """

    def __init__(self, game_config: GameConfig):
        """
        Class constructor

        :param game_config: the game configuration
        """
        self.game_config = game_config
        if self.game_config is None:
            self.game_config = GameConfig()

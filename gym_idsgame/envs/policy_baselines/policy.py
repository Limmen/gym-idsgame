from abc import ABC, abstractmethod
from gym_idsgame.envs.dao.game_state import GameState
from gym_idsgame.envs.dao.game_config import GameConfig
from typing import Union

class Policy(ABC):
    """
    Abstract class representing a policy
    """

    def __init__(self, game_config: GameConfig):
        """
        Class constructor

        :param game_config: the game configuration
        """
        self.game_config = game_config

    @abstractmethod
    def action(self, render_state: GameState) -> Union[int, int, int]:
        """
        Abstract method to be implemented by sub-classes.
        A method that takes in the current state and outputs an action

        :param render_state: the current state
        :return: (row, col, type)
        """
        pass
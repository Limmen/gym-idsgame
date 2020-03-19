from abc import ABC, abstractmethod
from gym_idsgame.envs.dao.game_state import GameState
from gym_idsgame.envs.dao.game_config import GameConfig
from typing import Union

class Policy(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def action(self, game_state: GameState, game_config: GameConfig) -> Union[int,int,int]:
        pass
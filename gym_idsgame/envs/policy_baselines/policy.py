from abc import ABC, abstractmethod
from gym_idsgame.envs.dao.render_state import RenderState
from gym_idsgame.envs.dao.game_config import GameConfig
from typing import Union

class Policy(ABC):

    def __init__(self, game_config: GameConfig):
        self.game_config = game_config

    @abstractmethod
    def action(self, render_state: RenderState) -> Union[int, int, int]:
        pass
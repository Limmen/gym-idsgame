from gym_idsgame.envs.policy_baselines.policy import Policy
from gym_idsgame.envs.dao.render_state import RenderState
from gym_idsgame.envs.dao.game_config import GameConfig
from typing import Union

class DeterministicDefense(Policy):

    def __init__(self, game_config: GameConfig, defend_type:int, defend_row:int, defend_col:int):
        super(DeterministicDefense, self).__init__(game_config)
        self.defend_type = defend_type
        self.defend_row = defend_row
        self.defend_col = defend_col
        if defend_type is None:
            raise ValueError("defend_type cannot be None for a Deterministic Defense Policy")
        if defend_row is None:
            raise ValueError("defend_row cannot be None for a Deterministic Defense Policy")
        if defend_col is None:
            raise ValueError("defend_row cannot be None for a Deterministic Defense Policy")

    def action(self, render_state: RenderState) -> Union[int, int, int]:
        return self.defend_row, self.defend_col, self.defend_type

"""
A Deterministic Defense Policy Baseline for the gym-idsgame environment
"""
from typing import Union
from gym_idsgame.envs.policy_baselines.policy import Policy
from gym_idsgame.envs.dao.game_state import GameState
from gym_idsgame.envs.dao.game_config import GameConfig

class DeterministicDefense(Policy):
    """
    Implements a deterministic defense policy: a policy where the defender always defends the same node using the same
    defense type
    """
    def __init__(self, game_config: GameConfig, defend_type: int, defend_row: int, defend_col: int):
        """
        Class constructor, intializes the policy

        :param game_config: configuration of the game
        :param defend_type: the type of defense
        :param defend_row: the row in the grid to defend
        :param defend_col: the columns in the grid to defend
        """
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

    def action(self, render_state: GameState) -> Union[int, int, int]:
        """
        Returns the action from the deterministic policy

        :param render_state: the current render state
        :return: (row, col, defend_type)
        """
        return self.defend_row, self.defend_col, self.defend_type

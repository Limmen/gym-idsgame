"""
A Deterministic Defense Agent Baseline for the gym-idsgame environment
"""
from typing import Union
from gym_idsgame.agents.bot_agent import BotAgent
from gym_idsgame.envs.dao.game_state import GameState
from gym_idsgame.envs.dao.game_config import GameConfig

class DeterministicDefenseBotAgent(BotAgent):
    """
    Implements a deterministic defense policy: a policy where the defender always defends the same node using the same
    defense type
    """
    def __init__(self, game_config: GameConfig, defend_type: int, defend_row: int, defend_col: int):
        """
        Class constructor, initializes the policy

        :param game_config: configuration of the game
        :param defend_type: the type of defense
        :param defend_row: the row in the grid to defend
        :param defend_col: the columns in the grid to defend
        """
        super(DeterministicDefenseBotAgent, self).__init__(game_config)
        self.defend_type = defend_type
        self.defend_row = defend_row
        self.defend_col = defend_col
        if defend_type is None:
            raise ValueError("defend_type cannot be None for a Deterministic Defense Policy")
        if defend_row is None:
            raise ValueError("defend_row cannot be None for a Deterministic Defense Policy")
        if defend_col is None:
            raise ValueError("defend_row cannot be None for a Deterministic Defense Policy")
        self.defende_node_id = self.game_config.network_config.get_node_id((self.defend_row, self.defend_col))

    def action(self, render_state: GameState) -> Union[int, int, int, int]:
        """
        Returns the action from the deterministic policy

        :param render_state: the current render state
        :return: (row, col, defend_type, defend_node_id)
        """
        return self.defend_row, self.defend_col, self.defend_type, self.defende_node_id

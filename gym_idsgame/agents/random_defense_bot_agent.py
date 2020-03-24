"""
A random defense agent for the gym-idsgame environment
"""
from typing import Union
import numpy as np
from gym_idsgame.agents.bot_agent import BotAgent
from gym_idsgame.envs.dao.game_state import GameState
from gym_idsgame.envs.dao.game_config import GameConfig
from gym_idsgame.envs.util import idsgame_util

class RandomDefenseBotAgent(BotAgent):
    """
    Class implementing a random defense policy: a policy where the defender selects a random node and random
    defense type in each iteration
    """

    def __init__(self, game_config: GameConfig):
        """
        Constructor, initializes the policy

        :param game_config: the game configuration
        """
        super(RandomDefenseBotAgent, self).__init__(game_config)


    def action(self, game_state: GameState) -> int:
        """
        Samples an action from the policy

        :param game_state: the game state
        :return: action_id
        """
        actions = list(range(self.game_config.num_actions))
        legal_actions = list(filter(lambda action: idsgame_util.is_defense_id_legal(action), actions))
        action_id = np.random.choice(legal_actions)
        return action_id

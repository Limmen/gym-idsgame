"""
A bot attack agent for the gym-idsgame environment that acts greedily according to a fixed Q-table
"""
import numpy as np
from gym_idsgame.agents.bot_agent import BotAgent
from gym_idsgame.envs.dao.game_state import GameState
from gym_idsgame.envs.dao.game_config import GameConfig
import gym_idsgame.envs.util.idsgame_util as util

class TabularQAttackerBotAgent(BotAgent):
    """
    Class implementing an attack policy that acts greedily according to a given Q-table
    """

    def __init__(self, game_config: GameConfig, q_table_path: str = None):
        """
        Constructor, initializes the policy

        :param game_config: the game configuration
        """
        super(TabularQAttackerBotAgent, self).__init__(game_config)
        if q_table_path is None:
            raise ValueError("Cannot create a TabularQAttackerBotAgent without specifying the path to the Q-table")
        self.q_table_path = q_table_path
        self.Q = np.load(q_table_path)

    def action(self, game_state: GameState) -> int:
        """
        Samples an action from the policy.

        :param game_state: the game state
        :return: action_id
        """
        actions = list(range(self.game_config.num_attack_actions))
        legal_actions = list(filter(lambda action: util.is_attack_id_legal(
            action, self.game_config, game_state.attacker_pos), actions))
        s = self.game_config.network_config.get_node_id(game_state.attacker_pos)
        max_legal_action_value = float("-inf")
        max_legal_action = float("-inf")
        for i in range(len(self.Q[s])):
            if i in legal_actions and self.Q[s][i] > max_legal_action_value:
                max_legal_action_value = self.Q[s][i]
                max_legal_action = i
        if max_legal_action == float("-inf") or max_legal_action_value == float("-inf"):
            raise AssertionError("Error when selecting action greedily according to the Q-function")
        return max_legal_action

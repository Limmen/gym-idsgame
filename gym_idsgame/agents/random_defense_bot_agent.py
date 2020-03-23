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


    def action(self, game_state: GameState) -> Union[int, int, int, int]:
        """
        Samples an action from the policy

        :param game_state: the current state
        :return: (row, col, defense_type, target_node_id)
        """
        actions = list(range(self.game_config.num_actions))
        legal_actions = list(filter(lambda action: idsgame_util.is_defense_id_legal(action), actions))
        action = np.random.choice(legal_actions)
        target_node_id, target_pos, attack_type = idsgame_util.interpret_attack(action, self.game_config)
        target_row, target_col = target_pos
        return target_row, target_col, attack_type, target_node_id

        # defend_type = np.random.randint(game_state.defense_values.shape[1])
        # defend_row = None
        # defend_col = None
        # for row in range(self.game_config.network_config.graph_layout.shape[0]):
        #     for col in range(self.game_config.network_config.graph_layout.shape[1]):
        #         if (self.game_config.network_config.graph_layout[row][col] == NodeType.SERVER.value
        #                 or self.game_config.network_config.graph_layout[row][col] == NodeType.DATA.value):
        #             if defend_row is None or defend_col is None:
        #                 defend_row, defend_col = row, col
        #             else:
        #                 if np.random.rand() >= 0.5:
        #                     defend_row, defend_col = row, col
        # if defend_row is None or defend_col is None:
        #     raise AssertionError("Invalid Network Config, could not find any node to defend")
        # return defend_row, defend_col, defend_type
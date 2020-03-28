"""
A deterministic bot attack agent for the gym-idsgame environment that always attacks the node with the maximal
attack value.
"""
import numpy as np
from gym_idsgame.agents.bot_agent import BotAgent
from gym_idsgame.envs.dao.game_state import GameState
from gym_idsgame.envs.dao.game_config import GameConfig
from gym_idsgame.envs.util import idsgame_util
from gym_idsgame.envs.dao.node_type import NodeType

class AttackMaximalValueBotAgent(BotAgent):
    """
    Class implementing a deterministic attack policy that always attacks the node with the maximal attack value
    """

    def __init__(self, game_config: GameConfig):
        """
        Constructor, initializes the policy

        :param game_config: the game configuration
        """
        super(AttackMaximalValueBotAgent, self).__init__(game_config)

    def action(self, game_state: GameState) -> int:
        """
        Samples an action from the policy.

        :param game_state: the game state
        :return: action_id
        """
        actions = list(range(self.game_config.num_attack_actions))
        legal_actions = list(filter(lambda action: idsgame_util.is_attack_id_legal(action, self.game_config,
                                                                                   game_state.attacker_pos), actions))
        attacker_row, attacker_col = game_state.attacker_pos
        max_node_value = float("-inf")
        max_action_id = -1
        for id, node in enumerate(self.game_config.network_config.node_list):
            if node == NodeType.SERVER.value or node == NodeType.DATA.value:
                max_idx = np.argmax(game_state.attack_values[id])
                action_id = idsgame_util.get_attack_action_id(id, max_idx, self.game_config)
                node_row, node_col = self.game_config.network_config.get_node_pos(id)
                if game_state.attack_values[id][max_idx] > max_node_value and action_id in legal_actions and \
                        node_row < attacker_row:
                    max_node_value = game_state.attack_values[id][max_idx]
                    max_action_id = action_id
        if max_action_id == -1:
            max_action_id = np.random.choice(legal_actions)
        return max_action_id

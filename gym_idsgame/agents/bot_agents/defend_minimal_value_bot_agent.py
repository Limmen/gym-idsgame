"""
A deterministic bot defense agent for the gym-idsgame environment that always defends the node with the minimal
value.
"""
import numpy as np
from gym_idsgame.agents.bot_agents.bot_agent import BotAgent
from gym_idsgame.envs.dao.game_state import GameState
from gym_idsgame.envs.dao.game_config import GameConfig
from gym_idsgame.envs.util import idsgame_util
from gym_idsgame.envs.dao.node_type import NodeType

class DefendMinimalValueBotAgent(BotAgent):
    """
    Class implementing a deterministic defense policy that always defends the node with the minial defense
    """

    def __init__(self, game_config: GameConfig):
        """
        Constructor, initializes the policy

        :param game_config: the game configuration
        """
        super(DefendMinimalValueBotAgent, self).__init__(game_config)

    def action(self, game_state: GameState) -> int:
        """
        Samples an action from the policy.

        :param game_state: the game state
        :return: action_id
        """
        actions = list(range(self.game_config.num_defense_actions))
        legal_actions = list(filter(lambda action: idsgame_util.is_defense_id_legal(action, self.game_config,
                                                                                    game_state), actions))
        min_node_value = float("inf")
        min_action_id = -1
        min_action_ids = []
        for id, node in enumerate(self.game_config.network_config.node_list):
            if node == NodeType.SERVER.value or node == NodeType.DATA.value:
                min_idx = np.argmin(game_state.defense_values[id])
                if game_state.defense_det[id] < game_state.defense_values[id][min_idx]:
                    action_id = idsgame_util.get_defense_action_id(id, self.game_config.num_attack_types,
                                                                   self.game_config)
                    if game_state.defense_det[id] < min_node_value and action_id in legal_actions:
                        min_node_value = game_state.defense_det[id]
                        min_action_id = action_id
                        min_action_ids = []
                    elif game_state.defense_det[id] < min_node_value and action_id in legal_actions:
                        min_action_ids.append(action_id)

                else:
                    action_id = idsgame_util.get_defense_action_id(id, min_idx, self.game_config)
                    if game_state.defense_values[id][min_idx] < min_node_value and action_id in legal_actions:
                        min_node_value = game_state.defense_values[id][min_idx]
                        min_action_id = action_id
                        min_action_ids = []
                    elif game_state.defense_values[id][min_idx] == min_node_value and action_id in legal_actions:
                        min_action_ids.append(action_id)
        if min_action_ids != -1 and len(min_action_ids) > 1:
          min_action_id = np.random.choice(min_action_ids)
        elif min_action_id == -1:
            if len(legal_actions) == 0:
                min_action_id = np.random.choice(actions)
            else:
                min_action_id = np.random.choice(legal_actions)
        return min_action_id

"""
A deterministic bot attack agent for the gym-idsgame environment that always attacks the node with the maximal
attack value.
"""
import numpy as np
from gym_idsgame.agents.bot_agents.bot_agent import BotAgent
from gym_idsgame.envs.dao.game_state import GameState
from gym_idsgame.envs.dao.game_config import GameConfig
from gym_idsgame.envs.dao.node_type import NodeType

class AttackMaximalValueBotAgent(BotAgent):
    """
    Class implementing a deterministic attack policy that always attacks the node with the maximal attack value
    """

    def __init__(self, game_config: GameConfig, env):
        """
        Constructor, initializes the policy

        :param game_config: the game configuration
        """
        super(AttackMaximalValueBotAgent, self).__init__(game_config)
        self.idsgame_env = env

    def action(self, game_state: GameState) -> int:
        """
        Samples an action from the policy.

        :param game_state: the game state
        :return: action_id
        """
        if not self.game_config.reconnaissance_actions:
            return self.non_rec_action(game_state)
        else:
            return self.rec_action(game_state)

    def rec_action(self, game_state: GameState) -> int:
        import gym_idsgame.envs.util.idsgame_util as util
        attacker_obs = game_state.get_attacker_observation(
            self.game_config.network_config, local_view=self.idsgame_env.local_view_features(),
            reconnaissance=self.game_config.reconnaissance_actions,
            reconnaissance_bool_features=self.idsgame_env.idsgame_config.reconnaissance_bool_features)
        actions = list(range(self.idsgame_env.num_attack_actions))
        # non_legal_actions = list(
        #     filter(lambda action: not self.is_attack_legal(action, attacker_obs, game_state), actions))
        legal_actions = list(
            filter(lambda action: self.is_attack_legal(action, attacker_obs, game_state), actions))
        min_attack_value = float("inf")
        min_action_id = -1
        for action in legal_actions:
            if self.idsgame_env.local_view_features():
                global_action_id = self.convert_local_attacker_action_to_global(action, attacker_obs)
            else:
                global_action_id = action
            server_id, server_pos, attack_type, reconnaissance = util.interpret_attack_action(global_action_id, self.game_config)
            if reconnaissance and server_id not in game_state.reconnaissance_actions:
                return global_action_id
            if not reconnaissance and server_id in game_state.reconnaissance_actions:
                attack_value = game_state.reconnaissance_state[server_id][attack_type] - game_state.attack_values[server_id][attack_type]
                if attack_value < min_attack_value:
                    min_action_id = global_action_id
                    min_attack_value = attack_value
        if min_action_id == -1:
            if len(legal_actions) > 0:
                min_action_id = np.random.choice(legal_actions)
            else:
                min_action_id = np.random.choice(actions)
        return min_action_id

    def is_attack_legal(self, action, obs, game_state):
        import gym_idsgame.envs.util.idsgame_util as util
        if self.idsgame_env.local_view_features():
            action = self.convert_local_attacker_action_to_global(action, obs)
            if action == -1:
                return False
        return util.is_attack_id_legal(action, self.game_config,
                                       game_state.attacker_pos, game_state, [])

    def convert_local_attacker_action_to_global(self, action_id, attacker_obs):
        num_attack_types = self.idsgame_env.idsgame_config.game_config.num_attack_types
        neighbor = action_id // (num_attack_types + 1)
        attack_type = action_id % (num_attack_types + 1)
        target_id = int(attacker_obs[neighbor][num_attack_types])
        if target_id == -1:
            return -1
        attacker_action = target_id * (num_attack_types + 1) + attack_type
        return attacker_action

    def non_rec_action(self, game_state: GameState) -> int:
        """
        Samples an action from the policy.

        :param game_state: the game state
        :return: action_id
        """
        import gym_idsgame.envs.util.idsgame_util as util
        actions = list(range(self.game_config.num_attack_actions))
        legal_actions = list(filter(lambda action: util.is_attack_id_legal(action, self.game_config,
                                                                                   game_state.attacker_pos,
                                                                                   game_state), actions))
        attacker_row, attacker_col = game_state.attacker_pos
        max_node_value = float("-inf")
        max_action_id = -1
        for id, node in enumerate(self.game_config.network_config.node_list):
            if node == NodeType.SERVER.value or node == NodeType.DATA.value:
                max_idx = np.argmax(game_state.attack_values[id])
                action_id = util.get_attack_action_id(id, max_idx, self.game_config)
                node_row, node_col = self.game_config.network_config.get_node_pos(id)
                if game_state.attack_values[id][max_idx] > max_node_value and action_id in legal_actions and \
                        node_row < attacker_row:
                    max_node_value = game_state.attack_values[id][max_idx]
                    max_action_id = action_id
        if max_action_id == -1:
            if len(legal_actions) > 0:
                max_action_id = np.random.choice(legal_actions)
            else:
                max_action_id = np.random.choice(actions)
        return max_action_id

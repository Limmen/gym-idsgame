"""
A random atatck agent for the gym-idsgame environment
"""
import numpy as np
from gym_idsgame.agents.bot_agents.bot_agent import BotAgent
from gym_idsgame.envs.dao.game_state import GameState
from gym_idsgame.envs.dao.game_config import GameConfig

class RandomAttackBotAgent(BotAgent):
    """
    Class implementing a random attack policy: a policy where the attacker selects a random node out of its neighbors
    and a random attack type in each iteration
    """

    def __init__(self, game_config: GameConfig, env):
        """
        Constructor, initializes the policy

        :param game_config: the game configuration
        """
        super(RandomAttackBotAgent, self).__init__(game_config)
        self.idsgame_env = env

    def action(self, game_state: GameState) -> int:
        """
        Samples an action from the policy

        :param game_state: the game state
        :return: action_id
        """
        from gym_idsgame.envs.util import idsgame_util
        actions = list(range(self.game_config.num_attack_actions))
        if not self.game_config.reconnaissance_actions:
            legal_actions = list(filter(lambda action: idsgame_util.is_attack_id_legal(action,
                                                                                       self.game_config,
                                                                                       game_state.attacker_pos,
                                                                                       game_state), actions))
            if len(legal_actions) > 0:
                action = np.random.choice(legal_actions)
            else:
                action = np.random.choice(actions)
        else:
            attacker_obs = game_state.get_attacker_observation(
                self.game_config.network_config, local_view=self.idsgame_env.local_view_features(),
                reconnaissance=self.game_config.reconnaissance_actions,
                reconnaissance_bool_features=self.idsgame_env.idsgame_config.reconnaissance_bool_features)
            legal_actions = list(
                filter(lambda action: self.is_attack_legal(action, attacker_obs, game_state), actions))
            if len(legal_actions) > 0:
                action = np.random.choice(legal_actions)
            else:
                action = np.random.choice(actions)
            if self.idsgame_env.local_view_features():
                action = self.convert_local_attacker_action_to_global(action, attacker_obs)
        return action

    def is_attack_legal(self, action, obs, game_state):
        if self.idsgame_env.local_view_features():
            action = self.convert_local_attacker_action_to_global(action, obs)
            if action == -1:
                return False
        return idsgame_util.is_attack_id_legal(action, self.game_config,
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

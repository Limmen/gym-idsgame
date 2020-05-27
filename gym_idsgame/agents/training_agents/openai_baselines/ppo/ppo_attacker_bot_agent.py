"""
A bot attack agent for the gym-idsgame environment that acts greedily according to a pre-trained policy network
"""
import numpy as np
import torch
from gym_idsgame.agents.bot_agents.bot_agent import BotAgent
from gym_idsgame.envs.dao.game_state import GameState
from gym_idsgame.envs.dao.game_config import GameConfig
from gym_idsgame.agents.training_agents.policy_gradient.pg_agent_config import PolicyGradientAgentConfig
from gym_idsgame.agents.training_agents.openai_baselines.common.ppo.ppo import PPO
from gym_idsgame.envs.idsgame_env import IdsGameEnv
import gym_idsgame.envs.util.idsgame_util as util

class PPOBaselineAttackerBotAgent(BotAgent):
    """
    Class implementing an attack policy that acts greedily according to a given Q-table
    """

    def __init__(self, pg_config: PolicyGradientAgentConfig, game_config: GameConfig, model_path: str = None,
                 env: IdsGameEnv = None):
        """
        Constructor, initializes the policy

        :param game_config: the game configuration
        """
        super(PPOBaselineAttackerBotAgent, self).__init__(game_config)
        if model_path is None:
            raise ValueError("Cannot create a PPOBaselineAttackerBotAgent without specifying the path to the model")
        self.idsgame_env = env
        self.config = pg_config
        self.model_path = model_path
        self.initialize_models()


    def initialize_models(self) -> None:
        """
        Initialize models
        :return: None
        """
        policy = "MlpPolicy"
        if self.config.cnn_feature_extractor:
            policy = "CnnPolicy"
        # Initialize models
        self.model = PPO.load(self.config.attacker_load_path, policy)

    def action(self, game_state: GameState) -> int:
        """
        Samples an action from the policy.

        :param game_state: the game state
        :return: action_id
        """
        # Feature engineering
        attacker_obs = game_state.get_attacker_observation(self.game_config.network_config, local_view=False,
                                                           reconnaissance=self.game_config.reconnaissance_actions)
        defender_obs = game_state.get_defender_observation(self.game_config.network_config)
        attacker_state = self.update_state(attacker_obs=attacker_obs, defender_obs=defender_obs, state=[],
                                           attacker=True)
        actions = list(range(self.idsgame_env.num_attack_actions))
        non_legal_actions = list(filter(lambda action: not util.is_attack_id_legal(action, self.game_config,
                                                                                   game_state.attacker_pos,
                                                                                   game_state, []), actions))
        obs_tensor_a = torch.as_tensor(attacker_state.flatten())
        attacker_actions, attacker_values, attacker_log_probs = self.model.attacker_policy.forward(
            obs_tensor_a, self.idsgame_env, device="cpu", attacker=True, non_legal_actions=non_legal_actions)
        return attacker_actions.item()


    def update_state(self, attacker_obs: np.ndarray = None, defender_obs: np.ndarray = None,
                     state: np.ndarray = None, attacker: bool = True, game_state : GameState = None) -> np.ndarray:
        """
        Update approximative Markov state

        :param attacker_obs: attacker obs
        :param defender_obs: defender observation
        :param state: current state
        :param attacker: boolean flag whether it is attacker or not
        :return: new state
        """
        if attacker and self.game_config.reconnaissance_actions:
            a_obs_len = self.game_config.num_attack_types + 1
            defender_obs = attacker_obs[:, a_obs_len:]
            attacker_obs = attacker_obs[:, 0:a_obs_len]

        if not attacker and self.idsgame_env.local_view_features():
            attacker_obs = game_state.get_attacker_observation(
                self.game_config.network_config,
                local_view=False,
            reconnaissance=self.idsgame_env.idsgame_config.reconnaissance_actions)

        # Zero mean
        if self.config.zero_mean_features:
            if not self.idsgame_env.local_view_features() or not attacker:
                attacker_obs_1 = attacker_obs[:, 0:-1]
            else:
                attacker_obs_1 = attacker_obs[:, 0:-2]
            zero_mean_attacker_features = []
            for idx, row in enumerate(attacker_obs_1):
                mean = np.mean(row)
                if mean != 0:
                    t = row - mean
                else:
                    t = row
                if np.isnan(t).any():
                    t = attacker_obs[idx]
                else:
                    t = t.tolist()
                    if not self.idsgame_env.local_view_features() or not attacker:
                        t.append(attacker_obs[idx][-1])
                    else:
                        t.append(attacker_obs[idx][-2])
                        t.append(attacker_obs[idx][-1])
                zero_mean_attacker_features.append(t)

            defender_obs_1 = defender_obs[:, 0:-1]
            zero_mean_defender_features = []
            for idx, row in enumerate(defender_obs_1):
                mean = np.mean(row)
                if mean != 0:
                    t = row - mean
                else:
                    t = row
                if np.isnan(t).any():
                    t = defender_obs[idx]
                else:
                    t = t.tolist()
                    t.append(defender_obs[idx][-1])
                zero_mean_defender_features.append(t)

            attacker_obs = np.array(zero_mean_attacker_features)
            defender_obs = np.array(zero_mean_defender_features)

        # Normalize
        if self.config.normalize_features:
            if not self.idsgame_env.local_view_features() or not attacker:
                attacker_obs_1 = attacker_obs[:, 0:-1] / np.linalg.norm(attacker_obs[:, 0:-1])
            else:
                attacker_obs_1 = attacker_obs[:, 0:-2] / np.linalg.norm(attacker_obs[:, 0:-2])
            normalized_attacker_features = []
            for idx, row in enumerate(attacker_obs_1):
                if np.isnan(attacker_obs_1).any():
                    t = attacker_obs[idx]
                else:
                    t = row.tolist()
                    if not self.idsgame_env.local_view_features() or not attacker:
                        t.append(attacker_obs[idx][-1])
                    else:
                        t.append(attacker_obs[idx][-2])
                        t.append(attacker_obs[idx][-1])
                normalized_attacker_features.append(t)

            if attacker and self.game_config.reconnaissance_actions:
                defender_obs_1 = defender_obs[:, 0:-1] / np.linalg.norm(defender_obs[:, 0:-1])
            else:
                defender_obs_1 = defender_obs / np.linalg.norm(defender_obs)
            normalized_defender_features = []
            for idx, row in enumerate(defender_obs_1):
                if np.isnan(defender_obs_1).any():
                    t = defender_obs[idx]
                else:
                    if attacker and self.game_config.reconnaissance_actions:
                        t = row.tolist()
                        t.append(defender_obs[idx][-1])
                    else:
                        t = row

                normalized_defender_features.append(t)

            attacker_obs = np.array(normalized_attacker_features)
            defender_obs = np.array(normalized_defender_features)

        if self.idsgame_env.local_view_features() and attacker:
            neighbor_defense_attributes = np.zeros((attacker_obs.shape[0], defender_obs.shape[1]))
            for node in range(attacker_obs.shape[0]):
                if int(attacker_obs[node][-1]) == 1:
                    id = int(attacker_obs[node][-2])
                    neighbor_defense_attributes[node] = defender_obs[id]

        if self.idsgame_env.fully_observed() or \
                (self.game_config.reconnaissance_actions and attacker):
            if self.config.merged_ad_features:
                if not self.idsgame_env.local_view_features() or not attacker:
                    a_pos = attacker_obs[:, -1]
                    if not self.game_config.reconnaissance_actions:
                        det_values = defender_obs[:, -1]
                        temp = defender_obs[:, 0:-1] - attacker_obs[:, 0:-1]
                    else:
                        temp = defender_obs[:, 0:] - attacker_obs[:, 0:-1]
                    features = []
                    for idx, row in enumerate(temp):
                        t = row.tolist()
                        t.append(a_pos[idx])
                        if not self.game_config.reconnaissance_actions:
                            t.append(det_values[idx])
                        features.append(t)
                else:
                    node_ids = attacker_obs[:, -2]
                    node_reachable = attacker_obs[:, -1]
                    if not self.game_config.reconnaissance_actions:
                        det_values = neighbor_defense_attributes[:, -1]
                    temp = neighbor_defense_attributes[:, 0:-1] - attacker_obs[:, 0:-2]
                    features = []
                    for idx, row in enumerate(temp):
                        t = row.tolist()
                        t.append(node_ids[idx])
                        t.append(node_reachable[idx])
                        if not self.game_config.reconnaissance_actions:
                            t.append(det_values[idx])
                        features.append(t)
                features = np.array(features)
                if self.config.state_length == 1:
                    return features
                if len(state) == 0:
                    s = np.array([features] * self.config.state_length)
                    return s
                state = np.append(state[1:], np.array([features]), axis=0)
            else:
                if self.config.state_length == 1:
                    if not self.idsgame_env.local_view_features() or not attacker:
                        if self.idsgame_env.idsgame_config.game_config.reconnaissance_actions and attacker:
                            combined_features = []
                            for idx, row in enumerate(attacker_obs):
                                combined_row = np.append(row, defender_obs[idx])
                                combined_features.append(combined_row)
                            return np.array(combined_features)
                            return np.append(attacker_obs, defender_obs)

                        return np.append(attacker_obs, defender_obs)
                    else:
                        return np.append(attacker_obs, neighbor_defense_attributes)
                if len(state) == 0:
                    if not self.idsgame_env.local_view_features() or not attacker:
                        temp = np.append(attacker_obs, defender_obs)
                    else:
                        temp = np.append(attacker_obs, neighbor_defense_attributes)
                    s = np.array([temp] * self.pg_agent_config.state_length)
                    return s
                if not self.idsgame_env.local_view_features() or not attacker:
                    temp = np.append(attacker_obs, defender_obs)
                else:
                    temp = np.append(attacker_obs, neighbor_defense_attributes)
                state = np.append(state[1:], np.array([temp]), axis=0)
            return state
        else:
            if self.config.state_length == 1:
                if attacker:
                    return np.array(attacker_obs)
                else:
                    return np.array(defender_obs)
            if len(state) == 0:
                if attacker:
                    return np.array([attacker_obs] * self.config.state_length)
                else:
                    return np.array([defender_obs] * self.config.state_length)
            if attacker:
                state = np.append(state[1:], np.array([attacker_obs]), axis=0)
            else:
                state = np.append(state[1:], np.array([defender_obs]), axis=0)
            return state



"""
A bot defender agent for the gym-idsgame environment that acts greedily according to a pre-trained policy network
"""
import numpy as np
import torch
import traceback
from sklearn import preprocessing
from gym_idsgame.agents.bot_agents.bot_agent import BotAgent
from gym_idsgame.envs.dao.game_state import GameState
from gym_idsgame.envs.dao.game_config import GameConfig
from gym_idsgame.agents.training_agents.policy_gradient.pg_agent_config import PolicyGradientAgentConfig
from gym_idsgame.agents.training_agents.openai_baselines.common.ppo.ppo import PPO
from gym_idsgame.envs.idsgame_env import IdsGameEnv
import gym_idsgame.envs.util.idsgame_util as util

class PPOBaselineDefenderBotAgent(BotAgent):
    """
    Class implementing an defense policy that acts greedily according to a given policy network
    """

    def __init__(self, pg_config: PolicyGradientAgentConfig, game_config: GameConfig, model_path: str = None,
                 env: IdsGameEnv = None):
        """
        Constructor, initializes the policy

        :param game_config: the game configuration
        """
        super(PPOBaselineDefenderBotAgent, self).__init__(game_config)
        if model_path is None:
            raise ValueError("Cannot create a PPOBaselineDefenderBotAgent without specifying the path to the model")
        self.idsgame_env = env
        self.config = pg_config
        self.model_path = model_path
        self.initialize_models()
        self.device = "cpu" if not self.config.gpu else "cuda:" + str(self.config.gpu_id)


    def initialize_models(self) -> None:
        """
        Initialize models
        :return: None
        """
        policy = "MlpPolicy"
        if self.config.cnn_feature_extractor:
            policy = "CnnPolicy"
        # Initialize models
        self.model = PPO.load(self.config.defender_load_path, policy, pg_agent_config=self.config)

    def action(self, game_state: GameState) -> int:
        """
        Samples an action from the policy.

        :param game_state: the game state
        :return: action_id
        """
        try:
            # Feature engineering
            attacker_obs = game_state.get_attacker_observation(
                self.game_config.network_config, local_view=self.idsgame_env.local_view_features(),
                reconnaissance=self.game_config.reconnaissance_actions,
                reconnaissance_bool_features=self.idsgame_env.idsgame_config.reconnaissance_bool_features)
            defender_obs = game_state.get_defender_observation(self.game_config.network_config)
            defender_state = self.update_state(attacker_obs=attacker_obs, defender_obs=defender_obs, state=[],
                                               attacker=False)
            if not self.config.ar_policy:
                actions = list(range(self.idsgame_env.num_defense_actions))
                non_legal_actions = list(filter(lambda action: not self.idsgame_env.is_defense_legal(action), actions))
                obs_tensor_d = torch.as_tensor(defender_state.flatten()).to(self.device)
                defender_actions, defender_values, defender_log_probs = self.model.defender_policy.forward(
                    obs_tensor_d, self.idsgame_env, device=self.device, attacker=False, non_legal_actions=non_legal_actions)
                defender_actions = defender_actions.item()
            else:
                actions = list(range(self.config.defender_node_net_output_dim))
                non_legal_actions = list(
                    filter(lambda action: not self.is_defense_legal(action, node=True, game_state=game_state), actions))
                if len(non_legal_actions) == len(actions):
                    non_legal_actions = []
                obs_tensor_d = torch.as_tensor(defender_state.flatten()).to(self.device)
                defender_node_actions, defender_node_values, defender_node_log_probs, defender_node_lstm_state = self.model.defender_node_policy.forward(
                    obs_tensor_d, self.idsgame_env, device=self.device, attacker=False,
                    non_legal_actions=non_legal_actions)
                defender_node_actions = defender_node_actions.cpu().numpy()
                node = defender_node_actions[0]
                obs_tensor_d_1 = obs_tensor_d.reshape(self.idsgame_env.idsgame_config.game_config.num_nodes,
                                                      self.config.defender_at_net_input_dim)
                obs_tensor_d_at = obs_tensor_d_1[node]
                actions = list(range(self.config.defender_at_net_output_dim))
                non_legal_actions = list(
                    filter(lambda action: not self.is_defense_legal(action, node=False, game_state=game_state, obs=obs_tensor_d_at), actions))
                if len(non_legal_actions) == len(actions):
                    non_legal_actions = []
                defender_at_actions, defender_at_values, defender_at_log_probs, defender_at_lstm_state = self.model.defender_at_policy.forward(
                    obs_tensor_d_at, self.idsgame_env, device=self.device, attacker=False, non_legal_actions=non_legal_actions)
                defender_at_actions = defender_at_actions.cpu().numpy()
                attack_id = util.get_defense_action_id(node, defender_at_actions[0], self.idsgame_env.idsgame_config.game_config)
                defender_actions = attack_id
        except Exception as e:
            print(str(e))
            traceback.print_exc()

        return defender_actions

    def is_defense_legal(self, defense_action: int, node: bool = False, obs : torch.Tensor = None,
                         game_state :GameState = None) -> bool:
        """
        Check if a given defense is legal or not.

        :param defense_action: the defense action to verify
        :return: True if legal otherwise False
        """
        if not self.config.ar_policy:
            return self.idsgame_env.is_defense_legal(defense_action)
        else:
            if node:
                return util.is_node_defense_legal(defense_action, self.game_config.network_config, game_state,
                                                  self.idsgame_env.idsgame_config.game_config.max_value)
            else:
                if obs is not None:
                    if obs[defense_action] >= self.idsgame_env.idsgame_config.game_config.max_value:
                        return False
                return True

    def update_state(self, attacker_obs: np.ndarray = None, defender_obs: np.ndarray = None,
                     state: np.ndarray = None, attacker: bool = True) -> np.ndarray:
        """
        Update approximative Markov state

        :param attacker_obs: attacker obs
        :param defender_obs: defender observation
        :param state: current state
        :param attacker: boolean flag whether it is attacker or not
        :return: new state
        """
        if attacker and self.idsgame_env.idsgame_config.game_config.reconnaissance_actions:
            a_obs_len = self.idsgame_env.idsgame_config.game_config.num_attack_types + 1
            defender_obs = attacker_obs[:,
                           a_obs_len:a_obs_len + self.idsgame_env.idsgame_config.game_config.num_attack_types]
            if self.idsgame_env.idsgame_config.reconnaissance_bool_features:
                d_bool_features = attacker_obs[:,
                                  a_obs_len + self.idsgame_env.idsgame_config.game_config.num_attack_types:]
            attacker_obs = attacker_obs[:, 0:a_obs_len]

        if not attacker and self.idsgame_env.local_view_features():
            attacker_obs = self.idsgame_env.state.get_attacker_observation(
                self.idsgame_env.idsgame_config.game_config.network_config,
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

            if attacker and self.idsgame_env.idsgame_config.game_config.reconnaissance_actions:
                defender_obs_1 = defender_obs[:, 0:-1] / np.linalg.norm(defender_obs[:, 0:-1])
            else:
                defender_obs_1 = defender_obs / np.linalg.norm(defender_obs)
            normalized_defender_features = []
            for idx, row in enumerate(defender_obs_1):
                if np.isnan(defender_obs_1).any():
                    t = defender_obs[idx]
                else:
                    if attacker and self.idsgame_env.idsgame_config.game_config.reconnaissance_actions:
                        t = row.tolist()
                        t.append(defender_obs[idx][-1])
                    else:
                        t = row

                normalized_defender_features.append(t)

            attacker_obs = np.array(normalized_attacker_features)
            defender_obs = np.array(normalized_defender_features)

        if self.idsgame_env.local_view_features() and attacker:
            if not self.idsgame_env.idsgame_config.game_config.reconnaissance_actions:
                neighbor_defense_attributes = np.zeros((attacker_obs.shape[0], defender_obs.shape[1]))
                for node in range(attacker_obs.shape[0]):
                    id = int(attacker_obs[node][-1])
                    neighbor_defense_attributes[node] = defender_obs[id]
            else:
                neighbor_defense_attributes = defender_obs

        if self.idsgame_env.fully_observed() or \
                (self.idsgame_env.idsgame_config.game_config.reconnaissance_actions and attacker):
            if self.config.merged_ad_features:
                if not self.idsgame_env.local_view_features() or not attacker:
                    a_pos = attacker_obs[:, -1]
                    if not self.idsgame_env.idsgame_config.game_config.reconnaissance_actions:
                        det_values = defender_obs[:, -1]
                        temp = defender_obs[:, 0:-1] - attacker_obs[:, 0:-1]
                    else:
                        temp = defender_obs[:, 0:] - attacker_obs[:, 0:-1]
                    features = []
                    for idx, row in enumerate(temp):
                        t = row.tolist()
                        t.append(a_pos[idx])
                        if not self.idsgame_env.idsgame_config.game_config.reconnaissance_actions:
                            t.append(det_values[idx])
                        features.append(t)
                else:
                    node_ids = attacker_obs[:, -1]
                    # node_reachable = attacker_obs[:, -1]
                    if not self.idsgame_env.idsgame_config.game_config.reconnaissance_actions:
                        det_values = neighbor_defense_attributes[:, -1]
                    if not self.idsgame_env.idsgame_config.game_config.reconnaissance_actions:
                        temp = neighbor_defense_attributes[:, 0:-1] - attacker_obs[:, 0:-1]
                    else:
                        temp = np.full(neighbor_defense_attributes.shape, -1)
                        for i in range(len(neighbor_defense_attributes)):
                            if np.sum(neighbor_defense_attributes[i]) > 0:
                                temp[i] = neighbor_defense_attributes[i] - attacker_obs[i, 0:-1]
                    features = []
                    for idx, row in enumerate(temp):
                        t = row.tolist()
                        t.append(node_ids[idx])
                        # t.append(node_reachable[idx])
                        if not self.idsgame_env.idsgame_config.game_config.reconnaissance_actions:
                            t.append(det_values[idx])
                        features.append(t)
                features = np.array(features)
                if self.idsgame_env.idsgame_config.reconnaissance_bool_features:
                    f = np.zeros((features.shape[0], features.shape[1] + d_bool_features.shape[1]))
                    for i in range(features.shape[0]):
                        f[i] = np.append(features[i], d_bool_features[i])
                    features = f
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
                            if self.idsgame_env.idsgame_config.reconnaissance_bool_features:
                                combined_features = np.array(combined_features)
                                f = np.zeros(
                                    (combined_features.shape[0], combined_features.shape[1] + d_bool_features.shape[1]))
                                for i in range(combined_features.shape[0]):
                                    f[i] = np.append(combined_features[i], d_bool_features[i])
                                combined_features = f
                            return np.array(combined_features)

                        return np.append(attacker_obs, defender_obs)
                    else:
                        if self.idsgame_env.idsgame_config.reconnaissance_bool_features:
                            f = np.zeros((attacker_obs.shape[0],
                                          attacker_obs.shape[1] + neighbor_defense_attributes.shape[1] +
                                          d_bool_features.shape[1]))
                            for i in range(f.shape[0]):
                                f[i] = np.append(np.append(attacker_obs[i], neighbor_defense_attributes[i]),
                                                 d_bool_features[i])
                        else:
                            f = np.zeros((attacker_obs.shape[0],
                                          attacker_obs.shape[1] + neighbor_defense_attributes.shape[1]))
                            for i in range(f.shape[0]):
                                f[i] = np.append(attacker_obs[i], neighbor_defense_attributes[i])
                        return f
                if len(state) == 0:
                    if not self.idsgame_env.local_view_features() or not attacker:
                        temp = np.append(attacker_obs, defender_obs)
                    else:
                        temp = np.append(attacker_obs, neighbor_defense_attributes)
                    s = np.array([temp] * self.config.state_length)
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

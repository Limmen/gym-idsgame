"""
A bot attack agent for the gym-idsgame environment that acts greedily according to a pre-trained policy network
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

class PPOBaselineAttackerBotAgent(BotAgent):
    """
    Class implementing an attack policy that acts greedily according to a given policy network
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
        self.model = PPO.load(self.config.attacker_load_path, policy)

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
            attacker_state = self.update_state(attacker_obs=attacker_obs, defender_obs=defender_obs, state=[],
                                               attacker=True)
            actions = list(range(self.idsgame_env.num_attack_actions))
            non_legal_actions = list(filter(lambda action: not self.is_attack_legal(action, attacker_obs, game_state), actions))
            obs_tensor_a = torch.as_tensor(attacker_state.flatten()).to(self.device)
            attacker_actions, attacker_values, attacker_log_probs = self.model.attacker_policy.forward(
                obs_tensor_a, self.idsgame_env, device=self.device, attacker=True, non_legal_actions=non_legal_actions)
        except Exception as e:
            print(str(e))
            traceback.print_exc()

        if self.idsgame_env.local_view_features():
            attack = self.convert_local_attacker_action_to_global(attacker_actions.item(), attacker_obs)
            print("predicted attacK.{}, legal:{}".format(attack, self.idsgame_env.is_attack_legal(attack)))
            return attack
        else:
            return attacker_actions.item()

    def is_attack_legal(self, action, obs, game_state):
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

    def grid_obs(self, attacker_obs, defender_obs, attacker=True):
        if attacker and self.idsgame_env.idsgame_config.game_config.reconnaissance_actions:
            a_obs_len = self.idsgame_env.idsgame_config.game_config.num_attack_types + 1
            defender_obs = attacker_obs[:, a_obs_len:]
            attacker_obs = attacker_obs[:, 0:a_obs_len]
            attacker_position = attacker_obs[:, -1]
            attacker_obs = attacker_obs[:, 0:-1]
        elif self.idsgame_env.fully_observed():
            a_obs_len = self.idsgame_env.idsgame_config.game_config.num_attack_types + 1
            attacker_position = attacker_obs[:, -1]
            attacker_obs = attacker_obs[:, 0:-1]
            defender_obs = defender_obs[:, 0:-1]

        attack_plane = attacker_obs
        if self.config.normalize_features:
            normalized_attack_plane = preprocessing.normalize(attack_plane)

        defense_plane = defender_obs
        if self.config.normalize_features:
            normalized_defense_plane = preprocessing.normalize(defense_plane)

        position_plane = np.zeros(attack_plane.shape)
        for idx, present in enumerate(attacker_position):
            position_plane[idx] = np.full(position_plane.shape[1], present)

        reachable_plane = np.zeros(attack_plane.shape)
        attacker_row, attacker_col = self.idsgame_env.state.attacker_pos
        attacker_matrix_id = self.idsgame_env.idsgame_config.game_config.network_config.get_adjacency_matrix_id(
            attacker_row, attacker_col)
        for node_id in range(len(attack_plane)):
            node_row, node_col = self.idsgame_env.idsgame_config.game_config.network_config.get_node_pos(node_id)
            adj_matrix_id = self.idsgame_env.idsgame_config.game_config.network_config.get_adjacency_matrix_id(node_row,
                                                                                                               node_col)
            reachable = self.idsgame_env.idsgame_config.game_config.network_config.adjacency_matrix[attacker_matrix_id][
                            adj_matrix_id] == int(1)
            if reachable:
                val = 1
            else:
                val = 0
            reachable_plane[node_id] = np.full(reachable_plane.shape[1], val)

        row_difference_plane = np.zeros(attack_plane.shape)
        for node_id in range(len(attack_plane)):
            node_row, node_col = self.idsgame_env.idsgame_config.game_config.network_config.get_node_pos(node_id)
            row_difference = attacker_row - node_row
            row_difference_plane[node_id] = np.full(row_difference_plane.shape[1], row_difference)

        if self.config.normalize_features:
            normalized_row_difference_plance = preprocessing.normalize(row_difference_plane)

        attack_defense_difference_plane = attacker_obs - defender_obs
        if self.config.normalize_features:
            normalized_attack_defense_difference_plane = preprocessing.normalize(attack_defense_difference_plane)

        if self.config.normalize_features:
            feature_frames = np.stack(
                [normalized_attack_plane, normalized_defense_plane, position_plane, reachable_plane,
                 normalized_row_difference_plance,
                 normalized_attack_defense_difference_plane],
                axis=0)
        else:
            feature_frames = np.stack(
                [attack_plane, defense_plane, position_plane, reachable_plane,
                 row_difference_plane,
                 attack_defense_difference_plane],
                axis=0)
        # print("feature_frames:")
        # print(feature_frames)
        # raise AssertionError("test")
        return feature_frames

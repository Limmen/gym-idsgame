import gym
import numpy as np
from gym_idsgame.envs.dao.idsgame_config import IdsGameConfig
from gym_idsgame.agents.training_agents.policy_gradient.pg_agent_config import PolicyGradientAgentConfig


class BaselineEnvWrapper(gym.Env):

    def __init__(self, env_name: str, idsgame_config: IdsGameConfig = None, save_dir: str = None,
                 initial_state_path: str = None,
                 pg_agent_config: PolicyGradientAgentConfig = None):
        super(BaselineEnvWrapper, self).__init__()
        self.idsgame_env = gym.make(env_name, idsgame_config=idsgame_config,
                                    save_dir=save_dir,
                                    initial_state_path=initial_state_path)
        #self.idsgame_env.idsgame_config.render_config.attacker_view = True
        self.pg_agent_config = pg_agent_config
        self.attacker_action_space = self.idsgame_env.attacker_action_space
        self.defender_action_space = self.idsgame_env.defender_action_space
        self.attacker_observation_space = gym.spaces.Box(low=0,
                                                         high=self.idsgame_env.idsgame_config.game_config.max_value,
                                                         shape=self.pg_agent_config.input_dim_attacker,
                                                         dtype=np.float32)
        self.defender_observation_space = gym.spaces.Box(low=0,
                                                         high=self.idsgame_env.idsgame_config.game_config.max_value,
                                                         shape=self.pg_agent_config.input_dim_defender,
                                                         dtype=np.float32)
        self.prev_episode_hacked = False
        self.prev_episode_detected = False
        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': 50  # Video rendering speed
        }
        # self.observation_space = self.idsgame_env.pg_agent_config

    def step(self, action):
        attacker_action = action[0][0]
        defender_action = action[1][0]
        joint_action = (attacker_action, defender_action)
        obs_prime, reward, done, info = self.idsgame_env.step(joint_action)
        attacker_reward, defender_reward = reward
        obs_prime_attacker, obs_prime_defender = obs_prime
        if self.pg_agent_config.cnn_feature_extractor:
            attacker_state = self.grid_obs(obs_prime_attacker, obs_prime_defender, attacker=True)
            defender_state = self.grid_obs(obs_prime_attacker, obs_prime_defender, attacker=True)
            return [attacker_state, defender_state], [attacker_reward, defender_reward], done, info
        else:
            attacker_state = self.update_state(attacker_obs=obs_prime_attacker, defender_obs=obs_prime_defender, state=[],
                                               attacker=True)
            defender_state = self.update_state(defender_obs=obs_prime_defender, attacker_obs=obs_prime_attacker, state=[],
                                               attacker=False)
            return [attacker_state.flatten(), defender_state.flatten()], [attacker_reward, defender_reward], done, info

    def reset(self, update_stats: False):
        self.prev_episode_hacked = self.idsgame_env.state.hacked
        self.prev_episode_detected = self.idsgame_env.state.detected
        obs = self.idsgame_env.reset(update_stats=update_stats)
        obs_attacker, obs_defender = obs

        if self.pg_agent_config.cnn_feature_extractor:
            attacker_state = self.grid_obs(obs_attacker, obs_defender, attacker=True)
            defender_state = self.grid_obs(obs_attacker, obs_defender, attacker=True)
            return [attacker_state, defender_state]
        else:
            attacker_state = self.update_state(attacker_obs=obs_attacker, defender_obs=obs_defender, state=[],
                                               attacker=True)
            defender_state = self.update_state(defender_obs=obs_defender, attacker_obs=obs_attacker, state=[],
                                               attacker=False)
            # print("attacker state:{}".format(attacker_state))
            # print("attacker state shape:{}".format(attacker_state.shape))
            # print("defender state shape:{}".format(defender_state.shape))
            # print("defender state:{}".format(defender_state))
            return [attacker_state.flatten(), defender_state.flatten()]

    def render(self, mode='human'):
        return self.idsgame_env.render(mode=mode)

    def close(self):
        return self.idsgame_env.close()

    def is_attack_legal(self, attack_action: int) -> bool:
        """
        Check if a given attack is legal or not.

        :param attack_action: the attack to verify
        :return: True if legal otherwise False
        """
        return self.idsgame_env.is_attack_legal(attack_action)

    def is_defense_legal(self, defense_action: int) -> bool:
        """
        Check if a given defense is legal or not.

        :param defense_action: the defense action to verify
        :return: True if legal otherwise False
        """
        return self.idsgame_env.is_defense_legal(defense_action)

    def num_attack_actions(self):
        return self.idsgame_env.num_attack_actions

    def num_defense_actions(self):
        return self.idsgame_env.num_defense_actions

    def hack_probability(self):
        if self.num_games > 0:
            return self.num_hacks / self.num_games
        else:
            return 0.0

    def games(self):
        return self.num_games

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
            defender_obs = attacker_obs[:, a_obs_len:]
            attacker_obs = attacker_obs[:, 0:a_obs_len]

        if not attacker and self.idsgame_env.local_view_features():
            attacker_obs = self.idsgame_env.state.get_attacker_observation(
                self.idsgame_env.idsgame_config.game_config.network_config,
                local_view=False,
            reconnaissance=self.idsgame_env.idsgame_config.reconnaissance_actions)

        # Zero mean
        if self.pg_agent_config.zero_mean_features:
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
        if self.pg_agent_config.normalize_features:
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
            neighbor_defense_attributes = np.zeros((attacker_obs.shape[0], defender_obs.shape[1]))
            for node in range(attacker_obs.shape[0]):
                if int(attacker_obs[node][-1]) == 1:
                    id = int(attacker_obs[node][-2])
                    neighbor_defense_attributes[node] = defender_obs[id]

        if self.idsgame_env.fully_observed() or \
                (self.idsgame_env.idsgame_config.game_config.reconnaissance_actions and attacker):
            if self.pg_agent_config.merged_ad_features:
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
                    node_ids = attacker_obs[:, -2]
                    node_reachable = attacker_obs[:, -1]
                    if not self.idsgame_env.idsgame_config.game_config.reconnaissance_actions:
                        det_values = neighbor_defense_attributes[:, -1]
                    temp = neighbor_defense_attributes[:, 0:-1] - attacker_obs[:, 0:-2]
                    features = []
                    for idx, row in enumerate(temp):
                        t = row.tolist()
                        t.append(node_ids[idx])
                        t.append(node_reachable[idx])
                        if not self.idsgame_env.idsgame_config.game_config.reconnaissance_actions:
                            t.append(det_values[idx])
                        features.append(t)
                features = np.array(features)
                if self.pg_agent_config.state_length == 1:
                    return features
                if len(state) == 0:
                    s = np.array([features] * self.pg_agent_config.state_length)
                    return s
                state = np.append(state[1:], np.array([features]), axis=0)
            else:
                if self.pg_agent_config.state_length == 1:
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
            if self.pg_agent_config.state_length == 1:
                if attacker:
                    return np.array(attacker_obs)
                else:
                    return np.array(defender_obs)
            if len(state) == 0:
                if attacker:
                    return np.array([attacker_obs] * self.pg_agent_config.state_length)
                else:
                    return np.array([defender_obs] * self.pg_agent_config.state_length)
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
        if self.pg_agent_config.normalize_features:
            attack_plane_row_sums = attack_plane.sum(axis=1)
            normalized_attack_plane = attack_plane / attack_plane_row_sums[:, np.newaxis]
            normalized_attack_plane = np.nan_to_num(normalized_attack_plane, nan=0.0)

        defense_plane = defender_obs
        if self.pg_agent_config.normalize_features:
            defense_plane_row_sums = defense_plane.sum(axis=1)
            normalized_defense_plane = defense_plane / defense_plane_row_sums[:, np.newaxis]
            normalized_defense_plane = np.nan_to_num(normalized_defense_plane, nan=0.0)

        position_plane = np.zeros(attack_plane.shape)
        for idx, present in enumerate(attacker_position):
            position_plane[idx] = np.full(position_plane.shape[1], present)

        reachable_plane = np.zeros(attack_plane.shape)
        attacker_row, attacker_col = self.idsgame_env.state.attacker_pos
        attacker_matrix_id = self.idsgame_env.idsgame_config.game_config.network_config.get_adjacency_matrix_id(attacker_row, attacker_col)
        for node_id in range(len(attack_plane)):
            node_row, node_col = self.idsgame_env.idsgame_config.game_config.network_config.get_node_pos(node_id)
            adj_matrix_id = self.idsgame_env.idsgame_config.game_config.network_config.get_adjacency_matrix_id(node_row,node_col)
            reachable = self.idsgame_env.idsgame_config.game_config.network_config.adjacency_matrix[attacker_matrix_id][adj_matrix_id] == int(1)
            if reachable:
                val = 1
            else:
                val = 0
            reachable_plane[node_id] = np.full(reachable_plane.shape[1], val)


        row_difference_plane = np.zeros(attack_plane.shape)
        for node_id in range(len(attack_plane)):
            node_row, node_col = self.idsgame_env.idsgame_config.game_config.network_config.get_node_pos(node_id)
            row_difference = attacker_row-node_row
            row_difference_plane[node_id] = np.full(row_difference_plane.shape[1], row_difference)

        attack_defense_difference_plane = attacker_obs - defender_obs
        if self.pg_agent_config.normalize_features:
            attack_defense_difference_plane_row_sums = attack_defense_difference_plane.sum(axis=1)
            normalized_attack_defense_difference_plane = attack_defense_difference_plane / attack_defense_difference_plane_row_sums[:, np.newaxis]
            normalized_attack_defense_difference_plane = np.nan_to_num(normalized_attack_defense_difference_plane, nan=0.0)

        # print("attack plane:")
        # print(attack_plane)
        # print("defense plane:")
        # print(defense_plane)
        # print("position plane:")
        # print(position_plane)
        if self.pg_agent_config.normalize_features:
            feature_frames = np.stack([normalized_attack_plane, normalized_defense_plane, position_plane, reachable_plane, row_difference_plane,
                                       normalized_attack_defense_difference_plane],
                                  axis=0)
        else:
            feature_frames = np.stack(
                [attack_plane, defense_plane, position_plane, reachable_plane,
                 row_difference_plane,
                 attack_defense_difference_plane],
                axis=0)
        print("feature_frames:")
        print(feature_frames)
        raise AssertionError("test")
        return feature_frames



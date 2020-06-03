"""
A wrapper environment to integrate idsgame-env with the OpenAI baselines library
"""
import gym
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
import matplotlib
from gym_idsgame.envs.dao.idsgame_config import IdsGameConfig
from gym_idsgame.agents.training_agents.policy_gradient.pg_agent_config import PolicyGradientAgentConfig
from gym_idsgame.envs.constants import constants
import gym_idsgame.envs.util.idsgame_util as util

class BaselineEnvWrapper(gym.Env):
    """
    A wrapper environment to integrate idsgame-env with the OpenAI baselines library
    """

    def __init__(self, env_name: str, idsgame_config: IdsGameConfig = None, save_dir: str = None,
                 initial_state_path: str = None,
                 pg_agent_config: PolicyGradientAgentConfig = None):
        super(BaselineEnvWrapper, self).__init__()
        self.idsgame_env = gym.make(env_name, idsgame_config=idsgame_config,
                                    save_dir=save_dir,
                                    initial_state_path=initial_state_path)
        self.pg_agent_config = pg_agent_config
        self.attacker_action_space = self.idsgame_env.attacker_action_space
        self.defender_action_space = self.idsgame_env.defender_action_space
        attacker_obs_shape = self.pg_agent_config.input_dim_attacker
        defender_obs_shape = self.pg_agent_config.input_dim_defender
        if type(attacker_obs_shape) != tuple:
            attacker_obs_shape = (attacker_obs_shape,)
        if type(defender_obs_shape) != tuple:
            defender_obs_shape = (defender_obs_shape,)
        self.attacker_observation_space = gym.spaces.Box(low=0,
                                                         high=self.idsgame_env.idsgame_config.game_config.max_value,
                                                         shape=attacker_obs_shape,
                                                         dtype=np.float32)
        self.observation_space = self.attacker_observation_space
        self.defender_observation_space = gym.spaces.Box(low=0,
                                                         high=self.idsgame_env.idsgame_config.game_config.max_value,
                                                         shape=defender_obs_shape,
                                                         dtype=np.float32)
        self.prev_episode_hacked = False
        self.prev_episode_detected = False
        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': 50  # Video rendering speed
        }
        self.num_attack_actions = self.pg_agent_config.output_dim_attacker
        self.num_defense_actions = self.idsgame_env.num_defense_actions
        self.latest_obs = False
        self.attacker_state = []
        self.defender_state = []

    def step(self, action):
        attacker_action = action[0][0]
        defender_action = action[1][0]
        if self.idsgame_env.local_view_features():
            attacker_obs, _ = self.idsgame_env.get_observation()
            attacker_action = self.convert_local_attacker_action_to_global(attacker_action, attacker_obs)
        joint_action = (attacker_action, defender_action)
        obs_prime, reward, done, info = self.idsgame_env.step(joint_action)
        self.latest_obs = obs_prime
        attacker_reward, defender_reward = reward
        obs_prime_attacker, obs_prime_defender = obs_prime
        if self.pg_agent_config.cnn_feature_extractor and not self.pg_agent_config.flatten_feature_planes and not self.pg_agent_config.seq_cnn and not self.pg_agent_config.grid_image_obs:
            attacker_state = self.grid_obs(obs_prime_attacker, obs_prime_defender, attacker=True)
            defender_state = self.grid_obs(obs_prime_attacker, obs_prime_defender, attacker=True)
            return [attacker_state, defender_state], [attacker_reward, defender_reward], done, info
        elif self.pg_agent_config.flatten_feature_planes and not self.pg_agent_config.seq_cnn and not self.pg_agent_config.grid_image_obs:
            attacker_state = self.grid_obs(obs_prime_attacker, obs_prime_defender, attacker=True)
            defender_state = self.grid_obs(obs_prime_attacker, obs_prime_defender, attacker=True)
            return [attacker_state.flatten(), defender_state.flatten()], [attacker_reward, defender_reward], done, info
        elif self.pg_agent_config.seq_cnn and not self.pg_agent_config.grid_image_obs:
            attacker_state = self.grid_seq_obs(obs_prime_attacker, obs_prime_defender, self.attacker_state, self.defender_state, attacker=True)
            defender_state = self.grid_seq_obs(obs_prime_attacker, obs_prime_defender, self.attacker_state, self.defender_state, attacker=True)
            self.attacker_state = attacker_state
            self.defender_state = defender_state
            return [attacker_state, defender_state], [attacker_reward, defender_reward], done, info
        elif self.pg_agent_config.one_hot_obs:
            attacker_state = self.one_hot_obs(attacker_obs=obs_prime_attacker, defender_obs=obs_prime_defender, attacker=True)
            defender_state = self.one_hot_obs(attacker_obs=obs_prime_attacker, defender_obs=obs_prime_defender, attacker=False)
            return [attacker_state.flatten(), defender_state.flatten()], [attacker_reward, defender_reward], done, info
        elif self.pg_agent_config.grid_image_obs:
            attacker_state = self.image_grid_obs(attacker_obs=obs_prime_attacker, defender_obs=obs_prime_defender, attacker=True)
            #defender_state = self.image_grid_obs(attacker_obs= obs_prime_attacker, defender_obs=obs_prime_defender, attacker=False)
            return [attacker_state, attacker_state], [attacker_reward, defender_reward], done, info
        else:
            attacker_state = self.update_state(attacker_obs=obs_prime_attacker, defender_obs=obs_prime_defender, state=self.attacker_state,
                                               attacker=True)
            defender_state = self.update_state(defender_obs=obs_prime_defender, attacker_obs=obs_prime_attacker, state=self.defender_state,
                                               attacker=False)
            self.attacker_state = attacker_state
            self.defender_state = defender_state
            return [attacker_state.flatten(), defender_state.flatten()], [attacker_reward, defender_reward], done, info

    def reset(self, update_stats: False):
        self.prev_episode_hacked = self.idsgame_env.state.hacked
        self.prev_episode_detected = self.idsgame_env.state.detected
        self.attacker_state = []
        self.defender_state = []
        obs = self.idsgame_env.reset(update_stats=update_stats)
        obs_attacker, obs_defender = obs
        self.latest_obs = obs

        if self.pg_agent_config.cnn_feature_extractor and not self.pg_agent_config.flatten_feature_planes and not self.pg_agent_config.seq_cnn and not self.pg_agent_config.grid_image_obs:
            attacker_state = self.grid_obs(obs_attacker, obs_defender, attacker=True)
            defender_state = self.grid_obs(obs_attacker, obs_defender, attacker=True)
            return [attacker_state, defender_state]
        elif self.pg_agent_config.flatten_feature_planes and not self.pg_agent_config.seq_cnn and not self.pg_agent_config.grid_image_obs:
            attacker_state = self.grid_obs(obs_attacker, obs_defender, attacker=True)
            defender_state = self.grid_obs(obs_attacker, obs_defender, attacker=True)
            return [attacker_state.flatten(), defender_state.flatten()]
        elif self.pg_agent_config.seq_cnn and not self.pg_agent_config.grid_image_obs:
            attacker_state = self.grid_seq_obs(obs_attacker, obs_defender, self.attacker_state,
                                               self.defender_state, attacker=True)
            defender_state = self.grid_seq_obs(obs_attacker, obs_defender, self.attacker_state,
                                               self.defender_state, attacker=True)
            self.attacker_state = attacker_state
            self.defender_state = defender_state
            return [self.attacker_state, self.defender_state]
        elif self.pg_agent_config.one_hot_obs:
            attacker_state = self.one_hot_obs(attacker_obs=obs_attacker, defender_obs=obs_defender, attacker=True)
            defender_state = self.one_hot_obs(attacker_obs=obs_attacker, defender_obs=obs_defender, attacker=False)
            return [attacker_state.flatten(), defender_state.flatten()]
        elif self.pg_agent_config.grid_image_obs:
            attacker_state = self.image_grid_obs(attacker_obs=obs_attacker, defender_obs=obs_defender, attacker=True)
            #defender_state = self.image_grid_obs(attacker_obs=obs_attacker, defender_obs=obs_defender, attacker=False)
            return [attacker_state, attacker_state]
        else:
            attacker_state = self.update_state(attacker_obs=obs_attacker, defender_obs=obs_defender, state=self.attacker_state,
                                               attacker=True)
            defender_state = self.update_state(defender_obs=obs_defender, attacker_obs=obs_attacker, state=self.defender_state,
                                               attacker=False)
            self.attacker_state = attacker_state
            self.defender_state = defender_state
            return [attacker_state.flatten(), defender_state.flatten()]

    def render(self, mode='human'):
        return self.idsgame_env.render(mode=mode)

    def close(self):
        return self.idsgame_env.close()

    def convert_local_attacker_action_to_global(self, action_id, attacker_obs):
        num_attack_types = self.idsgame_env.idsgame_config.game_config.num_attack_types
        neighbor = action_id // (num_attack_types+1)
        attack_type = action_id % (num_attack_types+1)
        target_id = int(attacker_obs[neighbor][num_attack_types])
        if target_id == -1:
            return -1
        attacker_action = target_id * (num_attack_types+1) + attack_type
        return attacker_action

    def is_attack_legal(self, attack_action: int) -> bool:
        """
        Check if a given attack is legal or not.

        :param attack_action: the attack to verify
        :return: True if legal otherwise False
        """
        if self.idsgame_env.local_view_features():
            #attacker_obs, _ = self.idsgame_env.get_observation()
            attack_action = self.convert_local_attacker_action_to_global(attack_action, self.latest_obs[0])
            if attack_action == -1:
                return False
        return self.idsgame_env.is_attack_legal(attack_action)

    def is_defense_legal(self, defense_action: int) -> bool:
        """
        Check if a given defense is legal or not.

        :param defense_action: the defense action to verify
        :return: True if legal otherwise False
        """
        return self.idsgame_env.is_defense_legal(defense_action)

    def hack_probability(self):
        if self.num_games > 0:
            return self.num_hacks / self.num_games
        else:
            return 0.0

    def is_reconnaissance(self, action):
        if self.idsgame_env.local_view_features():
            action = self.convert_local_attacker_action_to_global(action, self.latest_obs[0])
        return self.idsgame_env.is_reconnaissance(action)

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
            #if not self.idsgame_env.local_view_features():
            a_obs_len = self.idsgame_env.idsgame_config.game_config.num_attack_types + 1
            defender_obs = attacker_obs[:, a_obs_len:a_obs_len+self.idsgame_env.idsgame_config.game_config.num_attack_types]
            if self.idsgame_env.idsgame_config.reconnaissance_bool_features:
                d_bool_features = attacker_obs[:, a_obs_len+self.idsgame_env.idsgame_config.game_config.num_attack_types:]
            attacker_obs = attacker_obs[:, 0:a_obs_len]
            # else:
            #     a_obs_len = self.idsgame_env.idsgame_config.game_config.num_attack_types + 1
            #     defender_obs = attacker_obs[:,
            #                    a_obs_len:a_obs_len + self.idsgame_env.idsgame_config.game_config.num_attack_types]
            #     if self.idsgame_env.idsgame_config.reconnaissance_bool_features:
            #         d_bool_features = attacker_obs[:,
            #                           a_obs_len + self.idsgame_env.idsgame_config.game_config.num_attack_types:]
            #     attacker_obs = attacker_obs[:, 0:a_obs_len]

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
            if not self.idsgame_env.idsgame_config.game_config.reconnaissance_actions:
                neighbor_defense_attributes = np.zeros((attacker_obs.shape[0], defender_obs.shape[1]))
                for node in range(attacker_obs.shape[0]):
                    id = int(attacker_obs[node][-1])
                    neighbor_defense_attributes[node] = defender_obs[id]
            else:
                neighbor_defense_attributes = defender_obs

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
                        if self.idsgame_env.fully_observed():
                            t.append(det_values[idx])
                        features.append(t)
                else:
                    node_ids = attacker_obs[:, -1]
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
                        #t.append(node_reachable[idx])
                        if not self.idsgame_env.idsgame_config.game_config.reconnaissance_actions:
                            t.append(det_values[idx])
                        features.append(t)
                features = np.array(features)
                if self.idsgame_env.idsgame_config.reconnaissance_bool_features:
                    f = np.zeros((features.shape[0], features.shape[1] + d_bool_features.shape[1]))
                    for i in range(features.shape[0]):
                        f[i] = np.append(features[i], d_bool_features[i])
                    features = f
                if self.pg_agent_config.state_length == 1:
                    return features
                if len(state) == 0:
                    s = np.array([features] * self.pg_agent_config.state_length)
                    return s
                state = np.append(state[1:], np.array([features]), axis=0)
                return state
            else:
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
                if self.pg_agent_config.state_length == 1:
                    return f
                if len(state) == 0:
                    s = np.array([f] * self.pg_agent_config.state_length)
                    return s
                # if not self.idsgame_env.local_view_features() or not attacker:
                #     temp = np.append(attacker_obs, defender_obs)
                # else:
                #     temp = np.append(attacker_obs, neighbor_defense_attributes)
                state = np.append(state[1:], np.array([f]), axis=0)
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

    def grid_seq_obs(self, attacker_obs, defender_obs, attacker_state, defender_state, attacker=True):
        if attacker and self.idsgame_env.idsgame_config.game_config.reconnaissance_actions:
            # if not self.idsgame_env.local_view_features():
            a_obs_len = self.idsgame_env.idsgame_config.game_config.num_attack_types + 1
            defender_obs = attacker_obs[:,
                           a_obs_len:a_obs_len + self.idsgame_env.idsgame_config.game_config.num_attack_types]
            if self.idsgame_env.idsgame_config.reconnaissance_bool_features:
                d_bool_features = attacker_obs[:,
                                  a_obs_len + self.idsgame_env.idsgame_config.game_config.num_attack_types:]
            attacker_obs = attacker_obs[:, 0:a_obs_len-1]

        a_vec = attacker_obs.flatten()
        if len(attacker_state) == 0:
            attacker_plane = np.zeros((len(a_vec), self.pg_agent_config.state_length))
            for i in range(self.pg_agent_config.state_length):
                attacker_plane[:,i] = a_vec
        elif len(attacker_state) > 0 :
            attacker_plane = attacker_state[0]
            # attacker_plane[:, 1:] = attacker_plane[:, 0:-1]
            # attacker_plane[:, 0] = a_vec
            n_attacker_plane = np.zeros((len(a_vec), self.pg_agent_config.state_length))
            n_attacker_plane[:, 0] = a_vec
            for i in range(self.pg_agent_config.state_length-1):
                n_attacker_plane[:,i+1] = attacker_plane[:,i]
            attacker_plane = n_attacker_plane
        else:
            raise AssertionError("Invalid state")

        d_vec = defender_obs.flatten()
        if len(attacker_state) == 0:
            defense_plane = np.zeros((len(d_vec), self.pg_agent_config.state_length))
            for i in range(self.pg_agent_config.state_length):
                defense_plane[:, i] = d_vec
        elif len(attacker_state) > 0:
            defense_plane = attacker_state[1]
            # defense_plane[:, 1:] = defense_plane[:,0:-1]
            # defense_plane[:,0] = d_vec
            n_defense_plane = np.zeros((len(d_vec), self.pg_agent_config.state_length))
            n_defense_plane[:, 0] = d_vec
            for i in range(self.pg_agent_config.state_length - 1):
                n_defense_plane[:, i + 1] = defense_plane[:, i]
            defense_plane = n_defense_plane
        else:
            raise AssertionError("Invalid state")

        # rec_vec = d_bool_features.flatten()
        # if len(attacker_state) == 0:
        #     rec_plane = np.zeros((len(rec_vec), self.pg_agent_config.state_length))
        #     for i in range(self.pg_agent_config.state_length):
        #         rec_plane[:, i] = rec_vec
        # elif len(attacker_state) > 0:
        #     rec_plane = attacker_state[2]
        #     n_rec_plane = np.zeros((len(rec_vec), self.pg_agent_config.state_length))
        #     n_rec_plane[:, 0] = d_vec
        #     for i in range(self.pg_agent_config.state_length - 1):
        #         n_rec_plane[:, i + 1] = rec_plane[:, i]
        #     rec_plane = n_rec_plane
        # else:
        #     raise AssertionError("Invalid state")

        # print("attacker_plane:{}".format(attacker_plane.shape))
        # print("defender_plane:{}".format(defense_plane.shape))
        feature_frames = np.stack([attacker_plane, defense_plane], axis=0)
        # print("feature_frames:")
        # print(feature_frames)
        return feature_frames
        #raise AssertionError("test")




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
            normalized_attack_plane = preprocessing.normalize(attack_plane)

        defense_plane = defender_obs
        if self.pg_agent_config.normalize_features:
            normalized_defense_plane = preprocessing.normalize(defense_plane)

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

        if self.pg_agent_config.normalize_features:
            normalized_row_difference_plance = preprocessing.normalize(row_difference_plane)

        attack_defense_difference_plane = attacker_obs - defender_obs
        if self.pg_agent_config.normalize_features:
            normalized_attack_defense_difference_plane = preprocessing.normalize(attack_defense_difference_plane)

        if self.pg_agent_config.normalize_features:
            feature_frames = np.stack([normalized_attack_plane, normalized_defense_plane, position_plane, reachable_plane,
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

    def one_hot_obs(self, attacker_obs, defender_obs, attacker=True):
        attack_types = self.idsgame_env.idsgame_config.game_config.num_attack_types
        max_value = self.idsgame_env.idsgame_config.game_config.max_value
        num_nodes = self.idsgame_env.idsgame_config.game_config.num_nodes

        if attacker and self.idsgame_env.idsgame_config.game_config.reconnaissance_actions:
            #if not self.idsgame_env.local_view_features():
            a_obs_len = self.idsgame_env.idsgame_config.game_config.num_attack_types + 1
            defender_obs = attacker_obs[:, a_obs_len:a_obs_len+self.idsgame_env.idsgame_config.game_config.num_attack_types]
            if self.idsgame_env.idsgame_config.reconnaissance_bool_features:
                d_bool_features = attacker_obs[:, a_obs_len+self.idsgame_env.idsgame_config.game_config.num_attack_types:]
            attacker_obs = attacker_obs[:, 0:a_obs_len]

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
                        #t.append(a_pos[idx])
                        if not self.idsgame_env.idsgame_config.game_config.reconnaissance_actions:
                            t.append(det_values[idx])
                        features.append(t)
                else:
                    node_ids = attacker_obs[:, -1]
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
                        #t.append(node_reachable[idx])
                        if not self.idsgame_env.idsgame_config.game_config.reconnaissance_actions:
                            t.append(det_values[idx])
                        features.append(t)
                features = np.array(features)
                # if self.idsgame_env.idsgame_config.reconnaissance_bool_features:
                #     f = np.zeros((features.shape[0], features.shape[1] + d_bool_features.shape[1]))
                #     for i in range(features.shape[0]):
                #         f[i] = np.append(features[i], d_bool_features[i])
                #     features = f
        if attacker:
            a_obs = np.zeros((num_nodes, attack_types*((max_value+1)*2+1)))
            values_list = list(range(-max_value, max_value+1))
            for n in range(num_nodes):
                for t in range(attack_types):
                    for v in range(-max_value, max_value+1):
                        if features[n][t] == v:
                            a_obs[n][t*((max_value+1)*2) + values_list.index(v)] = 1
            a_obs[:,-1] = a_pos
        # print("attacker obs:{}, defender_obs:{}".format(attacker_obs, defender_obs))
        # print("a_obs:{}".format(a_obs))
        # print("a_obs shape:{}".format(a_obs.shape))
        # print("flatten shape:{}".format(a_obs.flatten().shape))
        # raise AssertionError("Test")
        if attacker:
            return a_obs
        else:
            return defender_obs

    def image_grid_obs(self, attacker_obs, defender_obs, attacker=True):
        if attacker and self.idsgame_env.idsgame_config.game_config.reconnaissance_actions:
            #if not self.idsgame_env.local_view_features():
            a_obs_len = self.idsgame_env.idsgame_config.game_config.num_attack_types + 1
            defender_obs = attacker_obs[:, a_obs_len:a_obs_len+self.idsgame_env.idsgame_config.game_config.num_attack_types]
            if self.idsgame_env.idsgame_config.reconnaissance_bool_features:
                d_bool_features = attacker_obs[:, a_obs_len+self.idsgame_env.idsgame_config.game_config.num_attack_types:]
            attacker_obs = attacker_obs[:, 0:a_obs_len]

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
                        #t.append(a_pos[idx])
                        if not self.idsgame_env.idsgame_config.game_config.reconnaissance_actions:
                            t.append(det_values[idx])
                        features.append(t)
                else:
                    node_ids = attacker_obs[:, -1]
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
                        #t.append(node_reachable[idx])
                        if not self.idsgame_env.idsgame_config.game_config.reconnaissance_actions:
                            t.append(det_values[idx])
                        features.append(t)
                features = np.array(features)

        colors = set()
        rank_ids = [2,3,4,5]
        features[0, :] = [1,2,3,4]
        for n in range(len(a_pos)):
            if a_pos[n] == 1:
                features[n,:] = np.full(features.shape[1], 0)
                colors.add(0)
            elif np.all(features[n] == constants.GAME_CONFIG.INITIAL_RECONNAISSANCE_STATE):
                features[n,:] = np.full(features.shape[1], 1)
                colors.add(1)
            else:
                values = features[n,:]
                sorted_values = sorted(values, key=lambda x: x)
                for i in range(len(values)):
                    idx = sorted_values.index(values[i])
                    values[i] = rank_ids[idx]
                    colors.add(rank_ids[idx])
                features[n, :] = values
        #print("features:{}".format(features))
        cmap = matplotlib.colors.ListedColormap(['white', 'red', "Blue", "gray", "yellow",
                                                  "green", "#A4940A", "#FFE600"], N=len(colors))
        fig = plt.figure(figsize=(3, 3))
        plt.pcolor(features[::-1], cmap=cmap, edgecolors='k', linewidths=3)
        plt.axis("off")
        data = util.get_img_from_fig(fig, dpi=20)
        data = np.rollaxis(data, 2, 0)
        plt.close()
        #print("data shape:{}".format(data.shape))
        #print("data shape:{}".format(data.shape))
        #fig2 = plt.figure(figsize=(4, 4))
        #ax = plt.Axes(fig2, [0., 0., 1., 1.])
        #ax.set_axis_off()
        #fig2.add_axes(ax)
        #ax.imshow(data)
        #plt.show()
        #raise AssertionError("test")
        return data


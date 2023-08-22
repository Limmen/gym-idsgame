from collections import OrderedDict
from copy import deepcopy
from typing import Sequence

import numpy as np

from gym_idsgame.agents.training_agents.openai_baselines.common.vec_env.base_vec_env import VecEnv
from gym_idsgame.agents.training_agents.openai_baselines.common.vec_env.util import copy_obs_dict, dict_to_obs, obs_space_info


class DummyVecEnv(VecEnv):
    """
    Creates a simple vectorized wrapper for multiple environments, calling each environment in sequence on the current
    Python process. This is useful for computationally simple environment such as ``cartpole-v1``,
    as the overhead of multiprocess or multithread outweighs the environment computation time.
    This can also be used for RL methods that
    require a vectorized environment, but that you want a single environments to train with.

    :param env_fns: ([Gym Environment]) the list of environments to vectorize
    """

    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        VecEnv.__init__(self, len(env_fns), env.attacker_observation_space, env.attacker_action_space,
                        env.defender_observation_space, env.defender_action_space)
        attacker_obs_space = env.attacker_observation_space
        self.attacker_keys, attacker_shapes, attacker_dtypes = obs_space_info(attacker_obs_space)
        defender_obs_space = env.defender_observation_space
        self.defender_keys, defender_shapes, defender_dtypes = obs_space_info(defender_obs_space)


        self.buf_a_obs = OrderedDict([
            (k, np.zeros((self.num_envs,) + tuple(attacker_shapes[k]), dtype=attacker_dtypes[k]))
            for k in self.attacker_keys])
        self.buf_d_obs = OrderedDict([
            (k, np.zeros((self.num_envs,) + tuple(defender_shapes[k]), dtype=defender_dtypes[k]))
            for k in self.defender_keys])
        self.buf_dones = np.zeros((self.num_envs,), dtype=np.bool_)
        self.buf_a_rews = np.zeros((self.num_envs,), dtype=np.float32)
        self.buf_d_rews = np.zeros((self.num_envs,), dtype=np.float32)
        self.buf_infos = [{} for _ in range(self.num_envs)]
        self.actions = None
        self.metadata = env.metadata

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self, update_stats : bool = False):
        for env_idx in range(self.num_envs):
            obs, rew, self.buf_dones[env_idx], _, self.buf_infos[env_idx] =\
                self.envs[env_idx].step(self.actions[env_idx])
            self.buf_a_rews[env_idx] = rew[0]
            self.buf_d_rews[env_idx] = rew[1]
            if self.buf_dones[env_idx]:
                # save final observation where user can get it, then reset
                self.buf_infos[env_idx]['terminal_observation'] = obs
                r_obs, _ = self.envs[env_idx].reset(update_stats=update_stats)
                if r_obs is not None:
                    obs = r_obs
            a_obs = obs[0]
            d_obs = obs[1]
            self._save_obs(env_idx, a_obs, d_obs)
        return (np.copy(a_obs), np.copy(d_obs), np.copy(self.buf_a_rews), np.copy(self.buf_d_rews), np.copy(self.buf_dones),
                deepcopy(self.buf_infos))

    def seed(self, seed=None):
        seeds = list()
        for idx, env in enumerate(self.envs):
            seeds.append(seed + idx)
        return seeds

    def reset(self, update_stats : bool = False):
        for env_idx in range(self.num_envs):
            obs, _ = self.envs[env_idx].reset(update_stats=update_stats)
            # self._save_obs(env_idx, a_obs, d_obs)
        return np.copy(obs)
        #return self._obs_from_buf()

    def close(self):
        for env in self.envs:
            env.close()

    def get_images(self, *args, **kwargs) -> Sequence[np.ndarray]:
        return [env.render(*args, mode='rgb_array', **kwargs) for env in self.envs]

    def render(self, *args, **kwargs):
        """
        Gym environment rendering. If there are multiple environments then
        they are tiled together in one image via ``BaseVecEnv.render()``.
        Otherwise (if ``self.num_envs == 1``), we pass the render call directly to the
        underlying environment.

        Therefore, some arguments such as ``mode`` will have values that are valid
        only when ``num_envs == 1``.

        :param mode: The rendering type.
        """
        if self.num_envs == 1:
            return self.envs[0].render(*args, **kwargs)
        else:
            return super().render(*args, **kwargs)

    def _save_obs(self, env_idx, a_obs, d_obs):
        pass
        # for key in self.attacker_keys:
        #     if key is None:
        #         # print("buf_a_obs:{}".format(self.buf_a_obs))
        #         # print("a_obs:{}".format(a_obs))
        #         self.buf_a_obs[key][env_idx] = a_obs
        #     else:
        #         self.buf_a_obs[key][env_idx] = a_obs[key]
        # for key in self.defender_keys:
        #     if key is None:
        #         self.buf_d_obs[key][env_idx] = d_obs
        #     else:
        #         self.buf_d_obs[key][env_idx] = d_obs[key]

    def _obs_from_buf(self):
        return dict_to_obs(self.attacker_observation_space, copy_obs_dict(self.buf_a_obs))

    def get_attr(self, attr_name, indices=None):
        """Return attribute from vectorized environment (see base class)."""
        target_envs = self._get_target_envs(indices)
        return [getattr(env_i, attr_name) for env_i in target_envs]

    def set_attr(self, attr_name, value, indices=None):
        """Set attribute inside vectorized environments (see base class)."""
        target_envs = self._get_target_envs(indices)
        for env_i in target_envs:
            setattr(env_i, attr_name, value)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        """Call instance methods of vectorized environments."""
        target_envs = self._get_target_envs(indices)
        return [getattr(env_i, method_name)(*method_args, **method_kwargs) for env_i in target_envs]

    def _get_target_envs(self, indices):
        indices = self._get_indices(indices)
        return [self.envs[i] for i in indices]

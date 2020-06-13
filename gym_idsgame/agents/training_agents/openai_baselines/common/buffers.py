from typing import Union, Optional, Generator

import numpy as np
import torch as th
from gym import spaces

from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.preprocessing import get_action_dim, get_obs_shape
from gym_idsgame.agents.training_agents.openai_baselines.common.type_aliases import RolloutBufferSamples, \
    ReplayBufferSamples, RolloutBufferSamplesRecurrent, RolloutBufferSamplesRecurrentMultiHead, RolloutBufferSamplesAR, \
    RolloutBufferSamplesARRecurrent, RolloutBufferSamplesARRecurrentMultiHead
from gym_idsgame.agents.training_agents.policy_gradient.pg_agent_config import PolicyGradientAgentConfig

class BaseBuffer(object):
    """
    Base class that represent a buffer (rollout or replay)

    :param buffer_size: (int) Max number of element in the buffer
    :param observation_space: (spaces.Space) Observation space
    :param action_space: (spaces.Space) Action space
    :param device: (Union[th.device, str]) PyTorch device
        to which the values will be converted
    :param n_envs: (int) Number of parallel environments
    """

    def __init__(self,
                 buffer_size: int,
                 observation_space: spaces.Space,
                 action_space: spaces.Space,
                 device: Union[th.device, str] = 'cpu',
                 n_envs: int = 1):
        super(BaseBuffer, self).__init__()
        self.buffer_size = buffer_size
        self.observation_space = observation_space
        self.action_space = action_space
        self.obs_shape = get_obs_shape(observation_space)
        self.action_dim = get_action_dim(action_space)
        self.pos = 0
        self.full = False
        self.device = device
        self.n_envs = n_envs

    @staticmethod
    def swap_and_flatten(arr: np.ndarray) -> np.ndarray:
        """
        Swap and then flatten axes 0 (buffer_size) and 1 (n_envs)
        to convert shape from [n_steps, n_envs, ...] (when ... is the shape of the features)
        to [n_steps * n_envs, ...] (which maintain the order)

        :param arr: (np.ndarray)
        :return: (np.ndarray)
        """
        shape = arr.shape
        if len(shape) < 3:
            shape = shape + (1,)
        return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])

    def size(self) -> int:
        """
        :return: (int) The current size of the buffer
        """
        if self.full:
            return self.buffer_size
        return self.pos

    def add(self, *args, **kwargs) -> None:
        """
        Add elements to the buffer.
        """
        raise NotImplementedError()

    def extend(self, *args, **kwargs) -> None:
        """
        Add a new batch of transitions to the buffer
        """
        # Do a for loop along the batch axis
        for data in zip(*args):
            self.add(*data)

    def reset(self) -> None:
        """
        Reset the buffer.
        """
        self.pos = 0
        self.full = False

    def sample(self,
               batch_size: int,
               env: Optional[VecNormalize] = None
               ):
        """
        :param batch_size: (int) Number of element to sample
        :param env: (Optional[VecNormalize]) associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return: (Union[RolloutBufferSamples, ReplayBufferSamples])
        """
        upper_bound = self.buffer_size if self.full else self.pos
        batch_inds = np.random.randint(0, upper_bound, size=batch_size)
        return self._get_samples(batch_inds, env=env)

    def _get_samples(self,
                     batch_inds: np.ndarray,
                     env: Optional[VecNormalize] = None
                     ):
        """
        :param batch_inds: (th.Tensor)
        :param env: (Optional[VecNormalize])
        :return: (Union[RolloutBufferSamples, ReplayBufferSamples])
        """
        raise NotImplementedError()

    def to_torch(self, array: np.ndarray, copy: bool = True) -> th.Tensor:
        """
        Convert a numpy array to a PyTorch tensor.
        Note: it copies the data by default

        :param array: (np.ndarray)
        :param copy: (bool) Whether to copy or not the data
            (may be useful to avoid changing things be reference)
        :return: (th.Tensor)
        """
        if copy:
            return th.tensor(array).to(self.device)
        return th.as_tensor(array).to(self.device)

    @staticmethod
    def _normalize_obs(obs: np.ndarray,
                       env: Optional[VecNormalize] = None) -> np.ndarray:
        if env is not None:
            return env.normalize_obs(obs).astype(np.float32)
        return obs

    @staticmethod
    def _normalize_reward(reward: np.ndarray,
                          env: Optional[VecNormalize] = None) -> np.ndarray:
        if env is not None:
            return env.normalize_reward(reward).astype(np.float32)
        return reward


class ReplayBuffer(BaseBuffer):
    """
    Replay buffer used in off-policy algorithms like SAC/TD3.

    :param buffer_size: (int) Max number of element in the buffer
    :param observation_space: (spaces.Space) Observation space
    :param action_space: (spaces.Space) Action space
    :param device: (th.device)
    :param n_envs: (int) Number of parallel environments
    """

    def __init__(self,
                 buffer_size: int,
                 observation_space: spaces.Space,
                 action_space: spaces.Space,
                 device: Union[th.device, str] = 'cpu',
                 n_envs: int = 1):
        super(ReplayBuffer, self).__init__(buffer_size, observation_space,
                                           action_space, device, n_envs=n_envs)

        assert n_envs == 1, "Replay buffer only support single environment for now"

        self.observations = np.zeros((self.buffer_size, self.n_envs,) + self.obs_shape, dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=np.float32)
        self.next_observations = np.zeros((self.buffer_size, self.n_envs,) + self.obs_shape, dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

    def add(self,
            obs: np.ndarray,
            next_obs: np.ndarray,
            action: np.ndarray,
            reward: np.ndarray,
            done: np.ndarray) -> None:
        # Copy to avoid modification by reference
        self.observations[self.pos] = np.array(obs).copy()
        self.next_observations[self.pos] = np.array(next_obs).copy()
        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.dones[self.pos] = np.array(done).copy()

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def _get_samples(self,
                     batch_inds: np.ndarray,
                     env: Optional[VecNormalize] = None
                     ) -> ReplayBufferSamples:
        data = (self._normalize_obs(self.observations[batch_inds, 0, :], env),
                self.actions[batch_inds, 0, :],
                self._normalize_obs(self.next_observations[batch_inds, 0, :], env),
                self.dones[batch_inds],
                self._normalize_reward(self.rewards[batch_inds], env))
        return ReplayBufferSamples(*tuple(map(self.to_torch, data)))


class RolloutBuffer(BaseBuffer):
    """
    Rollout buffer used in on-policy algorithms like A2C/PPO.

    :param buffer_size: (int) Max number of element in the buffer
    :param observation_space: (spaces.Space) Observation space
    :param action_space: (spaces.Space) Action space
    :param device: (th.device)
    :param gae_lambda: (float) Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        Equivalent to classic advantage when set to 1.
    :param gamma: (float) Discount factor
    :param n_envs: (int) Number of parallel environments
    """

    def __init__(self,
                 buffer_size: int,
                 observation_space: spaces.Space,
                 action_space: spaces.Space,
                 device: Union[th.device, str] = 'cpu',
                 gae_lambda: float = 1,
                 gamma: float = 0.99,
                 n_envs: int = 1):

        super(RolloutBuffer, self).__init__(buffer_size, observation_space,
                                            action_space, device, n_envs=n_envs)
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.observations, self.actions, self.rewards, self.advantages = None, None, None, None
        self.returns, self.dones, self.values, self.log_probs = None, None, None, None
        self.generator_ready = False
        self.reset()

    def reset(self) -> None:
        self.observations = np.zeros((self.buffer_size, self.n_envs,) + self.obs_shape, dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.generator_ready = False
        super(RolloutBuffer, self).reset()

    def compute_returns_and_advantage(self,
                                      last_value: th.Tensor,
                                      dones: np.ndarray) -> None:
        """
        Post-processing step: compute the returns (sum of discounted rewards)
        and GAE advantage.
        Adapted from Stable-Baselines PPO2.

        Uses Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)
        to compute the advantage. To obtain vanilla advantage (A(s) = R - V(S))
        where R is the discounted reward with value bootstrap,
        set ``gae_lambda=1.0`` during initialization.

        :param last_value: (th.Tensor)
        :param dones: (np.ndarray)

        """
        # convert to numpy
        last_value = last_value.clone().cpu().numpy().flatten()

        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[step + 1]
                next_value = self.values[step + 1]
            delta = self.rewards[step] + self.gamma * next_value * next_non_terminal - self.values[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam
        self.returns = self.advantages + self.values

    def add(self,
            obs: np.ndarray,
            action: np.ndarray,
            reward: np.ndarray,
            done: np.ndarray,
            value: th.Tensor,
            log_prob: th.Tensor) -> None:
        """
        :param obs: (np.ndarray) Observation
        :param action: (np.ndarray) Action
        :param reward: (np.ndarray)
        :param done: (np.ndarray) End of episode signal.
        :param value: (th.Tensor) estimated value of the current state
            following the current policy.
        :param log_prob: (th.Tensor) log probability of the action
            following the current policy.
        """
        if len(log_prob.shape) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob = log_prob.reshape(-1, 1)

        self.observations[self.pos] = np.array(obs).copy()
        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.dones[self.pos] = np.array(done).copy()
        self.values[self.pos] = value.clone().cpu().numpy().flatten()
        self.log_probs[self.pos] = log_prob.clone().cpu().numpy()
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def get(self, batch_size: Optional[int] = None) -> Generator[RolloutBufferSamples, None, None]:
        assert self.full, ''
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        # Prepare the data
        if not self.generator_ready:
            for tensor in ['observations', 'actions', 'values',
                           'log_probs', 'advantages', 'returns']:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx:start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(self, batch_inds: np.ndarray,
                     env: Optional[VecNormalize] = None) -> RolloutBufferSamples:
        data = (self.observations[batch_inds],
                self.actions[batch_inds],
                self.values[batch_inds].flatten(),
                self.log_probs[batch_inds].flatten(),
                self.advantages[batch_inds].flatten(),
                self.returns[batch_inds].flatten())
        return RolloutBufferSamples(*tuple(map(self.to_torch, data)))


class RolloutBufferRecurrent(BaseBuffer):
    """
    Rollout buffer used in on-policy algorithms like A2C/PPO.

    :param buffer_size: (int) Max number of element in the buffer
    :param observation_space: (spaces.Space) Observation space
    :param action_space: (spaces.Space) Action space
    :param device: (th.device)
    :param gae_lambda: (float) Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        Equivalent to classic advantage when set to 1.
    :param gamma: (float) Discount factor
    :param n_envs: (int) Number of parallel environments
    """

    def __init__(self,
                 buffer_size: int,
                 observation_space: spaces.Space,
                 action_space: spaces.Space,
                 device: Union[th.device, str] = 'cpu',
                 gae_lambda: float = 1,
                 gamma: float = 0.99,
                 n_envs: int = 1,
                 pg_agent_config : PolicyGradientAgentConfig = None):

        super(RolloutBufferRecurrent, self).__init__(buffer_size, observation_space,
                                            action_space, device, n_envs=n_envs)
        self.pg_agent_config = pg_agent_config
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.observations, self.actions, self.rewards, self.advantages = None, None, None, None
        self.returns, self.dones, self.values, self.log_probs, self.h_states, self.c_states = None, None, None, None, None, None
        self.generator_ready = False
        self.reset()

    def reset(self) -> None:
        self.observations = np.zeros((self.buffer_size, self.n_envs,) + self.obs_shape, dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.h_states = np.zeros((self.buffer_size, self.n_envs, self.pg_agent_config.num_lstm_layers, 1,
                                  self.pg_agent_config.lstm_hidden_dim), dtype=np.float32)
        self.c_states = np.zeros((self.buffer_size, self.n_envs, self.pg_agent_config.num_lstm_layers, 1,
                                  self.pg_agent_config.lstm_hidden_dim), dtype=np.float32)
        self.generator_ready = False
        super(RolloutBufferRecurrent, self).reset()

    def compute_returns_and_advantage(self,
                                      last_value: th.Tensor,
                                      dones: np.ndarray) -> None:
        """
        Post-processing step: compute the returns (sum of discounted rewards)
        and GAE advantage.
        Adapted from Stable-Baselines PPO2.

        Uses Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)
        to compute the advantage. To obtain vanilla advantage (A(s) = R - V(S))
        where R is the discounted reward with value bootstrap,
        set ``gae_lambda=1.0`` during initialization.

        :param last_value: (th.Tensor)
        :param dones: (np.ndarray)

        """
        # convert to numpy
        last_value = last_value.clone().cpu().numpy().flatten()

        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[step + 1]
                next_value = self.values[step + 1]
            delta = self.rewards[step] + self.gamma * next_value * next_non_terminal - self.values[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam
        self.returns = self.advantages + self.values

    def add(self,
            obs: np.ndarray,
            action: np.ndarray,
            reward: np.ndarray,
            done: np.ndarray,
            value: th.Tensor,
            log_prob: th.Tensor,
            state: th.Tensor) -> None:
        """
        :param obs: (np.ndarray) Observation
        :param action: (np.ndarray) Action
        :param reward: (np.ndarray)
        :param done: (np.ndarray) End of episode signal.
        :param value: (th.Tensor) estimated value of the current state
            following the current policy.
        :param log_prob: (th.Tensor) log probability of the action
            following the current policy.
        """
        if len(log_prob.shape) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob = log_prob.reshape(-1, 1)

        self.observations[self.pos] = np.array(obs).copy()
        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.dones[self.pos] = np.array(done).copy()
        self.values[self.pos] = value.clone().cpu().numpy().flatten()
        self.log_probs[self.pos] = log_prob.clone().cpu().numpy()
        self.h_states[self.pos] = state[0].clone().cpu().numpy()
        self.c_states[self.pos] = state[1].clone().cpu().numpy()
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def get(self, batch_size: Optional[int] = None) -> Generator[RolloutBufferSamplesRecurrent, None, None]:
        assert self.full, ''
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        # Prepare the data
        if not self.generator_ready:
            for tensor in ['observations', 'actions', 'values',
                           'log_probs', 'advantages', 'returns', 'h_states', 'c_states', 'dones']:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx:start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(self, batch_inds: np.ndarray,
                     env: Optional[VecNormalize] = None) -> RolloutBufferSamplesRecurrent:
        data = (self.observations[batch_inds],
                self.actions[batch_inds],
                self.values[batch_inds].flatten(),
                self.log_probs[batch_inds].flatten(),
                self.advantages[batch_inds].flatten(),
                self.returns[batch_inds].flatten(),
                self.h_states[batch_inds],
                self.c_states[batch_inds],
                self.dones[batch_inds]
                )
        return RolloutBufferSamplesRecurrent(*tuple(map(self.to_torch, data)))


class RolloutBufferRecurrentMultiHead(BaseBuffer):
    """
    Rollout buffer used in on-policy algorithms like A2C/PPO.

    :param buffer_size: (int) Max number of element in the buffer
    :param observation_space: (spaces.Space) Observation space
    :param action_space: (spaces.Space) Action space
    :param device: (th.device)
    :param gae_lambda: (float) Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        Equivalent to classic advantage when set to 1.
    :param gamma: (float) Discount factor
    :param n_envs: (int) Number of parallel environments
    """

    def __init__(self,
                 buffer_size: int,
                 observation_space: spaces.Space,
                 action_space: spaces.Space,
                 device: Union[th.device, str] = 'cpu',
                 gae_lambda: float = 1,
                 gamma: float = 0.99,
                 n_envs: int = 1,
                 pg_agent_config : PolicyGradientAgentConfig = None):

        super(RolloutBufferRecurrentMultiHead, self).__init__(buffer_size, observation_space,
                                            action_space, device, n_envs=n_envs)
        self.pg_agent_config = pg_agent_config
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.observations_1, self.observations_2, self.observations_3, self.observations_4, self.actions, self.rewards, \
        self.advantages = None, None, None, None, None, None, None
        self.returns, self.dones, self.values, self.log_probs, self.h_states, self.c_states = None, None, None, None, None, None
        self.generator_ready = False
        self.reset()

    def reset(self) -> None:
        self.observations_1 = np.zeros((self.buffer_size, self.n_envs, self.pg_agent_config.channel_1_input_dim), dtype=np.float32)
        self.observations_2 = np.zeros((self.buffer_size, self.n_envs, self.pg_agent_config.channel_2_input_dim), dtype=np.float32)
        self.observations_3 = np.zeros((self.buffer_size, self.n_envs, self.pg_agent_config.channel_3_input_dim), dtype=np.float32)
        self.observations_4 = np.zeros((self.buffer_size, self.n_envs, self.pg_agent_config.channel_4_input_dim), dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.h_states = np.zeros((self.buffer_size, self.n_envs, self.pg_agent_config.num_lstm_layers, 1,
                                  self.pg_agent_config.lstm_hidden_dim), dtype=np.float32)
        self.c_states = np.zeros((self.buffer_size, self.n_envs, self.pg_agent_config.num_lstm_layers, 1,
                                  self.pg_agent_config.lstm_hidden_dim), dtype=np.float32)
        self.generator_ready = False
        super(RolloutBufferRecurrentMultiHead, self).reset()

    def compute_returns_and_advantage(self,
                                      last_value: th.Tensor,
                                      dones: np.ndarray) -> None:
        """
        Post-processing step: compute the returns (sum of discounted rewards)
        and GAE advantage.
        Adapted from Stable-Baselines PPO2.

        Uses Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)
        to compute the advantage. To obtain vanilla advantage (A(s) = R - V(S))
        where R is the discounted reward with value bootstrap,
        set ``gae_lambda=1.0`` during initialization.

        :param last_value: (th.Tensor)
        :param dones: (np.ndarray)

        """
        # convert to numpy
        last_value = last_value.clone().cpu().numpy().flatten()

        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[step + 1]
                next_value = self.values[step + 1]
            delta = self.rewards[step] + self.gamma * next_value * next_non_terminal - self.values[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam
        self.returns = self.advantages + self.values

    def add(self,
            obs_1: np.ndarray,
            obs_2: np.ndarray,
            obs_3: np.ndarray,
            obs_4: np.ndarray,
            action: np.ndarray,
            reward: np.ndarray,
            done: np.ndarray,
            value: th.Tensor,
            log_prob: th.Tensor,
            state: th.Tensor) -> None:
        """
        :param obs: (np.ndarray) Observation
        :param action: (np.ndarray) Action
        :param reward: (np.ndarray)
        :param done: (np.ndarray) End of episode signal.
        :param value: (th.Tensor) estimated value of the current state
            following the current policy.
        :param log_prob: (th.Tensor) log probability of the action
            following the current policy.
        """
        if len(log_prob.shape) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob = log_prob.reshape(-1, 1)

        self.observations_1[self.pos] = np.array(obs_1).copy()
        self.observations_2[self.pos] = np.array(obs_2).copy()
        self.observations_3[self.pos] = np.array(obs_3).copy()
        self.observations_4[self.pos] = np.array(obs_4).copy()
        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.dones[self.pos] = np.array(done).copy()
        self.values[self.pos] = value.clone().cpu().numpy().flatten()
        self.log_probs[self.pos] = log_prob.clone().cpu().numpy()
        self.h_states[self.pos] = state[0].clone().cpu().numpy()
        self.c_states[self.pos] = state[1].clone().cpu().numpy()
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def get(self, batch_size: Optional[int] = None) -> Generator[RolloutBufferSamplesRecurrentMultiHead, None, None]:
        assert self.full, ''
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        # Prepare the data
        if not self.generator_ready:
            for tensor in ['observations_1', 'observations_2', 'observations_3', 'observations_4',
                           'actions', 'values',
                           'log_probs', 'advantages', 'returns', 'h_states', 'c_states', 'dones']:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx:start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(self, batch_inds: np.ndarray,
                     env: Optional[VecNormalize] = None) -> RolloutBufferSamplesRecurrentMultiHead:
        data = (self.observations_1[batch_inds],
                self.observations_2[batch_inds],
                self.observations_3[batch_inds],
                self.observations_4[batch_inds],
                self.actions[batch_inds],
                self.values[batch_inds].flatten(),
                self.log_probs[batch_inds].flatten(),
                self.advantages[batch_inds].flatten(),
                self.returns[batch_inds].flatten(),
                self.h_states[batch_inds],
                self.c_states[batch_inds],
                self.dones[batch_inds]
                )
        return RolloutBufferSamplesRecurrentMultiHead(*tuple(map(self.to_torch, data)))


class RolloutBufferAR(BaseBuffer):
    """
    Rollout buffer used in on-policy algorithms like A2C/PPO.

    :param buffer_size: (int) Max number of element in the buffer
    :param observation_space: (spaces.Space) Observation space
    :param action_space: (spaces.Space) Action space
    :param device: (th.device)
    :param gae_lambda: (float) Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        Equivalent to classic advantage when set to 1.
    :param gamma: (float) Discount factor
    :param n_envs: (int) Number of parallel environments
    """

    def __init__(self,
                 buffer_size: int,
                 observation_space: spaces.Space,
                 action_space: spaces.Space,
                 device: Union[th.device, str] = 'cpu',
                 gae_lambda: float = 1,
                 gamma: float = 0.99,
                 n_envs: int = 1,
                 pg_agent_config : PolicyGradientAgentConfig = None,):

        super(RolloutBufferAR, self).__init__(buffer_size, observation_space,
                                            action_space, device, n_envs=n_envs)
        self.pg_agent_config = pg_agent_config
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.node_observations, self.at_observations, self.node_actions, self.at_actions, self.rewards, \
        self.node_advantages, self.at_advantages = None, None, None, None, None, None, None
        self.node_returns, self.at_returns, self.dones, self.node_values, self.at_values, self.node_log_probs, self.at_log_probs = None, None, None, None, None, None, None
        self.generator_ready = False
        self.reset()

    def reset(self) -> None:
        self.node_observations = np.zeros((self.buffer_size, self.n_envs, self.pg_agent_config.node_net_input_dim), dtype=np.float32)
        self.at_observations = np.zeros((self.buffer_size, self.n_envs, self.pg_agent_config.at_net_input_dim), dtype=np.float32)
        self.node_actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=np.float32)
        self.at_actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.node_returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.at_returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.node_values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.at_values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.node_log_probs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.at_log_probs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.node_advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.at_advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.generator_ready = False
        super(RolloutBufferAR, self).reset()

    def compute_returns_and_advantage(self,
                                      last_value: th.Tensor,
                                      dones: np.ndarray, node :bool = True) -> None:
        """
        Post-processing step: compute the returns (sum of discounted rewards)
        and GAE advantage.
        Adapted from Stable-Baselines PPO2.

        Uses Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)
        to compute the advantage. To obtain vanilla advantage (A(s) = R - V(S))
        where R is the discounted reward with value bootstrap,
        set ``gae_lambda=1.0`` during initialization.

        :param last_value: (th.Tensor)
        :param dones: (np.ndarray)

        """
        # convert to numpy
        last_value = last_value.clone().cpu().numpy().flatten()

        last_gae_lam = 0
        if node:
            for step in reversed(range(self.buffer_size)):
                if step == self.buffer_size - 1:
                    next_non_terminal = 1.0 - dones
                    next_value = last_value
                else:
                    next_non_terminal = 1.0 - self.dones[step + 1]
                    next_value = self.node_values[step + 1]
                delta = self.rewards[step] + self.gamma * next_value * next_non_terminal - self.node_values[step]
                last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
                self.node_advantages[step] = last_gae_lam
            self.node_returns = self.node_advantages + self.node_values
        else:
            for step in reversed(range(self.buffer_size)):
                if step == self.buffer_size - 1:
                    next_non_terminal = 1.0 - dones
                    next_value = last_value
                else:
                    next_non_terminal = 1.0 - self.dones[step + 1]
                    next_value = self.at_values[step + 1]
                delta = self.rewards[step] + self.gamma * next_value * next_non_terminal - self.at_values[step]
                last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
                self.at_advantages[step] = last_gae_lam
            self.at_returns = self.at_advantages + self.at_values

    def add(self,
            node_obs: np.ndarray,
            at_obs: np.ndarray,
            node_action: np.ndarray,
            reward: np.ndarray,
            done: np.ndarray,
            node_value: th.Tensor,
            node_log_prob: th.Tensor,
            at_action : np.ndarray,
            at_log_prob: th.Tensor,
            at_value: th.Tensor,
            ) -> None:
        """
        :param node_obs: (np.ndarray) Observation
        :param node_action: (np.ndarray) Action
        :param reward: (np.ndarray)
        :param done: (np.ndarray) End of episode signal.
        :param node_value: (th.Tensor) estimated value of the current state
            following the current policy.
        :param node_log_prob: (th.Tensor) log probability of the action
            following the current policy.
        """
        if len(node_log_prob.shape) == 0:
            # Reshape 0-d tensor to avoid error
            node_log_prob = node_log_prob.reshape(-1, 1)

        self.node_observations[self.pos] = np.array(node_obs).copy()
        self.at_observations[self.pos] = np.array(at_obs).copy()
        self.node_actions[self.pos] = np.array(node_action).copy()
        self.at_actions[self.pos] = np.array(at_action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.dones[self.pos] = np.array(done).copy()
        self.node_values[self.pos] = node_value.clone().cpu().numpy().flatten()
        self.at_values[self.pos] = at_value.clone().cpu().numpy().flatten()
        self.node_log_probs[self.pos] = node_log_prob.clone().cpu().numpy()
        self.at_log_probs[self.pos] = at_log_prob.clone().cpu().numpy()
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def get(self, batch_size: Optional[int] = None) -> Generator[RolloutBufferSamplesAR, None, None]:
        assert self.full, ''
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        # Prepare the data
        if not self.generator_ready:
            for tensor in ['node_observations', 'node_actions', 'node_values',
                           'node_log_probs', 'node_advantages', 'node_returns', 'at_actions', 'at_log_probs', 'at_observations',
                           'at_values', 'at_returns', 'at_advantages']:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx:start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(self, batch_inds: np.ndarray,
                     env: Optional[VecNormalize] = None) -> RolloutBufferSamplesAR:
        data = (self.node_observations[batch_inds],
                self.node_actions[batch_inds],
                self.node_values[batch_inds].flatten(),
                self.node_log_probs[batch_inds].flatten(),
                self.node_advantages[batch_inds].flatten(),
                self.node_returns[batch_inds].flatten(),
                self.at_actions[batch_inds],
                self.at_log_probs[batch_inds].flatten(),
                self.at_observations[batch_inds],
                self.at_values[batch_inds].flatten(),
                self.at_returns[batch_inds].flatten(),
                self.at_advantages[batch_inds].flatten()
                )
        return RolloutBufferSamplesAR(*tuple(map(self.to_torch, data)))


class RolloutBufferARRecurrent(BaseBuffer):
    """
    Rollout buffer used in on-policy algorithms like A2C/PPO.

    :param buffer_size: (int) Max number of element in the buffer
    :param observation_space: (spaces.Space) Observation space
    :param action_space: (spaces.Space) Action space
    :param device: (th.device)
    :param gae_lambda: (float) Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        Equivalent to classic advantage when set to 1.
    :param gamma: (float) Discount factor
    :param n_envs: (int) Number of parallel environments
    """

    def __init__(self,
                 buffer_size: int,
                 observation_space: spaces.Space,
                 action_space: spaces.Space,
                 device: Union[th.device, str] = 'cpu',
                 gae_lambda: float = 1,
                 gamma: float = 0.99,
                 n_envs: int = 1,
                 pg_agent_config : PolicyGradientAgentConfig = None,):

        super(RolloutBufferARRecurrent, self).__init__(buffer_size, observation_space,
                                            action_space, device, n_envs=n_envs)
        self.pg_agent_config = pg_agent_config
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.node_observations, self.at_observations, self.node_actions, self.at_actions, self.rewards, \
        self.node_advantages, self.at_advantages = None, None, None, None, None, None, None
        self.node_returns, self.at_returns, self.dones, self.node_values, self.at_values, self.node_log_probs, \
        self.at_log_probs, self.node_h_states, self.node_c_states, self.at_h_states, \
        self.at_c_states = None, None, None, None, None, None, None, None, None, None, None
        self.generator_ready = False
        self.reset()

    def reset(self) -> None:
        self.node_observations = np.zeros((self.buffer_size, self.n_envs, self.pg_agent_config.node_net_input_dim), dtype=np.float32)
        self.at_observations = np.zeros((self.buffer_size, self.n_envs, self.pg_agent_config.at_net_input_dim), dtype=np.float32)
        self.node_actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=np.float32)
        self.at_actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.node_returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.at_returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.node_values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.at_values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.node_log_probs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.at_log_probs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.node_advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.at_advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.generator_ready = False
        self.node_h_states = np.zeros((self.buffer_size, self.n_envs, self.pg_agent_config.num_lstm_layers, 1,
                                       self.pg_agent_config.lstm_hidden_dim), dtype=np.float32)
        self.node_c_states = np.zeros((self.buffer_size, self.n_envs, self.pg_agent_config.num_lstm_layers, 1,
                                       self.pg_agent_config.lstm_hidden_dim), dtype=np.float32)
        self.at_h_states = np.zeros((self.buffer_size, self.n_envs, self.pg_agent_config.num_lstm_layers, 1,
                                       self.pg_agent_config.lstm_hidden_dim), dtype=np.float32)
        self.at_c_states = np.zeros((self.buffer_size, self.n_envs, self.pg_agent_config.num_lstm_layers, 1,
                                       self.pg_agent_config.lstm_hidden_dim), dtype=np.float32)
        super(RolloutBufferARRecurrent, self).reset()

    def compute_returns_and_advantage(self,
                                      last_value: th.Tensor,
                                      dones: np.ndarray, node :bool = True) -> None:
        """
        Post-processing step: compute the returns (sum of discounted rewards)
        and GAE advantage.
        Adapted from Stable-Baselines PPO2.

        Uses Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)
        to compute the advantage. To obtain vanilla advantage (A(s) = R - V(S))
        where R is the discounted reward with value bootstrap,
        set ``gae_lambda=1.0`` during initialization.

        :param last_value: (th.Tensor)
        :param dones: (np.ndarray)

        """
        # convert to numpy
        last_value = last_value.clone().cpu().numpy().flatten()

        last_gae_lam = 0
        if node:
            for step in reversed(range(self.buffer_size)):
                if step == self.buffer_size - 1:
                    next_non_terminal = 1.0 - dones
                    next_value = last_value
                else:
                    next_non_terminal = 1.0 - self.dones[step + 1]
                    next_value = self.node_values[step + 1]
                delta = self.rewards[step] + self.gamma * next_value * next_non_terminal - self.node_values[step]
                last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
                self.node_advantages[step] = last_gae_lam
            self.node_returns = self.node_advantages + self.node_values
        else:
            for step in reversed(range(self.buffer_size)):
                if step == self.buffer_size - 1:
                    next_non_terminal = 1.0 - dones
                    next_value = last_value
                else:
                    next_non_terminal = 1.0 - self.dones[step + 1]
                    next_value = self.at_values[step + 1]
                delta = self.rewards[step] + self.gamma * next_value * next_non_terminal - self.at_values[step]
                last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
                self.at_advantages[step] = last_gae_lam
            self.at_returns = self.at_advantages + self.at_values

    def add(self,
            node_obs: np.ndarray,
            at_obs: np.ndarray,
            node_action: np.ndarray,
            reward: np.ndarray,
            done: np.ndarray,
            node_value: th.Tensor,
            node_log_prob: th.Tensor,
            at_action : np.ndarray,
            at_log_prob: th.Tensor,
            at_value: th.Tensor,
            node_state: th.Tensor,
            at_state: th.Tensor) -> None:
        """
        :param node_obs: (np.ndarray) Observation
        :param node_action: (np.ndarray) Action
        :param reward: (np.ndarray)
        :param done: (np.ndarray) End of episode signal.
        :param node_value: (th.Tensor) estimated value of the current state
            following the current policy.
        :param node_log_prob: (th.Tensor) log probability of the action
            following the current policy.
        """
        if len(node_log_prob.shape) == 0:
            # Reshape 0-d tensor to avoid error
            node_log_prob = node_log_prob.reshape(-1, 1)

        self.node_observations[self.pos] = np.array(node_obs).copy()
        self.at_observations[self.pos] = np.array(at_obs).copy()
        self.node_actions[self.pos] = np.array(node_action).copy()
        self.at_actions[self.pos] = np.array(at_action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.dones[self.pos] = np.array(done).copy()
        self.node_values[self.pos] = node_value.clone().cpu().numpy().flatten()
        self.at_values[self.pos] = at_value.clone().cpu().numpy().flatten()
        self.node_log_probs[self.pos] = node_log_prob.clone().cpu().numpy()
        self.at_log_probs[self.pos] = at_log_prob.clone().cpu().numpy()
        self.node_h_states[self.pos] = node_state[0].clone().cpu().numpy()
        self.node_c_states[self.pos] = node_state[1].clone().cpu().numpy()
        self.at_h_states[self.pos] = at_state[0].clone().cpu().numpy()
        self.at_c_states[self.pos] = at_state[1].clone().cpu().numpy()
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def get(self, batch_size: Optional[int] = None) -> Generator[RolloutBufferSamplesARRecurrent, None, None]:
        assert self.full, ''
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        # Prepare the data
        if not self.generator_ready:
            for tensor in ['node_observations', 'node_actions', 'node_values',
                           'node_log_probs', 'node_advantages', 'node_returns', 'at_actions', 'at_log_probs', 'at_observations',
                           'at_values', 'at_returns', 'at_advantages', 'node_h_states', 'node_c_states', 'dones',
                           'at_h_states', 'at_c_states']:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx:start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(self, batch_inds: np.ndarray,
                     env: Optional[VecNormalize] = None) -> RolloutBufferSamplesARRecurrent:
        data = (self.node_observations[batch_inds],
                self.node_actions[batch_inds],
                self.node_values[batch_inds].flatten(),
                self.node_log_probs[batch_inds].flatten(),
                self.node_advantages[batch_inds].flatten(),
                self.node_returns[batch_inds].flatten(),
                self.at_actions[batch_inds],
                self.at_log_probs[batch_inds].flatten(),
                self.at_observations[batch_inds],
                self.at_values[batch_inds].flatten(),
                self.at_returns[batch_inds].flatten(),
                self.at_advantages[batch_inds].flatten(),
                self.node_h_states[batch_inds],
                self.node_c_states[batch_inds],
                self.dones[batch_inds],
                self.at_h_states[batch_inds],
                self.at_c_states[batch_inds]
                )
        return RolloutBufferSamplesARRecurrent(*tuple(map(self.to_torch, data)))


class RolloutBufferARRecurrentMultiHead(BaseBuffer):
    """
    Rollout buffer used in on-policy algorithms like A2C/PPO.

    :param buffer_size: (int) Max number of element in the buffer
    :param observation_space: (spaces.Space) Observation space
    :param action_space: (spaces.Space) Action space
    :param device: (th.device)
    :param gae_lambda: (float) Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        Equivalent to classic advantage when set to 1.
    :param gamma: (float) Discount factor
    :param n_envs: (int) Number of parallel environments
    """

    def __init__(self,
                 buffer_size: int,
                 observation_space: spaces.Space,
                 action_space: spaces.Space,
                 device: Union[th.device, str] = 'cpu',
                 gae_lambda: float = 1,
                 gamma: float = 0.99,
                 n_envs: int = 1,
                 pg_agent_config : PolicyGradientAgentConfig = None,):

        super(RolloutBufferARRecurrentMultiHead, self).__init__(buffer_size, observation_space,
                                            action_space, device, n_envs=n_envs)
        self.pg_agent_config = pg_agent_config
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.node_observations_1, self.node_observations_2, self.node_observations_3, self.node_observations_4, \
        self.at_observations, self.node_actions, self.at_actions, self.rewards, \
        self.node_advantages, self.at_advantages = None, None, None, None, None, None, None, None, None, None
        self.node_returns, self.at_returns, self.dones, self.node_values, self.at_values, self.node_log_probs, \
        self.at_log_probs, self.node_h_states, self.node_c_states, self.at_h_states, \
        self.at_c_states = None, None, None, None, None, None, None, None, None, None, None
        self.generator_ready = False
        self.reset()

    def reset(self) -> None:
        self.node_observations_1 = np.zeros((self.buffer_size, self.n_envs, self.pg_agent_config.channel_1_input_dim),dtype=np.float32)
        self.node_observations_2 = np.zeros((self.buffer_size, self.n_envs, self.pg_agent_config.channel_2_input_dim),dtype=np.float32)
        self.node_observations_3 = np.zeros((self.buffer_size, self.n_envs, self.pg_agent_config.channel_3_input_dim),dtype=np.float32)
        self.node_observations_4 = np.zeros((self.buffer_size, self.n_envs, self.pg_agent_config.channel_4_input_dim),dtype=np.float32)
        self.at_observations = np.zeros((self.buffer_size, self.n_envs, self.pg_agent_config.at_net_input_dim), dtype=np.float32)
        self.node_actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=np.float32)
        self.at_actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.node_returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.at_returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.node_values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.at_values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.node_log_probs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.at_log_probs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.node_advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.at_advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.generator_ready = False
        self.node_h_states = np.zeros((self.buffer_size, self.n_envs, self.pg_agent_config.num_lstm_layers, 1,
                                       self.pg_agent_config.lstm_hidden_dim), dtype=np.float32)
        self.node_c_states = np.zeros((self.buffer_size, self.n_envs, self.pg_agent_config.num_lstm_layers, 1,
                                       self.pg_agent_config.lstm_hidden_dim), dtype=np.float32)
        self.at_h_states = np.zeros((self.buffer_size, self.n_envs, self.pg_agent_config.num_lstm_layers, 1,
                                       self.pg_agent_config.lstm_hidden_dim), dtype=np.float32)
        self.at_c_states = np.zeros((self.buffer_size, self.n_envs, self.pg_agent_config.num_lstm_layers, 1,
                                       self.pg_agent_config.lstm_hidden_dim), dtype=np.float32)
        super(RolloutBufferARRecurrentMultiHead, self).reset()

    def compute_returns_and_advantage(self,
                                      last_value: th.Tensor,
                                      dones: np.ndarray, node :bool = True) -> None:
        """
        Post-processing step: compute the returns (sum of discounted rewards)
        and GAE advantage.
        Adapted from Stable-Baselines PPO2.

        Uses Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)
        to compute the advantage. To obtain vanilla advantage (A(s) = R - V(S))
        where R is the discounted reward with value bootstrap,
        set ``gae_lambda=1.0`` during initialization.

        :param last_value: (th.Tensor)
        :param dones: (np.ndarray)

        """
        # convert to numpy
        last_value = last_value.clone().cpu().numpy().flatten()

        last_gae_lam = 0
        if node:
            for step in reversed(range(self.buffer_size)):
                if step == self.buffer_size - 1:
                    next_non_terminal = 1.0 - dones
                    next_value = last_value
                else:
                    next_non_terminal = 1.0 - self.dones[step + 1]
                    next_value = self.node_values[step + 1]
                delta = self.rewards[step] + self.gamma * next_value * next_non_terminal - self.node_values[step]
                last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
                self.node_advantages[step] = last_gae_lam
            self.node_returns = self.node_advantages + self.node_values
        else:
            for step in reversed(range(self.buffer_size)):
                if step == self.buffer_size - 1:
                    next_non_terminal = 1.0 - dones
                    next_value = last_value
                else:
                    next_non_terminal = 1.0 - self.dones[step + 1]
                    next_value = self.at_values[step + 1]
                delta = self.rewards[step] + self.gamma * next_value * next_non_terminal - self.at_values[step]
                last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
                self.at_advantages[step] = last_gae_lam
            self.at_returns = self.at_advantages + self.at_values

    def add(self,
            node_obs_1: np.ndarray,
            node_obs_2: np.ndarray,
            node_obs_3: np.ndarray,
            node_obs_4: np.ndarray,
            at_obs: np.ndarray,
            node_action: np.ndarray,
            reward: np.ndarray,
            done: np.ndarray,
            node_value: th.Tensor,
            node_log_prob: th.Tensor,
            at_action : np.ndarray,
            at_log_prob: th.Tensor,
            at_value: th.Tensor,
            node_state: th.Tensor,
            at_state: th.Tensor) -> None:
        """
        :param node_obs_1: (np.ndarray) Observation
        :param node_action: (np.ndarray) Action
        :param reward: (np.ndarray)
        :param done: (np.ndarray) End of episode signal.
        :param node_value: (th.Tensor) estimated value of the current state
            following the current policy.
        :param node_log_prob: (th.Tensor) log probability of the action
            following the current policy.
        """
        if len(node_log_prob.shape) == 0:
            # Reshape 0-d tensor to avoid error
            node_log_prob = node_log_prob.reshape(-1, 1)

        self.node_observations_1[self.pos] = np.array(node_obs_1).copy()
        self.node_observations_2[self.pos] = np.array(node_obs_2).copy()
        self.node_observations_3[self.pos] = np.array(node_obs_3).copy()
        self.node_observations_4[self.pos] = np.array(node_obs_4).copy()
        self.at_observations[self.pos] = np.array(at_obs).copy()
        self.node_actions[self.pos] = np.array(node_action).copy()
        self.at_actions[self.pos] = np.array(at_action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.dones[self.pos] = np.array(done).copy()
        self.node_values[self.pos] = node_value.clone().cpu().numpy().flatten()
        self.at_values[self.pos] = at_value.clone().cpu().numpy().flatten()
        self.node_log_probs[self.pos] = node_log_prob.clone().cpu().numpy()
        self.at_log_probs[self.pos] = at_log_prob.clone().cpu().numpy()
        if node_state is not None:
            self.node_h_states[self.pos] = node_state[0].clone().cpu().numpy()
            self.node_c_states[self.pos] = node_state[1].clone().cpu().numpy()
        if at_state is not None:
            self.at_h_states[self.pos] = at_state[0].clone().cpu().numpy()
            self.at_c_states[self.pos] = at_state[1].clone().cpu().numpy()
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def get(self, batch_size: Optional[int] = None) -> Generator[RolloutBufferSamplesARRecurrentMultiHead, None, None]:
        assert self.full, ''
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        # Prepare the data
        if not self.generator_ready:
            for tensor in ['node_observations_1', 'node_observations_2', 'node_observations_3', 'node_observations_4',
                           'node_actions', 'node_values',
                           'node_log_probs', 'node_advantages', 'node_returns', 'at_actions', 'at_log_probs', 'at_observations',
                           'at_values', 'at_returns', 'at_advantages', 'node_h_states', 'node_c_states', 'dones',
                           'at_h_states', 'at_c_states']:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx:start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(self, batch_inds: np.ndarray,
                     env: Optional[VecNormalize] = None) -> RolloutBufferSamplesARRecurrentMultiHead:
        data = (self.node_observations_1[batch_inds],
                self.node_observations_2[batch_inds],
                self.node_observations_3[batch_inds],
                self.node_observations_4[batch_inds],
                self.node_actions[batch_inds],
                self.node_values[batch_inds].flatten(),
                self.node_log_probs[batch_inds].flatten(),
                self.node_advantages[batch_inds].flatten(),
                self.node_returns[batch_inds].flatten(),
                self.at_actions[batch_inds],
                self.at_log_probs[batch_inds].flatten(),
                self.at_observations[batch_inds],
                self.at_values[batch_inds].flatten(),
                self.at_returns[batch_inds].flatten(),
                self.at_advantages[batch_inds].flatten(),
                self.node_h_states[batch_inds],
                self.node_c_states[batch_inds],
                self.dones[batch_inds],
                self.at_h_states[batch_inds],
                self.at_c_states[batch_inds]
                )
        return RolloutBufferSamplesARRecurrentMultiHead(*tuple(map(self.to_torch, data)))
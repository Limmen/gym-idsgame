"""
Common aliases for type hint
"""
from typing import Union, Dict, Any, NamedTuple, List, Callable, Tuple

import numpy as np
import torch as th
import gym

from gym_idsgame.agents.training_agents.openai_baselines.common.vec_env import VecEnv
from gym_idsgame.agents.training_agents.openai_baselines.common.callbacks import BaseCallback


GymEnv = Union[gym.Env, VecEnv]
GymObs = Union[Tuple, Dict[str, Any], np.ndarray, int]
GymStepReturn = Tuple[GymObs, float, bool, Dict]
TensorDict = Dict[str, th.Tensor]
OptimizerStateDict = Dict[str, Any]
MaybeCallback = Union[None, Callable, List[BaseCallback], BaseCallback]


class RolloutBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor

class RolloutBufferSamplesRecurrent(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    h_states: th.Tensor
    c_states: th.Tensor
    dones: th.Tensor


class RolloutBufferSamplesRecurrentMultiHead(NamedTuple):
    observations_1: th.Tensor
    observations_2: th.Tensor
    observations_3: th.Tensor
    observations_4: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    h_states: th.Tensor
    c_states: th.Tensor
    dones: th.Tensor


class ReplayBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    next_observations: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor


class RolloutReturn(NamedTuple):
    episode_reward: float
    episode_timesteps: int
    n_episodes: int
    continue_training: bool


class RolloutBufferSamplesAR(NamedTuple):
    node_observations: th.Tensor
    node_actions: th.Tensor
    node_old_values: th.Tensor
    node_old_log_prob: th.Tensor
    node_advantages: th.Tensor
    node_returns: th.Tensor
    at_actions: th.Tensor
    at_old_log_prob: th.Tensor
    at_observations: th.Tensor
    at_old_values: th.Tensor
    at_returns: th.Tensor
    at_advantages: th.Tensor


class RolloutBufferSamplesARRecurrent(NamedTuple):
    node_observations: th.Tensor
    node_actions: th.Tensor
    node_old_values: th.Tensor
    node_old_log_prob: th.Tensor
    node_advantages: th.Tensor
    node_returns: th.Tensor
    at_actions: th.Tensor
    at_old_log_prob: th.Tensor
    at_observations: th.Tensor
    at_old_values: th.Tensor
    at_returns: th.Tensor
    at_advantages: th.Tensor
    node_h_states: th.Tensor
    node_c_states: th.Tensor
    dones: th.Tensor
    at_h_states: th.Tensor
    at_c_states: th.Tensor

class RolloutBufferSamplesARRecurrentMultiHead(NamedTuple):
    node_observations_1: th.Tensor
    node_observations_2: th.Tensor
    node_observations_3: th.Tensor
    node_observations_4: th.Tensor
    node_actions: th.Tensor
    node_old_values: th.Tensor
    node_old_log_prob: th.Tensor
    node_advantages: th.Tensor
    node_returns: th.Tensor
    at_actions: th.Tensor
    at_old_log_prob: th.Tensor
    at_observations: th.Tensor
    at_old_values: th.Tensor
    at_returns: th.Tensor
    at_advantages: th.Tensor
    node_h_states: th.Tensor
    node_c_states: th.Tensor
    dones: th.Tensor
    at_h_states: th.Tensor
    at_c_states: th.Tensor
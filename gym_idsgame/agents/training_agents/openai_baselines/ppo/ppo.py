import time
from typing import List, Tuple, Type, Union, Callable, Optional, Dict, Any

import gym
from gym import spaces
import torch as th
import torch.nn.functional as F

# Check if tensorboard is available for pytorch
# TODO: finish tensorboard integration
# try:
#     from torch.utils.tensorboard import SummaryWriter
# except ImportError:
#     SummaryWriter = None
import numpy as np

from torch.utils.tensorboard import SummaryWriter
from gym_idsgame.agents.training_agents.openai_baselines.base_class import BaseRLModel
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.utils import explained_variance, get_schedule_fn
from stable_baselines3.common.vec_env import VecEnv
from gym_idsgame.agents.training_agents.openai_baselines.callbacks import BaseCallback
from gym_idsgame.agents.training_agents.openai_baselines.ppo.ppo_policies import PPOPolicy
from gym_idsgame.agents.training_agents.policy_gradient.pg_agent_config import PolicyGradientAgentConfig


class PPO(BaseRLModel):
    """
    Proximal Policy Optimization algorithm (PPO) (clip version)

    Paper: https://arxiv.org/abs/1707.06347
    Code: This implementation borrows code from OpenAI Spinning Up (https://github.com/openai/spinningup/)
    https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail and
    and Stable Baselines (PPO2 from https://github.com/hill-a/stable-baselines)

    Introduction to PPO: https://spinningup.openai.com/en/latest/algorithms/ppo.html

    :param policy: (PPOPolicy or str) The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: (Gym environment or str) The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: (float or callable) The learning rate, it can be a function
        of the current progress (from 1 to 0)
    :param n_steps: (int) The number of steps to run for each environment per update
        (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
    :param batch_size: (int) Minibatch size
    :param n_epochs: (int) Number of epoch when optimizing the surrogate loss
    :param gamma: (float) Discount factor
    :param gae_lambda: (float) Factor for trade-off of bias vs variance for Generalized Advantage Estimator
    :param clip_range: (float or callable) Clipping parameter, it can be a function of the current progress
        (from 1 to 0).
    :param clip_range_vf: (float or callable) Clipping parameter for the value function,
        it can be a function of the current progress (from 1 to 0).
        This is a parameter specific to the OpenAI implementation. If None is passed (default),
        no clipping will be done on the value function.
        IMPORTANT: this clipping depends on the reward scaling.
    :param ent_coef: (float) Entropy coefficient for the loss calculation
    :param vf_coef: (float) Value function coefficient for the loss calculation
    :param max_grad_norm: (float) The maximum value for the gradient clipping
    :param use_sde: (bool) Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: (int) Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param target_kl: (float) Limit the KL divergence between updates,
        because the clipping is not enough to prevent large update
        see issue #213 (cf https://github.com/hill-a/stable-baselines/issues/213)
        By default, there is no limit on the kl div.
    :param tensorboard_log: (str) the log location for tensorboard (if None, no logging)
    :param create_eval_env: (bool) Whether to create a second environment that will be
        used for evaluating the agent periodically. (Only available when passing string for the environment)
    :param policy_kwargs: (dict) additional arguments to be passed to the policy on creation
    :param verbose: (int) the verbosity level: 0 no output, 1 info, 2 debug
    :param seed: (int) Seed for the pseudo random generators
    :param device: (str or th.device) Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: (bool) Whether or not to build the network at the creation of the instance
    """

    def __init__(self, policy: Union[str, Type[PPOPolicy]],
                 env: Union[GymEnv, str],
                 learning_rate: Union[float, Callable] = 3e-4,
                 n_steps: int = 2048,
                 batch_size: Optional[int] = 64,
                 n_epochs: int = 10,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_range: float = 0.2,
                 clip_range_vf: Optional[float] = None,
                 ent_coef: float = 0.0,
                 vf_coef: float = 0.5,
                 max_grad_norm: float = 0.5,
                 use_sde: bool = False,
                 sde_sample_freq: int = -1,
                 target_kl: Optional[float] = None,
                 tensorboard_log: Optional[str] = None,
                 create_eval_env: bool = False,
                 policy_kwargs: Optional[Dict[str, Any]] = None,
                 verbose: int = 0,
                 seed: Optional[int] = None,
                 device: Union[th.device, str] = 'auto',
                 _init_setup_model: bool = True,
                 pg_agent_config : PolicyGradientAgentConfig = None):

        super(PPO, self).__init__(policy, env, PPOPolicy, learning_rate, policy_kwargs=policy_kwargs,
                                  verbose=verbose, device=device, use_sde=use_sde, sde_sample_freq=sde_sample_freq,
                                  create_eval_env=create_eval_env, support_multi_env=True, seed=seed,
                                  pg_agent_config=pg_agent_config)

        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.n_steps = n_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.rollout_buffer = None
        self.target_kl = target_kl
        self.tensorboard_log = tensorboard_log
        self.tb_writer = None
        self.pg_agent_config = pg_agent_config
        self.iteration = 0
        self.tensorboard_writer = SummaryWriter(self.pg_agent_config.tensorboard_dir)
        self.tensorboard_writer.add_hparams(self.pg_agent_config.hparams_dict(), {})

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        self.rollout_buffer = RolloutBuffer(self.n_steps, self.observation_space,
                                            self.action_space, self.device,
                                            gamma=self.gamma, gae_lambda=self.gae_lambda,
                                            n_envs=self.n_envs)
        #self.policy_class = "experiments.training.v16.minimal_defense.openai_baselines.ppo_policies.PPOPolicy"
        self.policy = PPOPolicy(self.observation_space, self.action_space,
                                        self.lr_schedule, use_sde=self.use_sde, device=self.device,
                                        **self.policy_kwargs)
        self.policy = self.policy.to(self.device)

        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, ('`clip_range_vf` must be positive, '
                                                'pass `None` to deactivate vf clipping')

            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)

    def predict(self, observation: np.ndarray,
                state: Optional[np.ndarray] = None,
                mask: Optional[np.ndarray] = None,
                deterministic: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Get the model's action(s) from an observation

        :param observation: (np.ndarray) the input observation
        :param state: (Optional[np.ndarray]) The last states (can be None, used in recurrent policies)
        :param mask: (Optional[np.ndarray]) The last masks (can be None, used in recurrent policies)
        :param deterministic: (bool) Whether or not to return deterministic actions.
        :return: (Tuple[np.ndarray, Optional[np.ndarray]]) the model's action and the next state
            (used in recurrent policies)
        """
        return self.policy._predict(observation, self.env.envs[0], deterministic, device=self.device)

    def collect_rollouts(self,
                         env: VecEnv,
                         callback: BaseCallback,
                         rollout_buffer: RolloutBuffer,
                         n_rollout_steps: int = 256) -> Union[bool, List, List, List]:

        assert self._last_obs is not None, "No previous observation was provided"
        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        # Avg metrics
        episode_attacker_rewards = []
        episode_defender_rewards = []
        episode_steps = []

        # Per episode metrics
        episode_attacker_reward = 0
        episode_defender_reward = 0
        episode_step = 0

        callback.on_rollout_start()
        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor
                obs_tensor = th.as_tensor(self._last_obs).to(self.device)
                actions, values, log_probs = self.policy.forward(obs_tensor, self.env.envs[0], device=self.device)
            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, gym.spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            if callback.on_step() is False:
                return False


            # Record episode metrics
            self._update_info_buffer(infos)
            n_steps += 1
            self.num_timesteps += env.num_envs
            episode_attacker_reward += rewards
            episode_step +=1
            # Record episode metrics
            self.num_train_games += 1
            self.num_train_games_total += 1
            if env.envs[0].prev_episode_hacked:
                self.num_train_hacks += 1
                self.num_train_hacks_total += 1

            if isinstance(self.action_space, gym.spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)
            rollout_buffer.add(self._last_obs, actions, rewards, dones, values, log_probs)
            self._last_obs = new_obs

            if dones:
                episode_attacker_rewards.append(episode_attacker_reward)
                episode_defender_rewards.append(episode_defender_reward)
                episode_steps.append(episode_step)
                episode_attacker_reward = 0
                episode_defender_reward = 0
                episode_step = 0

        rollout_buffer.compute_returns_and_advantage(values, dones=dones)

        callback.on_rollout_end()

        return True, episode_attacker_rewards, episode_defender_rewards, episode_steps

    def train(self, n_epochs: int, batch_size: int = 64) -> None:
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        lr = self.policy.optimizer.param_groups[0]["lr"]
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress)
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress)

        entropy_losses, all_kl_divs = [], []
        pg_losses, value_losses = [], []
        clip_fractions = []

        # train for gradient_steps epochs
        for epoch in range(n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                # Re-sample the noise matrix because the log_std has changed
                # TODO: investigate why there is no issue with the gradient
                # if that line is commented (as in SAC)
                if self.use_sde:
                    self.policy.reset_noise(batch_size)

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)
                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the different between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(values - rollout_data.old_values, -clip_range_vf,
                                                                     clip_range_vf)
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -log_prob.mean()
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()
                approx_kl_divs.append(th.mean(rollout_data.old_log_prob - log_prob).detach().cpu().numpy())

            all_kl_divs.append(np.mean(approx_kl_divs))

            if self.target_kl is not None and np.mean(approx_kl_divs) > 1.5 * self.target_kl:
                print(f"Early stopping at step {epoch} due to reaching max kl: {np.mean(approx_kl_divs):.2f}")
                break

        self._n_updates += n_epochs
        explained_var = explained_variance(self.rollout_buffer.returns.flatten(),
                                           self.rollout_buffer.values.flatten())

        # logger.logkv("n_updates", self._n_updates)
        # logger.logkv("clip_fraction", np.mean(clip_fraction))
        # logger.logkv("clip_range", clip_range)
        # if self.clip_range_vf is not None:
        #     logger.logkv("clip_range_vf", clip_range_vf)
        #
        # logger.logkv("approx_kl", np.mean(approx_kl_divs))
        # logger.logkv("explained_variance", explained_var)
        # logger.logkv("entropy_loss", np.mean(entropy_losses))
        # logger.logkv("policy_gradient_loss", np.mean(pg_losses))
        # logger.logkv("value_loss", np.mean(value_losses))
        # if hasattr(self.policy, 'log_std'):
        #     logger.logkv("std", th.exp(self.policy.log_std).mean().item())

        return np.mean(entropy_losses), np.mean(pg_losses), np.mean(value_losses), lr

    def learn(self,
              total_timesteps: int,
              callback: MaybeCallback = None,
              log_interval: int = 1,
              eval_env: Optional[GymEnv] = None,
              eval_freq: int = -1,
              n_eval_episodes: int = 5,
              tb_log_name: str = "PPO",
              eval_log_path: Optional[str] = None,
              reset_num_timesteps: bool = True) -> 'PPO':

        self.pg_agent_config.logger.info("Setting up Training Configuration")
        print("Setting up Training Configuration")
        self.iteration = 0
        callback = self._setup_learn(eval_env, callback, eval_freq,
                                     n_eval_episodes, eval_log_path, reset_num_timesteps)

        # if self.tensorboard_log is not None and SummaryWriter is not None:
        #     self.tb_writer = SummaryWriter(log_dir=os.path.join(self.tensorboard_log, tb_log_name))

        callback.on_training_start(locals(), globals())
        self.pg_agent_config.logger.info("Starting training, max time steps:{}".format(total_timesteps))
        print("Starting training, max time steps:{}".format(total_timesteps))
        self.pg_agent_config.logger.info(self.pg_agent_config.to_str())

        # Tracking metrics

        # Avg metrics
        episode_attacker_rewards = []
        episode_defender_rewards = []
        episode_steps = []
        episode_avg_attacker_loss = []
        episode_avg_defender_loss = []
        attacker_lr = 0.0
        defender_lr = 0.0

        while self.num_timesteps < total_timesteps:
            continue_training, rollouts_attacker_rewards, rollouts_defender_rewards, rollouts_steps = \
                self.collect_rollouts(self.env, callback, self.rollout_buffer, n_rollout_steps=self.n_steps)
            episode_attacker_rewards.extend(rollouts_attacker_rewards)
            episode_defender_rewards.extend(rollouts_defender_rewards)
            episode_steps.extend(rollouts_defender_rewards)

            if continue_training is False:
                break

            self.iteration += 1
            self._update_current_progress(self.num_timesteps, total_timesteps)

            # Display training infos
            if self.iteration % self.pg_agent_config.train_log_frequency == 0:
                if self.num_train_games > 0 and self.num_train_games_total > 0:
                    self.train_hack_probability = self.num_train_hacks / self.num_train_games
                    self.train_cumulative_hack_probability = self.num_train_hacks_total / self.num_train_games_total
                else:
                    self.train_hack_probability = 0.0
                    self.train_cumulative_hack_probability = 0.0
                self.log_metrics(iteration=self.iteration, result=self.train_result,
                                 attacker_episode_rewards=episode_attacker_rewards,
                                 defender_episode_rewards=episode_defender_rewards, episode_steps=episode_steps,
                                 episode_avg_attacker_loss=episode_avg_attacker_loss,
                                 episode_avg_defender_loss=episode_avg_defender_loss,
                                 eval=False, update_stats=True, lr_attacker=attacker_lr, lr_defender=None,
                                 total_num_episodes=self.num_train_games_total)
                episode_attacker_rewards = []
                episode_defender_rewards = []
                episode_avg_attacker_loss = []
                episode_avg_defender_loss = []
                episode_steps = []
                self.num_train_games = 0
                self.num_train_hacks = 0

            # Save models every <self.config.checkpoint_frequency> iterations
            if self.iteration % self.pg_agent_config.checkpoint_freq == 0:
                self.save_model()
                if self.pg_agent_config.save_dir is not None:
                    time_str = str(time.time())
                    self.train_result.to_csv(
                        self.pg_agent_config.save_dir + "/" + time_str + "_train_results_checkpoint.csv")
                    self.eval_result.to_csv(self.pg_agent_config.save_dir + "/" + time_str + "_eval_results_checkpoint.csv")

            entropy_loss, pg_loss, value_loss, attacker_lr = self.train(self.n_epochs, batch_size=self.batch_size)
            episode_avg_attacker_loss.append(entropy_loss + pg_loss + value_loss)



            # For tensorboard integration
            # if self.tb_writer is not None:
            #     self.tb_writer.add_scalar('Eval/reward', mean_reward, self.num_timesteps)

        callback.on_training_end()

        return self

    def get_torch_variables(self) -> Tuple[List[str], List[str]]:
        """
        cf base class
        """
        state_dicts = ["policy", "policy.optimizer"]

        return state_dicts, []

    def save_model(self) -> None:
        """
        Saves the PyTorch Model Weights

        :return: None
        """
        time_str = str(time.time())
        if self.pg_agent_config.save_dir is not None:
            if self.pg_agent_config.attacker:
                path = self.pg_agent_config.save_dir + "/" + time_str + "_attacker_policy_network.zip"
                self.pg_agent_config.logger.info("Saving policy-network to: {}".format(path))
                print("Saving policy-network to: {}".format(path))
                self.save(path)
            # if self.config.defender:
            #     path = self.config.save_dir + "/" + time_str + "_defender_policy_network.pt"
            #     self.config.logger.info("Saving policy-network to: {}".format(path))
            #     torch.save(self.defender_policy_network.state_dict(), path)
        else:
            self.config.logger.warning("Save path not defined, not saving policy-networks to disk")
            print("Save path not defined, not saving policy-networks to disk")
"""
An agent for the IDSGameEnv that implements the REINFORCE with Baseline (Critic) Policy Gradient algorithm.
"""
from typing import Union, List
import numpy as np
import time
import tqdm
import torch
import copy
from scipy.special import softmax
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from gym_idsgame.envs.rendering.video.idsgame_monitor import IdsGameMonitor
from gym_idsgame.envs.idsgame_env import IdsGameEnv
from gym_idsgame.agents.dao.experiment_result import ExperimentResult
from gym_idsgame.envs.constants import constants
from gym_idsgame.agents.training_agents.policy_gradient.actor_critic.model import ActorCriticNN
from gym_idsgame.agents.training_agents.policy_gradient.pg_agent import PolicyGradientAgent
from gym_idsgame.agents.training_agents.policy_gradient.pg_agent_config import PolicyGradientAgentConfig
from gym_idsgame.envs.util import idsgame_util

class ActorCriticAgent(PolicyGradientAgent):
    """
    An implementation of the REINFORCE with Advantage Baseline (Actor Critic) Policy Gradient algorithm
    """
    def __init__(self, env:IdsGameEnv, config: PolicyGradientAgentConfig):
        """
        Initialize environment and hyperparameters

        :param config: the configuration
        """
        super(ActorCriticAgent, self).__init__(env, config)
        self.attacker_policy_network = None
        self.defender_policy_network = None
        self.critic_loss_fn = None
        self.attacker_optimizer = None
        self.defender_optimizer = None
        self.attacker_lr_decay = None
        self.defender_lr_decay = None
        self.tensorboard_writer = SummaryWriter(self.config.tensorboard_dir)
        if self.config.opponent_pool and self.config.opponent_pool_config is not None:
            self.attacker_pool = []
            self.defender_pool = []
        self.initialize_models()
        self.tensorboard_writer.add_hparams(self.config.hparams_dict(), {})
        self.machine_eps = np.finfo(np.float32).eps.item()
        self.env.idsgame_config.save_trajectories = False
        self.env.idsgame_config.save_attack_stats = False
        self.train_attacker = True
        self.train_defender = True

    def initialize_models(self) -> None:
        """
        Initialize models
        :return: None
        """

        # Initialize models
        self.attacker_policy_network = ActorCriticNN(self.config.input_dim, self.config.output_dim_attacker,
                                                     self.config.hidden_dim,
                                                     num_hidden_layers=self.config.num_hidden_layers,
                                                     hidden_activation=self.config.hidden_activation)
        self.defender_policy_network = ActorCriticNN(self.config.input_dim, self.config.output_dim_defender,
                                                     self.config.hidden_dim,
                                                     num_hidden_layers=self.config.num_hidden_layers,
                                                     hidden_activation=self.config.hidden_activation)

        # Specify device
        if torch.cuda.is_available() and self.config.gpu:
            device = torch.device("cuda:" + str(self.config.gpu_id))
            self.config.logger.info("Running on the GPU")
        else:
            device = torch.device("cpu")
            self.config.logger.info("Running on the CPU")

        self.attacker_policy_network.to(device)
        self.defender_policy_network.to(device)

        # Construct loss function
        if self.config.critic_loss_fn == "MSE":
            self.critic_loss_fn = torch.nn.MSELoss()
        elif self.config.critic_loss_fn == "Huber":
            self.critic_loss_fn = torch.nn.SmoothL1Loss()
        else:
            raise ValueError("Loss function not recognized")

        # Define Optimizer. The call to model.parameters() in the optimizer constructor will contain the learnable
        # parameters of the layers in the model
        if self.config.optimizer == "Adam":
            self.attacker_optimizer = torch.optim.Adam(self.attacker_policy_network.parameters(), lr=self.config.alpha_attacker)
            self.defender_optimizer = torch.optim.Adam(self.defender_policy_network.parameters(), lr=self.config.alpha_defender)
        elif self.config.optimizer == "SGD":
            self.attacker_optimizer = torch.optim.SGD(self.attacker_policy_network.parameters(), lr=self.config.alpha_attacker)
            self.defender_optimizer = torch.optim.SGD(self.defender_policy_network.parameters(), lr=self.config.alpha_defender)
        else:
            raise ValueError("Optimizer not recognized")

        # LR decay
        if self.config.lr_exp_decay:
            self.attacker_lr_decay = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.attacker_optimizer,
                                                                       gamma=self.config.lr_decay_rate)
            self.defender_lr_decay = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.defender_optimizer,
                                                                            gamma=self.config.lr_decay_rate)

        self.add_model_to_pool(attacker=True)
        self.add_model_to_pool(attacker=False)

    def add_model_to_pool(self, attacker=True) -> None:
        """
        Adds a model to the pool of opponents

        :param attacker: boolean flag indicating whether adding attacker model or defender model
        :return: None
        """
        if self.config.opponent_pool and self.config.opponent_pool_config is not None:
            if attacker:
                model_copy = copy.deepcopy(self.attacker_policy_network)
                if len(self.attacker_pool) >= self.config.opponent_pool_config.pool_maxsize:
                    self.attacker_pool.pop(0)
                if self.config.opponent_pool_config.quality_scores:
                    if len(self.attacker_pool) == 0:
                        self.attacker_pool.append([model_copy, self.config.opponent_pool_config.initial_quality])
                    elif len(self.attacker_pool) > 0:
                        qualities = self.get_attacker_pool_quality_scores()
                        max_q = max(qualities)
                        self.attacker_pool.append([model_copy, max_q])
                else:
                    self.attacker_pool.append(model_copy)
            else:
                model_copy = copy.deepcopy(self.defender_policy_network)
                if len(self.defender_pool) >= self.config.opponent_pool_config.pool_maxsize:
                    self.defender_pool.pop(0)
                if self.config.opponent_pool_config.quality_scores:
                    if len(self.defender_pool) == 0:
                        self.defender_pool.append([model_copy, self.config.opponent_pool_config.initial_quality])
                    elif len(self.defender_pool) > 0:
                        qualities = self.get_defender_pool_quality_scores()
                        max_q = max(qualities)
                        self.defender_pool.append([model_copy, max_q])
                else:
                    self.defender_pool.append(model_copy)

    def sample_opponent(self, attacker=True):
        if attacker:
            if self.config.opponent_pool_config.quality_scores:
                quality_scores = self.get_attacker_pool_quality_scores()
                softmax_dist = self.get_softmax_distribution(quality_scores)
                return np.random.choice(list(range(len(self.attacker_pool))), size=1, p=softmax_dist)[0]
            else:
                return np.random.choice(list(range(len(self.attacker_pool))), size=1)[0]
        else:
            if self.config.opponent_pool_config.quality_scores:
                quality_scores = self.get_defender_pool_quality_scores()
                softmax_dist = self.get_softmax_distribution(quality_scores)
                return np.random.choice(list(range(len(self.defender_pool))), size=1, p=softmax_dist)[0]
            else:
                return np.random.choice(list(range(len(self.defender_pool))), size=1)[0]

    def get_softmax_distribution(self, qualities) -> np.ndarray:
        """
        Converts a list of quality scores into a distribution with softmax

        :param qualities: the list of quality scores
        :return: the softmax distribution
        """
        return softmax(qualities)

    def get_attacker_pool_quality_scores(self):
        """
        :return: Returns the quality scores from the attacker pool
        """
        return list(map(lambda x: x[1], self.attacker_pool))

    def get_defender_pool_quality_scores(self):
        """
        :return: Returns the quality scores from the defender pool
        """
        return list(map(lambda x: x[1], self.defender_pool))

    def training_step(self, saved_rewards : List[List[float]], saved_log_probs : List[List[torch.Tensor]],
                      saved_state_values : List[List[torch.Tensor]], attacker=True) -> torch.Tensor:
        """
        Performs a training step of the Deep-Q-learning algorithm (implemented in PyTorch)

        :param saved_rewards list of rewards encountered in the latest episode trajectory
        :param saved_log_probs list of log-action probabilities (log p(a|s)) encountered in the latest episode trajectory
        :param saved_state_values list of state values encountered in the latest episode trajectory
        :return: loss
        """

        policy_loss = [] # list to save actor (policy) loss
        value_loss = []  # list to save critic (value) loss
        num_batches = len(saved_rewards)

        for batch in range(num_batches):
            R = 0
            returns = []  # list to save the true (observed) values
            # Create discounted returns. When episode is finished we can go back and compute the observed cumulative
            # discounted reward by using the observed rewards
            for r in saved_rewards[batch][::-1]:
                R = r + self.config.gamma * R
                returns.insert(0, R)
            num_rewards = len(returns)

            # convert list to torch tensor
            returns = torch.tensor(returns)

            # normalize
            std = returns.std()
            if num_rewards < 2:
                std = 0
            returns = (returns - returns.mean()) / (std + self.machine_eps)

            # Compute PG "loss" which in reality is the expected reward, which we want to maximize with gradient ascent
            for log_prob, state_value, R in zip(saved_log_probs[batch], saved_state_values[batch], returns):
                # Compute the advantage which will be used as a baseline in REINFORCE to reduce the gradient variance
                # Intuitively, the advantage tells us how much better the observed reward was compared to the expected reward
                # If the advantage of an action is high, it means that the current policy should be modified to reinforce
                # that action. That is, the advantage tells us for every action much better that action is than
                # the average action.
                advantage = R - state_value.item()

                # negative log probsince we are doing gradient descent (not ascent)
                policy_loss.append(-log_prob * advantage)

                R_tensor = torch.tensor([R])

                # Move to GPU if using GPU
                if torch.cuda.is_available() and self.config.gpu:
                    device = torch.device("cuda:" + str(self.config.gpu_id))
                    state_value = state_value.to(device)
                    R_tensor = R_tensor.to(device)


                # calculate critic loss using Huber loss
                value_loss.append(self.critic_loss_fn(state_value, R_tensor))


        # Compute gradient and update models
        if attacker:
            # reset gradients
            self.attacker_optimizer.zero_grad()
            # sum up all the values of policy losses and value losses
            total_loss = (torch.stack(policy_loss).sum() + torch.stack(value_loss).sum())
            loss = total_loss/num_batches
            # perform backprop
            loss.backward()
            # maybe clip gradient
            if self.config.clip_gradient:
                torch.nn.utils.clip_grad_norm_(self.attacker_policy_network.parameters(), 1)
            # gradient descent step
            self.attacker_optimizer.step()
        else:
            # reset gradients
            self.defender_optimizer.zero_grad()
            # sum up all the values of policy losses and value losses
            total_loss = torch.stack(policy_loss).sum() + torch.stack(value_loss).sum()
            loss = total_loss / num_batches
            # perform backprop
            loss.backward()
            # maybe clip gradient
            if self.config.clip_gradient:
                torch.nn.utils.clip_grad_norm_(self.defender_policy_network.parameters(), 1)
            # gradient descent step
            self.defender_optimizer.step()

        return loss


    def get_action(self, state: np.ndarray, attacker : bool = True, opponent_pool = False) -> Union[int, torch.Tensor, torch.Tensor, np.ndarray]:
        """
        Samples an action from the policy network

        :param state: the state to sample an action for
        :param attacker: boolean flag whether running in attacker mode (if false assume defender)
        :param opponent_pool: boolean flag, if true get model from opponent pool
        :return: The sampled action id, log probability of action id, state value, action distribution
        """
        state = torch.from_numpy(state.flatten()).float()

        # Move to GPU if using GPU
        if torch.cuda.is_available() and self.config.gpu:
            device = torch.device("cuda:" + str(self.config.gpu_id))
            state = state.to(device)

        # Calculate legal actions
        if attacker:
            actions = list(range(self.env.num_attack_actions))
            legal_actions = list(filter(lambda action: self.env.is_attack_legal(action), actions))
            non_legal_actions = list(filter(lambda action: not self.env.is_attack_legal(action), actions))
        else:
            actions = list(range(self.env.num_defense_actions))
            legal_actions = list(filter(lambda action: self.env.is_defense_legal(action), actions))
            non_legal_actions = list(filter(lambda action: not self.env.is_defense_legal(action), actions))

        # Forward pass using the current policy network to predict P(a|s)
        if attacker:
            if opponent_pool:
                action_probs, state_value = self.attacker_opponent(state)
            else:
                action_probs, state_value = self.attacker_policy_network(state)
        else:
            if opponent_pool:
                action_probs, state_value = self.defender_opponent(state)
            else:
                action_probs, state_value = self.defender_policy_network(state)

        # Set probability of non-legal actions to 0
        action_probs_1 = action_probs.clone()
        if len(legal_actions) > 0:
            action_probs_1[non_legal_actions] = 0

        # Use torch.distributions package to create a parameterizable probability distribution of the learned policy
        # PG uses a trick to turn the gradient into a stochastic gradient which we can sample from in order to
        # approximate the true gradient (which we can’t compute directly). It can be seen as an alternative to the
        # reparameterization trick
        policy_dist = Categorical(action_probs_1)

        # Sample an action from the probability distribution
        action = policy_dist.sample()

        # log_prob returns the log of the probability density/mass function evaluated at value.
        # save the log_prob as it will use later on for computing the policy gradient
        # policy gradient theorem says that the stochastic gradient of the expected return of the current policy is
        # the log gradient of the policy times the expected return, therefore we save the log of the policy distribution
        # now and use it later to compute the gradient once the episode has finished.
        log_prob = policy_dist.log_prob(action)

        return action.item(), log_prob, state_value, action_probs


    def train(self) -> ExperimentResult:
        """
        Runs the REINFORCE with Baseline algorithm

        :return: Experiment result
        """
        self.config.logger.info("Starting Training")
        self.config.logger.info(self.config.to_str())
        if len(self.train_result.avg_episode_steps) > 0:
            self.config.logger.warning("starting training with non-empty result object")
        done = False
        obs = self.env.reset(update_stats=False)
        attacker_obs, defender_obs = obs

        attacker_state = self.update_state(attacker_obs=attacker_obs, defender_obs=defender_obs, state=[], attacker=True)
        defender_state = self.update_state(defender_obs=defender_obs, attacker_obs=attacker_obs, state=[], attacker=False)

        # Tracking metrics
        episode_attacker_rewards = []
        episode_defender_rewards = []
        episode_steps = []
        episode_avg_attacker_loss = []
        episode_avg_defender_loss = []

        # Logging
        self.outer_train.set_description_str("[Train] epsilon:{:.2f},avg_a_R:{:.2f},avg_d_R:{:.2f},"
                                             "avg_t:{:.2f},avg_h:{:.2f},acc_A_R:{:.2f}," \
                                             "acc_D_R:{:.2f}".format(self.config.epsilon, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))

        train_attacker = True
        train_defender = True
        if self.config.alternating_optimization:
            train_attacker = False
            if self.config.opponent_pool and self.config.opponent_pool_config is not None:
                if np.random.rand() < self.config.opponent_pool_config.pool_prob:
                    self.defender_opponent_idx = self.sample_opponent(attacker=False)
                    if self.config.opponent_pool_config.quality_scores:
                        self.defender_opponent = self.defender_pool[self.defender_opponent_idx][0]
                    else:
                        self.defender_opponent = self.defender_pool[self.defender_opponent_idx]
                else:
                    self.defender_opponent = self.defender_policy_network
                    self.defender_opponent_idx = None

                if np.random.rand() < self.config.opponent_pool_config.pool_prob:
                    self.attacker_opponent_idx = self.sample_opponent(attacker=True)
                    if self.config.opponent_pool_config.quality_scores:
                        self.attacker_opponent = self.attacker_pool[self.attacker_opponent_idx][0]
                    else:
                        self.attacker_opponent = self.attacker_pool[self.attacker_opponent_idx]
                else:
                    self.attacker_opponent = self.attacker_policy_network
                    self.attacker_opponent_idx = None

        num_alt_iterations = 0
        num_attacker_pool_iterations = 0
        num_defender_pool_iterations = 0
        num_attacker_opponent_iterations = 0
        num_defender_opponent_iterations = 0

        attacker_initial_state_action_dist = np.zeros(self.config.output_dim_attacker)
        defender_initial_state_action_dist = np.zeros(self.config.output_dim_defender)

        num_batch_episode = 0
        saved_attacker_log_probs_batch = []
        saved_attacker_rewards_batch = []
        saved_attacker_state_values_batch = []
        saved_defender_log_probs_batch = []
        saved_defender_rewards_batch = []
        saved_defender_state_values_batch = []

        total_num_batches = 0

        # Training
        for episode in range(self.config.num_episodes):
            episode_attacker_reward = 0
            episode_defender_reward = 0
            episode_step = 0
            episode_attacker_loss = 0.0
            episode_defender_loss = 0.0
            saved_attacker_log_probs = []
            saved_attacker_rewards = []
            saved_attacker_state_values = []
            saved_defender_log_probs = []
            saved_defender_rewards = []
            saved_defender_state_values = []
            while not done:
                if self.config.render:
                    self.env.render(mode="human")

                if not self.config.attacker and not self.config.defender:
                    raise AssertionError("Must specify whether training an attacker agent or defender agent")

                # Default initialization
                attacker_action = 0
                defender_action = 0

                # Get attacker and defender actions
                if self.config.attacker:
                    if self.config.alternating_optimization and not train_attacker \
                            and self.config.opponent_pool and self.config.opponent_pool_config is not None:
                        attacker_action, attacker_log_prob, attacker_state_value, attacker_action_dist = self.get_action(
                            attacker_state, attacker=True, opponent_pool=True)
                    else:
                        attacker_action, attacker_log_prob, attacker_state_value, attacker_action_dist = \
                            self.get_action(attacker_state, attacker=True, opponent_pool=False)
                    saved_attacker_log_probs.append(attacker_log_prob)
                    saved_attacker_state_values.append(attacker_state_value)
                    if episode_step == 0:
                        attacker_initial_state_action_dist = attacker_action_dist

                if self.config.defender:
                    if self.config.alternating_optimization and not train_defender \
                            and self.config.opponent_pool and self.config.opponent_pool_config is not None:
                        defender_action, defender_log_prob, defender_state_value, defender_action_dist = self.get_action(
                            defender_state, attacker=False, opponent_pool=True)
                    else:
                        defender_action, defender_log_prob, defender_state_value, defender_action_dist = \
                            self.get_action(defender_state, attacker=False, opponent_pool=False)
                    saved_defender_log_probs.append(defender_log_prob)
                    saved_defender_state_values.append(defender_state_value)
                    if episode_step == 0:
                        defender_initial_state_action_dist = defender_action_dist

                action = (attacker_action, defender_action)

                # Take a step in the environment
                obs_prime, reward, done, _ = self.env.step(action)

                # Update metrics
                attacker_reward, defender_reward = reward
                obs_prime_attacker, obs_prime_defender = obs_prime
                episode_attacker_reward += attacker_reward
                saved_attacker_rewards.append(attacker_reward)
                episode_defender_reward += defender_reward
                saved_defender_rewards.append(defender_reward)
                episode_step += 1

                # Move to the next state
                obs = obs_prime
                attacker_obs = obs_prime_attacker
                defender_obs = obs_prime_defender
                attacker_state = self.update_state(attacker_obs=attacker_obs, defender_obs=defender_obs, state=attacker_state, attacker=True)
                defender_state = self.update_state(defender_obs=defender_obs, attacker_obs=attacker_obs, state=defender_state, attacker=False)


            # Render final frame
            if self.config.render:
                self.env.render(mode="human")

            saved_attacker_log_probs_batch.append(saved_attacker_log_probs)
            saved_attacker_rewards_batch.append(saved_attacker_rewards)
            saved_attacker_state_values_batch.append(saved_attacker_state_values)
            saved_defender_log_probs_batch.append(saved_defender_log_probs)
            saved_defender_rewards_batch.append(saved_defender_rewards)
            saved_defender_state_values_batch.append(saved_defender_state_values)
            num_batch_episode += 1

            if num_batch_episode >= self.config.batch_size:
                # Perform Policy Gradient updates
                if self.config.attacker:
                    if not self.config.alternating_optimization or \
                            (self.config.alternating_optimization and train_attacker):
                        loss = self.training_step(saved_attacker_rewards_batch, saved_attacker_log_probs_batch,
                                                  saved_attacker_state_values_batch, attacker=True)
                        episode_attacker_loss += loss.item()

                if self.config.defender:
                    if not self.config.alternating_optimization or \
                            (self.config.alternating_optimization and train_defender):
                        loss = self.training_step(saved_defender_rewards_batch, saved_defender_log_probs_batch,
                                                  saved_defender_state_values_batch, attacker=False)
                        episode_defender_loss += loss.item()

                saved_attacker_log_probs_batch = []
                saved_attacker_rewards_batch = []
                saved_attacker_state_values_batch = []
                saved_defender_log_probs_batch = []
                saved_defender_rewards_batch = []
                saved_defender_state_values_batch = []

                # Log values and gradients of the parameters (histogram summary) to tensorboard
                if self.config.attacker and train_attacker:
                    for tag, value in self.attacker_policy_network.named_parameters():
                        tag = tag.replace('.', '/')
                        self.tensorboard_writer.add_histogram(tag, value.data.cpu().numpy(), episode)
                        self.tensorboard_writer.add_histogram(tag + '_attacker/grad',
                                                              value.grad.data.cpu().numpy(),
                                                              episode)

                if self.config.defender and train_defender:
                    for tag, value in self.defender_policy_network.named_parameters():
                        tag = tag.replace('.', '/')
                        self.tensorboard_writer.add_histogram(tag, value.data.cpu().numpy(), episode)
                        self.tensorboard_writer.add_histogram(tag + '_defender/grad',
                                                              value.grad.data.cpu().numpy(),
                                                              episode)
                num_batch_episode = 0
                total_num_batches += 1
                num_alt_iterations += 1

                if episode_step > 0:
                    if self.config.attacker:
                        episode_avg_attacker_loss.append(episode_attacker_loss / episode_step)
                    if self.config.defender:
                        episode_avg_defender_loss.append(episode_defender_loss / episode_step)
                else:
                    if self.config.attacker:
                        episode_avg_attacker_loss.append(episode_attacker_loss)
                    if self.config.defender:
                        episode_avg_defender_loss.append(episode_defender_loss)

            # Decay LR after every episode
            lr_attacker = self.config.alpha_attacker
            if self.config.lr_exp_decay:
                self.attacker_lr_decay.step()
                lr_attacker = self.attacker_lr_decay.get_lr()[0]

            # Decay LR after every episode
            lr_defender = self.config.alpha_attacker
            if self.config.lr_exp_decay:
                self.defender_lr_decay.step()
                lr_defender = self.defender_lr_decay.get_lr()[0]

            # Record episode metrics
            self.num_train_games += 1
            self.num_train_games_total += 1
            if self.env.state.hacked:
                self.num_train_hacks += 1
                self.num_train_hacks_total += 1

            if self.config.alternating_optimization and self.config.opponent_pool:
                if train_attacker:
                    num_attacker_pool_iterations += 1
                    num_defender_opponent_iterations += 1

                if train_defender:
                    num_defender_pool_iterations += 1
                    num_attacker_opponent_iterations += 1

            episode_attacker_rewards.append(episode_attacker_reward)
            episode_defender_rewards.append(episode_defender_reward)
            episode_steps.append(episode_step)

            # Update opponent pool qualities
            if self.config.opponent_pool and self.config.opponent_pool_config is not None \
                    and self.config.opponent_pool_config.quality_scores:
                if train_attacker and self.defender_opponent_idx is not None and self.env.state.hacked:
                    self.update_quality_score(self.defender_opponent_idx, attacker=False)
                if train_defender and self.attacker_opponent_idx is not None and not self.env.state.hacked:
                    self.update_quality_score(self.attacker_opponent_idx, attacker=True)

            # Log average metrics every <self.config.train_log_frequency> episodes
            if episode % self.config.train_log_frequency == 0:
                if self.num_train_games > 0 and self.num_train_games_total > 0:
                    self.train_hack_probability = self.num_train_hacks / self.num_train_games
                    self.train_cumulative_hack_probability = self.num_train_hacks_total / self.num_train_games_total
                else:
                    self.train_hack_probability = 0.0
                    self.train_cumulative_hack_probability = 0.0
                a_pool = None
                d_pool = None
                if self.config.opponent_pool and self.config.opponent_pool_config is not None:
                    a_pool = len(self.attacker_pool)
                    d_pool = len(self.defender_pool)
                self.log_metrics(episode, self.train_result, episode_attacker_rewards, episode_defender_rewards, episode_steps,
                                 episode_avg_attacker_loss, episode_avg_defender_loss, lr_attacker=lr_attacker,
                                 lr_defender=lr_defender,
                                 train_attacker = (self.config.attacker and train_attacker),
                                 train_defender = (self.config.defender and train_defender),
                                 a_pool=a_pool, d_pool=d_pool, total_num_batches=total_num_batches)

                episode_attacker_rewards = []
                episode_defender_rewards = []
                episode_steps = []
                self.num_train_games = 0
                self.num_train_hacks = 0

            # Run evaluation every <self.config.eval_frequency> episodes
            if episode % self.config.eval_frequency == 0:
                self.eval(episode)
                if self.config.opponent_pool and self.config.opponent_pool_config is not None:
                    self.log_action_dist(attacker_initial_state_action_dist, attacker=True)
                    self.log_action_dist(defender_initial_state_action_dist, attacker=False)

            # Save models and other state every <self.config.checkpoint_frequency> episodes
            if episode % self.config.checkpoint_freq == 0:
                self.save_model()
                self.env.save_trajectories()
                self.env.save_attack_data(checkpoint=True)
                if self.config.save_dir is not None:
                    time_str = str(time.time())
                    self.train_result.to_csv(self.config.save_dir + "/" + time_str + "_train_results_checkpoint.csv")
                    self.eval_result.to_csv(self.config.save_dir + "/" + time_str + "_eval_results_checkpoint.csv")
                if self.config.opponent_pool and self.config.opponent_pool_config is not None:
                    self.create_policy_plot(attacker_initial_state_action_dist.data.cpu().numpy(), episode,
                                            attacker=True)
                    self.create_policy_plot(defender_initial_state_action_dist.data.cpu().numpy(), episode,
                                            attacker=False)

            # Reset environment for the next episode and update game stats
            done = False
            attacker_obs, defender_obs = self.env.reset(update_stats=True)
            attacker_state = self.update_state(attacker_obs=attacker_obs, defender_obs=defender_obs, state=[], attacker=True)
            defender_state = self.update_state(defender_obs=defender_obs, attacker_obs=attacker_obs, state=[], attacker=False)
            self.outer_train.update(1)

            # If using opponent pool, update the pool
            if self.config.opponent_pool and self.config.opponent_pool_config is not None:
                if train_attacker:
                    if num_attacker_pool_iterations > self.config.opponent_pool_config.pool_increment_period:
                        self.add_model_to_pool(attacker=True)
                        num_attacker_pool_iterations = 0

                    if num_defender_opponent_iterations > self.config.opponent_pool_config.head_to_head_period:
                        if np.random.rand() < self.config.opponent_pool_config.pool_prob:
                            self.defender_opponent_idx = self.sample_opponent(attacker=False)
                            if self.config.opponent_pool_config.quality_scores:
                                self.defender_opponent = self.defender_pool[self.defender_opponent_idx][0]
                            else:
                                self.defender_opponent = self.defender_pool[self.defender_opponent_idx]
                        else:
                            self.defender_opponent = self.defender_policy_network
                            self.defender_opponent_idx = None
                        num_defender_opponent_iterations = 0

                if train_defender:
                    if num_defender_pool_iterations > self.config.opponent_pool_config.pool_increment_period:
                        self.add_model_to_pool(attacker=False)
                        num_defender_pool_iterations = 0

                    if num_attacker_opponent_iterations > self.config.opponent_pool_config.head_to_head_period:
                        if np.random.rand() < self.config.opponent_pool_config.pool_prob:
                            self.attacker_opponent_idx = self.sample_opponent(attacker=True)
                            if self.config.opponent_pool_config.quality_scores:
                                self.attacker_opponent = self.attacker_pool[self.attacker_opponent_idx][0]
                            else:
                                self.attacker_opponent = self.attacker_pool[self.attacker_opponent_idx]
                        else:
                            self.attacker_opponent = self.attacker_policy_network
                            self.attacker_opponent_idx = None

                        num_attacker_opponent_iterations = 0

            # If doing alternating optimization and the alternating period is up, change agent that is optimized
            if self.config.alternating_optimization and num_alt_iterations > self.config.alternating_period:
                train_defender = not train_defender
                train_attacker = not train_attacker
                num_alt_iterations = 0

            # Anneal epsilon linearly
            self.anneal_epsilon()

        self.config.logger.info("Training Complete")

        # Final evaluation (for saving Gifs etc)
        self.eval(self.config.num_episodes-1, log=False)

        # Save networks
        self.save_model()

        # Save other game data
        self.env.save_trajectories(checkpoint=False)
        self.env.save_attack_data(checkpoint=False)
        if self.config.save_dir is not None:
            time_str = str(time.time())
            self.train_result.to_csv(self.config.save_dir + "/" + time_str + "_train_results_checkpoint.csv")
            self.eval_result.to_csv(self.config.save_dir + "/" + time_str + "_eval_results_checkpoint.csv")

        return self.train_result

    def create_policy_plot(self, distribution, episode, attacker = True) -> None:
        """
        Utility function for creating a density plot of the policy distribution p(a|s) and add to Tensorboard

        :param distribution: the distribution to plot
        :param episode: the episode when the distribution was recorded
        :param attacker: boolean flag whether it is the attacker or defender
        :return: None
        """
        sample = np.random.choice(list(range(len(distribution))), size=1000, p=distribution)
        tag = "Attacker"
        file_suffix = "initial_state_policy_attacker"
        if not attacker:
            tag = "Defender"
            file_suffix = "initial_state_policy_defender"
        title = tag + " Initial State Policy"
        data = idsgame_util.action_dist_hist(sample, title=title, xlabel="Action", ylabel=r"$\mathbb{P}(a|s)$",
                                      file_name=self.config.save_dir + "/" + file_suffix + "_" + str(episode))
        self.tensorboard_writer.add_image(str(episode) + "_initial_state_policy/" + tag,
                                          data, global_step=episode, dataformats="HWC")

    def update_quality_score(self, opponent_idx : int, attacker : bool = True) -> None:
        """
        Updates the quality score of an opponent in the opponent pool. Using same update rule as was used in
        "Dota 2 with Large Scale Deep Reinforcement Learning" by Berner et. al.

        :param opponent_idx: the index of the opponent in the pool
        :param attacker: boolean flag whether attacker or defender pool to be updated
        :return: None
        """
        if attacker:
            N = len(self.attacker_pool)
            qualities = self.get_attacker_pool_quality_scores()
            dist = self.get_softmax_distribution(qualities)
            p = dist[opponent_idx]
            self.attacker_pool[opponent_idx][1] = self.attacker_pool[opponent_idx][1] - \
                                                  (self.config.opponent_pool_config.quality_score_eta/(N*p))
        else:
            N = len(self.defender_pool)
            qualities = self.get_defender_pool_quality_scores()
            dist = self.get_softmax_distribution(qualities)
            p = dist[opponent_idx]
            self.defender_pool[opponent_idx][1] = self.defender_pool[opponent_idx][1] - \
                                                  (self.config.opponent_pool_config.quality_score_eta / (N * p))

    def update_state(self, attacker_obs: np.ndarray = None, defender_obs: np.ndarray = None,
                     state: np.ndarray = None, attacker : bool = True) -> np.ndarray:
        """
        Update approximative Markov state

        :param attacker_obs: attacker obs
        :param defender_obs: defender observation
        :param state: current state
        :param attacker: boolean flag whether it is attacker or not
        :return: new state
        """
        if self.env.fully_observed():
            a_pos = attacker_obs[:,-1]
            det_values = defender_obs[:, -1]
            temp = defender_obs[:,0:-1] - attacker_obs[:,0:-1]
            if self.config.normalize_features:
                det_values = det_values / np.linalg.norm(det_values)
                temp = temp / np.linalg.norm(temp)
            features = []
            for idx, row in enumerate(temp):
                t = row.tolist()
                t.append(a_pos[idx])
                t.append(det_values[idx])
                features.append(t)
            features = np.array(features)
            if self.config.state_length == 1:
                return features
            if len(state) == 0:
                s = np.array([features] * self.config.state_length)
                return s
            state = np.append(state[1:], np.array([features]), axis=0)
            return state
        else:
            if self.config.normalize_features:
                attacker_obs_1 = attacker_obs[:,0:-1] / np.linalg.norm(attacker_obs[:,0:-1])
                normalized_attacker_features = []
                for idx, row in enumerate(attacker_obs_1):
                    if np.isnan(attacker_obs_1).any():
                        t = attacker_obs[idx]
                    else:
                        t = attacker_obs_1.tolist()
                        t.append(attacker_obs[idx][-1])
                    normalized_attacker_features.append(t)

                defender_obs_1 = defender_obs[:, 0:-1] / np.linalg.norm(defender_obs[:, 0:-1])
                normalized_defender_features = []
                for idx, row in enumerate(defender_obs_1):
                    if np.isnan(defender_obs_1).any():
                        t= defender_obs[idx]
                    else:
                        t = defender_obs_1.tolist()
                        t.append(defender_obs[idx][-1])
                    normalized_defender_features.append(t)
                attacker_obs = np.array(normalized_attacker_features)
                defender_obs = np.array(normalized_defender_features)

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

    def eval(self, train_episode, log=True) -> ExperimentResult:
        """
        Performs evaluation with the greedy policy with respect to the learned Q-values

        :param train_episode: the train episode to keep track of logging
        :param log: whether to log the result
        :return: None
        """
        self.config.logger.info("Starting Evaluation")
        time_str = str(time.time())

        self.num_eval_games = 0
        self.num_eval_hacks = 0

        if len(self.eval_result.avg_episode_steps) > 0:
            self.config.logger.warning("starting eval with non-empty result object")
        if self.config.eval_episodes < 1:
            return
        done = False

        # Video config
        if self.config.video:
            if self.config.video_dir is None:
                raise AssertionError("Video is set to True but no video_dir is provided, please specify "
                                     "the video_dir argument")
            self.env = IdsGameMonitor(self.env, self.config.video_dir + "/" + time_str, force=True,
                                      video_frequency=self.config.video_frequency)
            self.env.metadata["video.frames_per_second"] = self.config.video_fps

        # Tracking metrics
        episode_attacker_rewards = []
        episode_defender_rewards = []
        episode_steps = []

        # Logging
        self.outer_eval = tqdm.tqdm(total=self.config.eval_episodes, desc='Eval Episode', position=1)
        self.outer_eval.set_description_str(
            "[Eval] avg_a_R:{:.2f},avg_d_R:{:.2f},avg_t:{:.2f},avg_h:{:.2f},acc_A_R:{:.2f}," \
            "acc_D_R:{:.2f}".format(0.0, 0,0, 0.0, 0.0, 0.0, 0.0))

        # Eval
        attacker_obs, defender_obs = self.env.reset(update_stats=False)
        attacker_state = self.update_state(attacker_obs=attacker_obs, defender_obs=defender_obs, state=[], attacker=True)
        defender_state = self.update_state(defender_obs=defender_obs, attacker_obs=attacker_obs, state=[], attacker=False)

        for episode in range(self.config.eval_episodes):
            episode_attacker_reward = 0
            episode_defender_reward = 0
            episode_step = 0
            while not done:
                if self.config.eval_render:
                    self.env.render()
                    time.sleep(self.config.eval_sleep)

                # Default initialization
                attacker_action = 0
                defender_action = 0

                # Get attacker and defender actions
                if self.config.attacker:
                    attacker_action, _, _, _ = self.get_action(attacker_state, attacker=True)
                if self.config.defender:
                    defender_action, _, _, _ = self.get_action(defender_state, attacker=False)
                action = (attacker_action, defender_action)

                # Take a step in the environment
                obs_prime, reward, done, _ = self.env.step(action)

                # Update state information and metrics
                attacker_reward, defender_reward = reward
                obs_prime_attacker, obs_prime_defender = obs_prime
                episode_attacker_reward += attacker_reward
                episode_defender_reward += defender_reward
                episode_step += 1
                attacker_obs = obs_prime_attacker
                defender_obs = obs_prime_defender
                attacker_state = self.update_state(attacker_obs=attacker_obs, defender_obs=defender_obs, state=attacker_state, attacker=True)
                defender_state = self.update_state(defender_obs=defender_obs, attacker_obs=attacker_obs, state=defender_state, attacker=False)

            # Render final frame when game completed
            if self.config.eval_render:
                self.env.render()
                time.sleep(self.config.eval_sleep)
            self.config.logger.info("Eval episode: {}, Game ended after {} steps".format(episode, episode_step))

            # Record episode metrics
            episode_attacker_rewards.append(episode_attacker_reward)
            episode_defender_rewards.append(episode_defender_reward)
            episode_steps.append(episode_step)

            # Update eval stats
            self.num_eval_games +=1
            self.num_eval_games_total += 1
            if self.env.state.detected:
                self.eval_attacker_cumulative_reward -= constants.GAME_CONFIG.POSITIVE_REWARD
                self.eval_defender_cumulative_reward += constants.GAME_CONFIG.POSITIVE_REWARD
            if self.env.state.hacked:
                self.eval_attacker_cumulative_reward += constants.GAME_CONFIG.POSITIVE_REWARD
                self.eval_defender_cumulative_reward -= constants.GAME_CONFIG.POSITIVE_REWARD
                self.num_eval_hacks += 1
                self.num_eval_hacks_total +=1

            # Log average metrics every <self.config.eval_log_frequency> episodes
            if episode % self.config.eval_log_frequency == 0 and log:
                if self.num_eval_hacks > 0:
                    self.eval_hack_probability = float(self.num_eval_hacks) / float(self.num_eval_games)
                if self.num_eval_games_total > 0:
                    self.eval_cumulative_hack_probability = float(self.num_eval_hacks_total) / float(
                        self.num_eval_games_total)
                a_pool = None
                d_pool = None
                if self.config.opponent_pool and self.config.opponent_pool_config is not None:
                    a_pool = len(self.attacker_pool)
                    d_pool = len(self.defender_pool)
                self.log_metrics(episode, self.eval_result, episode_attacker_rewards, episode_defender_rewards, episode_steps,
                                 eval = True, update_stats=False, a_pool=a_pool, d_pool = d_pool)

            # Save gifs
            if self.config.gifs and self.config.video:
                self.env.generate_gif(self.config.gif_dir + "/episode_" + str(train_episode) + "_"
                                      + time_str + ".gif", self.config.video_fps)

                # Add frames to tensorboard
                for idx, frame in enumerate(self.env.episode_frames):
                    self.tensorboard_writer.add_image(str(train_episode) + "_eval_frames/" + str(idx),
                                                       frame, global_step=train_episode,
                                                      dataformats = "HWC")


            # Reset for new eval episode
            done = False
            attacker_obs, defender_obs = self.env.reset(update_stats=False)
            attacker_state = self.update_state(attacker_obs=attacker_obs, defender_obs=defender_obs, state=attacker_state, attacker=True)
            defender_state = self.update_state(defender_obs=defender_obs, attacker_obs=attacker_obs, state=defender_state, attacker=False)
            self.outer_eval.update(1)

        # Log average eval statistics
        if log:
            if self.num_eval_hacks > 0:
                self.eval_hack_probability = float(self.num_eval_hacks) / float(self.num_eval_games)
            if self.num_eval_games_total > 0:
                self.eval_cumulative_hack_probability = float(self.num_eval_hacks_total) / float(
                    self.num_eval_games_total)

            self.log_metrics(train_episode, self.eval_result, episode_attacker_rewards, episode_defender_rewards,
                             episode_steps, eval=True, update_stats=True)

        self.env.close()
        self.config.logger.info("Evaluation Complete")
        return self.eval_result

    def save_model(self) -> None:
        """
        Saves the PyTorch Model Weights

        :return: None
        """
        time_str = str(time.time())
        if self.config.save_dir is not None:
            if self.config.attacker:
                path = self.config.save_dir + "/" + time_str + "_attacker_policy_network.pt"
                self.config.logger.info("Saving policy-network to: {}".format(path))
                torch.save(self.attacker_policy_network.state_dict(), path)
            if self.config.defender:
                path = self.config.save_dir + "/" + time_str + "_defender_policy_network.pt"
                self.config.logger.info("Saving policy-network to: {}".format(path))
                torch.save(self.defender_policy_network.state_dict(), path)
        else:
            self.config.logger.warning("Save path not defined, not saving policy-networks to disk")

"""
An agent for the IDSGameEnv that implements the DQN algorithm.
"""
from typing import Union
import numpy as np
import time
import tqdm
import logging
import torch
from torch.utils.tensorboard import SummaryWriter
from gym_idsgame.envs.rendering.video.idsgame_monitor import IdsGameMonitor
from gym_idsgame.agents.q_learning.tabular_q_learning.q_agent_config import QAgentConfig
from gym_idsgame.envs.idsgame_env import IdsGameEnv
from gym_idsgame.agents.dao.experiment_result import ExperimentResult
from gym_idsgame.envs.constants import constants
from gym_idsgame.agents.q_learning.dqn.model import SixLayerFNN
from gym_idsgame.agents.q_learning.experience_replay.replay_buffer import ReplayBuffer
from gym_idsgame.agents.q_learning.q_agent import QAgent

class DQNAgent(QAgent):
    """
    An implementation of the DQN algorithm from the paper 'Human-level control through deep reinforcement learning' by
    Mnih et. al.

    (DQN is originally Neural-fitted Q-iteration but with the addition of a separate target network)
    """
    def __init__(self, env:IdsGameEnv, config: QAgentConfig):
        """
        Initialize environment and hyperparameters

        :param config: the configuration
        """
        super(DQNAgent, self).__init__(env, config)
        self.attacker_q_network = None
        self.attacker_target_network = None
        self.defender_q_network = None
        self.defender_target_network = None
        self.loss_fn = None
        self.attacker_optimizer = None
        self.defender_optimizer = None
        self.initialize_models()
        self.buffer = ReplayBuffer(config.dqn_config.replay_memory_size)
        self.tensorboard_writer = SummaryWriter(self.config.dqn_config.tensorboard_dir)

    def warmup(self) -> None:
        """
        A warmup without any learning just to populate the replay buffer following a random strategy

        :return: None
        """

        # Setup logging
        outer_warmup = tqdm.tqdm(total=self.config.dqn_config.replay_start_size, desc='Warmup', position=0)
        outer_warmup.set_description_str("[Warmup] step:{}, buffer_size: {}".format(0, 0))

        # Reset env
        obs = self.env.reset(update_stats=False)
        self.config.logger.info("Starting warmup phase to fill replay buffer")

        # Perform <self.config.dqn_config.replay_start_size> steps and fill the replay memory
        for i in range(self.config.dqn_config.replay_start_size):

            if i % self.config.train_log_frequency == 0:
                log_str = "[Warmup] step:{}, buffer_size: {}".format(i, self.buffer.size())
                outer_warmup.set_description_str(log_str)
                self.config.logger.info(log_str)

            # Select random attacker and defender actions
            attacker_actions = list(range(self.env.num_attack_actions))
            defender_actions = list(range(self.env.num_defense_actions))
            legal_attack_actions = list(filter(lambda action: self.env.is_attack_legal(action), attacker_actions))
            legal_defense_actions = list(filter(lambda action: self.env.is_defense_legal(action), defender_actions))
            attacker_action = np.random.choice(legal_attack_actions)
            defender_action = np.random.choice(legal_defense_actions)
            action = (attacker_action, defender_action)

            # Take action in the environment
            obs_prime, reward, done, info = self.env.step(action)

            # Add transition to replay memory
            self.buffer.add_tuple(obs, action, reward, done, obs_prime)

            # Move to new state
            obs = obs_prime
            outer_warmup.update(1)

            if done:
                obs = self.env.reset(update_stats=False)

        self.config.logger.info("{} Warmup steps completed, replay buffer size: {}".format(
            self.config.dqn_config.replay_start_size, self.buffer.size()))
        self.env.close()

    def initialize_models(self) -> None:
        """
        Initialize models
        :return: None
        """

        # Initialize models
        self.attacker_q_network = SixLayerFNN(self.config.dqn_config.input_dim, self.config.dqn_config.output_dim,
                                              self.config.dqn_config.hidden_dim)
        self.attacker_target_network = SixLayerFNN(self.config.dqn_config.input_dim, self.config.dqn_config.output_dim,
                                                   self.config.dqn_config.hidden_dim)
        self.defender_q_network = SixLayerFNN(self.config.dqn_config.input_dim, self.config.dqn_config.output_dim,
                                              self.config.dqn_config.hidden_dim)
        self.defender_target_network = SixLayerFNN(self.config.dqn_config.input_dim, self.config.dqn_config.output_dim,
                                                   self.config.dqn_config.hidden_dim)


        # Specify device
        if torch.cuda.is_available() and self.config.dqn_config.gpu:
            device = torch.device("cuda:0")
            self.config.logger.info("Running on the GPU")
        else:
            device = torch.device("cpu")
            self.config.logger.info("Running on the CPU")

        self.attacker_q_network.to(device)
        self.attacker_target_network.to(device)
        self.defender_q_network.to(device)
        self.defender_target_network.to(device)

        # Set the target network to use the same weights initialization as the q-network
        self.attacker_target_network.load_state_dict(self.attacker_q_network.state_dict())
        self.defender_target_network.load_state_dict(self.defender_q_network.state_dict())
        # The target network is not trainable it is only used for predictions, therefore we set it to eval mode
        # to turn of dropouts, batch norms, gradient computations etc.
        self.attacker_target_network.eval()
        self.defender_target_network.eval()

        # Construct our loss function and an Optimizer. The call to model.parameters()
        # in the optimizer constructor will contain the learnable parameters of the layers in the model
        self.loss_fn = torch.nn.MSELoss(reduction='sum')
        self.attacker_optimizer = torch.optim.Adam(self.attacker_q_network.parameters(), lr=self.config.alpha)
        self.defender_optimizer = torch.optim.Adam(self.defender_q_network.parameters(), lr=self.config.alpha)

    def training_step(self, mini_batch: Union[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                                              np.ndarray, np.ndarray, np.ndarray]) -> torch.Tensor:
        """
        Performs a training step of the Deep-Q-learning algorithm (implemented in PyTorch)

        :param mini_batch: a minibatch to use for the training step
        :return: loss
        """

        # Unpack batch of transitions from the replay memory
        s_attacker_batch, s_defender_batch, a_attacker_batch, a_defender_batch, r_attacker_batch, r_defender_batch, \
        d_batch, s2_attacker_batch, s2_defender_batch = mini_batch

        # Convert batch into torch tensors and set Q-network in train mode and target network in eval mode
        self.attacker_q_network.train()
        self.attacker_target_network.eval()

        r_1 = torch.tensor(r_attacker_batch).float()
        s_1 = torch.tensor(s_attacker_batch).float()
        s_2 = torch.tensor(s2_attacker_batch).float()

        # Move to GPU if using GPU
        if torch.cuda.is_available() and self.config.dqn_config.gpu:
            device = torch.device("cuda:0")
            r_1 = r_1.to(device)
            s_1 = s_1.to(device)
            s_2 = s_2.to(device)

        # Set target baseline. We only want the loss to be computed for the Q-values of the actions taken, not the entire
        # vector of all Q-values. Therefore we initialize the target to the Q-values of the Q-network for s
        # and then we only update the Q-values for the affected actions with the real targets
        target = self.attacker_q_network(s_1)

        # Use the target network to compute the Q-values of s'
        with torch.no_grad():
            target_next = self.attacker_target_network(s_2).detach()

        for i in range(self.config.dqn_config.batch_size):
            # As defined by Mnih et. al. : For terminal states the Q-target should be equal to the immediate reward
            if d_batch[i]:
                target[i][a_attacker_batch[i]] = r_1[i]

            # For non terminal states the Q-target should be the immediate reward plus the discounted estimated future
            # reward when following Q* estimated by the target network.
            else:
                a = torch.argmax(target_next[i]).detach()
                target[i][a_attacker_batch[i]] = r_1[i] + self.config.gamma * (target_next[i][a])

        # Compute loss
        prediction = self.attacker_q_network(s_1)
        loss = self.loss_fn(prediction, target)

        # Zero gradients, perform a backward pass, and update the weights.
        self.attacker_optimizer.zero_grad()
        loss.backward()
        self.attacker_optimizer.step()

        return loss

    def get_action(self, state: np.ndarray, eval : bool = False, attacker : bool = True) -> int:
        """
        Samples an action according to a epsilon-greedy strategy using the Q-network

        :param state: the state to sample an action for
        :param eval: boolean flag whether running in evaluation mode
        :param attacker: boolean flag whether running in attacker mode (if false assume defender)
        :return: The sampled action id
        """
        state = torch.from_numpy(state.flatten()).float()

        # Move to GPU if using GPU
        if torch.cuda.is_available() and self.config.dqn_config.gpu:
            device = torch.device("cuda:0")
            state = state.to(device)

        if attacker:
            actions = list(range(self.env.num_attack_actions))
            legal_actions = list(filter(lambda action: self.env.is_attack_legal(action), actions))
        else:
            actions = list(range(self.env.num_defense_actions))
            legal_actions = list(filter(lambda action: self.env.is_defense_legal(action), actions))
        if np.random.rand() < self.config.epsilon and not eval:
            return np.random.choice(legal_actions)
        with torch.no_grad():
            act_values = self.attacker_q_network(state)
        return legal_actions[torch.argmax(act_values[legal_actions]).item()]

    def train(self) -> ExperimentResult:
        """
        Runs the DQN algorithm

        :return: Experiment result
        """
        self.config.logger.info("Starting Warmup")
        self.warmup()
        self.config.logger.info("Starting Training")
        self.config.logger.info(self.config.to_str())
        if len(self.train_result.avg_episode_steps) > 0:
            self.config.logger.warning("starting training with non-empty result object")
        done = False
        obs = self.env.reset(update_stats=False)
        attacker_obs, defender_obs = obs

        # Tracking metrics
        episode_attacker_rewards = []
        episode_defender_rewards = []
        episode_steps = []
        episode_avg_loss = []

        # Logging
        self.outer_train.set_description_str("[Train] epsilon:{:.2f},avg_a_R:{:.2f},avg_d_R:{:.2f},"
                                             "avg_t:{:.2f},avg_h:{:.2f},acc_A_R:{:.2f}," \
                                             "acc_D_R:{:.2f}".format(self.config.epsilon, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))

        # Training
        for episode in range(self.config.num_episodes):
            episode_attacker_reward = 0
            episode_defender_reward = 0
            episode_step = 0
            episode_loss = 0.0
            actions = []
            while not done:
                if self.config.render:
                    self.env.render(mode="human")

                if not self.config.attacker and not self.config.defender:
                    raise AssertionError("Must specify whether training an attacker agent or defender agent")

                # Default initialization
                defender_state_node_id = 0
                attacker_action = 0
                defender_action = 0

                # Get attacker and defender actions
                if self.config.attacker:
                    attacker_action = self.get_action(attacker_obs, attacker=True)
                    actions.append(attacker_action)
                if self.config.defender:
                    defender_action = self.get_action(defender_state_node_id, attacker=False)
                action = (attacker_action, defender_action)

                # Take a step in the environment
                obs_prime, reward, done, _ = self.env.step(action)

                # Add transition to replay memory
                self.buffer.add_tuple(obs, action, reward, done, obs_prime)

                # Sample random mini_batch of transitions from replay memory
                minibatch = self.buffer.sample(self.config.dqn_config.batch_size)

                # Perform a gradient descent step of the Q-network using targets produced by target network
                loss = self.training_step(minibatch)
                episode_loss += loss.item()

                # Update metrics
                attacker_reward, defender_reward = reward
                obs_prime_attacker, obs_prime_defender = obs_prime
                episode_attacker_reward += attacker_reward
                episode_defender_reward += defender_reward
                episode_step += 1

                # Move to the next state
                obs = obs_prime
                attacker_obs = obs_prime_attacker
                defender_obs = obs_prime_defender

            # Record episode metrics
            episode_attacker_rewards.append(episode_attacker_reward)
            episode_defender_rewards.append(episode_defender_reward)
            if episode_step > 0:
                episode_avg_loss.append(episode_loss/episode_step)
            else:
                episode_avg_loss.append(episode_loss)
            episode_steps.append(episode_step)

            # Log average metrics every <self.config.train_log_frequency> episodes
            if episode % self.config.train_log_frequency == 0:
                self.log_metrics(episode, self.train_result, episode_attacker_rewards, episode_defender_rewards, episode_steps,
                                 episode_avg_loss)
                episode_attacker_rewards = []
                episode_defender_rewards = []
                episode_steps = []

            # Update target network every <self.config.dqn_config.target_network_update_freq> episodes
            if episode % self.config.dqn_config.target_network_update_freq == 0:
                self.update_target_network()

            # Run evaluation every <self.config.eval_frequency> episodes
            if episode % self.config.eval_frequency == 0:
                self.eval()

            # Reset environment for the next episode and update game stats
            done = False
            attacker_obs, defender_obs = self.env.reset(update_stats=True)
            self.outer_train.update(1)

            # Anneal epsilon linearly
            self.anneal_epsilon()

        self.config.logger.info("Training Complete")

        # Final evaluation (for saving Gifs etc)
        self.eval(log=False)

        # Save Q-networks
        self.save_model()

        return self.train_result

    def update_target_network(self) -> None:
        """
        Updates the target networks. Delayed targets are used to stabilize training and partially remedy the
        problem with non-stationary targets in RL with function approximation.

        :return: None
        """
        self.config.logger.info("Updating target network")
        self.attacker_target_network.load_state_dict(self.attacker_q_network.state_dict())

    def eval(self, log=True) -> ExperimentResult:
        """
        Performs evaluation with the greedy policy with respect to the learned Q-values

        :param log: whether to log the result
        :return: None
        """
        self.config.logger.info("Starting Evaluation")
        time_str = str(time.time())

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

        for episode in range(self.config.eval_episodes):
            episode_attacker_reward = 0
            episode_defender_reward = 0
            episode_step = 0
            while not done:
                if self.config.eval_render:
                    self.env.render()
                    time.sleep(self.config.eval_sleep)

                # Default initialization
                attacker_state_node_id = 0
                defender_state_node_id = 0
                attacker_action = 0
                defender_action = 0

                # Get attacker and defender actions
                if self.config.attacker:
                    attacker_action = self.get_action(attacker_obs, eval=True, attacker=True)
                if self.config.defender:
                    defender_action = self.get_action(defender_state_node_id, eval=True, attacker=False)
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
            if self.env.state.detected:
                self.eval_attacker_cumulative_reward -= constants.GAME_CONFIG.POSITIVE_REWARD
                self.eval_defender_cumulative_reward += constants.GAME_CONFIG.POSITIVE_REWARD
            if self.env.state.hacked:
                self.eval_attacker_cumulative_reward += constants.GAME_CONFIG.POSITIVE_REWARD
                self.eval_defender_cumulative_reward -= constants.GAME_CONFIG.POSITIVE_REWARD
                self.num_eval_hacks += 1

            # Log average metrics every <self.config.eval_log_frequency> episodes
            if episode % self.config.eval_log_frequency == 0 and log:
                if self.num_eval_hacks > 0:
                    self.eval_hack_probability = float(self.num_eval_hacks) / float(self.num_eval_games)
                self.log_metrics(episode, self.eval_result, episode_attacker_rewards, episode_defender_rewards, episode_steps,
                                 eval = True)
                episode_attacker_rewards = []
                episode_steps = []

            # Save gifs
            if self.config.gifs and self.config.video:
                self.env.generate_gif(self.config.gif_dir + "/episode_" + str(episode) + "_"
                                      + time_str + ".gif", self.config.video_fps)

            # Reset for new eval episode
            done = False
            attacker_obs, defender_obs = self.env.reset(update_stats=False)
            self.outer_eval.update(1)

        self.env.close()
        self.config.logger.info("Evaluation Complete")
        return self.eval_result

    def save_model(self) -> None:
        """
        Saves the PyTorch Model Weights

        :return: None
        """
        if self.config.save_dir is not None:
            if self.config.attacker:
                self.config.logger.info("Saving Q-network to: {}".format(self.config.save_dir + "/attacker_q_network.pt"))
                torch.save(self.attacker_q_network.state_dict(), self.config.save_dir + "/attacker_q_network.pt")
            if self.config.defender:
                self.config.logger.info("Saving Q-network to: {}".format(self.config.save_dir + "/defender_q_network.pt"))
                torch.save(self.attacker_q_network.state_dict(), self.config.save_dir + "/defender_q_network.pt")
        else:
            self.config.logger.warning("Save path not defined, not saving Q-networks to disk")

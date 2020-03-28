"""
An agent for the IDSGameEnv that implements the tabular Q-learning algorithm.
"""
import numpy as np
import time
import tqdm
import logging
from gym_idsgame.envs.rendering.video.idsgame_monitor import IdsGameMonitor
from gym_idsgame.agents.dao.q_agent_config import QAgentConfig
from gym_idsgame.envs.idsgame_env import IdsGameEnv
from gym_idsgame.agents.dao.experiment_result import ExperimentResult
from gym_idsgame.agents.train_agent import TrainAgent
from gym_idsgame.envs.constants import constants

class TabularQAgent(TrainAgent):
    """
    A simple implementation of the Q(0)-learning algorithm (Sutton & Barto).
    Q-learning is a one-step off-policy algorithm
    """
    def __init__(self, env:IdsGameEnv, config: QAgentConfig):
        """
        Initialize environment and hyperparameters

        :param config: the configuration
        """
        self.env = env
        self.config = config
        if self.config.attacker:
            self.Q = np.random.rand(self.env.num_states, self.env.num_attack_actions)
        else:
            self.Q = np.random.rand(1, self.env.num_attack_actions+1)
        self.train_result = ExperimentResult()
        self.eval_result = ExperimentResult()
        self.outer_train = tqdm.tqdm(total=self.config.num_episodes, desc='Train Episode', position=0)
        if self.config.logger is None:
            self.config.logger = logging.getLogger('QAgent')
        self.num_eval_games = 0
        self.num_eval_hacks = 0
        self.eval_hack_probability = 0.0
        self.eval_attacker_cumulative_reward = 0
        self.eval_defender_cumulative_reward = 0

    def get_action(self, s, eval=False) -> int:
        """
        Sample an action using an epsilon-greedy policy with respect to the current Q-values

        :param s:  the state to sample an action for
        :return: a sampled action
        """
        actions = list(range(self.env.num_attack_actions))
        if self.config.attacker:
            legal_actions = list(filter(lambda action: self.env.is_attack_legal(action), actions))
        elif self.config.defender:
            legal_actions = list(filter(lambda action: self.env.is_defense_legal(action), actions))
        if len(legal_actions) == 0:
            print("Current state: {}, past moves: {}".format(s, self.env.past_moves))
            #raise AssertionError("Found no valid action from this state")
        if np.random.rand() < self.config.epsilon and not eval:
            return np.random.choice(legal_actions)
        max_legal_action_value = float("-inf")
        max_legal_action = float("-inf")
        for i in range(len(self.Q[s])):
            if i in legal_actions and self.Q[s][i] > max_legal_action_value:
                max_legal_action_value = self.Q[s][i]
                max_legal_action = i
        if max_legal_action == float("-inf") or max_legal_action_value == float("-inf"):
            raise AssertionError("Error when selecting action greedily according to the Q-function")
        return max_legal_action

    def train(self) -> ExperimentResult:
        """
        Runs the Q(0)-learning algorithm for estimating the state values under a given policy for a specific MDP

        :return: None
        """
        self.config.logger.info("Starting Training")
        self.config.logger.info(self.config.to_str())
        if len(self.train_result.avg_episode_steps) > 0:
            self.config.logger.warning("starting training with non-empty result object")
        done = False
        obs = self.env.reset(update_stats=False)

        # Tracking metrics
        episode_rewards = []
        episode_steps = []

        # Logging
        self.outer_train.set_description_str("[Train] epsilon:{:.2f},avg_R:{:.2f},avg_t:{:.2f}".format(self.config.epsilon, 0.0, 0.0))

        # Training
        for episode in range(self.config.num_episodes):
            episode_reward = 0
            episode_step = 0
            while not done:
                if self.config.render:
                    self.env.render(mode="human")
                if self.config.attacker:
                    state_node_id = self.env.get_attacker_node_from_observation(obs)
                elif self.config.defender:
                    state_node_id = 0
                else:
                    raise AssertionError("Must specify whether training an attacker agent or defender agent")
                action = self.get_action(state_node_id)
                obs_prime, reward, done, info = self.env.step(action)
                episode_reward += reward
                episode_step += 1
                if self.config.attacker:
                    state_prime_node_id = self.env.get_attacker_node_from_observation(obs_prime)
                elif self.config.defender:
                    state_prime_node_id = 0
                # Q-learning update
                self.Q[state_node_id, action] = self.Q[state_node_id, action] + \
                                                   self.config.alpha*(
                                                           reward +
                                                           self.config.gamma*np.max(self.Q[state_prime_node_id])
                                                           - self.Q[state_node_id,action])
                obs = obs_prime

            episode_rewards.append(episode_reward)
            episode_steps.append(episode_step)

            # Log average metrics every <self.config.train_log_frequency> episodes
            if episode % self.config.train_log_frequency == 0 and episode > 0:
                self.log_metrics(self.train_result, episode_rewards, episode_steps)
                episode_rewards = []
                episode_steps = []

            # Run evaluation every <self.config.eval_frequency> episodes
            if episode % self.config.eval_frequency == 0:
                self.eval()

            # Reset environment for the next episode and update game stats
            done = False
            obs = self.env.reset(update_stats=True)
            self.outer_train.update(1)

            # Anneal epsilon linearly
            self.anneal_epsilon()

        self.config.logger.info("Training Complete")

        # Final evaluation
        self.eval()

        # Log and return
        self.log_state_values()

        # Save Q Table
        self.save_q_table()

        return self.train_result

    def log_metrics(self, result: ExperimentResult, episode_rewards:list, episode_steps:list, eval:bool = False) \
            -> None:
        """
        Logs average metrics for the last <self.config.log_frequency> episodes

        :param result: the result object to add the results to
        :param episode_rewards: list of episode rewards for the last <self.config.log_frequency> episodes
        :param episode_steps: list of episode steps for the last <self.config.log_frequency> episodes
        :param eval: boolean flag whether the metrics are logged in an evaluation context.
        :return: None
        """
        avg_episode_reward = np.mean(episode_rewards)
        avg_episode_steps = np.mean(episode_steps)
        hack_probability = self.env.hack_probability() if not eval else self.eval_hack_probability
        attacker_cumulative_reward = self.env.state.attacker_cumulative_reward if not eval \
            else self.eval_attacker_cumulative_reward
        defender_cumulative_reward = self.env.state.defender_cumulative_reward if not eval \
            else self.eval_defender_cumulative_reward
        if eval:
            log_str = "[Eval] avg_R:{:.2f},avg_t:{:.2f},avg_h:{:.2f},acc_A_R:{:.2f}," \
                      "acc_D_R:{:.2f}".format(avg_episode_reward,
                                              avg_episode_steps,
                                              hack_probability,
                                              attacker_cumulative_reward,
                                              defender_cumulative_reward)
            self.outer_eval.set_description_str(log_str)
        else:
            log_str = "[Train] epsilon:{:.2f},avg_R:{:.2f},avg_t:{:.2f},avg_h:{:.2f},acc_A_R:{:.2f}," \
                      "acc_D_R:{:.2f}".format(self.config.epsilon, avg_episode_reward,
                                              avg_episode_steps,
                                              hack_probability,
                                              attacker_cumulative_reward,
                                              defender_cumulative_reward)
            self.outer_train.set_description_str(log_str)
        self.config.logger.info(log_str)
        result.avg_episode_steps.append(avg_episode_steps)
        result.avg_episode_rewards.append(avg_episode_reward)
        result.epsilon_values.append(self.config.epsilon)
        result.hack_probability.append(hack_probability)
        result.attacker_cumulative_reward.append(attacker_cumulative_reward)
        result.defender_cumulative_reward.append(defender_cumulative_reward)

    def eval(self) -> ExperimentResult:
        """
        Performs evaluation with the greedy policy with respect to the learned Q-values

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
        episode_rewards = []
        episode_steps = []

        # Logging
        self.outer_eval = tqdm.tqdm(total=self.config.eval_episodes, desc='Eval Episode', position=1)
        self.outer_eval.set_description_str(
            "[Eval] avg_R:{:.2f},avg_t:{:.2f},avg_h:{:.2f},acc_A_R:{:.2f}," \
            "acc_D_R:{:.2f}".format(0.0, 0.0,0.0,0.0,0.0))

        # Eval
        obs = self.env.reset(update_stats=False)
        for episode in range(self.config.eval_episodes):
            i = 0
            episode_reward = 0
            episode_step = 0
            while not done:
                if self.config.eval_render:
                    self.env.render()
                    time.sleep(self.config.eval_sleep)
                i = i+1
                if self.config.attacker:
                    s_node_id = self.env.get_attacker_node_from_observation(obs)
                elif self.config.defender:
                    s_node_id = 0
                action = self.get_action(s_node_id, eval=True)
                obs, reward, done, _ = self.env.step(action)
                episode_reward += reward
                episode_step += 1
            if self.config.eval_render:
                self.env.render()
                time.sleep(self.config.eval_sleep)
            self.config.logger.info("Eval episode: {}, Game ended after {} steps".format(episode, i))
            episode_rewards.append(episode_reward)
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
            if episode % self.config.eval_log_frequency == 0:
                if self.num_eval_hacks > 0:
                    self.eval_hack_probability = float(self.num_eval_hacks) / float(self.num_eval_games)
                self.log_metrics(self.eval_result, episode_rewards, episode_steps,
                                 eval = True)
                episode_rewards = []
                episode_steps = []
            if self.config.gifs and self.config.video:
                self.env.generate_gif(self.config.gif_dir + "/episode_" + str(episode) + "_"
                                      + time_str + ".gif", self.config.video_fps)
            done = False
            obs = self.env.reset(update_stats=False)
            self.outer_eval.update(1)

        self.env.close()
        self.config.logger.info("Evaluation Complete")
        return self.eval_result

    def anneal_epsilon(self) -> None:
        """
        Anneals the exploration rate slightly until it reaches the minimum value

        :return: None
        """
        if self.config.epsilon > self.config.min_epsilon:
            self.config.epsilon = self.config.epsilon*self.config.epsilon_decay

    def log_state_values(self) -> None:
        """
        Utility function for printing the state-values according to the learned Q-function
        :return:
        """
        self.config.logger.info("--- State Values ---")
        for i in range(len(self.Q)):
            state_value = sum(self.Q[i])
            node_id = i
            self.config.logger.info("s:{},V(s):{}".format(node_id, state_value))
        self.config.logger.info("--------------------")

    def save_q_table(self) -> None:
        """
        Saves Q table to disk in binary npy format

        :return: None
        """
        if self.config.save_dir is not None:
            self.config.logger.info("Saving Q-table to: {}".format(self.config.save_dir + "/q_table.npy"))
            np.save(self.config.save_dir + "/q_table.npy", self.Q)
        else:
            self.config.logger.warning("Save path not defined, not saving Q table to disk")

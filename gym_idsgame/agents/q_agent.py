"""
An agent for the IDSGameEnv that implements the tabular Q-learning algorithm.
"""
import numpy as np
import time
import tqdm
from gym import wrappers
from gym_idsgame.envs.rendering.video.idsgame_monitor import IdsGameMonitor
from gym_idsgame.agents.dao.q_agent_config import QAgentConfig
from gym_idsgame.envs.idsgame_env import IdsGameEnv
from gym_idsgame.agents.dao.train_result import TrainResult
from gym_idsgame.agents.train_agent import TrainAgent

class QAgent(TrainAgent):
    """
    A simple implementation of the Q(0)-learning algorithm (Sutton & Barto).
    Q-learning is a one-step off-policy algorithm
    """
    def __init__(self, env:IdsGameEnv, config: QAgentConfig):
        """
        Initalize environment and hyperparameters
        :param config: the configuration
        """
        self.env = env
        self.config = config
        self.Q = np.zeros((self.env.num_states, self.env.num_actions))
        self.train_result = TrainResult()
        self.eval_result = TrainResult()
        self.outer = tqdm.tqdm(total=self.config.num_episodes, desc='Episode', position=0)

    def get_action(self, s, eval=False):
        """
        Sample an action using an epsilon-greedy policy with respect to the current Q-values

        :param s:  the state to sample an action for
        :return: a sampled action
        """
        actions = list(range(self.env.num_actions))
        legal_actions = list(filter(lambda action: self.env.is_attack_legal(action), actions))
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

    def train(self):
        """
        Runs the Q(0)-learning algorithm for estimating the state values under a given policy for a specific MDP

        :return: None
        """
        self.config.logger.info("Starting Training")
        self.config.logger.info(self.config.to_str())
        if len(self.train_result.avg_episode_steps) > 0:
            self.config.logger.warning("starting training with non-empty result object")
        done = False
        obs = self.env.reset()

        # Tracking metrics
        episode_rewards = []
        episode_steps = []

        # Logging
        self.outer.set_description_str("epsilon:{:.2f},avg_R:{:.2f},avg_t:{:.2f}".format(self.config.epsilon, 0.0, 0.0))

        # Training
        for episode in range(self.config.num_episodes):
            episode_reward = 0
            episode_step = 0
            while not done:
                if self.config.render:
                    self.env.render(mode="human")
                attacker_node_id = self.env.get_attacker_node_from_observation(obs)
                action = self.get_action(attacker_node_id)
                while action >= (self.env.idsgame_config.game_config.num_nodes-1)*self.env.idsgame_config.game_config.num_attack_types:
                    action = self.get_action(attacker_node_id)
                obs_prime, reward, done, info = self.env.step(action)
                episode_reward += reward
                episode_step += 1
                attacker_node_id_prime = self.env.get_attacker_node_from_observation(obs_prime)
                # Q-learning update
                self.Q[attacker_node_id, action] = self.Q[attacker_node_id, action] + \
                                                   self.config.alpha*(
                                                           reward +
                                                           self.config.gamma*np.max(self.Q[attacker_node_id_prime])
                                                           - self.Q[attacker_node_id,action])
                obs = obs_prime

            episode_rewards.append(episode_reward)
            episode_steps.append(episode_step)

            # Log average metrics every <self.config.train_log_frequency> episodes
            if episode % self.config.train_log_frequency == 0:
                self.log_metrics(self.train_result, episode_rewards, episode_steps)
                episode_rewards = []
                episode_steps = []
            self.anneal_epsilon()
            done=False
            obs = self.env.reset()
            self.outer.update(1)

        self.config.logger.info("Training Complete")
        self.log_state_values()
        return self.train_result

    def log_metrics(self, result: TrainResult, episode_rewards:list, episode_steps:list) -> None:
        """
        Logs average metrics for the last <self.config.log_frequency> episodes

        :param result: the result object to add the results to
        :param episode_rewards: list of episode rewards for the last <self.config.log_frequency> episodes
        :param episode_steps: list of episode steps for the last <self.config.log_frequency> episodes
        :return: None
        """
        avg_episode_reward = np.mean(episode_rewards)
        avg_episode_steps = np.mean(episode_steps)
        log_str = "epsilon:{:.2f},avg_R:{:.2f},avg_t:{:.2f},avg_h:{:.2f},acc_A_R:{:.2f}," \
                  "acc_D_R:{:.2f}".format(self.config.epsilon, avg_episode_reward,
                                          avg_episode_steps,
                                          self.env.hack_probabiltiy(),
                                          self.env.state.attacker_cumulative_reward,
                                          self.env.state.defender_cumulative_reward)
        self.outer.set_description_str(log_str)
        self.config.logger.info(log_str)
        result.avg_episode_steps.append(avg_episode_steps)
        result.avg_episode_rewards.append(avg_episode_reward)
        result.epsilon_values.append(self.config.epsilon)
        result.hack_probability.append(self.env.hack_probabiltiy())
        result.attacker_cumulative_reward.append(self.env.state.attacker_cumulative_reward)
        result.defender_cumulative_reward.append(self.env.state.defender_cumulative_reward)

    def eval(self):
        """
        Performs evaluation with the greedy policy with respect to the learned Q-values

        :return: None
        """
        self.config.logger.info("Starting Evaluation")

        if len(self.eval_result.avg_episode_steps) > 0:
            self.config.logger.warning("starting eval with non-empty result object")
        if(self.config.eval_episodes < 1):
            return
        done = False
        mode = "human"

        # Video config
        if self.config.video:
            if self.config.video_dir is None:
                raise AssertionError("Video is set to True but no video_dir is provided, please specify "
                                     "the video_dir argument")
            self.env = IdsGameMonitor(self.env, self.config.video_dir, force=True)
            self.env.metadata["video.frames_per_second"] = self.config.video_fps

        # Tracking metrics
        episode_rewards = []
        episode_steps = []

        # Eval
        obs = self.env.reset()
        self.env.state.restart()
        for episode in range(self.config.eval_episodes):
            i = 0
            episode_reward = 0
            episode_step = 0
            while not done:
                self.env.render()
                time.sleep(self.config.eval_sleep)
                i = i+1
                attacker_node_id = self.env.get_attacker_node_from_observation(obs)
                action = self.get_action(attacker_node_id, eval=True)
                obs, reward, done, _ = self.env.step(action)
                episode_reward += reward
                episode_step += 1
            self.env.render()
            time.sleep(self.config.eval_sleep)
            self.config.logger.info("Eval episode: {}, Game ended after {} steps".format(episode, i))
            episode_rewards.append(episode_reward)
            episode_steps.append(episode_step)

            # Log average metrics every <self.config.eval_log_frequency> episodes
            if episode % self.config.eval_log_frequency == 0:
                self.log_metrics(self.eval_result, episode_rewards, episode_steps)
                episode_rewards = []
                episode_steps = []
            done = False
            obs = self.env.reset()

        self.config.logger.info("Evaluation Complete")
        return self.eval_result

    def anneal_epsilon(self):
        """
        Anneals the exploration rate slightly until it reaches the minimum value

        :return: None
        """
        if self.config.epsilon > self.config.min_epsilon:
            self.config.epsilon = self.config.epsilon*self.config.epsilon_decay

    def log_state_values(self):
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

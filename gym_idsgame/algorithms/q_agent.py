"""
An agent for the IDSGameEnv that implements the tabular Q-learning algorithm.
"""
import numpy as np
import time
import tqdm
from gym import wrappers
from gym_idsgame.algorithms.q_agent_config import QAgentConfig
from gym_idsgame.envs.idsgame_env import IdsGameEnv
from gym_idsgame.algorithms.train_result import TrainResult
from gym_idsgame.algorithms.train_agent import TrainAgent

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
        done = False
        obs = self.env.reset()

        # Tracking metrics
        episode_rewards = []
        episode_steps = []
        epsilon_values = []

        # Logging
        outer = tqdm.tqdm(total=self.config.num_episodes, desc='Epoch', position=0)
        outer.set_description_str("epsilon:{:.2f},avg_R:{:.2f},avg_t:{:.2f}".format(self.config.epsilon, 0.0, 0.0))

        # Training
        for episode in range(self.config.num_episodes):
            episode_reward = 0
            episode_step = 0
            epsilon_values.append(self.config.epsilon)
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
            if episode % self.config.log_frequency == 0 and episode > 0:
                outer.set_description_str("epsilon:{:.2f},avg_R:{:.2f},avg_t:{:.2f},avg_h:{:.2f},acc_A_R:{:.2f},"
                                          "acc_D_R:{:.2f}".format(self.config.epsilon, sum(episode_rewards)/episode,
                                                                  sum(episode_steps)/episode,
                                                                  self.env.hack_probabiltiy(),
                                                                  self.env.state.attacker_cumulative_reward,
                                                                  self.env.state.defender_cumulative_reward))
            self.anneal_epsilon()
            done=False
            obs = self.env.reset()
            outer.update(1)

        print("Training Complete")
        return TrainResult(episode_rewards=episode_rewards, episode_steps=episode_steps, epsilon_values=epsilon_values)

    def eval(self):
        """
        Performs evaluation with the greedy policy with respect to the learned Q-values

        :return: None
        """
        if(self.config.eval_episodes < 1):
            return
        done = False

        # Video config
        if self.config.video:
            if self.config.video_dir is None:
                raise AssertionError("Video is set to True but no video_dir is provided, please specify "
                                     "the video_dir argument")
            self.env = wrappers.Monitor(self.env, self.config.video_dir, force=True)
            self.env.metadata["video.frames_per_second"] = self.config.video_fps

        # Tracking metrics
        episode_rewards = []
        episode_steps = []

        # Eval
        obs = self.env.restart()
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
            print("Eval episode: {}, Game ended after {} steps".format(episode, i))
            episode_rewards.append(episode_reward)
            episode_steps.append(episode_step)
            done = False
            obs = self.env.reset()

        return TrainResult(episode_rewards = episode_rewards, episode_steps = episode_steps)

    def anneal_epsilon(self):
        """
        Anneals the exploration rate slightly until it reaches the minimum value

        :return: None
        """
        if self.config.epsilon > self.config.min_epsilon:
            self.config.epsilon = self.config.epsilon*self.config.epsilon_decay

    def print_state_values(self):
        """
        Utility function for printing the state-values according to the learned Q-function
        :return:
        """
        print("--- State Values ---")
        for i in range(len(self.Q)):
            state_value = sum(self.Q[i])
            node_id = i
            print("s:{},V(s):{}".format(node_id, state_value))
        print("--------------------")

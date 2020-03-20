from gym_idsgame.envs.idsgame_env import IdsGameEnv
import numpy as np
import random
import time
import tqdm
from gym import wrappers

class QAgent:
    """
    A simple implementation of the Q(0)-learning algorithm (Sutton & Barto).
    Q-learning is a one-step off-policy algorithm
    """
    def __init__(self, env:IdsGameEnv, gamma=0.8, alpha=0.1, epsilon=0.9, render = False, eval_sleep = 0.35,
                 epsilon_decay = 0.999, min_epsilon = 0.1, eval_epochs = 1, log_frequency = 100,
                 video = False, video_fps=5, video_dir = None):
        """
        Initalize environment and hyperparameters

        :param pi: the policy
        :param env: the environment
        :param gamma: the discount factor
        :param alpha: the learning rate
        :param epsilon: the exploration rate
        :param render: whether to render the environment *during training* (it will always render at evaluation)
        :param eval_sleep: amount of sleep between time-steps during evaluation and rendering
        :param epsilon_decay: rate of decay of epsilon
        :param min_epsilon: minimum epsilon rate
        :param eval_epochs: number of evaluation epochs
        :param log_frequency: number of episodes between logs
        :param video: boolean flag whether to record video of the evaluation.
        :param video_dir: path where to save videos (will overwrite)
        """
        self.gamma = gamma
        self.env = env
        self.alpha = alpha
        self.epsilon=epsilon
        self.Q = np.random.rand(self.env.num_states, self.env.num_actions)
        self.render = render
        self.eval_sleep = eval_sleep
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.eval_epochs = eval_epochs
        self.log_frequency = log_frequency
        self.video = video
        self.video_fps = video_fps
        self.video_dir = video_dir


    def get_action(self, s, eval=False):
        """
        Sample an action using an epsilon-greedy policy with respect to the current Q-values

        :param s:  the state to sample an action for
        :return: a sampled action
        """
        if np.random.rand() < self.epsilon and not eval:
            return random.randrange(self.env.num_actions)
        return np.argmax(self.Q[s]).item()


    def run(self, num_episodes):
        """
        Runs the Q(0)-learning algorithm for estimating the state values under a given policy for a specific MDP

        :param num_episodes: the number of iterations to run until stopping
        :return: None
        """
        done = False
        s = self.env.reset()

        # Tracking metrics
        episode_rewards = []
        episode_steps = []
        epsilon_values = []

        # Logging
        outer = tqdm.tqdm(total=num_episodes, desc='Epoch', position=0)
        outer.set_description_str("epsilon:{:.2f},avg_R:{:.2f},avg_t:{:.2f}".format(self.epsilon, 0.0, 0.0))

        # Training
        for episode in range(num_episodes):
            episode_reward = 0
            episode_step = 0
            epsilon_values.append(self.epsilon)
            while not done:
                if self.render:
                    self.env.render(mode="human")
                s_index = self.env.get_state_index(s)
                action = self.get_action(s_index)
                s_prime, reward, done, _ = self.env.step(action)
                episode_reward += reward
                episode_step += 1
                s_prime_index = self.env.get_state_index(s_prime)
                # Q-learning update
                self.Q[s_index, action] = self.Q[s_index, action] + self.alpha*(reward + self.gamma*np.max(self.Q[s_prime_index]) - self.Q[s_index,action])
                s = s_prime

            episode_rewards.append(episode_reward)
            episode_steps.append(episode_step)
            if episode % self.log_frequency == 0 and episode > 0:
                outer.set_description_str("epsilon:{:.2f},avg_R:{:.2f},avg_t:{:.2f}".format(
                    self.epsilon, sum(episode_rewards)/episode, sum(episode_steps)/episode))
            self.anneal_epsilon()
            done=False
            s = self.env.reset()
            outer.update(1)

        print("Training Complete")
        return episode_rewards, episode_steps, epsilon_values

    def print_state_values(self, width=5, height=5):
        """
        Utility function for pretty printing the state-values of the gridworld_v1 env

        :param width: the width of the grid
        :param height: the height of the grid
        :return:
        """
        print("State values:")
        state_values = list(map(lambda i: self.Q[i].sum(), range(0, self.env.num_states)))
        for j in range(height):
            str = ''
            for i in range(width):
                idx = i*width + j
                str = '{} {:+02.2f}'.format(str, state_values[idx])
            print(str)

    def eval(self):
        """
        Performs evaluation with the greedy policy with respect to the learned Q-values

        :return: None
        """
        if(self.eval_epochs < 1):
            return
        done = False
        if self.video:
            if self.video_dir is None:
                raise AssertionError("Video is set to True but no video_dir is provided, please specify "
                                     "the video_dir argument")
            self.env = wrappers.Monitor(self.env, self.video_dir, force=True)
            self.env.metadata["video.frames_per_second"] = self.video_fps
        s = self.env.reset()
        for epoch in range(self.eval_epochs):
            i = 0
            while not done:
                self.env.render()
                time.sleep(self.eval_sleep)
                i = i+1
                s_index = self.env.get_state_index(s)
                action = self.get_action(s_index, eval=True)
                s, reward, done, _ = self.env.step(action)
            print("Eval epoch: {}, Reached the goal after {} steps".format(epoch, i))
            done = False
            s = self.env.reset()

    def anneal_epsilon(self):
        """
        Anneals the exploration rate slightly until it reaches the minimum value

        :return: None
        """
        if self.epsilon > self.min_epsilon:
            self.epsilon = self.epsilon*self.epsilon_decay

# Program entrypoint, runs the Q(0)-learning algorithm
if __name__ == '__main__':
    pass
    # env = YagwEnv(height=8, width=8)
    # q_agent = QAgent(env, gamma=0.99, alpha=0.2, epsilon=1, render=False, eval_sleep=0.3,
    #                  min_epsilon=0.1, eval_epochs=2, log_frequency=100, epsilon_decay=0.999, video=True,
    #                  video_fps = 5, video_dir="./videos")
    # episode_rewards, episode_steps, epsilon_values = q_agent.run(5000)
    # q_agent.print_state_values(height=env.height, width=env.width)
    # q_agent.eval()
from collections import deque
import random
import numpy as np

class ReplayBuffer:
    """
    Implements the replay buffer from the paper 'Human-level control through deep reinforcement learning' by
    Mnih et. al.

    A fixed-length buffer with experience-tuples
    """
    def __init__(self, buffer_size:int):
        if buffer_size < 0:
            raise ValueError("The replay buffer size cannot be negative")
        self.BUFFER_SIZE = buffer_size
        self.count = 0
        self.buffer = deque() # efficient datastructure for pops and appends

    def add_tuple(self, state1: np.ndarray, action:int, reward:float, done:bool, state2: np.ndarray):
        """
        Adds a tuple of experience to the replay buffer

        :param state1: the starting state of the experience (4 84x84 images)
        :param action: the action taken in state1 (int)
        :param reward: the reward received with taking action in state1 and transitioning to state2 (float scalar)
        :param done: a boolean flag that indicates whether state2 is a terminal state and the episode is over
        :param state2: the state transitioned to when taking action in state1 (4 84x84 images)
        :return: None
        """
        # # Clip all positive rewards at 1 and all negative rewards at -1, leaving 0 rewards unchanged
        # reward = np.clip(reward, -1, 1)
        attacker_state_1, defender_state_1 = state1
        attacker_state_2, defender_state_2 = state2
        attacker_reward, defender_reward = reward
        attacker_action, defender_action = action
        exp_tuple = (attacker_state_1.flatten(), defender_state_1.flatten(), attacker_action, defender_action,
                     attacker_reward, defender_reward, done, attacker_state_2.flatten(), defender_state_2.flatten())
        if self.count < self.BUFFER_SIZE:
            self.buffer.append(exp_tuple)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(exp_tuple)

    def size(self):
        """
        :return: the size of the buffer
        """
        return self.count

    def sample(self, batch_size: int):
        """
        Samples a random batch from the replay  buffer

        :param batch_size: the size of the batch
        :return: Five arrays: s_batch, a_batch, r_batch, d_batch, s2_batch
        """
        if batch_size < 1:
            raise ValueError("Cannot sample a batch of size less than 1")

        sample_size = self.count if self.count < batch_size else batch_size
        # Gives a list of [(state1, action, reward, done, state2)]
        batch = random.sample(self.buffer, sample_size)

        # 1. Unzips the list of tuples [(state1, action, reward, done, state2)] into 5 independent tuples
        # 2. Converts each tuple into numpy arrays
        s_attacker_batch, s_defender_batch, a_attacker_batch, a_defender_batch, r_attacker_batch, r_defender_batch, \
        d_batch, s2_attacker_batch, s2_defender_batch = list(map(np.array, list(zip(*batch))))

        return s_attacker_batch, s_defender_batch, a_attacker_batch, a_defender_batch, r_attacker_batch, \
               r_defender_batch, d_batch, s2_attacker_batch, s2_defender_batch

    def clear(self):
        """
        Clears/resets the buffer

        :return: None
        """
        self.buffer.clear()
        self.count = 0
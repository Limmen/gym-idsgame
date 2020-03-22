"""
Training/Eval results
"""
from typing import List
import csv

class TrainResult:
    """
    DTO with training/eval result from an experiment in the IDSGameEnvironment
    """

    def __init__(self, avg_episode_rewards: List[float] = None, avg_episode_steps: List[int] = None,
                 epsilon_values: List[float] = None, hack_probability: List[float] = None,
                 attacker_cumulative_reward: List[int] = None, defender_cumulative_reward: List[int] = None):
        """
        Constructor, initializes the DTO

        :param avg_episode_rewards: list of episode rewards
        :param avg_episode_steps: list of episode steps
        :param epsilon_values: list of epsilon values
        :param hack_probability: list of hack probabilities
        :param attacker_cumulative_reward: list of attacker cumulative rewards
        :param defender_cumulative_reward: list of defender cumulative rewards
        """
        self.avg_episode_rewards = avg_episode_rewards
        self.avg_episode_steps = avg_episode_steps
        self.epsilon_values = epsilon_values
        self.hack_probability = hack_probability
        self.attacker_cumulative_reward = attacker_cumulative_reward
        self.defender_cumulative_reward = defender_cumulative_reward

        if avg_episode_steps is None:
            self.avg_episode_steps = []
        if avg_episode_rewards is None:
            self.avg_episode_rewards = []
        if epsilon_values is None:
            self.epsilon_values = []
        if hack_probability is None:
            self.hack_probability = []
        if attacker_cumulative_reward is None:
            self.attacker_cumulative_reward = []
        if defender_cumulative_reward is None:
            self.defender_cumulative_reward = []


    def to_csv(self, file_path):
        rows = zip(self.avg_episode_rewards, self.avg_episode_steps, self.epsilon_values, self.hack_probability,
                   self.attacker_cumulative_reward, self.defender_cumulative_reward)
        with open(file_path, "w") as f:
            writer = csv.writer(f)
            writer.writerow(["avg_episode_rewards", "avg_episode_steps", "epsilon_values", "hack_probability",
                             "attacker_cumulative_reward", "defender_cumulative_reward"])
            for row in rows:
                writer.writerow(row)
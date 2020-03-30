"""
Experiment results
"""
from typing import List
import csv

class ExperimentResult:
    """
    DTO with experiment result from an experiment in the IDSGameEnvironment
    """

    def __init__(self, avg_episode_rewards: List[float] = None, avg_episode_steps: List[int] = None,
                 epsilon_values: List[float] = None, hack_probability: List[float] = None,
                 attacker_cumulative_reward: List[int] = None, defender_cumulative_reward: List[int] = None,
                 attacker_wins: List[int] = None, defender_wins: List[int] = None
                 ):
        """
        Constructor, initializes the DTO

        :param avg_episode_rewards: list of episode rewards
        :param avg_episode_steps: list of episode steps
        :param epsilon_values: list of epsilon values
        :param hack_probability: list of hack probabilities
        :param attacker_cumulative_reward: list of attacker cumulative rewards
        :param defender_cumulative_reward: list of defender cumulative rewards
        :param attacker_wins: num episodes won by the attacker
        :param defender_wins: num episodes won by the defender
        """
        self.avg_episode_rewards = avg_episode_rewards
        self.avg_episode_steps = avg_episode_steps
        self.epsilon_values = epsilon_values
        self.hack_probability = hack_probability
        self.attacker_cumulative_reward = attacker_cumulative_reward
        self.defender_cumulative_reward = defender_cumulative_reward
        self.attacker_wins = attacker_wins
        self.defender_wins = defender_wins
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
        if attacker_wins is None:
            self.attacker_wins = []
        if defender_wins is None:
            self.defender_wins = []


    def to_csv(self, file_path):
        metrics = [self.avg_episode_rewards, self.avg_episode_steps, self.epsilon_values, self.hack_probability,
                   self.attacker_cumulative_reward, self.defender_cumulative_reward, self.attacker_wins,
                   self.defender_wins]
        metric_labels = ["avg_episode_rewards", "avg_episode_steps", "epsilon_values", "hack_probability",
                             "attacker_cumulative_reward", "defender_cumulative_reward","attacker_wins",
                             "defender_wins"]
        filtered_metric_labels = []
        filtered_metrics = []
        for i in range(len(metrics)):
            if len(metrics[i]) > 0:
                filtered_metrics.append(metrics[i])
                filtered_metric_labels.append(metric_labels[i])
        rows = zip(*filtered_metrics)
        with open(file_path, "w") as f:
            writer = csv.writer(f)
            writer.writerow(filtered_metric_labels)
            for row in rows:
                writer.writerow(row)
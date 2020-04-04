"""
Experiment results
"""
from typing import List
import csv

class ExperimentResult:
    """
    DTO with experiment result from an experiment in the IDSGameEnvironment
    """

    def __init__(self, avg_attacker_episode_rewards: List[float] = None,
                 avg_defender_episode_rewards: List[float] = None,
                 avg_episode_steps: List[int] = None,
                 epsilon_values: List[float] = None, hack_probability: List[float] = None,
                 attacker_cumulative_reward: List[int] = None, defender_cumulative_reward: List[int] = None,
                 attacker_wins: List[int] = None, defender_wins: List[int] = None,
                 avg_episode_loss_attacker: List[float] = None, avg_episode_loss_defender: List[float] = None,
                 lr_list : List[float] = None
                 ):
        """
        Constructor, initializes the DTO

        :param avg_attacker_episode_rewards: list of episode rewards for attacker
        :param avg_defender_episode_rewards: list of episode rewards for defender
        :param avg_episode_steps: list of episode steps
        :param epsilon_values: list of epsilon values
        :param hack_probability: list of hack probabilities
        :param attacker_cumulative_reward: list of attacker cumulative rewards
        :param defender_cumulative_reward: list of defender cumulative rewards
        :param attacker_wins: num episodes won by the attacker
        :param defender_wins: num episodes won by the defender
        :param avg_episode_loss_attacker: average loss for attacker
        :param avg_episode_loss_defender: average loss for defender
        :param lr_list: learning rates
        """
        self.avg_attacker_episode_rewards = avg_attacker_episode_rewards
        self.avg_defender_episode_rewards = avg_defender_episode_rewards
        self.avg_episode_steps = avg_episode_steps
        self.epsilon_values = epsilon_values
        self.hack_probability = hack_probability
        self.attacker_cumulative_reward = attacker_cumulative_reward
        self.defender_cumulative_reward = defender_cumulative_reward
        self.attacker_wins = attacker_wins
        self.defender_wins = defender_wins
        self.avg_episode_loss_attacker = avg_episode_loss_attacker
        self.avg_episode_loss_defender = avg_episode_loss_defender
        self.lr_list = lr_list
        if avg_episode_steps is None:
            self.avg_episode_steps = []
        if avg_attacker_episode_rewards is None:
            self.avg_attacker_episode_rewards = []
        if avg_defender_episode_rewards is None:
            self.avg_defender_episode_rewards = []
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
        if avg_episode_loss_attacker is None:
            self.avg_episode_loss_attacker = []
        if avg_episode_loss_defender is None:
            self.avg_episode_loss_defender = []
        if lr_list is None:
            self.lr_list = []


    def to_csv(self, file_path : str) -> None:
        """
        Save result to csv

        :param file_path: path to save the csv file
        :return: None
        """
        metrics = [self.avg_attacker_episode_rewards, self.avg_defender_episode_rewards,
                   self.avg_episode_steps, self.epsilon_values, self.hack_probability,
                   self.attacker_cumulative_reward, self.defender_cumulative_reward, self.attacker_wins,
                   self.defender_wins, self.avg_episode_loss_attacker, self.lr_list]
        metric_labels = ["avg_attacker_episode_rewards", "avg_defender_episode_rewards", "avg_episode_steps",
                         "epsilon_values", "hack_probability", "attacker_cumulative_reward",
                         "defender_cumulative_reward","attacker_wins", "defender_wins", "avg_episode_loss_attacker",
                         "avg_episode_loss_defender", "lr_list"]
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
"""
Training/Eval results
"""
from typing import List

class TrainResult:
    """
    DTO with training/eval result from an experiment in the IDSGameEnvironment
    """

    def __init__(self, episode_rewards: List[float] = [], episode_steps: List[int] = [],
                 epsilon_values: List[float] = [], hack_probability: List[float] = [],
                 attacker_cumulative_reward: List[int] = [], defender_cumulative_reward: List[int] = []):
        """
        Constructor, initializes the DTO

        :param episode_rewards: list of episode rewards
        :param episode_steps: list of episode steps
        :param epsilon_values: list of epsilon values
        :param hack_probability: list of hack probabilities
        :param attacker_cumulative_reward: list of attacker cumulative rewards
        :param defender_cumulative_reward: list of defender cumulative rewards
        """
        self.avg_episode_rewards = episode_rewards
        self.avg_episode_steps = episode_steps
        self.epsilon_values = epsilon_values
        self.hack_probability = hack_probability
        self.attacker_cumulative_reward = attacker_cumulative_reward
        self.defender_cumulative_reward = defender_cumulative_reward
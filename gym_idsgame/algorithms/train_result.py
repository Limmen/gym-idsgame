"""
Training/Eval results
"""
from typing import List

class TrainResult:
    """
    DTO with training/eval result from an experiment in the IDSGameEnvironment
    """

    def __init__(self, episode_rewards: List[float] = None, episode_steps: List[int] = None,
                 epsilon_values: List[float] = None):
        """
        Constructor, initializes the DTO

        :param episode_rewards: list of episode rewards
        :param episode_steps: list of episode steps
        :param epsilon_values: list of epsilon values
        """
        self.episode_rewards = episode_rewards
        self.episode_steps = episode_steps
        self.epsilon_values = epsilon_values
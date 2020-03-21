from typing import List

class TrainResult:

    def __init__(self, episode_rewards: List[float] = None, episode_steps: List[int] = None,
                 epsilon_values: List[float] = None):
        self.episode_rewards = episode_rewards
        self.episode_steps = episode_steps
        self.epsilon_values = epsilon_values
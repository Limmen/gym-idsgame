from abc import ABC

class Agent(ABC):

    def __init__(self):
        self.cumulative_reward = 0

    def add_reward(self, reward) -> None:
        self.cumulative_reward += reward
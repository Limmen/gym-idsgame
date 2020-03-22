"""
Abstract TrainAgent with an interface for plug-an-play algorithms for the IdsGame Environment
"""
from abc import ABC, abstractmethod
from gym_idsgame.agents.dao.train_result import TrainResult
class TrainAgent(ABC):
    """
    Abstract train agent
    """

    @abstractmethod
    def train(self) -> TrainResult:
        pass

    @abstractmethod
    def eval(self) -> TrainResult:
        pass


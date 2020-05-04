"""
Abstract TrainAgent with an interface for plug-an-play algorithms for the IdsGame Environment
"""
from abc import ABC, abstractmethod
from gym_idsgame.agents.dao.experiment_result import ExperimentResult
class TrainAgent(ABC):
    """
    Abstract train agent
    """

    @abstractmethod
    def train(self) -> ExperimentResult:
        pass

    @abstractmethod
    def eval(self) -> ExperimentResult:
        pass


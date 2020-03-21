from abc import ABC, abstractmethod
from gym_idsgame.algorithms.train_result import TrainResult
class TrainAgent(ABC):

    @abstractmethod
    def train(self) -> TrainResult:
        pass

    @abstractmethod
    def eval(self) -> TrainResult:
        pass


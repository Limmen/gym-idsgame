from typing import Union
import gym
from gym_idsgame.config.client_config import ClientConfig
from gym_idsgame.config.idsgame_mode import IdsGameMode
from gym_idsgame.agents.agent_type import AgentType
from gym_idsgame.algorithms.q_agent import QAgent
from gym_idsgame.algorithms.train_agent import TrainAgent
from gym_idsgame.algorithms.train_result import TrainResult
class Runner:

    @staticmethod
    def run(config: ClientConfig):
        if config.mode == IdsGameMode.TRAIN_ATTACKER.value:
            return Runner.train_attacker(config)
        elif config.mode == IdsGameMode.TRAIN_DEFENDER.value:
            return Runner.train_defender(config)
        elif config.mode == IdsGameMode.SIMULATE.value:
            return Runner.simulate(config)
        elif config.mode == IdsGameMode.MANUAL_ATTACKER.value:
            return Runner.manual_play_attacker(config)
        elif config.mode == IdsGameMode.MANUAL_DEFENDER.value:
            return Runner.manual_play_defender(config)
        else:
            raise AssertionError("Runner mode not recognized: {}".format(config.mode))

    @staticmethod
    def train_attacker(config: ClientConfig) -> Union[TrainResult, TrainResult]:
        env = gym.make(config.env_name)
        attacker: TrainAgent = None
        if config.attacker_type == AgentType.Q_AGENT.value:
            attacker = QAgent(env, config.q_agent_config)
        elif config.attacker_type == AgentType.RANDOM.value:
            pass
        elif config.attacker_type == AgentType.DETERMINISTIC.value:
            pass
        else:
            raise AssertionError("Attacker type not recognized: {}".format(config.attacker_type))
        train_result = attacker.train()
        eval_result = attacker.eval()
        return train_result, eval_result

    @staticmethod
    def train_defender(config: ClientConfig):
        env = gym.make(config.env_name)
        if config.defender_type == AgentType.Q_AGENT.value:
            defender = QAgent(env, config.q_agent_config)
        elif config.defender_type == AgentType.RANDOM.value:
            pass
        elif config.defender_type == AgentType.DETERMINISTIC.value:
            pass
        else:
            raise AssertionError("Defender type not recognized: {}".format(config.defender_type))

    @staticmethod
    def simulate(config:ClientConfig):
        pass

    @staticmethod
    def manual_play_attacker(config):
        pass

    @staticmethod
    def manual_play_defender(config):
        pass
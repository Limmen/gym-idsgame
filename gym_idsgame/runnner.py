"""
Generic runner for running experiments with idsgame environments
"""
from typing import Union
import gym
from gym_idsgame.config.client_config import ClientConfig
from gym_idsgame.config.runner_mode import RunnerMode
from gym_idsgame.agents.dao.agent_type import AgentType
from gym_idsgame.agents.q_agent import QAgent
from gym_idsgame.agents.train_agent import TrainAgent
from gym_idsgame.agents.dao.experiment_result import ExperimentResult
from gym_idsgame.agents.manual_attack_agent import ManualAttackAgent
from gym_idsgame.agents.manual_defense_agent import ManualDefenseAgent
from gym_idsgame.envs.idsgame_env import IdsGameEnv, AttackDefenseEnv, AttackerEnv, DefenderEnv
from gym_idsgame.simulation.simulator import Simulator

class Runner:
    """
    Class with utility methods for running structured experiments with idsgame environments
    """

    @staticmethod
    def run(config: ClientConfig):
        """
        Run entrypoint

        :param config: configuration for the run
        :return: the result
        """
        if config.mode == RunnerMode.TRAIN_ATTACKER.value:
            return Runner.train_attacker(config)
        elif config.mode == RunnerMode.TRAIN_DEFENDER.value:
            return Runner.train_defender(config)
        elif config.mode == RunnerMode.SIMULATE.value:
            return Runner.simulate(config)
        elif config.mode == RunnerMode.MANUAL_ATTACKER.value:
            return Runner.manual_play_attacker(config)
        elif config.mode == RunnerMode.MANUAL_DEFENDER.value:
            return Runner.manual_play_defender(config)
        else:
            raise AssertionError("Runner mode not recognized: {}".format(config.mode))

    @staticmethod
    def train_attacker(config: ClientConfig) -> Union[ExperimentResult, ExperimentResult]:
        """
        Trains an attacker agent in the environment

        :param config: Training configuration
        :return: trainresult, evalresult
        """
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
    def simulate(config: ClientConfig):
        env: IdsGameEnv = None
        env = gym.make(config.env_name)
        if not issubclass(type(env), AttackDefenseEnv):
            raise AssertionError("Simulations can only be run for Attack-Defense environments")
        env.idsgame_config.defender_agent = config.simulation_config.defender_agent
        env.idsgame_config.attacker_agent = config.simulation_config.attacker_agent
        simulator = Simulator(env, config.simulation_config)
        return simulator.simulate()

    @staticmethod
    def manual_play_attacker(config: ClientConfig):
        env: IdsGameEnv = gym.make(config.env_name)
        if not issubclass(type(env), AttackerEnv):
            raise AssertionError("Manual attacker play is only supported for attacker-envs")
        env.idsgame_config.game_config.manual_attacker = True
        ManualAttackAgent(env.idsgame_config)

    @staticmethod
    def manual_play_defender(config: ClientConfig):
        env: IdsGameEnv = gym.make(config.env_name)
        if not issubclass(type(env), DefenderEnv):
            raise AssertionError("Manual defender play is only supported for defender-envs")
        env.idsgame_config.game_config.manual_defender = True
        ManualDefenseAgent(env.idsgame_config)
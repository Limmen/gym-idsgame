"""
Generic runner for running experiments with idsgame environments
"""
from typing import Union
import gym

# In case running on server without screen
try:
    from gym_idsgame.agents.manual_attack_agent import ManualAttackAgent
    from gym_idsgame.agents.manual_defense_agent import ManualDefenseAgent
except:
    pass

from gym_idsgame.config.client_config import ClientConfig
from gym_idsgame.config.runner_mode import RunnerMode
from gym_idsgame.agents.dao.agent_type import AgentType
from gym_idsgame.agents.tabular_q_learning.tabular_q_agent import TabularQAgent
from gym_idsgame.agents.train_agent import TrainAgent
from gym_idsgame.agents.bot_agent import BotAgent
from gym_idsgame.agents.dao.experiment_result import ExperimentResult
from gym_idsgame.envs.idsgame_env import IdsGameEnv, AttackDefenseEnv, AttackerEnv, DefenderEnv
from gym_idsgame.simulation.simulator import Simulator
from gym_idsgame.agents.random_defense_bot_agent import RandomDefenseBotAgent
from gym_idsgame.agents.random_attack_bot_agent import RandomAttackBotAgent
from gym_idsgame.agents.defend_minimal_value_bot_agent import DefendMinimalValueBotAgent
from gym_idsgame.agents.attack_maximal_value_bot_agent import AttackMaximalValueBotAgent
from gym_idsgame.agents.tabular_q_learning.tabular_q_attacker_bot_agent import TabularQAttackerBotAgent
from gym_idsgame.agents.tabular_q_learning.tabular_q_defender_bot_agent import TabularQDefenderBotAgent

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
        elif config.mode == RunnerMode.TRAIN_DEFENDER_AND_ATTACKER.value:
            return Runner.train_attacker_and_defender(config)
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
        env: IdsGameEnv = None
        env = gym.make(config.env_name, idsgame_config = config.idsgame_config, save_dir=config.output_dir + "/data",
                       initial_state_path = config.initial_state_path)
        if config.title is not None:
            env.idsgame_config.render_config.title = config.title
        attacker: TrainAgent = None
        if config.attacker_type == AgentType.TABULAR_Q_AGENT.value:
            attacker = TabularQAgent(env, config.q_agent_config)
        else:
            raise AssertionError("Attacker train agent type not recognized: {}".format(config.attacker_type))
        train_result = attacker.train()
        eval_result = attacker.eval()
        return train_result, eval_result

    @staticmethod
    def train_defender(config: ClientConfig) -> Union[ExperimentResult, ExperimentResult]:
        """
        Trains a defender agent in the environment

        :param config: the training configuration
        :return: trainresult, evalresult
        """
        env = gym.make(config.env_name, idsgame_config = config.idsgame_config, save_dir=config.output_dir + "/data",
                       initial_state_path = config.initial_state_path)
        if config.title is not None:
            env.idsgame_config.render_config.title = config.title
        defender: TrainAgent = None
        if config.defender_type == AgentType.TABULAR_Q_AGENT.value:
            defender = TabularQAgent(env, config.q_agent_config)
        else:
            raise AssertionError("Defender train agent type not recognized: {}".format(config.defender_type))
        train_result = defender.train()
        eval_result = defender.eval()
        return train_result, eval_result


    @staticmethod
    def train_attacker_and_defender(config: ClientConfig) -> Union[ExperimentResult, ExperimentResult]:
        """
        Trains an attacker agent and a defender agent simultaneously in the environment

        :param config: Training configuration
        :return: trainresult, evalresult
        """
        env: IdsGameEnv = None
        env = gym.make(config.env_name, idsgame_config = config.idsgame_config, save_dir=config.output_dir + "/data",
                       initial_state_path = config.initial_state_path)
        if config.title is not None:
            env.idsgame_config.render_config.title = config.title
        agent: TrainAgent = None
        if config.attacker_type == AgentType.TABULAR_Q_AGENT.value:
            agent = TabularQAgent(env, config.q_agent_config)
        else:
            raise AssertionError("Train agent type not recognized: {}".format(config.attacker_type))
        train_result = agent.train()
        eval_result = agent.eval()
        return train_result, eval_result

    @staticmethod
    def simulate(config: ClientConfig) -> ExperimentResult:
        """
        Runs a simulation with two pre-defined policies against each other

        :param config: the simulation config
        :return: experiment result
        """
        env: IdsGameEnv = None
        env = gym.make(config.env_name, idsgame_config = config.idsgame_config, save_dir=config.output_dir + "/data",
                       initial_state_path = config.initial_state_path)
        if config.title is not None:
            env.idsgame_config.render_config.title = config.title
        if not issubclass(type(env), AttackDefenseEnv):
            raise AssertionError("Simulations can only be run for Attack-Defense environments")

        defender: BotAgent = None
        if config.defender_type == AgentType.RANDOM.value:
            defender = RandomDefenseBotAgent(env.idsgame_config.game_config)
        elif config.defender_type == AgentType.DEFEND_MINIMAL_VALUE.value:
            defender = DefendMinimalValueBotAgent(env.idsgame_config.game_config)
        elif config.defender_type == AgentType.TABULAR_Q_AGENT.value:
            if config.q_agent_config is None or config.q_agent_config.load_path is None:
                raise ValueError("To run a simulation with a tabular Q-agent, the path to the saved "
                                 "Q-table must be specified")
            defender = TabularQDefenderBotAgent(env.idsgame_config.game_config, config.q_agent_config.load_path)
        else:
            raise AssertionError("Defender type not recognized: {}".format(config.defender_type))

        attacker: BotAgent = None
        if config.attacker_type == AgentType.TABULAR_Q_AGENT.value:
            if config.q_agent_config is None or config.q_agent_config.load_path is None:
                raise ValueError("To run a simulation with a tabular Q-agent, the path to the saved "
                                 "Q-table must be specified")
            attacker = TabularQAttackerBotAgent(env.idsgame_config.game_config, config.q_agent_config.load_path)
        elif config.attacker_type == AgentType.RANDOM.value:
            attacker = RandomAttackBotAgent(env.idsgame_config.game_config)
        elif config.attacker_type == AgentType.ATTACK_MAXIMAL_VALUE.value:
            attacker = AttackMaximalValueBotAgent(env.idsgame_config.game_config)
        else:
            raise AssertionError("Attacker type not recognized: {}".format(config.attacker_type))
        env.idsgame_config.defender_agent = defender
        env.idsgame_config.attacker_agent = attacker
        simulator = Simulator(env, config.simulation_config)
        return simulator.simulate()

    @staticmethod
    def manual_play_attacker(config: ClientConfig) -> IdsGameEnv:
        """
        Starts an experiment with a manual attacker player against some bot

        :param config: configuration of the experiment
        :return: the created environment
        """
        env: IdsGameEnv = gym.make(config.env_name, idsgame_config = config.idsgame_config,
                                   save_dir=config.output_dir + "/data", initial_state_path = config.initial_state_path)
        if config.title is not None:
            env.idsgame_config.render_config.title = config.title
        if not issubclass(type(env), AttackerEnv):
            raise AssertionError("Manual attacker play is only supported for attacker-envs")
        env.idsgame_config.game_config.manual_attacker = True
        ManualAttackAgent(env.idsgame_config)
        return env

    @staticmethod
    def manual_play_defender(config: ClientConfig) -> IdsGameEnv:
        """
        Starts an experiment with a manual defender player against some bot

        :param config: the configuration of the experiment
        :return: the created environment
        """
        env: IdsGameEnv = gym.make(config.env_name, idsgame_config = config.idsgame_config,
                                   save_dir=config.output_dir + "/data", initial_state_path = config.initial_state_path)
        if config.title is not None:
            env.idsgame_config.render_config.title = config.title
        if not issubclass(type(env), DefenderEnv):
            raise AssertionError("Manual defender play is only supported for defender-envs")
        env.idsgame_config.game_config.manual_defender = True
        ManualDefenseAgent(env.idsgame_config)
        return env
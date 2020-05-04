"""
Client configuration for running experiments (parsed from JSON)
"""
from gym_idsgame.agents.training_agents.q_learning.q_agent_config import QAgentConfig
from gym_idsgame.agents.training_agents.policy_gradient.pg_agent_config import PolicyGradientAgentConfig
from gym_idsgame.envs.dao.idsgame_config import IdsGameConfig
from gym_idsgame.simulation.dao.simulation_config import SimulationConfig
from gym_idsgame.config.hp_tuning_config import HpTuningConfig

class ClientConfig:
    """
    DTO with client config for running experiments
    """

    def __init__(self, env_name:str, attacker_type: int = 0,
                 defender_type: int = 1, mode: int = 0, q_agent_config: QAgentConfig = None,
                 pg_agent_config: PolicyGradientAgentConfig = None,
                 output_dir:str = None, simulation_config: SimulationConfig = None, title = None,
                 idsgame_config : IdsGameConfig = None, initial_state_path: str = None, run_many :bool = False,
                 random_seeds : list = None, random_seed = 0, hp_tuning_config : HpTuningConfig = None,
                 hp_tuning : bool = False):
        """
        Class constructor, initializes the DTO

        :param env_name: name of the environment for the experiment
        :param attacker_type: type of attacker
        :param defender_type: type of defender
        :param mode: type of experiment
        :param q_agent_config: configuration in case QAgent is used for one of the agents
        :param simulation_config: configuration for running a simulation (no training)
        :param output_dir: directory to save outputs (results)
        :param title: title in the GUI
        :param idsgame_config: idsgame configuration
        :param initial_state_path: path to initial state
        :param run_many: if this is true, it will try to run many experiments in a row, using different random seeds
        :param random_seeds: list of random seeds when running several experiments in a row
        :param random_seed: specific random seed
        :param hp_tuning_config: hyperparameter tuning config
        :param hp_tuning: boolean flag, if true runs hyperparameter tuning, otherwise run regular experiment
        :param pg_agent_config: policy gradient agent config
        """
        self.env_name = env_name
        self.attacker_type = attacker_type
        self.defender_type = defender_type
        self.mode = mode
        self.q_agent_config = q_agent_config
        self.logger = None
        self.output_dir = output_dir
        self.simulation_config = simulation_config
        self.title = title
        self.idsgame_config = idsgame_config
        self.initial_state_path = initial_state_path
        self.run_many = run_many
        self.random_seeds = random_seeds
        self.random_seed = random_seed
        self.hp_tuning_config = hp_tuning_config
        self.hp_tuning = hp_tuning
        self.pg_agent_config = pg_agent_config
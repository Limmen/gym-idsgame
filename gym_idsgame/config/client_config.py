"""
Client configuration for running experiments (parsed from JSON)
"""
from gym_idsgame.agents.dao.q_agent_config import QAgentConfig

class ClientConfig:
    """
    DTO with client config for running experiments
    """

    def __init__(self, env_name:str, attacker_type: int = 0,
                 defender_type: int = 1, mode: int = 0, q_agent_config: QAgentConfig = None,
                 output_dir:str = None):
        """
        Class constructor, initializes the DTO

        :param env_name: name of the environment for the experiment
        :param attacker_type: type of attacker
        :param defender_type: type of defender
        :param mode: type of experiment
        :param q_agent_config: configuration in case QAgent is used for one of the agents
        :param output_dir: directory to save outputs (results)g
        """
        self.env_name = env_name
        self.attacker_type = attacker_type
        self.defender_type = defender_type
        self.mode = mode
        self.q_agent_config = q_agent_config
        self.logger = None
        self.output_dir = output_dir
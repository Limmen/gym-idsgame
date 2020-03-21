from gym_idsgame.algorithms.q_agent_config import QAgentConfig

class ClientConfig:

    def __init__(self, env_name:str, attacker_type: int = 0,
                 defender_type: int = 1,
                 mode: int = 0, q_agent_config: QAgentConfig = None):
        self.env_name = env_name
        self.attacker_type = attacker_type
        self.defender_type = defender_type
        self.mode = mode
        self.q_agent_config = q_agent_config
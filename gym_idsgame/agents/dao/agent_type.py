from enum import Enum

class AgentType(Enum):
    TABULAR_Q_AGENT = 0
    RANDOM = 1
    DEFEND_MINIMAL_VALUE = 2
    MANUAL_ATTACK = 3
    MANUAL_DEFENSE = 4
    ATTACK_MAXIMAL_VALUE = 5
    DQN_AGENT = 6

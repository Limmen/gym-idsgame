from enum import Enum

class AgentType(Enum):
    Q_AGENT = 0
    RANDOM = 1
    DETERMINISTIC = 2
    MANUAL_ATTACK = 3
    MANUAL_DEFENSE = 4

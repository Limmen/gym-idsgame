from enum import Enum

class PolicyType(Enum):
    """
    Enum representing different baseline policy types
    """
    DETERMINISTIC_DEFENSE = 0
    RANDOM_DEFENSE = 1
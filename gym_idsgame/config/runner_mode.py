"""
Different types of experiments for the runner
"""
from enum import Enum

class RunnerMode(Enum):
    """
    Mode for the experiments runner
    """
    TRAIN_ATTACKER = 0
    TRAIN_DEFENDER = 1
    SIMULATE = 2
    MANUAL_ATTACKER = 3
    MANUAL_DEFENDER = 4

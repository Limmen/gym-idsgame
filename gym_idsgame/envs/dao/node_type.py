"""
Type of nodes in the gym-idsgame environment
"""
from enum import Enum

class NodeType(Enum):
    """
    Enum representing the different node types in the network. Empty means that there is slot in the
    grid-network where there is no node.
    """
    EMPTY = 0
    START = 1
    SERVER = 2
    DATA = 3

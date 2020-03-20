"""
Utility functions for the gym-idsgame environment
"""
from typing import Union
import numpy as np
from gym_idsgame.envs.dao.idsgame_config import IdsGameConfig

def validate_config(idsgame_config: IdsGameConfig) -> None:
    """
    Validates the configuration for the environment

    :param idsgame_config: the config to validate
    :return: None
    """
    if idsgame_config.game_config.num_layers < 1:
        raise AssertionError("The number of layers cannot be less than 1")
    if idsgame_config.game_config.num_attack_types < 1:
        raise AssertionError("The number of attack types cannot be less than 1")
    if idsgame_config.game_config.max_value < 3:
        raise AssertionError("The max attack/defense value cannot be less than 3")

def is_attack_legal(target_pos: Union[int, int], attacker_pos: Union[int, int], num_cols: int,
                    adjacency_matrix: np.ndarray) -> bool:
    """
    Checks whether an attack is legal. That is, can the attacker reach the target node from its current
    position in 1 step given the network configuration?

    :param attacker_pos: the position of the attacker
    :param target_pos: the position of the target node
    :param num_cols: number of columns in the grid
    :param adjacency_matrix: the adjacency matrix
    :return: True if the attack is legal, otherwise False
    """
    attacker_row, attacker_col = attacker_pos
    attacker_adjacency_matrix_id = attacker_row * num_cols + attacker_col
    target_row, target_col = target_pos
    target_adjacency_matrix_id = target_row * num_cols + target_col
    return adjacency_matrix[attacker_adjacency_matrix_id][target_adjacency_matrix_id] == int(1)
"""
Utility functions for the gym-idsgame environment
"""
from typing import Union, List
from gym_idsgame.envs.dao.idsgame_config import IdsGameConfig
from gym_idsgame.envs.dao.game_config import GameConfig
from gym_idsgame.envs.dao.node_type import NodeType
from gym_idsgame.envs.dao.network_config import NetworkConfig

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


def is_defense_id_legal(defense_id: int, game_config: GameConfig) -> bool:
    """
    Check if a given defense is legal or not.

    :param defense_id: the defense to verify
    :return: True if legal otherwise False
    """
    server_id, server_pos, defense_type = interpret_defense_action(defense_id, game_config)
    if (game_config.network_config.node_list[server_id] == NodeType.SERVER.value
        or game_config.network_config.node_list[server_id] == NodeType.DATA.value):
        return True
    return False


def is_attack_legal(target_pos: Union[int, int], attacker_pos: Union[int, int], network_config: NetworkConfig,
                    past_positions: List[int] = None) -> bool:
    """
    Checks whether an attack is legal. That is, can the attacker reach the target node from its current
    position in 1 step given the network configuration?

    :param attacker_pos: the position of the attacker
    :param target_pos: the position of the target node
    :param network_config: the network configuration
    :param past_positions: if not None, used to check whether the agent is in a periodic policy, e.g. a circle.
    :return: True if the attack is legal, otherwise False
    """
    if target_pos == attacker_pos:
        return False
    target_row, target_col = target_pos
    attacker_row, attacker_col = attacker_pos
    if target_row > attacker_row:
        return False
    if past_positions is not None and len(past_positions) >=2:
        if target_pos in past_positions[-3:]:
            return False
    attacker_adjacency_matrix_id = attacker_row * network_config.num_cols + attacker_col
    target_adjacency_matrix_id = target_row * network_config.num_cols + target_col
    return network_config.adjacency_matrix[attacker_adjacency_matrix_id][target_adjacency_matrix_id] == int(1)


def is_attack_id_legal(attack_id: int, game_config: GameConfig, attacker_pos: Union[int, int],
                       past_positions: List[int] = None) -> bool:
    """
    Check if a given attack is legal or not.

    :param attack_id: the attack to verify
    :param game_config: game configuration
    :param attacker_pos: the current position of the attacker
    :param past_positions: if not None, used to check whether the agent is in a periodic policy, e.g. a circle.
    :return: True if legal otherwise False
    """
    server_id, server_pos, attack_type = interpret_attack_action(attack_id, game_config)
    return is_attack_legal(server_pos, attacker_pos, game_config.network_config, past_positions)


def interpret_attack_action(action: int, game_config: GameConfig) -> Union[int, Union[int, int], int]:
    """
    Utility method for interpreting the given attack action, converting it into server_id,pos,type

    :param action: the attack action-id
    :param game_config: game configuration
    :return: server-id, server-position, attack-type
    """
    server_id = action // game_config.num_attack_types
    server_pos = game_config.network_config.get_node_pos(server_id)
    attack_type = get_attack_type(action, game_config)
    # print("interpret action: {}, attack_type:{},server_pos:{}, server_id:{}".format(action, attack_type, server_pos,
    #                                                                                 server_id))
    return server_id, server_pos, attack_type

def interpret_defense_action(action: int, game_config: GameConfig) -> Union[int, Union[int, int], int]:
    """
    Utility method for interpreting the given action, converting it into server_id,pos,type

    :param action: the attack action-id
    :param game_config: game configuration
    :return: server-id, server-position, attack-type
    """
    server_id = action // (game_config.num_attack_types+1) # +1 for detection type attack
    server_pos = game_config.network_config.get_node_pos(server_id)
    defense_type = get_defense_type(action, game_config)
    return server_id, server_pos, defense_type


def get_attack_action_id(server_id, attack_type, game_config: GameConfig):
    """
    Gets the attack action id from a given server position, attack_type, and game config

    :param server_id: id of the server
    :param attack_type: attack type
    :param game_config: game config
    :return: attack id
    """
    action_id = server_id * game_config.num_attack_types + attack_type
    return action_id


def get_defense_action_id(server_id, defense_type, game_config: GameConfig):
    """
    Gets the defense action id from a given server position, defense_type, and game config

    :param server_id: id of the server
    :param defense_type: defense type
    :param game_config: game config
    :return: attack id
    """
    action_id = server_id * (game_config.num_attack_types+1) + defense_type
    return action_id


def get_attack_type(action: int, game_config: GameConfig) -> int:
    """
    Utility method for getting the attack type of action-id

    :param action: action-id
    :param game_config: game configuration
    :return: action type
    """
    attack_defense_type = action % game_config.num_attack_types
    return attack_defense_type


def get_defense_type(action: int, game_config: GameConfig) -> int:
    """
    Utility method for getting the defense type of action-id

    :param action: action-id
    :param game_config: game configuration
    :return: action type
    """
    defense_type = action % (game_config.num_attack_types+1) # +1 for detection
    return defense_type
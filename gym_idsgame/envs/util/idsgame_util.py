"""
Utility functions for the gym-idsgame environment
"""
from typing import Union, List
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import io
import cv2
from gym_idsgame.envs.dao.idsgame_config import IdsGameConfig
from gym_idsgame.envs.dao.game_config import GameConfig
from gym_idsgame.envs.dao.game_state import GameState
from gym_idsgame.envs.dao.node_type import NodeType
from gym_idsgame.envs.dao.network_config import NetworkConfig

def validate_config(idsgame_config: IdsGameConfig) -> None:
    """
    Validates the configuration for the environment

    :param idsgame_config: the config to validate
    :return: None
    """
    if idsgame_config.game_config.num_layers < 0:
        raise AssertionError("The number of layers cannot be less than 0")
    if idsgame_config.game_config.num_attack_types < 1:
        raise AssertionError("The number of attack types cannot be less than 1")
    if idsgame_config.game_config.max_value < 1:
        raise AssertionError("The max attack/defense value cannot be less than 1")


def is_defense_id_legal(defense_id: int, game_config: GameConfig, state : GameState) -> bool:
    """
    Check if a given defense is legal or not.

    :param defense_id: the defense to verify
    :param game_config: the game config
    :param state: the game state
    :return: True if legal otherwise False
    """
    server_id, server_pos, defense_type = interpret_defense_action(defense_id, game_config)

    # if defense_type < game_config.num_attack_types:
    #     if state.defense_values[server_id][defense_type] >= game_config.max_value:
    #         return False

    # if defense_type >= game_config.num_attack_types:
    #     if state.defense_det[server_id] >= game_config.max_value:
    #         return False

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
    # if past_positions is not None and len(past_positions) >=2:
    #     if target_pos in past_positions[-3:]:
    #         return False
    attacker_adjacency_matrix_id = attacker_row * network_config.num_cols + attacker_col
    target_adjacency_matrix_id = target_row * network_config.num_cols + target_col
    return network_config.adjacency_matrix[attacker_adjacency_matrix_id][target_adjacency_matrix_id] == int(1)


def is_node_attack_legal(target_node : int, attacker_pos : Union[int, int], network_config : NetworkConfig) -> bool:
    target_pos = network_config.get_node_pos(target_node)

    target_row, target_col = target_pos
    attacker_row, attacker_col = attacker_pos

    attacker_adjacency_matrix_id = attacker_row * network_config.num_cols + attacker_col
    target_adjacency_matrix_id = target_row * network_config.num_cols + target_col

    return network_config.adjacency_matrix[attacker_adjacency_matrix_id][target_adjacency_matrix_id] == int(1)


def is_node_defense_legal(target_node : int, network_config : NetworkConfig, state : GameState, max_value:int) -> bool:
    if (network_config.node_list[target_node] == NodeType.SERVER.value
            or network_config.node_list[target_node] == NodeType.DATA.value):
        if state.defense_det[target_node] < max_value:
            return True
        for i in range(len(state.defense_values[target_node])):
            if state.defense_values[target_node][i] < max_value:
                return True
        return False
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
    # if past_positions is not None and len(past_positions) >=2:
    #     if target_pos in past_positions[-3:]:
    #         return False
    attacker_adjacency_matrix_id = attacker_row * network_config.num_cols + attacker_col
    target_adjacency_matrix_id = target_row * network_config.num_cols + target_col
    return network_config.adjacency_matrix[attacker_adjacency_matrix_id][target_adjacency_matrix_id] == int(1)


def is_attack_id_legal(attack_id: int, game_config: GameConfig, attacker_pos: Union[int, int], game_state : GameState,
                       past_positions: List[int] = None, past_reconnaissance_activities: List = None) -> bool:
    """
    Check if a given attack is legal or not.

    :param attack_id: the attack to verify
    :param game_config: game configuration
    :param attacker_pos: the current position of the attacker
    :param game_state: the game state
    :param past_positions: if not None, used to check whether the agent is in a periodic policy, e.g. a circle.
    :return: True if legal otherwise False
    """
    server_id, server_pos, attack_type, reconnaissance = interpret_attack_action(attack_id, game_config)
    if not reconnaissance:
        if game_state.attack_values[server_id][attack_type] >= game_config.max_value:
            return False
    # if reconnaissance and past_reconnaissance_activities is not None:
    #     for rec_act in past_reconnaissance_activities[-5:]:
    #         node_id, rec_type = rec_act
    #         if node_id == server_id and rec_type == attack_type:
    #             #print("illegal rec type, past:{}".format(past_reconnaissance_activities))
    #             return False
    return is_attack_legal(server_pos, attacker_pos, game_config.network_config, past_positions)


def interpret_attack_action(action: int, game_config: GameConfig) -> Union[int, Union[int, int], int, bool]:
    """
    Utility method for interpreting the given attack action, converting it into server_id,pos,type

    :param action: the attack action-id
    :param game_config: game configuration
    :return: server-id, server-position, attack-type
    """
    if not game_config.reconnaissance_actions:
        server_id = action // game_config.num_attack_types
    else:
        server_id = action // (game_config.num_attack_types +1)
        #server_id = action // (game_config.num_attack_types*2)

    server_pos = game_config.network_config.get_node_pos(server_id)
    attack_type = get_attack_type(action, game_config)
    reconnaissance = attack_type >= game_config.num_attack_types
    if reconnaissance:
        attack_type = attack_type - game_config.num_attack_types
    #print("server:{},pos:{},a_type:{},rec:{}".format(server_id, server_pos, attack_type, reconnaissance))
    return server_id, server_pos, attack_type, reconnaissance

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
    if not game_config.reconnaissance_actions:
        action_id = server_id * game_config.num_attack_types + attack_type
    else:
        action_id = server_id * (game_config.num_attack_types+1) + attack_type
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
    if not game_config.reconnaissance_actions:
        attack_defense_type = action % game_config.num_attack_types
    else:
        attack_defense_type = action % (game_config.num_attack_types+1)
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


def get_img_from_fig(fig, dpi=180):
    """
    Convert matplotlib fig to numpy array

    :param fig: fig to convert
    :param dpi: dpi of conversion
    :return: np array of the image
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img

def action_dist_hist(data: np.ndarray,
                           title: str = "Test", xlabel: str = "test", ylabel: str = "test",
                           file_name: str = "test.eps", xlims: Union[float, float] = None) -> np.ndarray:
    """
    Plot a distribution of the policy

    :param data: the data to plot
    :param title: title of the plot
    :param xlabel: xlabel
    :param ylabel: ylabel
    :param file_name: path where to save file
    :param xlims: xlimits (optional)
    :return: numpy array of the figure
    """
    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{amsfonts}')
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 3))
    if xlims is None:
        xlims = (min(data),
                 max(data))

    sns.distplot(data, kde=True,
              color = 'darkblue',
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 0.5}, bins=xlims[1], fit=None)

    ax.set_xlim(xlims)
    ax.set_xticks(list(range(xlims[1]+1)))

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # set the grid on
    ax.grid('on')

    # tweak the axis labels
    xlab = ax.xaxis.get_label()
    ylab = ax.yaxis.get_label()
    xlab.set_size(10)
    ylab.set_size(10)

    # change the color of the top and right spines to opaque gray
    ax.spines['right'].set_color((.8, .8, .8))
    ax.spines['top'].set_color((.8, .8, .8))

    fig.tight_layout()
    fig.savefig(file_name + ".png", format="png", dpi=600)
    fig.savefig(file_name + ".pdf", format='pdf', dpi=600, bbox_inches='tight', transparent=True)
    data = get_img_from_fig(fig, dpi=100)
    plt.close(fig)
    return data


def defense_score(game_sate : GameState, network_config : NetworkConfig, game_config : GameConfig):
    total_min_def = 0
    for row in range(network_config.num_rows):
        min_def = float("inf")
        for col in range(network_config.num_cols):
            node_id = network_config.get_node_id((row, col))
            if (game_config.network_config.node_list[node_id] == NodeType.SERVER.value
                    or game_config.network_config.node_list[node_id] == NodeType.DATA.value):
                d = np.min(game_sate.defense_values[node_id])
                if d < min_def:
                    min_def = d
                #print("row:{}, col:{}, node_id:{}, type:{}, min_def:{}".format(row, col, node_id, game_config.network_config.node_list[node_id], min_def))
        total_min_def = total_min_def + min_def
        if min_def < game_config.max_value:
            return total_min_def
    return total_min_def
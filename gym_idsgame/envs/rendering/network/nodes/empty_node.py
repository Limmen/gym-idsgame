"""
Represents an empty node in the gym-idsgame environment
"""
from typing import Union
from gym_idsgame.envs.rendering.network.nodes.node import Node
from gym_idsgame.envs.dao.idsgame_config import IdsGameConfig
from gym_idsgame.envs.dao.node_type import NodeType

class EmptyNode(Node):
    """
    Represents an empty node in the grid-network
    """

    def __init__(self, idsgame_config: IdsGameConfig, row: int, col: int, id: int):
        """
        Class constructor, initializes the node

        :param idsgame_config: configuration for the IdsGameEnv
        :param row: the row in the grid
        :param col: the column in the grid
        :param id: the id of the node
        """
        super(EmptyNode, self).__init__()
        self.idsgame_config = idsgame_config
        self.row = row
        self.col = col
        self._id = id

    @property
    def node_type(self) -> NodeType:
        """
        :return: the node type
        """
        return NodeType.EMPTY

    @property
    def id(self) -> int:
        """
        :return: the id of the node
        """
        return self._id

    # --- Inherited methods----
    # empty nodes cannot be attacked or defended so simply do nothing when they are called

    def manual_blink_defense(self, i):
        pass

    def manual_blink_attack(self, i, attacker_pos: Union[int, int], edges=None):
        pass

    def set_state(self, attack_values, defense_values, det_value):
        pass

    def visualize_defense(self, detect):
        pass

    def reset(self):
        pass

    def add_in_edge(self, edge):
        pass

    def add_out_edge(self, edge):
        pass

    def add_horizontal_edge(self, edge):
        pass

    def get_link_coords(self, upper=True, lower=False):
        pass

    def get_coords(self):
        pass

    def get_node(self):
        pass

    def unschedule(self):
        pass

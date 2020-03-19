from gym_idsgame.envs.policy_baselines.policy import Policy
from gym_idsgame.envs.dao.game_state import GameState
from gym_idsgame.envs.dao.game_config import GameConfig
from gym_idsgame.envs.constants import constants
from gym_idsgame.envs.dao.node_type import NodeType
from typing import Union
import numpy as np

class RandomDefense(Policy):

    def __init__(self):
        super(RandomDefense, self).__init__()


    def action(self, game_state: GameState, game_config: GameConfig) -> Union[int,int,int]:
        defend_type = np.random.randint(len(self.defense_values))
        defend_row = None
        defend_col = None
        for row in game_config.graph_layout.shape[0]:
            for col in game_config.graph_layout.shape[0]:
                if (game_config.graph_layout[row][col] == NodeType.SERVER
                        or game_config.graph_layout[row][col] == NodeType.DATA):
                    if defend_row == None or defend_col == None:
                        defend_row, defend_col = row, col
                    else:
                        if np.random.rand() >= 0.5:
                            defend_row, defend_col = row, col
        if defend_row == None or defend_col == None:
            raise AssertionError("Invalid Network Config, could not find any node to defend")
        return defend_row, defend_col, defend_type

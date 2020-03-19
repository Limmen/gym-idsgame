from gym_idsgame.envs.policy_baselines.policy import Policy
from gym_idsgame.envs.dao.render_state import RenderState
from gym_idsgame.envs.dao.game_config import GameConfig
from gym_idsgame.envs.dao.node_type import NodeType
from typing import Union
import numpy as np

class RandomDefense(Policy):

    def __init__(self, game_config: GameConfig):
        super(RandomDefense, self).__init__(game_config)


    def action(self, render_state: RenderState) -> Union[int, int, int]:
        defend_type = np.random.randint(len(render_state.defense_values))
        defend_row = None
        defend_col = None
        for row in range(self.game_config.network_config.graph_layout.shape[0]):
            for col in range(self.game_config.network_config.graph_layout.shape[1]):
                if (self.game_config.network_config.graph_layout[row][col] == NodeType.SERVER.value
                        or self.game_config.network_config.graph_layout[row][col] == NodeType.DATA.value):
                    if defend_row is None or defend_col is None:
                        defend_row, defend_col = row, col
                    else:
                        if np.random.rand() >= 0.5:
                            defend_row, defend_col = row, col
        if defend_row is None or defend_col is None:
            raise AssertionError("Invalid Network Config, could not find any node to defend")
        return defend_row, defend_col, defend_type

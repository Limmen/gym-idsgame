"""
Configuration for the gym-idsgame environment
"""
from gym_idsgame.envs.dao.render_config import RenderConfig
from gym_idsgame.envs.dao.game_config import GameConfig
from gym_idsgame.envs.policy_baselines.policy import Policy
from gym_idsgame.envs.policy_baselines.random_defense import RandomDefense

class IdsGameConfig:
    """
    DTO representing the configuration of the IdsGameEnv:
    """

    def __init__(self, render_config: RenderConfig = None, game_config: GameConfig = None,
                 defender_policy: Policy = None):
        """
        Constructor, initializes the config

        :param render_config: render config, e.g colors, size, line width etc.
        :param game_config: game configuration, e.g. number of nodes
        :param defender_policy: the defender policy
        """
        self.render_config = render_config
        self.game_config = game_config
        self.defender_policy = defender_policy
        if self.render_config is None:
            self.render_config = RenderConfig()
        if self.game_config is None:
            self.game_config = GameConfig()
        if self.defender_policy is None:
            self.defender_policy = RandomDefense(game_config)
        self.render_config.set_height(self.game_config.num_rows)
        self.render_config.set_width(self.game_config.num_cols)

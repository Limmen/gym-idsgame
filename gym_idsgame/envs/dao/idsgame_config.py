"""
Configuration for the gym-idsgame environment
"""
from gym_idsgame.envs.dao.render_config import RenderConfig
from gym_idsgame.envs.dao.game_config import GameConfig
from gym_idsgame.agents.agent import Agent

class IdsGameConfig:
    """
    DTO representing the configuration of the IdsGameEnv:
    """

    def __init__(self, render_config: RenderConfig = None, game_config: GameConfig = None,
                 defender_agent: Agent = None, attacker_agent: Agent = None):
        """
        Constructor, initializes the config

        :param render_config: render config, e.g colors, size, line width etc.
        :param game_config: game configuration, e.g. number of nodes
        :param defender_agent: the defender agent
        :param attacker_agent: the attacker agent
        """
        self.render_config = render_config
        self.game_config = game_config
        self.defender_agent = defender_agent
        self.attacker_agent = attacker_agent
        if self.render_config is None:
            self.render_config = RenderConfig()
        if self.game_config is None:
            self.game_config = GameConfig()
        self.render_config.set_height(self.game_config.num_rows)
        self.render_config.set_width(self.game_config.num_cols)

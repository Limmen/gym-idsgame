"""
Manual attacker agent
"""
from gym_idsgame.agents.agent import Agent
from gym_idsgame.envs.dao.idsgame_config import IdsGameConfig
from gym_idsgame.envs.rendering.viewer import Viewer

class ManualAttackAgent(Agent):
    """
    Class representing a manual attacker agent, controlled in the GUI by keyboard and mouse
    """

    def __init__(self, idsgame_config: IdsGameConfig):
        """
        Sets up the GUI with the manual attacker

        :param idsgame_config: the configuration
        """
        super(ManualAttackAgent, self).__init__(idsgame_config.game_config)
        self.idsgame_config = idsgame_config
        self.idsgame_config.render_config.manual_default()
        viewer = Viewer(self.idsgame_config)
        viewer.manual_start_attacker()

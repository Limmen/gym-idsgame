from gym_idsgame.agents.agent import Agent
from gym_idsgame.envs.dao.idsgame_config import IdsGameConfig
from gym_idsgame.envs.rendering.viewer import Viewer

class ManualDefenseAgent(Agent):

    def __init__(self, idsgame_config: IdsGameConfig):
        super(ManualDefenseAgent, self).__init__(idsgame_config.game_config)
        self.idsgame_config = idsgame_config
        self.idsgame_config.render_config.manual_default()
        viewer = Viewer(self.idsgame_config)
        viewer.manual_start_defender()

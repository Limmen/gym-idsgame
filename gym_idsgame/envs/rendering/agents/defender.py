from gym_idsgame.envs.rendering.agents.agent import Agent
from gym_idsgame.envs.dao.render_config import RenderConfig

class Defender(Agent):

    def __init__(self, render_config: RenderConfig):
        super(Agent, self).__init__()
        self.render_config = render_config
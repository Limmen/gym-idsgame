from gym_idsgame.agents.agent import Agent

class SimulationConfig:

    def __init__(self, attacker_agent: Agent = None, defender_agent: Agent = None, num_episodes: int = 10, video_fps=5,
                 video=False, gif_dir=None, video_dir=None, gifs=False, render=False, eval_sleep=0.35,
                 log_frequency = 1):
        self.attacker_agent = attacker_agent
        self.defender_agent = defender_agent
        self.num_episodes = num_episodes
        self.video_fps = video_fps
        self.video = video
        self.gif_dir = gif_dir
        self.video_dir = video_dir
        self.gifs = gifs
        self.render = render
        self.eval_sleep = eval_sleep
        self.log_frequency = log_frequency
        self.logger = None
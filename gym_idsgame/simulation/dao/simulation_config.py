
class SimulationConfig:

    def __init__(self, num_episodes: int = 10, video_fps=5,
                 video=False, gif_dir=None, video_dir=None, gifs=False, render=False, sleep=0.35,
                 log_frequency = 1, video_frequency = 100):
        self.num_episodes = num_episodes
        self.video_fps = video_fps
        self.video = video
        self.gif_dir = gif_dir
        self.video_dir = video_dir
        self.gifs = gifs
        self.render = render
        self.sleep = sleep
        self.log_frequency = log_frequency
        self.logger = None
        self.video_frequency = video_frequency
"""
Configuratin for QAgent
"""
import csv
import logging


class QAgentConfig:
    """
    DTO with configuration for QAgent
    """

    def __init__(self, gamma=0.8, alpha=0.1, epsilon=0.9, render=False, eval_sleep=0.35,
                 epsilon_decay=0.999, min_epsilon=0.1, eval_episodes=1, train_log_frequency=100,
                 eval_log_frequency=1, video=False, video_fps=5, video_dir=None, num_episodes=5000,
                 eval_render=False, gifs=False, gif_dir=None, eval_frequency=1000, video_frequency = 1):
        """
        Initialize environment and hyperparameters

        :param gamma: the discount factor
        :param alpha: the learning rate
        :param epsilon: the exploration rate
        :param render: whether to render the environment *during training*
        :param eval_sleep: amount of sleep between time-steps during evaluation and rendering
        :param epsilon_decay: rate of decay of epsilon
        :param min_epsilon: minimum epsilon rate
        :param eval_episodes: number of evaluation episodes
        :param train_log_frequency: number of episodes between logs during train
        :param eval_log_frequency: number of episodes between logs during eval
        :param video: boolean flag whether to record video of the evaluation.
        :param video_dir: path where to save videos (will overwrite)
        :param gif_dir: path where to save gifs (will overwrite)
        :param num_episodes: number of training epochs
        :param eval_render: whether to render the game during evaluation or not
                            (perhaps set to False if video is recorded instead)
        :param gifs: boolean flag whether to save gifs during evaluation or not
        :param eval_frequency: the frequency (episodes) when running evaluation
        :param video_frequency: the frequency (eval episodes) to record video and gif
        """
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.render = render
        self.eval_sleep = eval_sleep
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.eval_episodes = eval_episodes
        self.train_log_frequency = train_log_frequency
        self.eval_log_frequency = eval_log_frequency
        self.video = video
        self.video_fps = video_fps
        self.video_dir = video_dir
        self.num_episodes = num_episodes
        self.eval_render = eval_render
        self.gifs = gifs
        self.gif_dir = gif_dir
        self.eval_frequency = eval_frequency
        self.logger = None
        self.video_frequency = video_frequency

    def to_str(self) -> str:
        """
        :return: a string with information about all of the parameters
        """
        return "Hyperparameters: gamma:{0},alpha:{1},epsilon:{2},render:{3},eval_sleep:{4}," \
               "epsilon_decay:{5},min_epsilon:{6},eval_episodes:{7},train_log_frequency:{8}," \
               "eval_log_frequency:{9},video:{10},video_fps:{11}," \
               "video_dir:{12},num_episodes:{13},eval_render:{14},gifs:{15}," \
               "gifdir:{16},eval_frequency:{17},video_frequency:{18}".format(
            self.gamma, self.alpha, self.epsilon, self.render, self.eval_sleep, self.epsilon_decay,
            self.min_epsilon, self.eval_episodes, self.train_log_frequency, self.eval_log_frequency, self.video,
            self.video_fps, self.video_dir, self.num_episodes, self.eval_render, self.gifs, self.gif_dir,
            self.eval_frequency, self.video_frequency)

    def to_csv(self, file_path: str) -> None:
        """
        Write parameters to csv file

        :param file_path: path to the file
        :return: None
        """
        with open(file_path, "w") as f:
            writer = csv.writer(f)
            writer.writerow(["parameter", "value"])
            writer.writerow(["gamma", str(self.gamma)])
            writer.writerow(["alpha", str(self.alpha)])
            writer.writerow(["epsilon", str(self.epsilon)])
            writer.writerow(["render", str(self.render)])
            writer.writerow(["eval_sleep", str(self.eval_sleep)])
            writer.writerow(["epsilon_decay", str(self.epsilon_decay)])
            writer.writerow(["min_epsilon", str(self.min_epsilon)])
            writer.writerow(["eval_episodes", str(self.eval_episodes)])
            writer.writerow(["train_log_frequency", str(self.train_log_frequency)])
            writer.writerow(["eval_log_frequency", str(self.eval_log_frequency)])
            writer.writerow(["video", str(self.video)])
            writer.writerow(["video_fps", str(self.video_fps)])
            writer.writerow(["video_dir", str(self.video_dir)])
            writer.writerow(["num_episodes", str(self.num_episodes)])
            writer.writerow(["eval_render", str(self.eval_render)])
            writer.writerow(["gifs", str(self.gifs)])
            writer.writerow(["gifdir", str(self.gif_dir)])
            writer.writerow(["eval_frequency", str(self.eval_frequency)])
            writer.writerow(["video_frequency", str(self.video_frequency)])

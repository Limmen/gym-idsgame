class QAgentConfig:

    def __init__(self, gamma=0.8, alpha=0.1, epsilon=0.9, render = False, eval_sleep = 0.35,
                 epsilon_decay = 0.999, min_epsilon = 0.1, eval_episodes = 1, log_frequency = 100,
                 video = False, video_fps=5, video_dir = None, num_episodes = 5000):
        """
        Initalize environment and hyperparameters

        :param gamma: the discount factor
        :param alpha: the learning rate
        :param epsilon: the exploration rate
        :param render: whether to render the environment *during training* (it will always render at evaluation)
        :param eval_sleep: amount of sleep between time-steps during evaluation and rendering
        :param epsilon_decay: rate of decay of epsilon
        :param min_epsilon: minimum epsilon rate
        :param eval_episodes: number of evaluation episodes
        :param log_frequency: number of episodes between logs
        :param video: boolean flag whether to record video of the evaluation.
        :param video_dir: path where to save videos (will overwrite)
        :param num_episodes: number of training epochs
        """
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.render = render
        self.eval_sleep = eval_sleep
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.eval_episodes = eval_episodes
        self.log_frequency = log_frequency
        self.video = video
        self.video_fps = video_fps
        self.video_dir = video_dir
        self.num_episodes = num_episodes
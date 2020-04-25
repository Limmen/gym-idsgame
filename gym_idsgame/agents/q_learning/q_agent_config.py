"""
Configuratin for QAgent
"""
import csv
from gym_idsgame.agents.q_learning.dqn.dqn_config import DQNConfig

class QAgentConfig:
    """
    DTO with configuration for QAgent
    """

    def __init__(self, gamma :float = 0.8, alpha:float = 0.1, epsilon :float =0.9, render :bool =False,
                 eval_sleep :float = 0.35,
                 epsilon_decay :float = 0.999, min_epsilon :float = 0.1, eval_episodes :int = 1,
                 train_log_frequency :int =100,
                 eval_log_frequency :int =1, video :bool = False, video_fps :int = 5, video_dir :bool = None,
                 num_episodes :int = 5000,
                 eval_render :bool = False, gifs :bool = False, gif_dir: str = None, eval_frequency :int =1000,
                 video_frequency :int = 101, attacker :bool = True, defender :bool = False,
                 save_dir :str = None, attacker_load_path : str = None, defender_load_path : str = None,
                 dqn_config: DQNConfig = None,
                 checkpoint_freq : int = 100000, random_seed: int = 0):
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
        :param attacker: True if the QAgent is an attacker
        :param attacker: True if the QAgent is a defender
        :param save_dir: dir to save Q-table
        :param attacker_load_path: path to load a saved Q-table of the attacker
        :param defender_load_path: path to load a saved Q-table of the defender
        :param dqn_config: configuration for DQN
        :param checkpoint_freq: frequency of checkpointing the model (episodes)
        :param random_seed: the random seed for reproducibility
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
        self.attacker = attacker
        self.defender = defender
        self.save_dir = save_dir
        self.attacker_load_path = attacker_load_path
        self.defender_load_path = defender_load_path
        self.dqn_config = dqn_config
        self.checkpoint_freq = checkpoint_freq
        self.random_seed = random_seed

    def to_str(self) -> str:
        """
        :return: a string with information about all of the parameters
        """
        return "Hyperparameters: gamma:{0},alpha:{1},epsilon:{2},render:{3},eval_sleep:{4}," \
               "epsilon_decay:{5},min_epsilon:{6},eval_episodes:{7},train_log_frequency:{8}," \
               "eval_log_frequency:{9},video:{10},video_fps:{11}," \
               "video_dir:{12},num_episodes:{13},eval_render:{14},gifs:{15}," \
               "gifdir:{16},eval_frequency:{17},video_frequency:{18},attacker{19},defender:{20}," \
               "checkpoint_freq:{21},random_seed:{22}".format(
            self.gamma, self.alpha, self.epsilon, self.render, self.eval_sleep, self.epsilon_decay,
            self.min_epsilon, self.eval_episodes, self.train_log_frequency, self.eval_log_frequency, self.video,
            self.video_fps, self.video_dir, self.num_episodes, self.eval_render, self.gifs, self.gif_dir,
            self.eval_frequency, self.video_frequency, self.attacker, self.defender, self.checkpoint_freq,
            self.random_seed)

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
            writer.writerow(["attacker", str(self.attacker)])
            writer.writerow(["defender", str(self.defender)])
            writer.writerow(["checkpoint_freq", str(self.checkpoint_freq)])
            writer.writerow(["random_seed", str(self.random_seed)])
            if self.dqn_config is not None:
                writer.writerow(["input_dim", str(self.dqn_config.input_dim)])
                writer.writerow(["output_dim", str(self.dqn_config.output_dim)])
                writer.writerow(["hidden_dim", str(self.dqn_config.hidden_dim)])
                writer.writerow(["replay_memory_size", str(self.dqn_config.replay_memory_size)])
                writer.writerow(["replay_start_size", str(self.dqn_config.replay_start_size)])
                writer.writerow(["batch_size", str(self.dqn_config.batch_size)])
                writer.writerow(["target_network_update_freq", str(self.dqn_config.target_network_update_freq)])
                writer.writerow(["gpu", str(self.dqn_config.gpu)])
                writer.writerow(["tensorboard", str(self.dqn_config.tensorboard)])
                writer.writerow(["tensorboard_dir", str(self.dqn_config.tensorboard_dir)])
                writer.writerow(["loss_fn", str(self.dqn_config.loss_fn)])
                writer.writerow(["optimizer", str(self.dqn_config.optimizer)])
                writer.writerow(["num_hidden_layers", str(self.dqn_config.num_hidden_layers)])
                writer.writerow(["lr_exp_decay", str(self.dqn_config.lr_exp_decay)])
                writer.writerow(["lr_decay_rate", str(self.dqn_config.lr_decay_rate)])
                writer.writerow(["hidden_activation", str(self.dqn_config.hidden_activation)])

    def hparams_dict(self):
        hparams = {}
        hparams["gamma"] = self.gamma
        hparams["alpha"] = self.alpha
        hparams["epsilon"] = self.epsilon
        hparams["epsilon_decay"] = self.epsilon_decay
        hparams["min_epsilon"] = self.min_epsilon
        hparams["eval_episodes"] = self.eval_episodes
        hparams["train_log_frequency"] = self.train_log_frequency
        hparams["eval_log_frequency"] = self.eval_log_frequency
        hparams["num_episodes"] = self.num_episodes
        hparams["eval_frequency"] = self.eval_frequency
        hparams["attacker"] = self.attacker
        hparams["defender"] = self.defender
        hparams["checkpoint_freq"] = self.checkpoint_freq
        hparams["random_seed"] = self.random_seed
        if self.dqn_config is not None:
            hparams["input_dim"] = self.dqn_config.input_dim
            hparams["output_dim"] = self.dqn_config.output_dim
            hparams["hidden_dim"] = self.dqn_config.hidden_dim
            hparams["replay_memory_size"] = self.dqn_config.replay_memory_size
            hparams["replay_start_size"] = self.dqn_config.replay_start_size
            hparams["batch_size"] = self.dqn_config.batch_size
            hparams["num_hidden_layers"] = self.dqn_config.num_hidden_layers
            hparams["target_network_update_freq"] = self.dqn_config.target_network_update_freq
            hparams["gpu"] = self.dqn_config.gpu
            hparams["loss_fn"] = self.dqn_config.loss_fn
            hparams["optimizer"] = self.dqn_config.optimizer
            hparams["lr_exp_decay"] = self.dqn_config.lr_exp_decay
            hparams["lr_decay_rate"] = self.dqn_config.lr_decay_rate
            hparams["hidden_activation"] = self.dqn_config.hidden_activation
        return hparams

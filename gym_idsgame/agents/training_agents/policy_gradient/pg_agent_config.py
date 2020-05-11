"""
Configuration for Policy gradient agents
"""
import csv
from gym_idsgame.agents.training_agents.common.opponent_pool_config import OpponentPoolConfig

class PolicyGradientAgentConfig:
    """
    DTO with configuration for PolicyGradientAgent
    """

    def __init__(self, gamma :float = 0.8, alpha_attacker:float = 0.1, alpha_defender:float = 0.1,
                 epsilon :float =0.9, render :bool =False,
                 eval_sleep :float = 0.35,
                 epsilon_decay :float = 0.999, min_epsilon :float = 0.1, eval_episodes :int = 1,
                 train_log_frequency :int =100,
                 eval_log_frequency :int =1, video :bool = False, video_fps :int = 5, video_dir :bool = None,
                 num_episodes :int = 5000,
                 eval_render :bool = False, gifs :bool = False, gif_dir: str = None, eval_frequency :int =1000,
                 video_frequency :int = 101, attacker :bool = True, defender :bool = False,
                 save_dir :str = None, attacker_load_path : str = None, defender_load_path : str = None,
                 checkpoint_freq : int = 100000, random_seed: int = 0, eval_epsilon : float = 0.0,
                 input_dim: int = 30, output_dim_attacker: int = 30, output_dim_defender: int = 33,
                 hidden_dim: int = 64,
                 batch_size: int = 64, num_hidden_layers=2,
                 gpu: bool = False, tensorboard: bool = False, tensorboard_dir: str = "",
                 optimizer: str = "Adam", lr_exp_decay: bool = False,
                 lr_decay_rate: float = 0.96, hidden_activation: str = "ReLU", clip_gradient = False,
                 max_gradient_norm = 40, critic_loss_fn : str = "MSE", state_length = 1,
                 alternating_optimization : bool = False, alternating_period : int = 15000,
                 opponent_pool : bool = False, opponent_pool_config : OpponentPoolConfig = None,
                 normalize_features : bool = False, gpu_id: int = 0
                 ):
        """
        Initialize environment and hyperparameters

        :param gamma: the discount factor
        :param alpha_attacker: the learning rate of the attacker
        :param alpha_defender: the learning rate of the defender
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
        :param checkpoint_freq: frequency of checkpointing the model (episodes)
        :param random_seed: the random seed for reproducibility
        :param eval_epsilon: evaluation epsilon for implementing a "soft policy" rather than a "greedy policy"
        :param input_dim: input dimension of the policy network
        :param output_dim_attacker: output dimensions of the policy network of the attacker
        :param output_dim_defender: output dimensions of the policy network of the defender
        :param hidden_dim: hidden dimension of the policy network
        :param num_hidden_layers: number of hidden layers
        :param batch_size: the batch size during training
        :param gpu: boolean flag whether using GPU or not
        :param tensorboard: boolean flag whether using tensorboard logging or not
        :param tensorboard_dir: tensorboard logdir
        :param optimizer: optimizer
        :param lr_exp_decay: whether to use exponential decay of learning rate or not
        :param lr_decay_rate: decay rate of lr
        :param hidden_activation: the activation function for hidden units
        :param clip_gradient: boolean flag whether to clip gradient or not
        :param max_gradient_norm: max norm of gradient before clipping it
        :param critic_loss_fn: loss function for the critic
        :param state_length: length of observations to use for approximative Markov state
        :param alternating_optimization: boolean flag whether using alteranting optimization or not
        :param alternating_period: period for alternating between training attacker and defender
        :param opponent_pool: boolean flag whether using opponent pool or not
        :param opponent_pool_config: DTO with config when training against opponent pool
        :param normalize_features: boolean flag that indicates whether features should be normalized or not
        :param gpu_id: id of the GPU to use
        """
        self.gamma = gamma
        self.alpha_attacker = alpha_attacker
        self.alpha_defender = alpha_defender
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
        self.checkpoint_freq = checkpoint_freq
        self.random_seed = random_seed
        self.eval_epsilon = eval_epsilon
        self.input_dim = input_dim
        self.output_dim_attacker = output_dim_attacker
        self.output_dim_defender = output_dim_defender
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_hidden_layers = num_hidden_layers
        self.gpu = gpu
        self.tensorboard = tensorboard
        self.tensorboard_dir = tensorboard_dir
        self.optimizer = optimizer
        self.lr_exp_decay = lr_exp_decay
        self.lr_decay_rate = lr_decay_rate
        self.hidden_activation = hidden_activation
        self.clip_gradient = clip_gradient
        self.max_gradient_norm = max_gradient_norm
        self.critic_loss_fn = critic_loss_fn
        self.state_length = state_length
        self.alternating_optimization = alternating_optimization
        self.alternating_period = alternating_period
        self.opponent_pool = opponent_pool
        self.opponent_pool_config = opponent_pool_config
        self.normalize_features = normalize_features
        self.gpu_id = gpu_id


    def to_str(self) -> str:
        """
        :return: a string with information about all of the parameters
        """
        return "Hyperparameters: gamma:{0},alpha_attacker:{1},epsilon:{2},render:{3},eval_sleep:{4}," \
               "epsilon_decay:{5},min_epsilon:{6},eval_episodes:{7},train_log_frequency:{8}," \
               "eval_log_frequency:{9},video:{10},video_fps:{11}," \
               "video_dir:{12},num_episodes:{13},eval_render:{14},gifs:{15}," \
               "gifdir:{16},eval_frequency:{17},video_frequency:{18},attacker{19},defender:{20}," \
               "checkpoint_freq:{21},random_seed:{22},eval_epsilon:{23},clip_gradient:{24},max_gradient_norm:{25}," \
               "output_dim_defender:{26},critic_loss_fn:{27},state_length:{28},alternating_optimization:{29}," \
               "alternating_period:{30},normalize_features:{31},alpha_defender:{32},gpu_id:{33}".format(
            self.gamma, self.alpha_attacker, self.epsilon, self.render, self.eval_sleep, self.epsilon_decay,
            self.min_epsilon, self.eval_episodes, self.train_log_frequency, self.eval_log_frequency, self.video,
            self.video_fps, self.video_dir, self.num_episodes, self.eval_render, self.gifs, self.gif_dir,
            self.eval_frequency, self.video_frequency, self.attacker, self.defender, self.checkpoint_freq,
            self.random_seed, self.eval_epsilon, self.clip_gradient, self.max_gradient_norm, self.output_dim_defender,
            self.critic_loss_fn, self.state_length, self.alternating_optimization, self.alternating_period,
            self.normalize_features, self.alpha_defender, self.gpu_id)

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
            writer.writerow(["alpha_attacker", str(self.alpha_attacker)])
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
            writer.writerow(["eval_epsilon", str(self.eval_epsilon)])
            writer.writerow(["input_dim", str(self.input_dim)])
            writer.writerow(["output_dim_attacker", str(self.output_dim_attacker)])
            writer.writerow(["hidden_dim", str(self.hidden_dim)])
            writer.writerow(["batch_size", str(self.batch_size)])
            writer.writerow(["gpu", str(self.gpu)])
            writer.writerow(["tensorboard", str(self.tensorboard)])
            writer.writerow(["tensorboard_dir", str(self.tensorboard_dir)])
            writer.writerow(["optimizer", str(self.optimizer)])
            writer.writerow(["num_hidden_layers", str(self.num_hidden_layers)])
            writer.writerow(["lr_exp_decay", str(self.lr_exp_decay)])
            writer.writerow(["lr_decay_rate", str(self.lr_decay_rate)])
            writer.writerow(["hidden_activation", str(self.hidden_activation)])
            writer.writerow(["clip_gradient", str(self.clip_gradient)])
            writer.writerow(["max_gradient_norm", str(self.max_gradient_norm)])
            writer.writerow(["output_dim_defender", str(self.output_dim_defender)])
            writer.writerow(["critic_loss_fn", str(self.critic_loss_fn)])
            writer.writerow(["state_length", str(self.state_length)])
            writer.writerow(["alternating_optimization", str(self.alternating_optimization)])
            writer.writerow(["alternating_period", str(self.alternating_period)])
            writer.writerow(["normalize_features", str(self.normalize_features)])
            writer.writerow(["alpha_defender", str(self.normalize_features)])
            writer.writerow(["gpu_id", str(self.gpu_id)])
            if self.opponent_pool and self.opponent_pool_config is not None:
                writer.writerow(["pool_maxsize", str(self.opponent_pool_config.pool_maxsize)])
                writer.writerow(["pool_increment_period", str(self.opponent_pool_config.pool_increment_period)])
                writer.writerow(["head_to_head_period", str(self.opponent_pool_config.head_to_head_period)])
                writer.writerow(["quality_scores", str(self.opponent_pool_config.quality_scores)])
                writer.writerow(["quality_score_eta", str(self.opponent_pool_config.quality_score_eta)])
                writer.writerow(["pool_prob", str(self.opponent_pool_config.pool_prob)])
                writer.writerow(["initial_quality", str(self.opponent_pool_config.initial_quality)])


    def hparams_dict(self):
        hparams = {}
        hparams["gamma"] = self.gamma
        hparams["alpha_attacker"] = self.alpha_attacker
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
        hparams["eval_epsilon"] = self.eval_epsilon
        hparams["input_dim"] = self.input_dim
        hparams["output_dim_attacker"] = self.output_dim_attacker
        hparams["hidden_dim"] = self.hidden_dim
        hparams["batch_size"] = self.batch_size
        hparams["num_hidden_layers"] = self.num_hidden_layers
        hparams["gpu"] = self.gpu
        hparams["optimizer"] = self.optimizer
        hparams["lr_exp_decay"] = self.lr_exp_decay
        hparams["lr_decay_rate"] = self.lr_decay_rate
        hparams["hidden_activation"] = self.hidden_activation
        hparams["clip_gradient"] = self.clip_gradient
        hparams["max_gradient_norm"] = self.max_gradient_norm
        hparams["output_dim_defender"] = self.output_dim_defender
        hparams["critic_loss_fn"] = self.critic_loss_fn
        hparams["state_length"] = self.state_length
        hparams["alternating_optimization"] = self.alternating_optimization
        hparams["alternating_period"] = self.alternating_period
        hparams["normalize_features"] = self.normalize_features
        hparams["alpha_defender"] = self.alpha_defender
        hparams["gpu_id"] = self.gpu_id
        if self.opponent_pool and self.opponent_pool_config is not None:
            hparams["pool_maxsize"] = self.opponent_pool_config.pool_maxsize
            hparams["pool_increment_period"] = self.opponent_pool_config.pool_increment_period
            hparams["head_to_head_period"] = self.opponent_pool_config.head_to_head_period
            hparams["quality_scores"] = self.opponent_pool_config.quality_scores
            hparams["quality_score_eta"] = self.opponent_pool_config.quality_score_eta
            hparams["pool_prob"] = self.opponent_pool_config.pool_prob
            hparams["initial_quality"] = self.opponent_pool_config.initial_quality
        return hparams

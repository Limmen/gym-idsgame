"""
DTO  class holding config parameters for DQN training
"""

import csv

class DQNConfig:
    """
    Configuration parameters for DQN
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 64, replay_memory_size: int = 100000,
                 replay_start_size : int = 10000, batch_size: int = 64, num_hidden_layers = 2,
                 target_network_update_freq : int = 10,
                 gpu : bool = False, tensorboard : bool = False, tensorboard_dir: str = "",
                 loss_fn : str = "MSE", optimizer : str = "Adam", lr_exp_decay : bool = False,
                 lr_decay_rate : float = 0.96):
        """
        Initializes the config

        :param input_dim: input dimension of the DQN networks
        :param output_dim: output dimensions of the DQN networks
        :param hidden_dim: hidden dimension of the DQN networks
        :param num_hidden_layers: number of hidden layers
        :param replay_memory_size: replay memory size
        :param replay_start_size: start size of the replay memory (populated with warmup)
        :param batch_size: the batch size during training
        :param target_network_update_freq: the frequency (in episodes) of updating the target network
        :param gpu: boolean flag whether using GPU or not
        :param tensorboard: boolean flag whether using tensorboard logging or not
        :param tensorboard_dir: tensorboard logdir
        :param loss_fn: loss function
        :param optimizer: optimizer
        :param lr_exp_decay: whether to use exponential decay of learning rate or not
        :param lr_decay_rate: decay rate of lr
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.replay_memory_size = replay_memory_size
        self.replay_start_size = replay_start_size
        self.batch_size = batch_size
        self.target_network_update_freq = target_network_update_freq
        self.gpu = gpu
        self.tensorboard = tensorboard
        self.tensorboard_dir = tensorboard_dir
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.num_hidden_layers = num_hidden_layers
        self.lr_exp_decay = lr_exp_decay
        self.lr_decay_rate = lr_decay_rate

    def to_str(self) -> str:
        """
        :return: a string with information about all of the parameters
        """
        return "DQN Hyperparameters: input_dim:{0},output_dim:{1},hidden_dim:{2},replay_memory_size:{3}," \
               "replay_start_size:{4}," \
               "batch_size:{5},target_network_update_freq:{6},gpu:{7},tensorboard:{8}," \
               "tensorboard_dir:{9},loss_fn:{10},optimizer:{11},num_hidden_layers:{12}," \
               "lr_exp_decay:{13},lr_decay_rate:{14}".format(
            self.input_dim, self.output_dim, self.hidden_dim, self.replay_memory_size,
            self.replay_start_size, self.batch_size, self.target_network_update_freq,
            self.batch_size, self.target_network_update_freq, self.gpu, self.tensorboard, self.tensorboard_dir,
            self.loss_fn, self.optimizer, self.num_hidden_layers, self.lr_exp_decay, self.lr_decay_rate)

    def to_csv(self, file_path: str) -> None:
        """
        Write parameters to csv file

        :param file_path: path to the file
        :return: None
        """
        with open(file_path, "w") as f:
            writer = csv.writer(f)
            writer.writerow(["parameter", "value"])
            writer.writerow(["input_dim", str(self.input_dim)])
            writer.writerow(["output_dim", str(self.output_dim)])
            writer.writerow(["hidden_dim", str(self.hidden_dim)])
            writer.writerow(["replay_memory_size", str(self.replay_memory_size)])
            writer.writerow(["replay_start_size", str(self.replay_start_size)])
            writer.writerow(["batch_size", str(self.batch_size)])
            writer.writerow(["target_network_update_freq", str(self.target_network_update_freq)])
            writer.writerow(["gpu", str(self.gpu)])
            writer.writerow(["tensorboard", str(self.tensorboard)])
            writer.writerow(["tensorboard_dir", str(self.tensorboard_dir)])
            writer.writerow(["loss_fn", str(self.loss_fn)])
            writer.writerow(["optimizer", str(self.optimizer)])
            writer.writerow(["num_hidden_layers", str(self.num_hidden_layers)])
            writer.writerow(["lr_exp_decay", str(self.lr_exp_decay)])
            writer.writerow(["lr_decay_rate", str(self.lr_decay_rate)])
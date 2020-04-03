
class DQNConfig:
    """
    Configuration parameters for DQN
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 64, replay_memory_size: int = 100000,
                 replay_start_size : int = 10000, batch_size: int = 64, target_network_update_freq : int = 10):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.replay_memory_size = replay_memory_size
        self.replay_start_size = replay_start_size
        self.batch_size = batch_size
        self.target_network_update_freq = target_network_update_freq
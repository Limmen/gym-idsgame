
class HpTuningConfig:
    """
    Config for hparam tuning
    """

    def __init__(self, param_1 : str, param_2 : str, alpha : list = None, epsilon_decay : list = None,
                 num_hidden_layers : list = None):
        self.param_1 = param_1
        self.param_2 = param_2
        self.alpha = alpha
        self.epsilon_decay = epsilon_decay
        self.num_hidden_layers = num_hidden_layers
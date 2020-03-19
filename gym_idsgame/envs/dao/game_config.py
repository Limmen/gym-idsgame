from gym_idsgame.envs.dao.render_state import RenderState
from gym_idsgame.envs.dao.network_config import NetworkConfig

class GameConfig():
    """
    DTO with game configuration parameters
    """
    def __init__(self, network_config:NetworkConfig = None, manual:bool = True, num_layers:int = 1,
                 num_servers_per_layer:int = 2, num_attack_types:int = 10, max_value:int = 10,
                 initial_state:bool = None):
        self.manual = manual
        self.num_layers = num_layers
        self.num_servers_per_layer = num_servers_per_layer
        self.num_attack_types = num_attack_types
        self.max_value = max_value
        self.num_rows = self.num_layers + 2
        self.num_cols = self.num_servers_per_layer
        self.network_config = network_config
        if network_config is None:
            self.network_config = NetworkConfig(self.num_rows, self.num_cols)
        self.initial_state = initial_state
        if self.initial_state is None:
            self.initial_state = RenderState()
            self.initial_state.default_state(self.network_config.graph_layout, self.num_rows, self.num_cols, self.num_attack_types)
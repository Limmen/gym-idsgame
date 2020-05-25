"""
Configuration for the gym-idsgame environment
"""
from gym_idsgame.envs.dao.render_config import RenderConfig
from gym_idsgame.envs.dao.game_config import GameConfig
from gym_idsgame.agents.agent import Agent

class IdsGameConfig:
    """
    DTO representing the configuration of the IdsGameEnv:
    """

    def __init__(self, render_config: RenderConfig = None, game_config: GameConfig = None,
                 defender_agent: Agent = None, attacker_agent: Agent = None, initial_state_path: str = None,
                 save_trajectories :bool = False, save_attack_stats : bool = False,
                 randomize_env : bool = False, local_view_observations : bool = False,
                 reconnaissance_actions : bool = False, randomize_starting_position : bool = False):
        """
        Constructor, initializes the config

        :param render_config: render config, e.g colors, size, line width etc.
        :param game_config: game configuration, e.g. number of nodes
        :param defender_agent: the defender agent
        :param attacker_agent: the attacker agent
        :param initial_state_path: path to the initial state
        :param save_trajectories: boolean flag whether trajectories should be saved to create a self-play-dataset
        :param save_attack_stats: boolean flag whether to save attack statistics or not
        :param randomize_env: boolean flag whether to randomize the environment creation before each episode
        :param local_view_observations: boolean flag whether features are provided in a "local view" mode
        :param reconnaissance_actions: a boolean flag that indicates whether reconnaissance activities are enabled for
                                       the attacker
        :param randomize_starting_position: if true, the starting position of the attacker is randomized
        """
        self.render_config = render_config
        self.game_config = game_config
        self.defender_agent = defender_agent
        self.attacker_agent = attacker_agent
        if self.render_config is None:
            self.render_config = RenderConfig()
        if self.game_config is None:
            self.game_config = GameConfig(initial_state_path=initial_state_path)
        self.render_config.set_height(self.game_config.num_rows)
        self.render_config.set_width(self.game_config.num_cols)
        self.save_trajectories = save_trajectories
        self.save_attack_stats = save_attack_stats
        self.randomize_env = randomize_env
        self.local_view_observations = local_view_observations
        self.reconnaissance_actions = reconnaissance_actions
        self.randomize_starting_position = randomize_starting_position

"""
Runner for the IDS Game GUI Manual Game
"""
from gym_idsgame.envs.rendering.viewer import Viewer
from gym_idsgame.envs.dao.game_config import GameConfig
from gym_idsgame.envs.dao.render_config import RenderConfig
from gym_idsgame.envs.policy_baselines.random_defense import RandomDefense

if __name__ == '__main__':
    game_config = GameConfig(num_layers = 2, num_servers_per_layer = 3, num_attack_types = 10, max_value = 10)
    defender_policy = RandomDefense(game_config)
    render_config = RenderConfig(game_config=game_config, defender_policy=defender_policy)
    viewer = Viewer(render_config)
    viewer.manual_start()
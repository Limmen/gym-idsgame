"""
Runner for the IDS Game GUI Manual Game
"""
from gym_idsgame.envs.rendering.viewer import Viewer
from gym_idsgame.envs.dao.game_config import GameConfig
from gym_idsgame.envs.dao.render_config import RenderConfig

if __name__ == '__main__':
    game_config = GameConfig(num_layers = 2, num_servers_per_layer = 3, num_attack_types = 10, max_value = 10)
    render_config = RenderConfig(game_config)
    viewer = Viewer(render_config)
    viewer.manual_start()
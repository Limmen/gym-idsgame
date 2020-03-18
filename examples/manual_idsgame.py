"""
Runner for the IDS Game GUI Manual Game
"""
from gym_idsgame.envs.rendering.viewer import Viewer

if __name__ == '__main__':
    viewer = Viewer(num_layers = 3, num_servers_per_layer = 3, num_attack_types = 5, max_value = 5)
    viewer.manual_start()
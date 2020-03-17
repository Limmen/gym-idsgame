"""
Runner for the IDS Game GUI Manual Game
"""
from gym_idsgame.envs.rendering.viewer import Viewer
from gym_idsgame.envs.rendering import constants
import os
if __name__ == '__main__':
    script_dir = os.path.dirname(__file__)
    resource_path = os.path.join(script_dir, './../gym_idsgame/envs/rendering/', constants.IDSGAME.RESOURCES_DIR)
    resource_path = './../gym_idsgame/envs/rendering/resources'
    viewer = Viewer(num_layers = 1, num_servers_per_layer = 3, resources_dir = resource_path)
    viewer.manual_start()
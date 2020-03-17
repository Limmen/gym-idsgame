"""
Runner for the IDS Game GUI Manual Game
"""
from gym_idsgame.envs.idsgame_env import IdsGameEnv
from gym_idsgame.envs.rendering.viewer import Viewer

if __name__ == '__main__':
    env = IdsGameEnv()
    #env.step(12)
    viewer = Viewer(num_layers=2, num_servers_per_layer=3, num_attack_types=10, max_value=10,
                    adjacency_matrix=env.adjacency_matrix, graph_layout=env.graph_layout)
    viewer.manual_start()
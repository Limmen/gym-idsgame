"""
Runner for the IDS Game GUI Manual Game
"""
from gym_idsgame.envs.idsgame_env import IdsGameEnv
from gym_idsgame.envs.rendering.viewer import Viewer
import time
from gym_idsgame.envs.rendering import constants
import numpy as np



if __name__ == '__main__':
    env = IdsGameEnv(num_layers = 2, num_servers_per_layer = 3, num_attack_types = 10, max_value = 10)
    render_state = env.convert_state_to_render_state()

    obs = env.reset()
    done = False
    for i in range(3):
        action = 12
        env.render()
        time.sleep(0.5)
        action = 12
        s, reward, done, _ = env.step(action)
    for i in range(3):
        action = 42
        env.render()
        time.sleep(0.5)
        s, reward, done, _ = env.step(action)
    for i in range(3):
        action = 73
        env.render()
        time.sleep(0.5)
        s, reward, done, _ = env.step(action)
    while not done:
        env.render()
    #     for i in range(3):
    #
    #     env.render()
    #     time.sleep(0.5)
    #     action = 12
    #     s, reward, done, _ = env.step(action)

    #render_state = env.convert_state_to_render_state()
    # viewer = Viewer(num_layers=2, num_servers_per_layer=3, num_attack_types=10, max_value=10,
    #                 adjacency_matrix=env.adjacency_matrix, graph_layout=env.graph_layout)
    #viewer.agent_start()
    #viewer.gameframe.set_state(render_state)
    #viewer.render()


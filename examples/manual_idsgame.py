"""
Runner for the IDS Game GUI Manual Game
"""
from gym_idsgame.envs.rendering.viewer import Viewer

if __name__ == '__main__':
    viewer = Viewer(width=300, height=400, rect_size=50, manual=True)
    viewer.manual_start()
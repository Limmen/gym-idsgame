import os
import gym
import sys
from gym_idsgame.agents.manual_agents.manual_defense_agent import ManualDefenseAgent

def get_script_path():
    """
    :return: the script path
    """
    return os.path.dirname(os.path.realpath(sys.argv[0]))


def default_output_dir() -> str:
    """
    :return: the default output dir
    """
    script_dir = get_script_path()
    return script_dir

# Program entrypoint
if __name__ == '__main__':
    random_seed = 0
    env_name = "idsgame-random_attack-v2"
    env = gym.make(env_name)
    ManualDefenseAgent(env.idsgame_config)

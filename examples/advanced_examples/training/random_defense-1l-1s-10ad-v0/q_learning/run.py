import argparse
import jsonpickle
import json
import io
import os
from gym_idsgame.config.runner_mode import RunnerMode
from gym_idsgame.algorithms.q_agent_config import QAgentConfig
from gym_idsgame.agents.agent_type import AgentType
from gym_idsgame.config.client_config import ClientConfig
from gym_idsgame.runnner import Runner

def default_config_path():
    """
    :return: the default path to configuration file
    """
    script_dir = os.path.dirname(__file__)
    config_path = os.path.join(script_dir, './config.json')
    return config_path

def parse_args():
    """
    Parses the commandline arguments with argparse
    """
    parser = argparse.ArgumentParser(description='Parse flags to configure the json parsing')
    parser.add_argument("-cp", "--configpath", help="Path to configuration file",
                        default=default_config_path(), type=str)
    args = parser.parse_args()
    return args

def default_config():
    """
    :return: Default configuration for the experiment
    """
    q_agent_config = QAgentConfig(gamma=0.9, alpha=0.3, epsilon=1, render=False, eval_sleep=0.5,
                                  min_epsilon=0.1, eval_episodes=5, log_frequency=100, epsilon_decay=0.999, video=False,
                                  video_fps=5, video_dir="./videos", num_episodes=5000)
    env_name = "idsgame-random_defense-1l-1s-10ad-v0"
    client_config = ClientConfig(env_name=env_name, attacker_type=AgentType.Q_AGENT.value,
                                 mode=RunnerMode.TRAIN_ATTACKER.value,
                                 q_agent_config=q_agent_config)
    return client_config


def write_default_config(path:str = None):
    """
    Writes the default configuration to a json file

    :param path: the path to write the configuration to
    :return: None
    """
    if path is None:
        path = default_config_path()
    config = default_config()
    json_str = json.dumps(json.loads(jsonpickle.encode(config)), indent=4, sort_keys=True)
    with io.open(path, 'w', encoding='utf-8') as f:
        f.write(json_str)


def read_config(config_path) -> ClientConfig:
    """
    Reads configuration of the experiment from a json file

    :param config_path: the path to the configuration file
    :return: the configuration
    """
    with io.open(config_path, 'r', encoding='utf-8') as f:
        json_str = f.read()
    client_config: ClientConfig = jsonpickle.decode(json_str)
    return client_config

# Program entrypoint
if __name__ == '__main__':
    args = parse_args()
    if args.configpath is not None:
        config = read_config(args.configpath)
    else:
        config = default_config()
    train_result, eval_result = Runner.run(config)
    print(train_result, eval_result)




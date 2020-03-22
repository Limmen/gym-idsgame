import os
from gym_idsgame.config.runner_mode import RunnerMode
from gym_idsgame.algorithms.q_agent_config import QAgentConfig
from gym_idsgame.agents.agent_type import AgentType
from gym_idsgame.config.client_config import ClientConfig
from gym_idsgame.runnner import Runner
from examples.util import util

def default_config_path():
    """
    :return: the default path to configuration file
    """
    script_dir = os.path.dirname(__file__)
    config_path = os.path.join(script_dir, './config.json')
    return config_path


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
    util.write_config_file(config, path)


# Program entrypoint
if __name__ == '__main__':
    args = util.parse_args(default_config_path())
    logger = util.setup_logger("random_defense-1l-1s-10ad-v0-Q_learning", os.path.dirname(__file__))
    if args.configpath is not None:
        config = util.read_config(args.configpath)
    else:
        config = default_config()
    config.logger = logger
    config.q_agent_config.logger = logger
    train_result, eval_result = Runner.run(config)
    print(train_result, eval_result)




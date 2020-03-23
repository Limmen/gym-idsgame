import os
from gym_idsgame.config.runner_mode import RunnerMode
from gym_idsgame.agents.dao.q_agent_config import QAgentConfig
from gym_idsgame.agents.dao.agent_type import AgentType
from gym_idsgame.config.client_config import ClientConfig
from gym_idsgame.runnner import Runner
from experiments.util import util

def default_output_dir() -> str:
    """
    :return: the default output dir
    """
    script_dir = os.path.dirname(__file__)
    return script_dir


def default_config_path() -> str:
    """
    :return: the default path to configuration file
    """
    config_path = os.path.join(default_output_dir(), './config.json')
    return config_path


def default_config() -> ClientConfig:
    """
    :return: Default configuration for the experiment
    """
    env_name = "idsgame-random_attack-1l-1s-10ad-v0"
    client_config = ClientConfig(env_name=env_name, defender_type=AgentType.MANUAL_DEFENSE.value,
                                 mode=RunnerMode.MANUAL_DEFENDER.value, output_dir=default_output_dir())
    return client_config


def write_default_config(path:str = None) -> None:
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
    if args.configpath is not None:
        if not os.path.exists(args.configpath):
            write_default_config()
        config = util.read_config(args.configpath)
    else:
        config = default_config()
    Runner.run(config)




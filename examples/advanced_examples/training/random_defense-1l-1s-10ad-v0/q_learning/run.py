import os
import time
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
    dir = os.path.dirname(__file__)
    q_agent_config = QAgentConfig(gamma=0.9, alpha=0.3, epsilon=1, render=False, eval_sleep=0.5,
                                  min_epsilon=0.1, eval_episodes=5, train_log_frequency=100,
                                  epsilon_decay=0.999, video=False, eval_log_frequency=1,
                                  video_fps=5, video_dir=dir + "/videos", num_episodes=5000)
    env_name = "idsgame-random_defense-1l-1s-10ad-v0"
    client_config = ClientConfig(env_name=env_name, attacker_type=AgentType.Q_AGENT.value,
                                 mode=RunnerMode.TRAIN_ATTACKER.value,
                                 q_agent_config=q_agent_config, output_dir=dir)
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
#    write_default_config()
    args = util.parse_args(default_config_path())
    if args.configpath is not None:
        config = util.read_config(args.configpath)
    else:
        config = default_config()
    logger = util.setup_logger("random_defense-1l-1s-10ad-v0-Q_learning", config.output_dir)
    config.logger = logger
    config.q_agent_config.logger = logger
    train_result, eval_result = Runner.run(config)
    train_result.to_csv(config.output_dir + "/" + str(time.time()) + "_train" + ".csv")
    eval_result.to_csv(config.output_dir + "/" + str(time.time()) + "_eval" + ".csv")
    # print(train_result, eval_result)
    # print(eval_result.avg_episode_rewards)
    # print(eval_result.defender_cumulative_reward)
    # print(eval_result.attacker_cumulative_reward)




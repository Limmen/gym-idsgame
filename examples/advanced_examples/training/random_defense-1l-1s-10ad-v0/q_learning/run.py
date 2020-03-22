import os
import time
from gym_idsgame.config.runner_mode import RunnerMode
from gym_idsgame.algorithms.q_agent_config import QAgentConfig
from gym_idsgame.agents.agent_type import AgentType
from gym_idsgame.config.client_config import ClientConfig
from gym_idsgame.runnner import Runner
from examples.util import util
from examples.util import plotting_util

def default_config_path() -> str:
    """
    :return: the default path to configuration file
    """
    script_dir = os.path.dirname(__file__)
    config_path = os.path.join(script_dir, './config.json')
    return config_path


def default_config() -> ClientConfig:
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

def plot_csv(config: ClientConfig, eval_csv_path:str, train_csv_path: str) -> None:
    """
    Plot results from csv files

    :param config: client config
    :param eval_csv_path: path to the csv file with evaluation results
    :param train_csv_path: path to the csv file with training results
    :return: None
    """
    eval_df, train_df = plotting_util.read_data(eval_csv_path, train_csv_path)
    plotting_util.plot_results(train_df["avg_episode_rewards"].values, train_df["avg_episode_steps"].values,
                               train_df["epsilon_values"], train_df["hack_probability"],
                               train_df["attacker_cumulative_reward"], train_df["defender_cumulative_reward"],
                               config.q_agent_config.train_log_frequency,
                               config.output_dir, eval=False)
    plotting_util.plot_results(eval_df["avg_episode_rewards"].values, eval_df["avg_episode_steps"].values,
                               eval_df["epsilon_values"], eval_df["hack_probability"],
                               eval_df["attacker_cumulative_reward"], eval_df["defender_cumulative_reward"],
                               config.q_agent_config.train_log_frequency,
                               config.output_dir, eval=True)


# Program entrypoint
if __name__ == '__main__':
    args = util.parse_args(default_config_path())
    if args.configpath is not None:
        if not os.path.exists(args.configpath):
            write_default_config()
        config = util.read_config(args.configpath)
    else:
        config = default_config()
    util.create_artefact_dirs(config.output_dir)
    logger = util.setup_logger("random_defense-1l-1s-10ad-v0-Q_learning", config.output_dir + "/logs/")
    config.logger = logger
    config.q_agent_config.logger = logger
    train_result, eval_result = Runner.run(config)
    train_csv_path = config.output_dir + "/data/" + str(time.time()) + "_train" + ".csv"
    train_result.to_csv(train_csv_path)
    eval_csv_path = config.output_dir + "/data/" + str(time.time()) + "_eval" + ".csv"
    eval_result.to_csv(eval_csv_path)
    plot_csv(config, eval_csv_path, train_csv_path)




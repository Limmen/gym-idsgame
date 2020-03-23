import os
import time
from gym_idsgame.config.runner_mode import RunnerMode
from gym_idsgame.simulation.dao.simulation_config import SimulationConfig
from gym_idsgame.agents.dao.agent_type import AgentType
from gym_idsgame.config.client_config import ClientConfig
from gym_idsgame.runnner import Runner
from experiments.util import plotting_util, util
from gym_idsgame.agents.random_defense_bot_agent import RandomDefenseBotAgent
from gym_idsgame.agents.random_attack_bot_agent import RandomAttackBotAgent

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
    simulation_config = SimulationConfig(render=False, sleep=0.5, video=True, log_frequency=100,
                                         video_fps=5, video_dir=default_output_dir() + "/videos", num_episodes=1000,
                                         gifs=True, gif_dir=default_output_dir() + "/gifs", video_frequency = 100)
    env_name = "idsgame-1l-1s-10ad-v0"
    client_config = ClientConfig(env_name=env_name, attacker_type=AgentType.RANDOM.value,
                                 defender_type=AgentType.RANDOM.value, mode=RunnerMode.SIMULATE.value,
                                 simulation_config=simulation_config, output_dir=default_output_dir())
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
                               config.simulation_config.log_frequency,
                               config.output_dir, eval=False)
    plotting_util.plot_results(eval_df["avg_episode_rewards"].values, eval_df["avg_episode_steps"].values,
                               eval_df["epsilon_values"], eval_df["hack_probability"],
                               eval_df["attacker_cumulative_reward"], eval_df["defender_cumulative_reward"],
                               config.simulation_config.log_frequency,
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
    time_str = str(time.time())
    util.create_artefact_dirs(config.output_dir)
    logger = util.setup_logger("idsgame-1l-1s-10ad-v0-random_vs_random", config.output_dir + "/logs/",
                               time_str=time_str)
    config.logger = logger
    config.simulation_config.logger = logger
    result = Runner.run(config)
    # config.q_agent_config.logger = logger
    # config.q_agent_config.to_csv(config.output_dir + "/hyperparameters/" + time_str + ".csv")
    # train_result, eval_result = Runner.run(config)
    # train_csv_path = config.output_dir + "/data/" + time_str + "_train" + ".csv"
    # train_result.to_csv(train_csv_path)
    # eval_csv_path = config.output_dir + "/data/" + time_str + "_eval" + ".csv"
    # eval_result.to_csv(eval_csv_path)
    # plot_csv(config, eval_csv_path, train_csv_path)




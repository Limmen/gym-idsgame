import os
import time
from gym_idsgame.config.runner_mode import RunnerMode
from gym_idsgame.simulation.dao.simulation_config import SimulationConfig
from gym_idsgame.agents.dao.agent_type import AgentType
from gym_idsgame.config.client_config import ClientConfig
from gym_idsgame.runnner import Runner
from experiments.util import plotting_util, util

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
    simulation_config = SimulationConfig(render=False, sleep=0.8, video=True, log_frequency=1,
                                         video_fps=5, video_dir=default_output_dir() + "/videos", num_episodes=1000,
                                         gifs=True, gif_dir=default_output_dir() + "/gifs", video_frequency = 1)
    env_name = "idsgame-v4"
    client_config = ClientConfig(env_name=env_name, attacker_type=AgentType.RANDOM.value,
                                 defender_type=AgentType.DEFEND_MINIMAL_VALUE.value, mode=RunnerMode.SIMULATE.value,
                                 simulation_config=simulation_config, output_dir=default_output_dir(),
                                 title="RandomAttacker vs DefendMinimalDefender")
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


def plot_csv(config: ClientConfig, csv_path:str) -> None:
    """
    Plot results from csv files

    :param config: client config
    :param csv_path: path to the csv file with results
    :return: None
    """
    df = plotting_util.read_data(csv_path)
    plotting_util.plot_results(avg_episode_rewards=None, avg_episode_steps=df["avg_episode_steps"].values,
                               epsilon_values=None, hack_probability=df["hack_probability"],
                               attacker_cumulative_reward=df["attacker_cumulative_reward"],
                               defender_cumulative_reward=df["defender_cumulative_reward"],
                               log_frequency=config.simulation_config.log_frequency,
                               output_dir=config.output_dir, eval=False, sim=True)


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
    logger = util.setup_logger("idsgame-v0-random_vs_defend_minimal", config.output_dir + "/logs/",
                               time_str=time_str)
    config.logger = logger
    config.simulation_config.logger = logger
    config.simulation_config.to_csv(config.output_dir + "/hyperparameters/" + time_str + ".csv")
    result = Runner.run(config)
    if len(result.avg_episode_steps) > 0:
        csv_path = config.output_dir + "/data/" + time_str + "_simulation" + ".csv"
        result.to_csv(csv_path)
        plot_csv(config, csv_path)




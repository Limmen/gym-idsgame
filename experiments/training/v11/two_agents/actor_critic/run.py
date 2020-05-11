import os
import time
import sys
import glob
from gym_idsgame.config.runner_mode import RunnerMode
from gym_idsgame.agents.training_agents.policy_gradient.pg_agent_config import PolicyGradientAgentConfig
from gym_idsgame.agents.dao.agent_type import AgentType
from gym_idsgame.config.client_config import ClientConfig
from gym_idsgame.config.hp_tuning_config import HpTuningConfig
from gym_idsgame.runnner import Runner
from gym_idsgame.agents.training_agents.common.opponent_pool_config import OpponentPoolConfig
from experiments.util import plotting_util, util, hp_tuning


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


def default_config_path() -> str:
    """
    :return: the default path to configuration file
    """
    config_path = os.path.join(default_output_dir(), './config.json')
    return config_path

def hp_tuning_config(client_config: ClientConfig) -> ClientConfig:
    """
    Setup config for hparam tuning

    :param client_config: the client config
    :return: the updated client config
    """
    client_config.hp_tuning = True
    client_config.hp_tuning_config = HpTuningConfig(param_1="alpha", param_2="num_hidden_layers",
                                                    alpha=[0.000001, 0.00001, 0.0001, 0.001, 0.01],
                                                    num_hidden_layers=[1, 2, 4, 8, 16])
    client_config.run_many = False
    return client_config

def default_config() -> ClientConfig:
    """
    :return: Default configuration for the experiment
    """
    opponent_pool_config = OpponentPoolConfig(pool_maxsize=1000,
                                              pool_increment_period=500,
                                              head_to_head_period=1)

    pg_agent_config = PolicyGradientAgentConfig(gamma=0.999, alpha_attacker=0.0001, epsilon=1, render=False, eval_sleep=0.9,
                                                min_epsilon=0.01, eval_episodes=100, train_log_frequency=100,
                                                epsilon_decay=0.9999, video=True, eval_log_frequency=1,
                                                video_fps=5, video_dir=default_output_dir() + "/results/videos",
                                                num_episodes=1350001,
                                                eval_render=False, gifs=True,
                                                gif_dir=default_output_dir() + "/results/gifs",
                                                eval_frequency=10000, attacker=True, defender=True, video_frequency=101,
                                                save_dir=default_output_dir() + "/results/data",
                                                checkpoint_freq=10000, input_dim=6*2, output_dim_attacker=4,
                                                output_dim_defender=6,
                                                hidden_dim=16,
                                                num_hidden_layers=1, batch_size=32,
                                                gpu=False, tensorboard=True,
                                                tensorboard_dir=default_output_dir() + "/results/tensorboard",
                                                optimizer="Adam", lr_exp_decay=False, lr_decay_rate=0.999,
                                                state_length=1, alternating_optimization=True,
                                                alternating_period=5000, opponent_pool=True,
                                                opponent_pool_config=opponent_pool_config)
    env_name = "idsgame-v11"
    client_config = ClientConfig(env_name=env_name, attacker_type=AgentType.ACTOR_CRITIC_AGENT.value,
                                 defender_type=AgentType.ACTOR_CRITIC_AGENT.value,
                                 mode=RunnerMode.TRAIN_DEFENDER_AND_ATTACKER.value,
                                 pg_agent_config=pg_agent_config, output_dir=default_output_dir(),
                                 title="Actor-Critic vs Actor-Critic",
                                 run_many=False, random_seeds=[0, 999, 299, 399, 499])
    #client_config = hp_tuning_config(client_config)
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


def plot_csv(config: ClientConfig, eval_csv_path:str, train_csv_path: str, random_seed : int = 0) -> None:
    """
    Plot results from csv files

    :param config: client config
    :param eval_csv_path: path to the csv file with evaluation results
    :param train_csv_path: path to the csv file with training results
    :param random_seed: the random seed of the experiment
    :return: None
    """
    plotting_util.read_and_plot_results(train_csv_path, eval_csv_path, config.pg_agent_config.train_log_frequency,
                                        config.pg_agent_config.eval_frequency, config.pg_agent_config.eval_log_frequency,
                                        config.pg_agent_config.eval_episodes, config.output_dir, sim=False,
                                        random_seed = random_seed)


def plot_average_results(experiment_title :str, config: ClientConfig, eval_csv_paths:list,
                         train_csv_paths: str) -> None:
    """
    Plots average results after training with different seeds

    :param experiment_title: title of the experiment
    :param config: experiment config
    :param eval_csv_paths: paths to csv files with evaluation data
    :param train_csv_paths: path to csv files with training data
    :return: None
    """
    plotting_util.read_and_plot_average_results(experiment_title, train_csv_paths, eval_csv_paths,
                                                config.pg_agent_config.train_log_frequency,
                                                config.pg_agent_config.eval_frequency,
                                                config.output_dir,
                                                plot_attacker_loss = True, plot_defender_loss = False)

def run_experiment(configpath: str, random_seed: int, noconfig: bool):
    """
    Runs one experiment and saves results and plots

    :param configpath: path to configfile
    :param noconfig: whether to override config
    :return: (train_csv_path, eval_csv_path)
    """
    if configpath is not None and not noconfig:
        if not os.path.exists(args.configpath):
            write_default_config()
        config = util.read_config(args.configpath)
    else:
        config = default_config()
    time_str = str(time.time())
    util.create_artefact_dirs(config.output_dir, random_seed)
    logger = util.setup_logger("actor_critic_vs_random_defense-v11", config.output_dir + "/results/logs/" +
                               str(random_seed) + "/",
                               time_str=time_str)
    config.pg_agent_config.save_dir = default_output_dir() + "/results/data/" + str(random_seed) + "/"
    config.pg_agent_config.video_dir= default_output_dir() + "/results/videos/" + str(random_seed) + "/"
    config.pg_agent_config.gif_dir= default_output_dir() + "/results/gifs/" + str(random_seed) + "/"
    config.pg_agent_config.tensorboard_dir = default_output_dir() + "/results/tensorboard/" \
                                                       + str(random_seed) + "/"
    config.logger = logger
    config.pg_agent_config.logger = logger
    config.pg_agent_config.random_seed = random_seed
    config.random_seed = random_seed
    config.pg_agent_config.to_csv(config.output_dir + "/results/hyperparameters/" + str(random_seed) + "/" + time_str + ".csv")
    train_csv_path = ""
    eval_csv_path = ""
    if config.hp_tuning:
        hp_tuning.hype_grid(config)
    else:
        train_result, eval_result = Runner.run(config)
        if len(train_result.avg_episode_steps) > 0 and len(eval_result.avg_episode_steps) > 0:
            train_csv_path = config.output_dir + "/results/data/" + str(random_seed) + "/" + time_str + "_train" + ".csv"
            train_result.to_csv(train_csv_path)
            eval_csv_path = config.output_dir + "/results/data/" + str(random_seed) + "/" + time_str + "_eval" + ".csv"
            eval_result.to_csv(eval_csv_path)
            plot_csv(config, eval_csv_path, train_csv_path, random_seed)

    return train_csv_path, eval_csv_path

# Program entrypoint
if __name__ == '__main__':
    args = util.parse_args(default_config_path())
    experiment_title = "Actor-Critic vs minimal defense"
    if args.configpath is not None and not args.noconfig:
        if not os.path.exists(args.configpath):
            write_default_config()
        config = util.read_config(args.configpath)
    else:
        config = default_config()
    if args.plotonly:
        base_dir = default_output_dir() + "/results/data/"
        train_csv_paths = []
        eval_csv_paths = []
        for seed in config.random_seeds:
            train_csv_path = glob.glob(base_dir + str(seed) + "/*_train.csv")[0]
            eval_csv_path = glob.glob(base_dir + str(seed) + "/*_eval.csv")[0]
            train_csv_paths.append(train_csv_path)
            eval_csv_paths.append(eval_csv_path)
            plot_csv(config, eval_csv_path, train_csv_path, random_seed=seed)
        try:
            plot_average_results(experiment_title, config, eval_csv_paths, train_csv_paths)
        except Exception as e:
            print("Error when trying to plot summary: " + str(e))
    else:
        if not config.run_many:
            run_experiment(args.configpath, 0, args.noconfig)
        else:
            train_csv_paths = []
            eval_csv_paths = []
            for seed in config.random_seeds:
                train_csv_path, eval_csv_path = run_experiment(args.configpath, seed, args.noconfig)
                train_csv_paths.append(train_csv_path)
                eval_csv_paths.append(eval_csv_path)
            try:
                plot_average_results(experiment_title, config, eval_csv_paths, train_csv_paths)
            except Exception as e:
                print("Error when trying to plot summary: " + str(e))


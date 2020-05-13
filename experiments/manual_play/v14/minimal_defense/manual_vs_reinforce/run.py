import os
from gym_idsgame.config.runner_mode import RunnerMode
from gym_idsgame.agents.dao.agent_type import AgentType
from gym_idsgame.config.client_config import ClientConfig
from gym_idsgame.agents.training_agents.policy_gradient.pg_agent_config import PolicyGradientAgentConfig
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
    env_name = "idsgame-v14"
    pg_agent_config = PolicyGradientAgentConfig(gamma=0.999, alpha_attacker=0.001, epsilon=1, render=False,
                                                eval_sleep=0.9,
                                                min_epsilon=0.01, eval_episodes=100, train_log_frequency=100,
                                                epsilon_decay=0.9999, video=True, eval_log_frequency=1,
                                                video_fps=5, video_dir=default_output_dir() + "/results/videos",
                                                num_episodes=10000,
                                                eval_render=False, gifs=True,
                                                gif_dir=default_output_dir() + "/results/gifs",
                                                eval_frequency=1000, attacker=True, defender=False, video_frequency=101,
                                                save_dir=default_output_dir() + "/results/data",
                                                checkpoint_freq=1000, input_dim_attacker=(4 + 3) * 2,
                                                output_dim_attacker=4 * 2,
                                                hidden_dim=32,
                                                num_hidden_layers=1, batch_size=8,
                                                gpu=True, tensorboard=True,
                                                tensorboard_dir=default_output_dir() + "/results/tensorboard",
                                                optimizer="Adam", lr_exp_decay=False, lr_decay_rate=0.999,
                                                state_length=1, normalize_features=False, merged_ad_features=True,
                                                zero_mean_features=False, gpu_id=0,
                                                attacker_load_path="/Users/kimham/workspace/rl/gym-idsgame/experiments/training/v14/minimal_defense/reinforce/results/data/0/1589370657.761538_attacker_policy_network.pt")
    client_config = ClientConfig(env_name=env_name, attacker_type=AgentType.REINFORCE_AGENT.value,
                                 mode=RunnerMode.MANUAL_DEFENDER.value, output_dir=default_output_dir(),
                                 title="REINFORCE vs ManualDefender", pg_agent_config=pg_agent_config,
                                 bot_attacker=True)
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




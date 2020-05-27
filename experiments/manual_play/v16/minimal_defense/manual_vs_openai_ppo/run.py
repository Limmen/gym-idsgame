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
    env_name = "idsgame-v16"
    pg_agent_config = PolicyGradientAgentConfig(gamma=1, alpha_attacker=0.00001, epsilon=1, render=False,
                                                alpha_defender=0.0001,
                                                eval_sleep=0.9,
                                                min_epsilon=0.01, eval_episodes=100, train_log_frequency=1,
                                                epsilon_decay=0.9999, video=True, eval_log_frequency=1,
                                                video_fps=5, video_dir=default_output_dir() + "/results/videos",
                                                num_episodes=100000000,
                                                eval_render=False, gifs=True,
                                                gif_dir=default_output_dir() + "/results/gifs",
                                                eval_frequency=100000, attacker=True, defender=False,
                                                video_frequency=101,
                                                save_dir=default_output_dir() + "/results/data",
                                                checkpoint_freq=5000,
                                                input_dim_attacker=(4 + 2) * 3,
                                                output_dim_attacker=4 * 3,
                                                input_dim_defender=(4 + 2) * 3,
                                                output_dim_defender=5 * 3,
                                                hidden_dim=64,
                                                num_hidden_layers=4, batch_size=2000,
                                                gpu=False, tensorboard=True,
                                                tensorboard_dir=default_output_dir() + "/results/tensorboard",
                                                optimizer="Adam", lr_exp_decay=False, lr_decay_rate=0.999,
                                                state_length=1, normalize_features=False, merged_ad_features=True,
                                                zero_mean_features=False, gpu_id=1, lstm_network=False,
                                                lstm_seq_length=4, num_lstm_layers=2, optimization_iterations=10,
                                                eps_clip=0.2, max_gradient_norm=0.5, gae_lambda=0.95,
                                                cnn_feature_extractor=False, features_dim=512,
                                                flatten_feature_planes=False,
                                                attacker_load_path="/Users/kimham/workspace/rl/gym-idsgame/experiments/manual_play/v16/minimal_defense/manual_vs_openai_ppo/1590564756.5454807_attacker_policy_network.zip")
    client_config = ClientConfig(env_name=env_name, attacker_type=AgentType.PPO_OPENAI_AGENT.value,
                                 mode=RunnerMode.MANUAL_DEFENDER.value, output_dir=default_output_dir(),
                                 title="OpenAI PPO vs ManualDefender", pg_agent_config=pg_agent_config,
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




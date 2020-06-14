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
    env_name = "idsgame-v19"
    pg_agent_config = PolicyGradientAgentConfig(gamma=1, alpha_attacker=0.0001, epsilon=1, render=False,
                                                alpha_defender=0.0001,
                                                eval_sleep=0.9,
                                                min_epsilon=0.01, eval_episodes=1000, train_log_frequency=1,
                                                epsilon_decay=0.9999, video=True, eval_log_frequency=500,
                                                video_fps=5, video_dir=default_output_dir() + "/results/videos",
                                                num_episodes=100000000,
                                                eval_render=False, gifs=True,
                                                gif_dir=default_output_dir() + "/results/gifs",
                                                eval_frequency=55000, attacker=False, defender=True,
                                                video_frequency=1001,
                                                save_dir=default_output_dir() + "/results/data",
                                                checkpoint_freq=250,
                                                input_dim_attacker=((4 + 2) * 4),
                                                output_dim_attacker=(4 + 1) * 4,
                                                input_dim_defender=((4 + 1) * 4),
                                                output_dim_defender=5 * 4,
                                                hidden_dim=128, num_hidden_layers=2,
                                                pi_hidden_layers=1, pi_hidden_dim=128, vf_hidden_layers=1,
                                                vf_hidden_dim=128,
                                                batch_size=2000,
                                                gpu=False, tensorboard=True,
                                                tensorboard_dir=default_output_dir() + "/results/tensorboard",
                                                optimizer="Adam", lr_exp_decay=False, lr_decay_rate=0.999,
                                                state_length=1, normalize_features=False, merged_ad_features=True,
                                                zero_mean_features=False, gpu_id=0, lstm_network=False,
                                                lstm_seq_length=4, num_lstm_layers=2, optimization_iterations=10,
                                                eps_clip=0.2, max_gradient_norm=0.5, gae_lambda=0.95,
                                                cnn_feature_extractor=False, features_dim=512,
                                                flatten_feature_planes=False, cnn_type=5, vf_coef=0.5, ent_coef=0.001,
                                                render_attacker_view=False, lr_progress_power_decay=4,
                                                lr_progress_decay=True, use_sde=False, sde_sample_freq=4,
                                                one_hot_obs=False, lstm_core=False, lstm_hidden_dim=32,
                                                multi_channel_obs=False,
                                                channel_1_dim=32, channel_1_layers=2, channel_1_input_dim=16,
                                                channel_2_dim=32, channel_2_layers=2, channel_2_input_dim=16,
                                                channel_3_dim=32, channel_3_layers=2, channel_3_input_dim=4,
                                                channel_4_dim=32, channel_4_layers=2, channel_4_input_dim=4,
                                                mini_batch_size=64, ar_policy=True,
                                                attacker_node_input_dim=((4 + 2) * 4),
                                                attacker_at_net_input_dim=(4 + 2), attacker_at_net_output_dim=(4 + 1),
                                                attacker_node_net_output_dim=4,
                                                defender_node_input_dim=((4 + 1) * 4),
                                                defender_at_net_input_dim=(4 + 1),
                                                defender_node_net_output_dim=4, defender_at_net_output_dim=5,
                                                defender_load_path="/home/kim/workspace/gym-idsgame/experiments/manual_play/v19/maximal_attack/manual_vs_openai_ppo/1592125075.4390159_defender_node_policy_network.zip")
    client_config = ClientConfig(env_name=env_name, defender_type=AgentType.PPO_OPENAI_AGENT.value,
                                 mode=RunnerMode.MANUAL_ATTACKER.value, output_dir=default_output_dir(),
                                 title="ManualAttacker vs OpenAI PPO", pg_agent_config=pg_agent_config,
                                 bot_defender=True)
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




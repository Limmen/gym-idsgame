import os
import sys
import gymnasium as gym
from gym_idsgame.agents.training_agents.policy_gradient.pg_agent_config import PolicyGradientAgentConfig
from gym_idsgame.agents.training_agents.openai_baselines.ppo.ppo import OpenAiPPOAgent
from gym_idsgame.agents.training_agents.openai_baselines.common.baseline_env_wrapper import BaselineEnvWrapper
import gym_idsgame
from experiments.util import util

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

# Program entrypoint
if __name__ == '__main__':
    random_seed = 0
    util.create_artefact_dirs(default_output_dir(), random_seed)
    pg_agent_config = PolicyGradientAgentConfig(gamma=1, alpha_attacker=0.0001, epsilon=1, render=False,
                                                alpha_defender=0.0001,
                                                eval_sleep=0.9,
                                                min_epsilon=0.01, eval_episodes=1000, train_log_frequency=1,
                                                epsilon_decay=0.9999, video=False, eval_log_frequency=500,
                                                video_fps=5, video_dir=default_output_dir() + "/results/videos",
                                                num_episodes=100000000,
                                                eval_render=False, gifs=True,
                                                gif_dir=default_output_dir() + "/results/gifs/" + str(random_seed),
                                                eval_frequency=55000, attacker=True, defender=False,
                                                video_frequency=1001,
                                                save_dir=default_output_dir() + "/results/data/" + str(random_seed),
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
                                                tensorboard_dir=default_output_dir() + "/results/tensorboard/" + str(random_seed),
                                                optimizer="Adam", lr_exp_decay=False, lr_decay_rate=0.999,
                                                state_length=1, normalize_features=False, merged_ad_features=True,
                                                zero_mean_features=False, gpu_id=0, lstm_network=False,
                                                lstm_seq_length=4, num_lstm_layers=2, optimization_iterations=10,
                                                eps_clip=0.2, max_gradient_norm=0.5, gae_lambda=0.95,
                                                cnn_feature_extractor=False, features_dim=512,
                                                flatten_feature_planes=False, cnn_type=5, vf_coef=0.5, ent_coef=0.001,
                                                render_attacker_view=True, lr_progress_power_decay=4,
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
                                                attacker_node_net_output_dim=4)
    env_name = "idsgame-minimal_defense-v19"
    wrapper_env = BaselineEnvWrapper(env_name, idsgame_config=None,
                                     save_dir=default_output_dir() + "/results/data/" + str(random_seed),
                                     pg_agent_config=pg_agent_config)
    attacker_agent = OpenAiPPOAgent(wrapper_env, pg_agent_config)

    attacker_agent.train()
    train_result = attacker_agent.train_result
    eval_result = attacker_agent.eval_result
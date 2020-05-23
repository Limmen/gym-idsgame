import time
import os
import sys
import torch
from gym_idsgame.config.runner_mode import RunnerMode
from gym_idsgame.agents.training_agents.policy_gradient.pg_agent_config import PolicyGradientAgentConfig
from gym_idsgame.agents.dao.agent_type import AgentType
from gym_idsgame.config.client_config import ClientConfig
from experiments.util import util
from gym_idsgame.agents.training_agents.openai_baselines.lib.baseline_env_wrapper import BaselineEnvWrapper
from gym_idsgame.agents.training_agents.openai_baselines.lib.ppo.ppo import PPO
from gym_idsgame.envs.rendering.video.idsgame_monitor import IdsGameMonitor

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

def default_config() -> ClientConfig:
    """
    :return: Default configuration for the experiment
    """
    pg_agent_config = PolicyGradientAgentConfig(gamma=1, alpha_attacker=0.0001, alpha_defender=0.0001, epsilon=1, render=False,
                                                eval_sleep=0.9,
                                                min_epsilon=0.01, eval_episodes=100, train_log_frequency=1,
                                                epsilon_decay=0.9999, video=True, eval_log_frequency=1,
                                                video_fps=5, video_dir=default_output_dir() + "/results/videos",
                                                num_episodes=100000000,
                                                eval_render=False, gifs=True,
                                                gif_dir=default_output_dir() + "/results/gifs",
                                                eval_frequency=25000, attacker=False, defender=True, video_frequency=101,
                                                save_dir=default_output_dir() + "/results/data",
                                                checkpoint_freq=5000, input_dim_attacker=(4 + 2) * 3,
                                                input_dim_defender=(4 + 2) * 3,
                                                output_dim_attacker=4 * 3,
                                                output_dim_defender=5 * 3,
                                                hidden_dim=32,
                                                num_hidden_layers=1, batch_size=2000,
                                                gpu=True, tensorboard=True,
                                                tensorboard_dir=default_output_dir() + "/results/tensorboard",
                                                optimizer="Adam", lr_exp_decay=False, lr_decay_rate=0.999,
                                                state_length=1, normalize_features=False, merged_ad_features=True,
                                                zero_mean_features=True, gpu_id=0, lstm_network=False,
                                                lstm_seq_length=4, num_lstm_layers=2, optimization_iterations=10,
                                                eps_clip=0.2, max_gradient_norm=0.5, gae_lambda=0.95)
    env_name = "idsgame-maximal_attack-v16"
    client_config = ClientConfig(env_name=env_name, attacker_type=AgentType.REINFORCE_AGENT.value,
                                 mode=RunnerMode.TRAIN_DEFENDER,
                                 pg_agent_config=pg_agent_config, output_dir=default_output_dir(),
                                 title="REINFORCE vs DefendMinimalDefender",
                                 run_many=False, random_seeds=[0, 999, 299, 399, 499])
    #client_config = hp_tuning_config(client_config)
    return client_config

def test():
    random_seed = 0
    config = default_config()
    time_str = str(time.time())
    util.create_artefact_dirs(config.output_dir, random_seed)
    logger = util.setup_logger("reinforce_vs_minimal_defense-v14", config.output_dir + "/results/logs/" +
                               str(random_seed) + "/",
                               time_str=time_str)
    config.pg_agent_config.save_dir = default_output_dir() + "/results/data/" + str(random_seed) + "/"
    config.pg_agent_config.video_dir = default_output_dir() + "/results/videos/" + str(random_seed) + "/"
    config.pg_agent_config.gif_dir = default_output_dir() + "/results/gifs/" + str(random_seed) + "/"
    config.pg_agent_config.tensorboard_dir = default_output_dir() + "/results/tensorboard/" \
                                             + str(random_seed) + "/"
    config.logger = logger
    config.pg_agent_config.logger = logger
    config.pg_agent_config.random_seed = random_seed
    config.random_seed = random_seed
    config.pg_agent_config.to_csv(
        config.output_dir + "/results/hyperparameters/" + str(random_seed) + "/" + time_str + ".csv")
    train_csv_path = ""
    eval_csv_path = ""
    wrapper_env = BaselineEnvWrapper(config.env_name, idsgame_config=config.idsgame_config,
                   save_dir=config.output_dir + "/results/data/" + str(config.random_seed),
                   initial_state_path=config.initial_state_path, pg_agent_config=config.pg_agent_config)

    # Custom MLP policy of two layers of size 32 each with ReLu activation function
    policy_kwargs = dict(activation_fn=torch.nn.ReLU, net_arch=[32, 32])

    #check_env(wrapper_env)
    device = "cpu" if not config.pg_agent_config.gpu else "cuda:" + str(config.pg_agent_config.gpu_id)
    model = PPO("MlpPolicy", wrapper_env,
                learning_rate=config.pg_agent_config.alpha_attacker,
                n_steps=config.pg_agent_config.batch_size,
                n_epochs=config.pg_agent_config.optimization_iterations,
                gamma = config.pg_agent_config.gamma,
                gae_lambda=config.pg_agent_config.gae_lambda,
                clip_range=config.pg_agent_config.eps_clip,
                max_grad_norm = config.pg_agent_config.max_gradient_norm,
                verbose=1, tensorboard_log=config.pg_agent_config.tensorboard_dir,
                seed=config.random_seed,
                policy_kwargs=policy_kwargs,
                device=device,
                pg_agent_config = config.pg_agent_config)

    # Video config
    if config.pg_agent_config.video:
        if config.pg_agent_config.video_dir is None:
            raise AssertionError("Video is set to True but no video_dir is provided, please specify "
                                 "the video_dir argument")
        eval_env = IdsGameMonitor(wrapper_env, config.pg_agent_config.video_dir + "/" + time_str, force=True,
                             video_frequency= config.pg_agent_config.video_frequency, openai_baseline=True)
        eval_env.metadata["video.frames_per_second"] = config.pg_agent_config.video_fps

    model.learn(total_timesteps=config.pg_agent_config.num_episodes,
                log_interval=config.pg_agent_config.train_log_frequency,
                eval_freq=config.pg_agent_config.eval_frequency,
                n_eval_episodes=config.pg_agent_config.eval_episodes,
                eval_env=eval_env)


if __name__ == '__main__':
    test()
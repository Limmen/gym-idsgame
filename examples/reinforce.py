import os
import gymnasium as gym
import sys
from gym_idsgame.agents.training_agents.policy_gradient.pg_agent_config import PolicyGradientAgentConfig
from gym_idsgame.agents.training_agents.policy_gradient.reinforce.reinforce import ReinforceAgent
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
    pg_agent_config = PolicyGradientAgentConfig(gamma=0.999, alpha_attacker=0.00001, epsilon=1, render=False,
                                                eval_sleep=0.9,
                                                min_epsilon=0.01, eval_episodes=100, train_log_frequency=100,
                                                epsilon_decay=0.9999, video=True, eval_log_frequency=1,
                                                video_fps=5, video_dir=default_output_dir() + "/results/videos/" + str(random_seed),
                                                num_episodes=200001,
                                                eval_render=False, gifs=True,
                                                gif_dir=default_output_dir() + "/results/gifs/" + str(random_seed),
                                                eval_frequency=10000, attacker=True, defender=False,
                                                video_frequency=101,
                                                save_dir=default_output_dir() + "/results/data/" + str(random_seed),
                                                checkpoint_freq=5000, input_dim_attacker=44, output_dim_attacker=40,
                                                hidden_dim=64,
                                                num_hidden_layers=1, batch_size=32,
                                                gpu=False, tensorboard=True,
                                                tensorboard_dir=default_output_dir() + "/results/tensorboard/" + str(random_seed),
                                                optimizer="Adam", lr_exp_decay=False, lr_decay_rate=0.999)
    env_name = "idsgame-minimal_defense-v9"
    env = gym.make(env_name, save_dir=default_output_dir() + "/results/data/" + str(random_seed))
    attacker_agent = ReinforceAgent(env, pg_agent_config)
    attacker_agent.train()
    train_result = attacker_agent.train_result
    eval_result = attacker_agent.eval_result
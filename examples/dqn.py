import os
import gym
import sys
from gym_idsgame.agents.training_agents.q_learning.q_agent_config import QAgentConfig
from gym_idsgame.agents.training_agents.q_learning.dqn.dqn import DQNAgent
from gym_idsgame.agents.training_agents.q_learning.dqn.dqn_config import DQNConfig
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
    dqn_config = DQNConfig(input_dim=88, defender_output_dim=88, hidden_dim=64, replay_memory_size=10000,
                           num_hidden_layers=1,
                           replay_start_size=1000, batch_size=32, target_network_update_freq=1000,
                           gpu=True, tensorboard=True, tensorboard_dir=default_output_dir() + "/results/tensorboard/" + str(random_seed),
                           loss_fn="Huber", optimizer="Adam", lr_exp_decay=True, lr_decay_rate=0.9999)
    q_agent_config = QAgentConfig(gamma=0.999, alpha=0.00001, epsilon=1, render=False, eval_sleep=0.9,
                                  min_epsilon=0.01, eval_episodes=100, train_log_frequency=100,
                                  epsilon_decay=0.9999, video=True, eval_log_frequency=1,
                                  video_fps=5, video_dir=default_output_dir() + "/results/videos/" + str(random_seed), num_episodes=20001,
                                  eval_render=False, gifs=True, gif_dir=default_output_dir() + "/results/gifs/" + str(random_seed),
                                  eval_frequency=1000, attacker=False, defender=True, video_frequency=101,
                                  save_dir=default_output_dir() + "/results/data/" + str(random_seed), dqn_config=dqn_config,
                                  checkpoint_freq=5000)
    env_name = "idsgame-maximal_attack-v3"
    env = gym.make(env_name, save_dir=default_output_dir() + "/results/data/" + str(random_seed))
    defender_agent = DQNAgent(env, q_agent_config)
    defender_agent.train()
    train_result = defender_agent.train_result
    eval_result = defender_agent.eval_result
"""
Abstract Q Agent
"""
import numpy as np
import tqdm
import logging
from abc import ABC, abstractmethod
from gym_idsgame.agents.q_learning.q_agent_config import QAgentConfig
from gym_idsgame.envs.idsgame_env import IdsGameEnv
from gym_idsgame.agents.dao.experiment_result import ExperimentResult
from gym_idsgame.agents.train_agent import TrainAgent

class QAgent(TrainAgent, ABC):
    """
    Abstract QAgent
    """
    def __init__(self, env:IdsGameEnv, config: QAgentConfig):
        """
        Initialize environment and hyperparameters

        :param config: the configuration
        """
        self.env = env
        self.config = config
        self.train_result = ExperimentResult()
        self.eval_result = ExperimentResult()
        self.num_eval_games = 0
        self.num_eval_hacks = 0
        self.eval_hack_probability = 0.0
        self.eval_attacker_cumulative_reward = 0
        self.eval_defender_cumulative_reward = 0
        self.outer_train = tqdm.tqdm(total=self.config.num_episodes, desc='Train Episode', position=0)
        if self.config.logger is None:
            self.config.logger = logging.getLogger('QAgent')

    def log_metrics(self, episode: int, result: ExperimentResult, attacker_episode_rewards: list,
                    defender_episode_rewards: list,
                    episode_steps: list, episode_avg_loss: list = None, eval: bool = False) -> None:
        """
        Logs average metrics for the last <self.config.log_frequency> episodes

        :param episode: the episode
        :param result: the result object to add the results to
        :param attacker_episode_rewards: list of attacker episode rewards for the last <self.config.log_frequency> episodes
        :param defender_episode_rewards: list of defender episode rewards for the last <self.config.log_frequency> episodes
        :param episode_steps: list of episode steps for the last <self.config.log_frequency> episodes
        :param episode_avg_loss: list of episode loss for the last <self.config.log_frequency> episodes
        :param eval: boolean flag whether the metrics are logged in an evaluation context.
        :return: None
        """
        avg_attacker_episode_rewards = np.mean(attacker_episode_rewards)
        avg_defender_episode_rewards = np.mean(defender_episode_rewards)
        if not eval and episode_avg_loss is not None:
            avg_episode_loss = np.mean(episode_avg_loss)
        else:
            avg_episode_loss = 0.0
        avg_episode_steps = np.mean(episode_steps)
        hack_probability = self.env.hack_probability() if not eval else self.eval_hack_probability
        attacker_cumulative_reward = self.env.state.attacker_cumulative_reward if not eval \
            else self.eval_attacker_cumulative_reward
        defender_cumulative_reward = self.env.state.defender_cumulative_reward if not eval \
            else self.eval_defender_cumulative_reward
        if eval:
            log_str = "[Eval] avg_a_R:{:.2f},avg_d_R:{:.2f},avg_t:{:.2f},avg_h:{:.2f},acc_A_R:{:.2f}," \
                      "acc_D_R:{:.2f}".format(avg_attacker_episode_rewards,
                                              avg_defender_episode_rewards,
                                              avg_episode_steps,
                                              hack_probability,
                                              attacker_cumulative_reward,
                                              defender_cumulative_reward)
            self.outer_eval.set_description_str(log_str)
        else:
            log_str = "[Train] epsilon:{:.2f},avg_a_R:{:.2f},avg_d_R:{:.2f},avg_t:{:.2f},avg_h:{:.2f},acc_A_R:{:.2f}," \
                      "acc_D_R:{:.2f},loss:{:.5f}".format(self.config.epsilon, avg_attacker_episode_rewards,
                                                          avg_defender_episode_rewards,
                                                          avg_episode_steps,
                                                          hack_probability,
                                                          attacker_cumulative_reward,
                                                          defender_cumulative_reward,
                                                          avg_episode_loss)
            self.outer_train.set_description_str(log_str)
        self.config.logger.info(log_str)
        if self.config.dqn_config is not None and self.config.dqn_config.tensorboard:
            self.log_tensorboard(episode, avg_attacker_episode_rewards, avg_defender_episode_rewards, avg_episode_steps,
                                 avg_episode_loss, hack_probability, attacker_cumulative_reward,
                                 defender_cumulative_reward, eval=eval)
        result.avg_episode_steps.append(avg_episode_steps)
        result.avg_attacker_episode_rewards.append(avg_attacker_episode_rewards)
        result.avg_defender_episode_rewards.append(avg_defender_episode_rewards)
        result.epsilon_values.append(self.config.epsilon)
        result.hack_probability.append(hack_probability)
        result.attacker_cumulative_reward.append(attacker_cumulative_reward)
        result.defender_cumulative_reward.append(defender_cumulative_reward)
        result.avg_episode_loss_attacker.append(avg_episode_loss)

    def log_tensorboard(self, episode: int, avg_attacker_episode_rewards: float, avg_defender_episode_rewards: float,
                        avg_episode_steps: float, episode_avg_loss: float, hack_probability: float,
                        attacker_cumulative_reward: int, defender_cumulative_reward: int, eval=False) -> None:
        """
        Log metrics to tensorboard

        :param episode: the episode
        :param avg_attacker_episode_rewards: the average attacker episode reward
        :param avg_defender_episode_rewards: the average defender episode reward
        :param avg_episode_steps: the average number of episode steps
        :param episode_avg_loss: the average episode loss
        :param hack_probability: the hack probability
        :param attacker_cumulative_reward: the cumulative attacker reward
        :param defender_cumulative_reward: the cumulative defender reward
        :param eval: boolean flag whether eval or not
        :return: Nont
        """
        train_or_eval = "eval" if eval else "train"
        self.tensorboard_writer.add_scalar('avg_episode_rewards/' + train_or_eval + "/attacker",
                                           avg_attacker_episode_rewards, episode)
        self.tensorboard_writer.add_scalar('avg_episode_rewards/' + train_or_eval + "/defender",
                                           avg_defender_episode_rewards, episode)
        self.tensorboard_writer.add_scalar('episode_steps/' + train_or_eval, avg_episode_steps, episode)
        self.tensorboard_writer.add_scalar('episode_avg_loss/' + train_or_eval, episode_avg_loss, episode)
        self.tensorboard_writer.add_scalar('hack_probability/' + train_or_eval, hack_probability, episode)
        self.tensorboard_writer.add_scalar('cumulative_reward/attacker/' + train_or_eval,
                                           attacker_cumulative_reward, episode)
        self.tensorboard_writer.add_scalar('cumulative_reward/defender/' + train_or_eval,
                                           defender_cumulative_reward, episode)

    def anneal_epsilon(self) -> None:
        """
        Anneals the exploration rate slightly until it reaches the minimum value

        :return: None
        """
        if self.config.epsilon > self.config.min_epsilon:
            self.config.epsilon = self.config.epsilon*self.config.epsilon_decay

    @abstractmethod
    def get_action(self, s, eval=False, attacker=True) -> int:
        pass

    @abstractmethod
    def train(self) -> ExperimentResult:
        pass

    @abstractmethod
    def eval(self, log=True) -> ExperimentResult:
        pass
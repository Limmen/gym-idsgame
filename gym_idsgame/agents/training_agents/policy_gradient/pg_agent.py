"""
Abstract PolicyGradient Agent
"""
import numpy as np
import tqdm
import logging
import random
import torch
from abc import ABC, abstractmethod
from gym_idsgame.envs.idsgame_env import IdsGameEnv
from gym_idsgame.agents.dao.experiment_result import ExperimentResult
from gym_idsgame.agents.training_agents.train_agent import TrainAgent
from gym_idsgame.agents.training_agents.policy_gradient.pg_agent_config import PolicyGradientAgentConfig

class PolicyGradientAgent(TrainAgent, ABC):
    """
    Abstract PolicyGradient Agent
    """
    def __init__(self, env:IdsGameEnv, config: PolicyGradientAgentConfig):
        """
        Initialize environment and hyperparameters

        :param config: the configuration
        """
        self.env = env
        self.config = config
        self.train_result = ExperimentResult()
        self.eval_result = ExperimentResult()
        self.num_eval_games_total = 0
        self.num_eval_hacks_total = 0
        self.num_eval_games = 0
        self.num_eval_hacks = 0
        self.num_train_games = 0
        self.num_train_hacks = 0
        self.num_train_games_total = 0
        self.num_train_hacks_total = 0
        self.train_hack_probability = 0.0
        self.train_cumulative_hack_probability = 0.0
        self.eval_hack_probability = 0.0
        self.eval_cumulative_hack_probability = 0.0
        self.eval_attacker_cumulative_reward = 0
        self.eval_defender_cumulative_reward = 0
        self.outer_train = tqdm.tqdm(total=self.config.num_episodes, desc='Train Episode', position=0)
        if self.config.logger is None:
            self.config.logger = logging.getLogger('PolicyGradient Agent')
        random.seed(self.config.random_seed)
        np.random.seed(self.config.random_seed)
        torch.manual_seed(self.config.random_seed)

    def log_action_dist(self, dist, attacker = True):
        suffix = "[Attacker]"
        if not attacker:
            suffix = "[Defender]"
        log_str = suffix + " Initial State Action Dist: ["
        dist_str = ",".join(list(map(lambda x: str(x), dist.data.cpu().numpy().tolist())))
        log_str = log_str + dist_str + "]"
        self.config.logger.info(log_str)

    def log_metrics(self, episode: int, result: ExperimentResult, attacker_episode_rewards: list,
                    defender_episode_rewards: list,
                    episode_steps: list, episode_avg_attacker_loss: list = None,
                    episode_avg_defender_loss: list = None,
                    eval: bool = False,
                    update_stats : bool = True, lr_attacker: float = None, lr_defender: float = None,
                    train_attacker : bool = False,
                    train_defender : bool = False, a_pool: int = 0, d_pool : int = 0, total_num_batches : int = 0) -> None:
        """
        Logs average metrics for the last <self.config.log_frequency> episodes

        :param episode: the episode
        :param result: the result object to add the results to
        :param attacker_episode_rewards: list of attacker episode rewards for the last <self.config.log_frequency> episodes
        :param defender_episode_rewards: list of defender episode rewards for the last <self.config.log_frequency> episodes
        :param episode_steps: list of episode steps for the last <self.config.log_frequency> episodes
        :param episode_avg_attacker_loss: list of episode attacker loss for the last <self.config.log_frequency> episodes
        :param episode_avg_defender_loss: list of episode defedner loss for the last <self.config.log_frequency> episodes
        :param eval: boolean flag whether the metrics are logged in an evaluation context.
        :param update_stats: boolean flag whether to update stats
        :param lr_attacker: the learning rate of the attacker
        :param lr_defender: the learning rate of the defender
        :param train_attacker: boolean flag indicating whether the attacker is being trained
        :param train_defender: boolean flag indicating whether the defender is being trained
        :param a_pool: size of the attacker pool (if using opponent pools)
        :param d_pool: size of the defender pool (if using opponent pools)
        :param total_num_batches: number of training batches
        :return: None
        """
        avg_attacker_episode_rewards = np.mean(attacker_episode_rewards)
        avg_defender_episode_rewards = np.mean(defender_episode_rewards)
        if lr_attacker is None:
            lr_attacker = 0.0
        if lr_defender is None:
            lr_defender = 0.0
        if not eval and episode_avg_attacker_loss is not None:
            avg_episode_attacker_loss = np.mean(episode_avg_attacker_loss)
        else:
            avg_episode_attacker_loss = 0.0
        if not eval and episode_avg_defender_loss is not None:
            avg_episode_defender_loss = np.mean(episode_avg_defender_loss)
        else:
            avg_episode_defender_loss = 0.0

        avg_episode_steps = np.mean(episode_steps)
        hack_probability = self.train_hack_probability if not eval else self.eval_hack_probability
        hack_probability_total = self.train_cumulative_hack_probability if not eval else self.eval_cumulative_hack_probability
        attacker_cumulative_reward = self.env.state.attacker_cumulative_reward if not eval \
            else self.eval_attacker_cumulative_reward
        defender_cumulative_reward = self.env.state.defender_cumulative_reward if not eval \
            else self.eval_defender_cumulative_reward
        if eval:
            log_str = "[Eval] episode:{},avg_a_R:{:.2f},avg_d_R:{:.2f},avg_t:{:.2f},avg_h:{:.2f},acc_A_R:{:.2f}," \
                      "acc_D_R:{:.2f},lr_a:{:.2E},lr_d:{:.2E},c_h:{:.2f}".format(
                episode, avg_attacker_episode_rewards, avg_defender_episode_rewards, avg_episode_steps, hack_probability,
                attacker_cumulative_reward, defender_cumulative_reward, lr_attacker, lr_defender,
                hack_probability_total)
            self.outer_eval.set_description_str(log_str)
        else:
            log_str = "[Train] episode: {:.2f} epsilon:{:.2f},avg_a_R:{:.2f},avg_d_R:{:.2f},avg_t:{:.2f},avg_h:{:.2f},acc_A_R:{:.2f}," \
                      "acc_D_R:{:.2f},A_loss:{:.6f},D_loss:{:.6f},lr_a:{:.2E},lr_d:{:.2E},c_h:{:.2f},Tr_A:{},Tr_D:{}," \
                      "a_pool:{},d_pool:{},batch:{}".format(
                episode, self.config.epsilon, avg_attacker_episode_rewards, avg_defender_episode_rewards,
                avg_episode_steps, hack_probability, attacker_cumulative_reward, defender_cumulative_reward,
                avg_episode_attacker_loss, avg_episode_defender_loss, lr_attacker, lr_defender, hack_probability_total,
                train_attacker,
                train_defender,a_pool, d_pool, total_num_batches)
            self.outer_train.set_description_str(log_str)
        self.config.logger.info(log_str)
        if update_stats and self.config.tensorboard:
            self.log_tensorboard(episode, avg_attacker_episode_rewards, avg_defender_episode_rewards, avg_episode_steps,
                                 avg_episode_attacker_loss, avg_episode_defender_loss, hack_probability,
                                 attacker_cumulative_reward, defender_cumulative_reward, self.config.epsilon, lr_attacker,
                                 lr_defender, hack_probability_total, a_pool, d_pool, eval=eval)
        if update_stats:
            result.avg_episode_steps.append(avg_episode_steps)
            result.avg_attacker_episode_rewards.append(avg_attacker_episode_rewards)
            result.avg_defender_episode_rewards.append(avg_defender_episode_rewards)
            result.epsilon_values.append(self.config.epsilon)
            result.hack_probability.append(hack_probability)
            result.cumulative_hack_probabiltiy.append(hack_probability_total)
            result.attacker_cumulative_reward.append(attacker_cumulative_reward)
            result.defender_cumulative_reward.append(defender_cumulative_reward)
            result.avg_episode_loss_attacker.append(avg_episode_attacker_loss)
            result.avg_episode_loss_defender.append(avg_episode_defender_loss)
            result.lr_list.append(lr_attacker)

    def log_tensorboard(self, episode: int, avg_attacker_episode_rewards: float, avg_defender_episode_rewards: float,
                        avg_episode_steps: float, episode_avg_loss_attacker: float, episode_avg_loss_defender: float,
                        hack_probability: float, attacker_cumulative_reward: int, defender_cumulative_reward: int,
                        epsilon: float, lr_attacker: float, lr_defender: float, cumulative_hack_probability : float,
                        a_pool : int, d_pool : int, eval=False) -> None:
        """
        Log metrics to tensorboard

        :param episode: the episode
        :param avg_attacker_episode_rewards: the average attacker episode reward
        :param avg_defender_episode_rewards: the average defender episode reward
        :param avg_episode_steps: the average number of episode steps
        :param episode_avg_loss_attacker: the average episode loss of the attacker
        :param episode_avg_loss_defender: the average episode loss of the defender
        :param hack_probability: the hack probability
        :param attacker_cumulative_reward: the cumulative attacker reward
        :param defender_cumulative_reward: the cumulative defender reward
        :param epsilon: the exploration rate
        :param lr_attacker: the learning rate of the attacker
        :param lr_defender: the learning rate of the defender
        :param cumulative_hack_probability: the cumulative hack probability
        :param eval: boolean flag whether eval or not
        :param a_pool: size of the attacker opponent pool
        :param d_pool: size of the defender opponent pool
        :return: None
        """
        train_or_eval = "eval" if eval else "train"
        self.tensorboard_writer.add_scalar('avg_episode_rewards/' + train_or_eval + "/attacker",
                                           avg_attacker_episode_rewards, episode)
        self.tensorboard_writer.add_scalar('avg_episode_rewards/' + train_or_eval + "/defender",
                                           avg_defender_episode_rewards, episode)
        self.tensorboard_writer.add_scalar('episode_steps/' + train_or_eval, avg_episode_steps, episode)
        self.tensorboard_writer.add_scalar('episode_avg_loss/' + train_or_eval + "/attacker", episode_avg_loss_attacker,
                                           episode)
        self.tensorboard_writer.add_scalar('episode_avg_loss/' + train_or_eval + "/defender", episode_avg_loss_defender,
                                           episode)
        self.tensorboard_writer.add_scalar('hack_probability/' + train_or_eval, hack_probability, episode)
        self.tensorboard_writer.add_scalar('cumulative_hack_probability/' + train_or_eval, cumulative_hack_probability, episode)
        self.tensorboard_writer.add_scalar('cumulative_reward/attacker/' + train_or_eval,
                                           attacker_cumulative_reward, episode)
        self.tensorboard_writer.add_scalar('cumulative_reward/defender/' + train_or_eval,
                                           defender_cumulative_reward, episode)
        self.tensorboard_writer.add_scalar('epsilon', epsilon, episode)
        if self.config.opponent_pool and a_pool is not None and d_pool is not None and not eval:
            self.tensorboard_writer.add_scalar('opponent_pool_size/attacker', a_pool, episode)
            self.tensorboard_writer.add_scalar('opponent_pool_size/defender', d_pool, episode)
        if not eval:
            self.tensorboard_writer.add_scalar('lr/attacker', lr_attacker, episode)
            self.tensorboard_writer.add_scalar('lr/defender', lr_defender, episode)

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
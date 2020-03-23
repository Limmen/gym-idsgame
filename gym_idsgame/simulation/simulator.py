import tqdm
import logging
import time
import numpy as np
from gym_idsgame.envs.idsgame_env import IdsGameEnv
from gym_idsgame.agents.dao.experiment_result import ExperimentResult
from gym_idsgame.simulation.dao.simulation_config import SimulationConfig
from gym_idsgame.envs.rendering.video.idsgame_monitor import IdsGameMonitor

class Simulator:

    def __init__(self, env: IdsGameEnv, config: SimulationConfig):
        self.config = config
        self.env = env
        self.experiment_result = ExperimentResult()
        self.outer = tqdm.tqdm(total=self.config.num_episodes, desc='Episode', position=0)
        if self.config.logger is None:
            self.config.logger = logging.getLogger('Simulation')
        self.attacker = self.env.idsgame_config.attacker_agent
        self.defender = self.env.idsgame_config.defender_agent

    def simulate(self):
        self.config.logger.info("Starting Simulation")
        time_str = str(time.time())

        if len(self.experiment_result.avg_episode_steps) > 0:
            self.config.logger.warning("starting simulation with a non-empty result object")
        if self.config.num_episodes < 1:
            return
        done = False

        # Video config
        if self.config.video:
            if self.config.video_dir is None:
                raise AssertionError("Video is set to True but no video_dir is provided, please specify "
                                     "the video_dir argument")
            self.env = IdsGameMonitor(self.env, self.config.video_dir + "/" + time_str, force=True,
                                      video_frequency=self.config.video_frequency)
            self.env.metadata["video.frames_per_second"] = self.config.video_fps

        # Tracking metrics
        episode_steps = []

        # Simulate
        obs = self.env.reset()
        for episode in range(self.config.num_episodes):
            i = 0
            episode_step = 0
            while not done:
                if self.config.render:
                    self.env.render()
                    time.sleep(self.config.sleep)
                i = i + 1
                attacker_node_id = self.env.get_attacker_node_from_observation(obs)
                attacker_node_pos = self.env.idsgame_config.game_config.network_config.get_node_pos(attacker_node_id)
                defense_id = self.defender.action(attacker_node_pos)
                attack_id = self.attacker.action(attacker_node_pos)
                obs, _, done, _ = self.env.step((attack_id, defense_id))
                episode_step += 1
            if self.config.render:
                self.env.render()
                time.sleep(self.config.sleep)
            self.config.logger.info("Simulation episode: {}, Game ended after {} steps".format(episode, i))
            episode_steps.append(episode_step)

            # Log average metrics every <self.config.eval_log_frequency> episodes
            if episode % self.config.log_frequency == 0:
                self.log_metrics(self.experiment_result, episode_steps)
                episode_steps = []
            if self.config.gifs and self.config.video:
                self.env.generate_gif(self.config.gif_dir + "/episode_" + str(episode) + "_"
                                      + time_str + ".gif", self.config.video_fps)

            done = False
            obs = self.env.reset()
            self.outer.update(1)

        self.env.close()
        self.config.logger.info("Simulation Complete")
        return self.experiment_result

    def log_metrics(self, result: ExperimentResult, episode_steps: list) -> None:
        """
        Logs average metrics for the last <self.config.log_frequency> episodes

        :param result: the result object to add the results to
        :param episode_steps: list of episode steps for the last <self.config.log_frequency> episodes
        :return: None
        """
        avg_episode_steps = np.mean(episode_steps)
        log_str = "avg_t:{:.2f},avg_h:{:.2f},acc_A_R:{:.2f}," \
                  "acc_D_R:{:.2f}".format(avg_episode_steps,
                                          self.env.hack_probability(),
                                          self.env.state.attacker_cumulative_reward,
                                          self.env.state.defender_cumulative_reward)
        self.outer.set_description_str(log_str)
        self.config.logger.info(log_str)
        result.avg_episode_steps.append(avg_episode_steps)
        result.hack_probability.append(self.env.hack_probability())
        result.attacker_cumulative_reward.append(self.env.state.attacker_cumulative_reward)
        result.defender_cumulative_reward.append(self.env.state.defender_cumulative_reward)

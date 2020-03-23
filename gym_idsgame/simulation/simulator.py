import tqdm
import logging
import time
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
        pass

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
            self.env = IdsGameMonitor(self.env, self.config.video_dir + "/" + time_str, force=True)
            self.env.metadata["video.frames_per_second"] = self.config.video_fps


        # Tracking metrics
        episode_rewards = []
        episode_steps = []

        # Simulate
        obs = self.env.reset()
        for episode in range(self.config.eval_episodes):
            i = 0
            episode_reward = 0
            episode_step = 0
            while not done:
                if self.config.eval_render:
                    self.env.render()
                    time.sleep(self.config.eval_sleep)
                i = i + 1
                attacker_node_id = self.env.get_attacker_node_from_observation(obs)
                action = self.get_action(attacker_node_id, eval=True)
                obs, reward, done, _ = self.env.step(action)
                episode_reward += reward
                episode_step += 1
            if self.config.eval_render:
                self.env.render()
                time.sleep(self.config.eval_sleep)
            self.config.logger.info("Eval episode: {}, Game ended after {} steps".format(episode, i))
            episode_rewards.append(episode_reward)
            episode_steps.append(episode_step)

            # Log average metrics every <self.config.eval_log_frequency> episodes
            if episode % self.config.eval_log_frequency == 0:
                self.log_metrics(self.eval_result, episode_rewards, episode_steps)
                episode_rewards = []
                episode_steps = []
            if self.config.gifs and self.config.video:
                self.env.generate_gif(self.config.gif_dir + "/episode_" + str(episode) + "_"
                                      + time_str + ".gif", self.config.video_fps)

            done = False
            obs = self.env.reset()
        self.env.close()
        self.config.logger.info("Evaluation Complete")
        return self.eval_result

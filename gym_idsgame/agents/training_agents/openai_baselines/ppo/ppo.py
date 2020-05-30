"""
An agent for the IDSGameEnv that uses the PPO Policy Gradient algorithm from OpenAI stable baselines
"""
import time
import torch
import math
from gym_idsgame.envs.rendering.video.idsgame_monitor import IdsGameMonitor
from gym_idsgame.agents.training_agents.openai_baselines.common.baseline_env_wrapper import BaselineEnvWrapper
from gym_idsgame.agents.dao.experiment_result import ExperimentResult
from gym_idsgame.agents.training_agents.policy_gradient.pg_agent import PolicyGradientAgent
from gym_idsgame.agents.training_agents.policy_gradient.pg_agent_config import PolicyGradientAgentConfig
from gym_idsgame.agents.training_agents.openai_baselines.common.ppo.ppo import PPO

class OpenAiPPOAgent(PolicyGradientAgent):
    """
    An agent for the IDSGameEnv that uses the PPO Policy Gradient algorithm from OpenAI stable baselines
    """

    def __init__(self, env: BaselineEnvWrapper, config: PolicyGradientAgentConfig):
        """
        Initialize environment and hyperparameters

        :param config: the configuration
        """
        super(OpenAiPPOAgent, self).__init__(env, config)


    def train(self) -> ExperimentResult:
        """
        Starts the training loop and returns the result when complete

        :return: the training result
        """
        self.env.idsgame_env.idsgame_config.render_config.attacker_view = self.config.render_attacker_view
        # Custom MLP policy
        net_arch = []
        for l in range(self.config.num_hidden_layers):
            net_arch.append(self.config.hidden_dim)
        policy_kwargs = dict(activation_fn=self.get_hidden_activation(), net_arch=net_arch)
        device = "cpu" if not self.config.gpu else "cuda:" + str(self.config.gpu_id)
        policy = "MlpPolicy"
        if self.config.cnn_feature_extractor:
            policy = "CnnPolicy"
        print("policy:{}".format(policy))

        if self.config.lr_progress_decay:
            temp = self.config.alpha_attacker
            lr_decay_func = lambda x: temp*math.pow(x, self.config.lr_progress_power_decay)
            self.config.alpha_attacker = lr_decay_func

        model = PPO(policy, self.env,
                    learning_rate=self.linear_schedule,
                    n_steps=self.config.batch_size,
                    n_epochs=self.config.optimization_iterations,
                    gamma=self.config.gamma,
                    gae_lambda=self.config.gae_lambda,
                    clip_range=self.config.eps_clip,
                    max_grad_norm=self.config.max_gradient_norm,
                    verbose=1, tensorboard_log=self.config.tensorboard_dir,
                    seed=self.config.random_seed,
                    policy_kwargs=policy_kwargs,
                    device=device,
                    pg_agent_config=self.config,
                    vf_coef=self.config.vf_coef,
                    ent_coef=self.config.ent_coef)
        if self.config.attacker_load_path is not None:
            PPO.load(self.config.attacker_load_path, policy)


        # Video config
        if self.config.video:
            time_str = str(time.time())
            if self.config.video_dir is None:
                raise AssertionError("Video is set to True but no video_dir is provided, please specify "
                                     "the video_dir argument")
            eval_env = IdsGameMonitor(self.env, self.config.video_dir + "/" + time_str, force=True,
                                      video_frequency=self.config.video_frequency, openai_baseline=True)
            eval_env.metadata["video.frames_per_second"] = self.config.video_fps

        model.learn(total_timesteps=self.config.num_episodes,
                    log_interval=self.config.train_log_frequency,
                    eval_freq=self.config.eval_frequency,
                    n_eval_episodes=self.config.eval_episodes,
                    eval_env=eval_env)

        self.config.logger.info("Training Complete")

        # Save networks
        model.save_model()

        # Save other game data
        if self.config.save_dir is not None:
            time_str = str(time.time())
            model.train_result.to_csv(self.config.save_dir + "/" + time_str + "_train_results_checkpoint.csv")
            model.eval_result.to_csv(self.config.save_dir + "/" + time_str + "_eval_results_checkpoint.csv")

        self.train_result = model.train_result
        self.eval_result = model.eval_result
        return model.train_result

    def linear_schedule(self, initial_value):
        """
        Linear learning rate schedule.
        :param initial_value: (float or str)
        :return: (function)
        """
        if isinstance(initial_value, str):
            initial_value = float(initial_value)

        def func(progress):
            """
            Progress will decrease from 1 (beginning) to 0
            :param progress: (float)
            :return: (float)
            """
            print("progress:{}".format(progress))
            return progress * initial_value

        return func


    def get_hidden_activation(self):
        """
        Interprets the hidden activation

        :return: the hidden activation function
        """
        if self.config.hidden_activation == "ReLU":
            return torch.nn.ReLU
        elif self.config.hidden_activation == "LeakyReLU":
            return torch.nn.LeakyReLU
        elif self.config.hidden_activation == "LogSigmoid":
            return torch.nn.LogSigmoid
        elif self.config.hidden_activation == "PReLU":
            return torch.nn.PReLU
        elif self.config.hidden_activation == "Sigmoid":
            return torch.nn.Sigmoid
        elif self.config.hidden_activation == "Softplus":
            return torch.nn.Softplus
        elif self.config.hidden_activation == "Tanh":
            return torch.nn.Tanh
        else:
            raise ValueError("Activation type: {} not recognized".format(self.config.hidden_activation))

    def get_action(self, s, eval=False, attacker=True) -> int:
        raise NotImplemented("not implemented")

    def eval(self, log=True) -> ExperimentResult:
        raise NotImplemented("not implemented")
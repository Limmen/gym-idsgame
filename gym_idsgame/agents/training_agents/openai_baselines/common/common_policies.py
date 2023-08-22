from typing import Union, Type, Dict, List, Tuple, Optional, Any

from itertools import zip_longest

import gymnasium as gym
import torch as th
import torch.nn as nn
import numpy as np

from stable_baselines3.common.preprocessing import preprocess_obs, get_flattened_obs_dim, is_image_space
from gym_idsgame.agents.training_agents.openai_baselines.common.utils import get_device
from gym_idsgame.agents.training_agents.openai_baselines.common.vec_env.vec_transpose import VecTransposeImage
from gym_idsgame.agents.training_agents.policy_gradient.pg_agent_config import PolicyGradientAgentConfig
from gym_idsgame.agents.training_agents.models.idsgame_resnet import IdsGameResNet

class BaseFeaturesExtractor(nn.Module):
    """
    Base class that represents a features extractor.

    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
    """

    def __init__(self, pg_agent_config : PolicyGradientAgentConfig, observation_space: gym.Space, features_dim: int = 0,
                 node_net : bool = False, at_net : bool = False):
        super(BaseFeaturesExtractor, self).__init__()
        assert features_dim > 0
        self._observation_space = observation_space
        self._features_dim = features_dim
        self.pg_agent_config = pg_agent_config
        self.node_net = node_net
        self.at_net = at_net

    @property
    def features_dim(self) -> int:
        return self._features_dim

    def forward(self, observations: th.Tensor) -> th.Tensor:
        raise NotImplementedError()


class FlattenExtractor(BaseFeaturesExtractor):
    """
    Feature extract that flatten the input.
    Used as a placeholder when feature extraction is not needed.

    :param observation_space: (gym.Space)
    """

    def __init__(self, pg_agent_config : PolicyGradientAgentConfig, observation_space: gym.Space,
                 node_net : bool = False, at_net : bool = False):
        super(FlattenExtractor, self).__init__(pg_agent_config, observation_space,
                                               get_flattened_obs_dim(observation_space), at_net=at_net,
                                               node_net=node_net)
        self.flatten = nn.Flatten()

    def forward(self, observations: th.Tensor) -> th.Tensor:
        if len(observations.shape) > 1:
            return self.flatten(observations)
        else:
            return observations


class NatureCNN(BaseFeaturesExtractor):
    """
    CNN from DQN nature paper: https://arxiv.org/abs/1312.5602

    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, pg_agent_config : PolicyGradientAgentConfig, observation_space: gym.spaces.Box,
                 features_dim: int = 512, node_net : bool = False, at_net : bool = False):
        super(NatureCNN, self).__init__(pg_agent_config, observation_space, pg_agent_config.features_dim,
                                        at_net=at_net, node_net=node_net)

        n_input_channels = observation_space.shape[0]
        if pg_agent_config.cnn_type == 0:
            self.cnn = nn.Sequential(nn.Conv2d(n_input_channels, out_channels=2, kernel_size=1, stride=1, padding=0),
                                     nn.ReLU(),
                                     nn.Conv2d(in_channels=2, out_channels=2, kernel_size=2, stride=1, padding=0),
                                     nn.ReLU(),
                                     nn.Conv2d(in_channels=2, out_channels=2, kernel_size=2, stride=1, padding=0),
                                     nn.ReLU(),
                                     nn.Flatten())
        elif pg_agent_config.cnn_type == 1:
            self.cnn = nn.Sequential(nn.Conv2d(n_input_channels, out_channels=8, kernel_size=1, stride=1, padding=0),
                                     nn.ReLU(),
                                     nn.Conv2d(in_channels=8, out_channels=8, kernel_size=1, stride=1, padding=0),
                                     nn.ReLU(),
                                     nn.Conv2d(in_channels=8, out_channels=8, kernel_size=1, stride=1, padding=0),
                                     nn.ReLU(),
                                     nn.Flatten())
        elif pg_agent_config.cnn_type == 2:
            self.cnn = nn.Sequential(nn.Conv2d(n_input_channels, out_channels=64, kernel_size=1, stride=1, padding=0),
                                     nn.MaxPool2d(kernel_size=2, stride=1, padding=0),
                                     nn.ReLU(),
                                     nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0),
                                     nn.MaxPool2d(kernel_size=2, stride=1, padding=0),
                                     nn.ReLU(),
                                     nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0),
                                     nn.MaxPool2d(kernel_size=1, stride=1, padding=0),
                                     nn.ReLU(),
                                     nn.Flatten())
        elif pg_agent_config.cnn_type == 3:
            self.cnn = nn.Sequential(nn.Conv2d(n_input_channels, out_channels=64, kernel_size=1, stride=1, padding=0),
                                     nn.MaxPool2d(kernel_size=1, stride=1, padding=0),
                                     nn.ReLU(),
                                     nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0),
                                     nn.MaxPool2d(kernel_size=1, stride=1, padding=0),
                                     nn.ReLU(),
                                     nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0),
                                     nn.MaxPool2d(kernel_size=1, stride=1, padding=0),
                                     nn.ReLU(),
                                     nn.Flatten())

        elif pg_agent_config.cnn_type == 4:
            self.cnn = IdsGameResNet(in_channels=6, output_dim=44)
        elif pg_agent_config.cnn_type == 5:
            self.cnn = nn.Sequential(nn.Conv2d(n_input_channels, out_channels=64, kernel_size=1, stride=1, padding=0),
                                     nn.ReLU(),
                                     nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0),
                                     nn.ReLU(),
                                     nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0),
                                     nn.ReLU(),
                                     nn.Flatten())

        elif pg_agent_config.cnn_type == 6:
            self.cnn = nn.Sequential(nn.Conv2d(n_input_channels, out_channels=2, kernel_size=3, stride=1, padding=0),
                                           nn.ReLU(),
                                           nn.Conv2d(in_channels=2, out_channels=2, kernel_size=3, stride=1,
                                                           padding=0),
                                           nn.ReLU(),
                                           nn.Conv2d(in_channels=2, out_channels=2, kernel_size=2, stride=1,
                                                           padding=0),
                                           nn.ReLU(),
                                           nn.Conv2d(in_channels=2, out_channels=2, kernel_size=1, stride=1,
                                                           padding=0),
                                           nn.ReLU(),
                                           nn.Flatten())

        if not self.pg_agent_config.cnn_type == 4:
            # Compute shape by doing one forward pass
            with th.no_grad():
                n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1]

            self.linear = nn.Sequential(nn.Linear(n_flatten, pg_agent_config.features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        if len(observations.shape) == 3:
            observations = observations.unsqueeze(0)
        if not self.pg_agent_config.cnn_type == 4:
            return self.linear(self.cnn(observations))
        else:
            return self.cnn(observations)


class BasePolicy(nn.Module):
    """
    The base policy object

    :param observation_space: (gym.spaces.Space) The observation space of the environment
    :param action_space: (gym.spaces.Space) The action space of the environment
    :param device: (Union[th.device, str]) Device on which the code should run.
    :param squash_output: (bool) For continuous actions, whether the output is squashed
        or not using a ``tanh()`` function.
    :param features_extractor_class: (Type[BaseFeaturesExtractor]) Features extractor to use.
    :param features_extractor_kwargs: (Optional[Dict[str, Any]]) Keyword arguments
        to pass to the feature extractor.
    :param features_extractor: (nn.Module) Network to extract features
        (a CNN when using images, a nn.Flatten() layer otherwise)
    :param normalize_images: (bool) Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: (Type[th.optim.Optimizer]) The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: (Optional[Dict[str, Any]]) Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """

    def __init__(self, pg_agent_config : PolicyGradientAgentConfig, observation_space: gym.spaces.Space,
                 action_space: gym.spaces.Space,
                 device: Union[th.device, str] = 'auto',
                 features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
                 features_extractor_kwargs: Optional[Dict[str, Any]] = None,
                 features_extractor: Optional[nn.Module] = None,
                 normalize_images: bool = False,
                 optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
                 optimizer_kwargs: Optional[Dict[str, Any]] = None,
                 squash_output: bool = False,
                 node_net : bool = False,
                 at_net : bool = False):
        super(BasePolicy, self).__init__()
        if optimizer_kwargs is None:
            optimizer_kwargs = {}

        if features_extractor_kwargs is None:
            features_extractor_kwargs = {}
        self.node_net = node_net
        self.at_net = at_net
        self.pg_agent_config = pg_agent_config
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = get_device(device, self.pg_agent_config)
        self.features_extractor = features_extractor
        self.normalize_images = normalize_images
        self._squash_output = squash_output

        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs
        self.optimizer = None  # type: Optional[th.optim.Optimizer]

        self.features_extractor_class = features_extractor_class
        self.features_extractor_kwargs = features_extractor_kwargs

    def extract_features(self, obs: th.Tensor) -> th.Tensor:
        """
        Preprocess the observation if needed and extract features.

        :param obs: (th.Tensor)
        :return: (th.Tensor)
        """
        assert self.features_extractor is not None, 'No feature extractor was set'
        preprocessed_obs = preprocess_obs(obs, self.observation_space, normalize_images=self.normalize_images)

        return self.features_extractor(preprocessed_obs)

    @property
    def squash_output(self) -> bool:
        """ (bool) Getter for squash_output."""
        return self._squash_output

    @staticmethod
    def init_weights(module: nn.Module, gain: float = 1) -> None:
        """
        Orthogonal initialization (used in PPO and A2C)
        """
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(module.weight, gain=gain)
            module.bias.data.fill_(0.0)

    @staticmethod
    def _dummy_schedule(_progress: float) -> float:
        """ (float) Useful for pickling policy."""
        return 0.0

    def forward(self, *_args, **kwargs):
        raise NotImplementedError()

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        """
        Get the action according to the policy for a given observation.

        :param observation: (th.Tensor)
        :param deterministic: (bool) Whether to use stochastic or deterministic actions
        :return: (th.Tensor) Taken action according to the policy
        """
        raise NotImplementedError()

    def predict(self, observation: np.ndarray,
                state: Optional[np.ndarray] = None,
                mask: Optional[np.ndarray] = None,
                deterministic: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Get the policy action and state from an observation (and optional state).

        :param observation: (np.ndarray) the input observation
        :param state: (Optional[np.ndarray]) The last states (can be None, used in recurrent policies)
        :param mask: (Optional[np.ndarray]) The last masks (can be None, used in recurrent policies)
        :param deterministic: (bool) Whether or not to return deterministic actions.
        :return: (Tuple[np.ndarray, Optional[np.ndarray]]) the model's action and the next state
            (used in recurrent policies)
        """
        # if state is None:
        #     state = self.initial_state
        # if mask is None:
        #     mask = [False for _ in range(self.n_envs)]
        observation = np.array(observation)

        # Handle the different cases for images
        # as PyTorch use channel first format
        if is_image_space(self.observation_space):
            if (observation.shape == self.observation_space.shape
                    or observation.shape[1:] == self.observation_space.shape):
                pass
            else:
                # Try to re-order the channels
                transpose_obs = VecTransposeImage.transpose_image(observation)
                if (transpose_obs.shape == self.observation_space.shape
                        or transpose_obs.shape[1:] == self.observation_space.shape):
                    observation = transpose_obs

        vectorized_env = self._is_vectorized_observation(observation, self.observation_space)

        observation = observation.reshape((-1,) + self.observation_space.shape)

        observation = th.as_tensor(observation).to(self.device)
        with th.no_grad():
            actions = self._predict(observation, deterministic=deterministic)
        # Convert to numpy
        actions = actions.cpu().numpy()

        # Rescale to proper domain when using squashing
        if isinstance(self.action_space, gym.spaces.Box) and self.squash_output:
            actions = self.unscale_action(actions)

        clipped_actions = actions
        # Clip the actions to avoid out of bound error when using gaussian distribution
        if isinstance(self.action_space, gym.spaces.Box) and not self.squash_output:
            clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

        if not vectorized_env:
            if state is not None:
                raise ValueError("Error: The environment must be vectorized when using recurrent policies.")
            clipped_actions = clipped_actions[0]

        return clipped_actions, state

    def scale_action(self, action: np.ndarray) -> np.ndarray:
        """
        Rescale the action from [low, high] to [-1, 1]
        (no need for symmetric action space)

        :param action: (np.ndarray) Action to scale
        :return: (np.ndarray) Scaled action
        """
        low, high = self.action_space.low, self.action_space.high
        return 2.0 * ((action - low) / (high - low)) - 1.0

    def unscale_action(self, scaled_action: np.ndarray) -> np.ndarray:
        """
        Rescale the action from [-1, 1] to [low, high]
        (no need for symmetric action space)

        :param scaled_action: Action to un-scale
        """
        low, high = self.action_space.low, self.action_space.high
        return low + (0.5 * (scaled_action + 1.0) * (high - low))

    @staticmethod
    def _is_vectorized_observation(observation: np.ndarray, observation_space: gym.spaces.Space) -> bool:
        """
        For every observation type, detects and validates the shape,
        then returns whether or not the observation is vectorized.

        :param observation: (np.ndarray) the input observation to validate
        :param observation_space: (gym.spaces) the observation space
        :return: (bool) whether the given observation is vectorized or not
        """
        if isinstance(observation_space, gym.spaces.Box):
            if observation.shape == observation_space.shape:
                return False
            elif observation.shape[1:] == observation_space.shape:
                return True
            else:
                raise ValueError(f"Error: Unexpected observation shape {observation.shape} for "
                                 + f"Box environment, please use {observation_space.shape} "
                                 + "or (n_env, {}) for the observation shape."
                                 .format(", ".join(map(str, observation_space.shape))))
        elif isinstance(observation_space, gym.spaces.Discrete):
            if observation.shape == ():  # A numpy array of a number, has shape empty tuple '()'
                return False
            elif len(observation.shape) == 1:
                return True
            else:
                raise ValueError(f"Error: Unexpected observation shape {observation.shape} for "
                                 + "Discrete environment, please use (1,) or (n_env, 1) for the observation shape.")

        elif isinstance(observation_space, gym.spaces.MultiDiscrete):
            if observation.shape == (len(observation_space.nvec),):
                return False
            elif len(observation.shape) == 2 and observation.shape[1] == len(observation_space.nvec):
                return True
            else:
                raise ValueError(f"Error: Unexpected observation shape {observation.shape} for MultiDiscrete "
                                 + f"environment, please use ({len(observation_space.nvec)},) or "
                                 + f"(n_env, {len(observation_space.nvec)}) for the observation shape.")
        elif isinstance(observation_space, gym.spaces.MultiBinary):
            if observation.shape == (observation_space.n,):
                return False
            elif len(observation.shape) == 2 and observation.shape[1] == observation_space.n:
                return True
            else:
                raise ValueError(f"Error: Unexpected observation shape {observation.shape} for MultiBinary "
                                 + f"environment, please use ({observation_space.n},) or "
                                 + f"(n_env, {observation_space.n}) for the observation shape.")
        else:
            raise ValueError("Error: Cannot determine if the observation is vectorized "
                             + f" with the space type {observation_space}.")

    def _get_data(self) -> Dict[str, Any]:
        """
        Get data that need to be saved in order to re-create the policy.
        This corresponds to the arguments of the constructor.

        :return: (Dict[str, Any])
        """
        return dict(
            observation_space=self.observation_space,
            action_space=self.action_space,
            # Passed to the constructor by child class
            # squash_output=self.squash_output,
            # features_extractor=self.features_extractor
            normalize_images=self.normalize_images,
        )

    def save(self, path: str) -> None:
        """
        Save policy to a given location.

        :param path: (str)
        """
        th.save({'state_dict': self.state_dict(), 'data': self._get_data()}, path)

    @classmethod
    def load(cls, path: str, device: Union[th.device, str] = 'auto', pg_agent_config: PolicyGradientAgentConfig = None) -> 'BasePolicy':
        """
        Load policy from path.

        :param path: (str)
        :param device: ( Union[th.device, str]) Device on which the policy should be loaded.
        :return: (BasePolicy)
        """
        device = get_device(device, pg_agent_config)
        saved_variables = th.load(path, map_location=device)
        # Create policy object
        model = cls(**saved_variables['data'])
        # Load weights
        model.load_state_dict(saved_variables['state_dict'])
        model.to(device)
        return model

    def load_from_vector(self, vector: np.ndarray):
        """
        Load parameters from a 1D vector.

        :param vector: (np.ndarray)
        """
        th.nn.utils.vector_to_parameters(th.FloatTensor(vector).to(self.device), self.parameters())

    def parameters_to_vector(self) -> np.ndarray:
        """
        Convert the parameters to a 1D vector.

        :return: (np.ndarray)
        """
        return th.nn.utils.parameters_to_vector(self.parameters()).detach().cpu().numpy()


def create_mlp(input_dim: int,
               output_dim: int,
               net_arch: List[int],
               activation_fn: Type[nn.Module] = nn.ReLU,
               squash_output: bool = False) -> List[nn.Module]:
    """
    Create a multi layer perceptron (MLP), which is
    a collection of fully-connected layers each followed by an activation function.

    :param input_dim: (int) Dimension of the input vector
    :param output_dim: (int)
    :param net_arch: (List[int]) Architecture of the neural net
        It represents the number of units per layer.
        The length of this list is the number of layers.
    :param activation_fn: (Type[nn.Module]) The activation function
        to use after each layer.
    :param squash_output: (bool) Whether to squash the output using a Tanh
        activation function
    :return: (List[nn.Module])
    """
    if len(net_arch) > 0:
        modules = [nn.Linear(input_dim, net_arch[0]), activation_fn()]
    else:
        modules = []

    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
        modules.append(activation_fn())

    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(nn.Linear(last_layer_dim, output_dim))
    if squash_output:
        modules.append(nn.Tanh())
    return modules


def create_sde_features_extractor(features_dim: int,
                                  sde_net_arch: List[int],
                                  activation_fn: Type[nn.Module]) -> Tuple[nn.Sequential, int]:
    """
    Create the neural network that will be used to extract features
    for the gSDE exploration function.

    :param features_dim: (int)
    :param sde_net_arch: ([int])
    :param activation_fn: (Type[nn.Module])
    :return: (nn.Sequential, int)
    """
    # Special case: when using states as features (i.e. sde_net_arch is an empty list)
    # don't use any activation function
    sde_activation = activation_fn if len(sde_net_arch) > 0 else None
    latent_sde_net = create_mlp(features_dim, -1, sde_net_arch, activation_fn=sde_activation, squash_output=False)
    latent_sde_dim = sde_net_arch[-1] if len(sde_net_arch) > 0 else features_dim
    sde_features_extractor = nn.Sequential(*latent_sde_net)
    return sde_features_extractor, latent_sde_dim


_policy_registry = dict()  # type: Dict[Type[BasePolicy], Dict[str, Type[BasePolicy]]]


def get_policy_from_name(base_policy_type: Type[BasePolicy], name: str) -> Type[BasePolicy]:
    """
    Returns the registered policy from the base type and name

    :param base_policy_type: (Type[BasePolicy]) the base policy class
    :param name: (str) the policy name
    :return: (Type[BasePolicy]) the policy
    """
    if base_policy_type not in _policy_registry:
        raise ValueError(f"Error: the policy type {base_policy_type} is not registered!")
    if name not in _policy_registry[base_policy_type]:
        raise ValueError(f"Error: unknown policy type {name},"
                         f"the only registed policy type are: {list(_policy_registry[base_policy_type].keys())}!")
    return _policy_registry[base_policy_type][name]


def register_policy(name: str, policy: Type[BasePolicy]) -> None:
    """
    Register a policy, so it can be called using its name.
    e.g. SAC('MlpPolicy', ...) instead of SAC(MlpPolicy, ...)

    :param name: (str) the policy name
    :param policy: (Type[BasePolicy]) the policy class
    """
    sub_class = None
    for cls in BasePolicy.__subclasses__():
        if issubclass(policy, cls):
            sub_class = cls
            break
    if sub_class is None:
        raise ValueError(f"Error: the policy {policy} is not of any known subclasses of BasePolicy!")

    if sub_class not in _policy_registry:
        _policy_registry[sub_class] = {}
    if name in _policy_registry[sub_class]:
        raise ValueError(f"Error: the name {name} is alreay registered for a different policy, will not override.")
    _policy_registry[sub_class][name] = policy


class MlpExtractor(nn.Module):
    """
    Constructs an MLP that receives observations as an input and outputs a latent representation for the policy and
    a value network. The ``net_arch`` parameter allows to specify the amount and size of the hidden layers and how many
    of them are shared between the policy network and the value network. It is assumed to be a list with the following
    structure:

    1. An arbitrary length (zero allowed) number of integers each specifying the number of units in a shared layer.
       If the number of ints is zero, there will be no shared layers.
    2. An optional dict, to specify the following non-shared layers for the value network and the policy network.
       It is formatted like ``dict(vf=[<value layer sizes>], pi=[<policy layer sizes>])``.
       If it is missing any of the keys (pi or vf), no non-shared layers (empty list) is assumed.

    For example to construct a network with one shared layer of size 55 followed by two non-shared layers for the value
    network of size 255 and a single non-shared layer of size 128 for the policy network, the following layers_spec
    would be used: ``[55, dict(vf=[255, 255], pi=[128])]``. A simple shared network topology with two layers of size 128
    would be specified as [128, 128].

    Adapted from Stable Baselines.

    :param feature_dim: (int) Dimension of the feature vector (can be the output of a CNN)
    :param net_arch: ([int or dict]) The specification of the policy and value networks.
        See above for details on its formatting.
    :param activation_fn: (Type[nn.Module]) The activation function to use for the networks.
    :param device: (th.device)
    """

    def __init__(self, feature_dim: int,
                 net_arch: List[Union[int, Dict[str, List[int]]]],
                 activation_fn: Type[nn.Module],
                 device: Union[th.device, str] = 'auto',
                 pg_agent_config: PolicyGradientAgentConfig = None,
                 at_net :bool = False,
                 node_net :bool = False
                 ):
        super(MlpExtractor, self).__init__()
        self.pg_agent_config = pg_agent_config
        self.at_net = at_net
        self.node_net = node_net
        device = get_device(device, pg_agent_config)
        shared_net, policy_net, value_net = [], [], []
        policy_only_layers = []  # Layer sizes of the network that only belongs to the policy network
        value_only_layers = []  # Layer sizes of the network that only belongs to the value network
        last_layer_dim_shared = feature_dim

        if (not self.pg_agent_config.ar_policy and self.pg_agent_config.multi_channel_obs) or \
            (self.pg_agent_config.ar_policy and self.node_net and self.pg_agent_config.attacker_node_net_multi_channel) or \
                (self.pg_agent_config.ar_policy and self.at_net and self.pg_agent_config.attacker_at_net_multi_channel):
            attack_encoder_net = []
            last_layer_dim_attack_encoder = self.pg_agent_config.channel_1_input_dim
            for i in range(self.pg_agent_config.channel_1_layers):
                attack_encoder_net.append(nn.Linear(last_layer_dim_attack_encoder, self.pg_agent_config.channel_1_dim))
                attack_encoder_net.append(activation_fn())
                last_layer_dim_attack_encoder = self.pg_agent_config.channel_1_dim
            attack_encoder_net.append(nn.Linear(last_layer_dim_attack_encoder, self.pg_agent_config.channel_1_dim))
            self.attack_encoder = nn.Sequential(*attack_encoder_net).to(device)

            defense_encoder_net = []
            last_layer_dim_defense_encoder = self.pg_agent_config.channel_2_input_dim
            for i in range(self.pg_agent_config.channel_2_layers):
                defense_encoder_net.append(nn.Linear(last_layer_dim_defense_encoder, self.pg_agent_config.channel_2_dim))
                defense_encoder_net.append(activation_fn())
                last_layer_dim_defense_encoder = self.pg_agent_config.channel_2_dim
            defense_encoder_net.append(nn.Linear(last_layer_dim_defense_encoder, self.pg_agent_config.channel_2_dim))
            self.defense_encoder = nn.Sequential(*defense_encoder_net).to(device)

            position_encoder_net = []
            last_layer_dim_position_encoder = self.pg_agent_config.channel_3_input_dim
            for i in range(self.pg_agent_config.channel_3_layers):
                position_encoder_net.append(nn.Linear(last_layer_dim_position_encoder, self.pg_agent_config.channel_3_dim))
                position_encoder_net.append(activation_fn())
                last_layer_dim_position_encoder = self.pg_agent_config.channel_3_dim
            position_encoder_net.append(nn.Linear(last_layer_dim_position_encoder, self.pg_agent_config.channel_3_dim))
            self.position_encoder = nn.Sequential(*position_encoder_net).to(device)

            rec_encoder_net = []
            last_layer_dim_rec_encoder = self.pg_agent_config.channel_4_input_dim
            for i in range(self.pg_agent_config.channel_4_layers):
                rec_encoder_net.append(
                    nn.Linear(last_layer_dim_rec_encoder, self.pg_agent_config.channel_4_dim))
                rec_encoder_net.append(activation_fn())
                last_layer_dim_rec_encoder = self.pg_agent_config.channel_4_dim
            rec_encoder_net.append(nn.Linear(last_layer_dim_rec_encoder, self.pg_agent_config.channel_4_dim))
            self.rec_encoder = nn.Sequential(*rec_encoder_net).to(device)

            if (not self.pg_agent_config.ar_policy and self.pg_agent_config.lstm_core) or \
                    (self.pg_agent_config.ar_policy and self.node_net and self.pg_agent_config.attacker_node_net_lstm_core) or \
                    (self.pg_agent_config.ar_policy and self.at_net and self.pg_agent_config.attacker_at_net_lstm_core):
                self.core_lstm = th.nn.LSTM(input_size=(self.pg_agent_config.channel_1_dim +
                                                       self.pg_agent_config.channel_2_dim +
                                                       self.pg_agent_config.channel_3_dim +
                                                       self.pg_agent_config.channel_4_dim),
                                            hidden_size=self.pg_agent_config.lstm_hidden_dim,
                                            num_layers=self.pg_agent_config.num_lstm_layers)
                # initialize the hidden state.
                self.lstm_hidden = (th.zeros(self.pg_agent_config.num_lstm_layers, 1,
                                             self.pg_agent_config.lstm_hidden_dim),
                                    th.zeros(self.pg_agent_config.num_lstm_layers, 1,
                                             self.pg_agent_config.lstm_hidden_dim))
                last_layer_dim_shared = self.pg_agent_config.lstm_hidden_dim
            else:
                last_layer_dim_shared = self.pg_agent_config.channel_1_dim + self.pg_agent_config.channel_2_dim + \
                                        self.pg_agent_config.channel_3_dim + self.pg_agent_config.channel_4_dim
        else:
            if (not self.pg_agent_config.ar_policy and self.pg_agent_config.lstm_core) or \
                    (self.pg_agent_config.ar_policy and self.node_net and self.pg_agent_config.attacker_node_net_lstm_core) or \
                    (self.pg_agent_config.ar_policy and self.at_net and self.pg_agent_config.attacker_at_net_lstm_core):
                self.core_lstm = th.nn.LSTM(input_size=last_layer_dim_shared,
                                            hidden_size=self.pg_agent_config.lstm_hidden_dim,
                                            num_layers=self.pg_agent_config.num_lstm_layers)
                # initialize the hidden state.
                self.lstm_hidden = (th.zeros(self.pg_agent_config.num_lstm_layers, 1,
                                             self.pg_agent_config.lstm_hidden_dim),
                                    th.zeros(self.pg_agent_config.num_lstm_layers, 1,
                                             self.pg_agent_config.lstm_hidden_dim))
                last_layer_dim_shared = self.pg_agent_config.lstm_hidden_dim

        # Iterate through the shared layers and build the shared parts of the network
        for idx, layer in enumerate(net_arch):
            if isinstance(layer, int):  # Check that this is a shared layer
                if not self.pg_agent_config.lstm_core:
                    layer_size = layer
                    shared_net.append(nn.Linear(last_layer_dim_shared, layer_size))
                    shared_net.append(activation_fn())
                    last_layer_dim_shared = layer_size
            else:
                assert isinstance(layer, dict), "Error: the net_arch list can only contain ints and dicts"
                if 'pi' in layer:
                    assert isinstance(layer['pi'], list), "Error: net_arch[-1]['pi'] must contain a list of integers."
                    policy_only_layers = layer['pi']

                if 'vf' in layer:
                    assert isinstance(layer['vf'], list), "Error: net_arch[-1]['vf'] must contain a list of integers."
                    value_only_layers = layer['vf']
                break  # From here on the network splits up in policy and value network

        last_layer_dim_pi = last_layer_dim_shared
        last_layer_dim_vf = last_layer_dim_shared

        # Build the non-shared part of the network
        for idx, (pi_layer_size, vf_layer_size) in enumerate(zip_longest(policy_only_layers, value_only_layers)):
            if pi_layer_size is not None:
                assert isinstance(pi_layer_size, int), "Error: net_arch[-1]['pi'] must only contain integers."
                policy_net.append(nn.Linear(last_layer_dim_pi, pi_layer_size))
                policy_net.append(activation_fn())
                last_layer_dim_pi = pi_layer_size

            if vf_layer_size is not None:
                assert isinstance(vf_layer_size, int), "Error: net_arch[-1]['vf'] must only contain integers."
                value_net.append(nn.Linear(last_layer_dim_vf, vf_layer_size))
                value_net.append(activation_fn())
                last_layer_dim_vf = vf_layer_size

        # Save dim, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Create networks
        # If the list of layers is empty, the network will just act as an Identity module
        if not self.pg_agent_config.lstm_core or \
                (self.pg_agent_config.ar_policy and self.node_net and not self.pg_agent_config.attacker_node_net_lstm_core) or \
                (self.pg_agent_config.ar_policy and self.at_net and not self.pg_agent_config.attacker_at_net_lstm_core):
            self.shared_net = nn.Sequential(*shared_net).to(device)
        else:
            if not self.pg_agent_config.ar_policy or \
                (self.pg_agent_config.ar_policy and self.node_net and self.pg_agent_config.attacker_node_net_lstm_core) or \
                (self.pg_agent_config.ar_policy and self.at_net and self.pg_agent_config.attacker_at_net_lstm_core):
                self.core_lstm = self.core_lstm.to(device)
        self.policy_net = nn.Sequential(*policy_net).to(device)
        self.value_net = nn.Sequential(*value_net).to(device)

    def forward(self, features: th.Tensor, lstm_state = None, masks = None, channel_1_features = None,
                channel_2_features = None, channel_3_features = None, channel_4_features = None) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        if (self.pg_agent_config.multi_channel_obs and not self.pg_agent_config.ar_policy) or \
                (self.pg_agent_config.ar_policy and self.node_net and self.pg_agent_config.attacker_node_net_multi_channel):
            channel_1_latent = self.attack_encoder(channel_1_features.float())
            channel_2_latent = self.defense_encoder(channel_2_features.float())
            channel_3_latent = self.position_encoder(channel_3_features.float())
            channel_4_latent = self.rec_encoder(channel_4_features.float())
            if len(channel_1_latent.shape) == 1:
                features = th.cat([channel_1_latent, channel_2_latent, channel_3_latent, channel_4_latent], dim=0)
            elif len(channel_1_latent.shape) == 2:
                features = th.cat([channel_1_latent, channel_2_latent, channel_3_latent, channel_4_latent], dim=1)
            else:
                raise AssertionError("Do not recognize the shape")
        if not self.pg_agent_config.lstm_core or \
                (self.pg_agent_config.ar_policy and self.node_net and not self.pg_agent_config.attacker_node_net_lstm_core) or \
                (self.pg_agent_config.ar_policy and self.at_net and not self.pg_agent_config.attacker_at_net_lstm_core):
            shared_latent = self.shared_net(features)
            return self.policy_net(shared_latent), self.value_net(shared_latent), None
        else:
            if lstm_state is None:
                if len(features.shape) == 1:
                    lstm_input_latent = features.reshape(1, 1, features.shape[0])
                elif len(features.shape) == 2:
                    lstm_input_latent = features.reshape(1, features.shape[0], features.shape[1])
                lstm_latent, lstm_state = self.core_lstm(lstm_input_latent, self.lstm_hidden)
                self.lstm_hidden = lstm_state
            else:
                outputs = []
                h_states = lstm_state[0][0]
                c_states = lstm_state[1][0]
                hiddden_state = (h_states, c_states)
                for i in range(len(masks)):
                    latent_input = features[i]
                    # print("latent input shape:{}".format(latent_input.shape))
                    # print("features shape:{}".format(features.shape))
                    latent_input = latent_input.reshape(1, 1, features.shape[1])
                    if masks[0] == 1:
                        hiddden_state = (
                            th.zeros(self.pg_agent_config.num_lstm_layers, 1,
                                     self.pg_agent_config.lstm_hidden_dim),
                            th.zeros(self.pg_agent_config.num_lstm_layers, 1,
                                     self.pg_agent_config.lstm_hidden_dim))
                    lstm_latent, hidden_state = self.core_lstm(latent_input, hiddden_state)
                    outputs.append(lstm_latent)

                x = th.cat(outputs, dim=0)
                x = x.reshape((x.shape[0], x.shape[2]))
                lstm_latent = x

            return self.policy_net(lstm_latent), self.value_net(lstm_latent), lstm_state
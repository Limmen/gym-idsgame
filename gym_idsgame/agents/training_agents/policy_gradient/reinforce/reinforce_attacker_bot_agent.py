"""
A bot attack agent for the gym-idsgame environment that acts greedily according to a pre-trained policy network
"""
import numpy as np
import torch
from gym_idsgame.agents.training_agents.models.fnn_w_softmax import FNNwithSoftmax
from gym_idsgame.agents.bot_agents.bot_agent import BotAgent
from gym_idsgame.envs.dao.game_state import GameState
from gym_idsgame.envs.dao.game_config import GameConfig
from gym_idsgame.agents.training_agents.policy_gradient.pg_agent_config import PolicyGradientAgentConfig
from gym_idsgame.agents.training_agents.policy_gradient.pg_agent import PolicyGradientAgent
import gym_idsgame.envs.util.idsgame_util as util
from torch.distributions import Categorical

class ReinforceAttackerBotAgent(BotAgent):
    """
    Class implementing an attack policy that acts greedily according to a given Q-table
    """

    def __init__(self, pg_config: PolicyGradientAgentConfig, game_config: GameConfig, model_path: str = None):
        """
        Constructor, initializes the policy

        :param game_config: the game configuration
        """
        super(ReinforceAttackerBotAgent, self).__init__(game_config)
        if model_path is None:
            raise ValueError("Cannot create a ReinforceAttackerbotAgent without specifying the path to the Q-table")
        self.config = pg_config
        self.model_path = model_path
        self.initialize_models()


    def initialize_models(self) -> None:
        """
        Initialize models
        :return: None
        """

        # Initialize models
        self.attacker_policy_network = FNNwithSoftmax(self.config.input_dim_attacker, self.config.output_dim_attacker,
                                                      self.config.hidden_dim,
                                                      num_hidden_layers=self.config.num_hidden_layers,
                                                      hidden_activation=self.config.hidden_activation)

        # Specify device
        if torch.cuda.is_available() and self.config.gpu:
            device = torch.device("cuda:" + str(self.config.gpu_id))
            #self.config.logger.info("Running on the GPU")
        else:
            device = torch.device("cpu")
            #self.config.logger.info("Running on the CPU")
        self.attacker_policy_network.to(device)

        self.attacker_policy_network.load_state_dict(torch.load(self.model_path))
        self.attacker_policy_network.eval()

    def action(self, game_state: GameState) -> int:
        """
        Samples an action from the policy.

        :param game_state: the game state
        :return: action_id
        """

        # Feature engineering
        attacker_obs = game_state.get_attacker_observation(self.game_config.network_config, local_view=True)
        defender_obs = game_state.get_defender_observation(self.game_config.network_config)
        neighbor_defense_attributes = np.zeros((attacker_obs.shape[0], defender_obs.shape[1]))
        for node in range(attacker_obs.shape[0]):
            if int(attacker_obs[node][-1]) == 1:
                id = int(attacker_obs[node][-2])
                neighbor_defense_attributes[node] = defender_obs[id]
        node_ids = attacker_obs[:, -2]
        node_reachable = attacker_obs[:, -1]
        det_values = neighbor_defense_attributes[:, -1]
        temp = neighbor_defense_attributes[:, 0:-1] - attacker_obs[:, 0:-2]
        features = []
        for idx, row in enumerate(temp):
            t = row.tolist()
            t.append(node_ids[idx])
            t.append(node_reachable[idx])
            t.append(det_values[idx])
            features.append(t)
        features = np.array(features)

        state = torch.from_numpy(features.flatten()).float()
        # Move to GPU if using GPU
        if torch.cuda.is_available() and self.config.gpu:
            device = torch.device("cuda:" + str(self.config.gpu_id))
            state = state.to(device)
        legal_actions, non_legal_actions = self.get_legal_attacker_actions(attacker_obs, game_state)
        # Forward pass using the current policy network to predict P(a|s)
        action_probs = self.attacker_policy_network(state)
        # Set probability of non-legal actions to 0
        action_probs_1 = action_probs.clone()
        if len(legal_actions) > 0 and len(non_legal_actions) < len(action_probs_1):
            action_probs_1[non_legal_actions] = 0
        # Use torch.distributions package to create a parameterizable probability distribution of the learned policy
        policy_dist = Categorical(action_probs_1)
        # Sample an action from the probability distribution
        action = policy_dist.sample()

        global_action = PolicyGradientAgent.convert_local_attacker_action_to_global(action.item(), attacker_obs)
        return global_action

    def get_legal_attacker_actions(self, attacker_obs, state):
        legal_actions = []
        illegal_actions = []
        num_attack_types = attacker_obs[:,0:-2].shape[1]
        for i in range(len(attacker_obs)):
            if int(attacker_obs[i][-1]) == 1:
                for ac in range(num_attack_types):
                    legal_actions.append(i*num_attack_types + ac)
            else:
                for ac in range(num_attack_types):
                    illegal_actions.append(i * num_attack_types + ac)
        legal_actions_2 = []
        for action in legal_actions:
            global_action = PolicyGradientAgent.convert_local_attacker_action_to_global(action, attacker_obs)
            if util.is_attack_id_legal(global_action, self.game_config, state.attacker_pos, state, []):
                legal_actions_2.append(global_action)
            else:
                illegal_actions.append(action)
        return legal_actions_2, illegal_actions



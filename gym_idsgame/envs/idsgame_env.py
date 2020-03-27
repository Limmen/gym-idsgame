"""
RL environment for an intrusion detection Markov game
"""
from typing import Union
import numpy as np
import gym
import os
from abc import ABC, abstractmethod
from gym_idsgame.envs.dao.game_config import GameConfig
from gym_idsgame.agents.random_defense_bot_agent import RandomDefenseBotAgent
from gym_idsgame.agents.random_attack_bot_agent import RandomAttackBotAgent
from gym_idsgame.agents.defend_minimal_value_bot_agent import DefendMinimalValueBotAgent
from gym_idsgame.agents.attack_maximal_value_bot_agent import AttackMaximalValueBotAgent
from gym_idsgame.envs.dao.game_state import GameState
from gym_idsgame.envs.dao.idsgame_config import IdsGameConfig
import gym_idsgame.envs.util.idsgame_util as util
from gym_idsgame.envs.constants import constants

class IdsGameEnv(gym.Env, ABC):
    """
    Implementation of the RL environment from the paper
    "Adversarial Reinforcement Learning in a Cyber Security Simulation" by Edlerman et. al.

    It is an abstract very simple cyber security simulation where there is an attacker agent that tries to penetrate
    a network and a defender agent that tries to defend the network.
    """

    def __init__(self, idsgame_config: IdsGameConfig = None):
        """
        Initializes the environment

        Observation:
            Type: Box(num_nodes*num_attack_types)
        Actions:
            Type: Discrete(num_nodes*num_action_types)
        Reward:
            Reward is 0 for all steps except the final step which is either +100 (win) or -100 (loss)
        Starting State:
            Start node, all attack values are 0
        Episode Termination:
            When attacker reaches DATA node or when attacker is detected

        :param idsgame_config: configuration of the environment
        """
        if idsgame_config is None:
            idsgame_config = IdsGameConfig()
        util.validate_config(idsgame_config)
        self.idsgame_config: IdsGameConfig = idsgame_config
        self.state: GameState = self.idsgame_config.game_config.initial_state.copy()
        self.observation_space = self.idsgame_config.game_config.get_attacker_observation_space()
        self.action_space = self.idsgame_config.game_config.get_action_space()
        self.viewer = None
        self.steps_beyond_done = None
        self.metadata = {
         'render.modes': ['human', 'rgb_array'],
         'video.frames_per_second' : 50 # Video rendering speed
        }
        self.reward_range = (float(constants.GAME_CONFIG.NEGATIVE_REWARD), float(constants.GAME_CONFIG.POSITIVE_REWARD))
        self.num_states = self.idsgame_config.game_config.num_nodes
        self.num_actions = self.idsgame_config.game_config.num_actions

    # -------- API ------------
    def step(self, action: int) -> Union[np.ndarray, int, bool, dict]:
        """
        Takes a step in the environment using the given action.

        When end of episode is reached, the caller is responsible for calling `reset()`
        to reset this environment's state.

        :param action: the action to take in the environment
        :return:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        # Initialization
        reward = 0
        info = {}
        self.state.attack_events = []
        self.state.defense_events = []

        if self.state.game_step > constants.GAME_CONFIG.MAX_GAME_STEPS:
            return self.get_observation(), 0, True, info

        # 1. Interpret attacker action
        attacker_pos = self.state.attacker_pos
        target_node_id, target_pos, attack_type = self.get_attacker_action(action)

        # 2. Interpret defense action
        defense_node_id, defense_pos, defense_type,  = self.get_defender_action(action)

        # 3. Defend
        self.state.defend(defense_node_id, defense_type, self.idsgame_config.game_config.max_value,
                          self.idsgame_config.game_config.network_config)
        self.state.add_defense_event(defense_pos, defense_type)

        if util.is_attack_legal(target_pos, attacker_pos, self.idsgame_config.game_config.network_config):
            # 4. Attack
            self.state.attack(target_node_id, attack_type, self.idsgame_config.game_config.max_value,
                              self.idsgame_config.game_config.network_config)
            self.state.add_attack_event(target_pos, attack_type)

            # 5. Simulate attack outcome
            attack_successful = self.state.simulate_attack(target_node_id, attack_type,
                                                           self.idsgame_config.game_config.network_config)
            # 6. Update state based on attack outcome
            if attack_successful:
                self.state.attacker_pos = target_pos
                if target_pos == self.idsgame_config.game_config.network_config.data_pos:
                    self.state.done = True
                    self.state.hacked = True
                    reward = self.get_hack_reward()
            else:
                detected = self.state.simulate_detection(target_node_id)
                if detected:
                    self.state.done = True
                    self.state.detected = True
                    reward = self.get_detect_reward()
        if self.state.done:
            if self.steps_beyond_done is None:
                self.steps_beyond_done = 0
            else:
                gym.logger.warn(
                    "You are calling 'step()' even though this environment has already returned done = True. "
                    "You should always call 'reset()' once you receive 'done = True' -- "
                    "any further steps are undefined behavior.")
                self.steps_beyond_done += 1
        self.state.game_step += 1
        observation = self.get_observation()
        if self.viewer is not None:
            self.viewer.gameframe.set_state(self.state)
        return observation, reward, self.state.done, info

    def reset(self, update_stats = False) -> np.ndarray:
        """
        Resets the environment and returns the initial state

        :param update_stats: whether the game count should be incremented or not
        :return: the initial state
        """
        self.steps_beyond_done = None
        self.state.new_game(self.idsgame_config.game_config.initial_state, update_stats=update_stats)
        if self.viewer is not None:
            self.viewer.gameframe.reset()
        observation = self.get_observation()
        return observation

    def restart(self) -> np.ndarray:
        """
        Restarts the game, and all the history

        :return: the observation from the first state
        """
        obs = self.reset()
        self.state.restart()
        return obs

    def render(self, mode: str ='human'):
        """
        Renders the environment

        Supported rendering modes:

        - human: render to the current display or terminal and
          return nothing. Usually for human consumption.
        - rgb_array: Return an numpy.ndarray with shape (x, y, 3),
          representing RGB values for an x-by-y pixel image, suitable
          for turning into a video.

        :param mode: the rendering mode
        :return: True (if human mode) otherwise an rgb array
        """
        if mode not in self.metadata["render.modes"]:
            raise NotImplemented("mode: {} is not supported".format(mode))
        if self.viewer is None:
            self.__setup_viewer()
            self.viewer.gameframe.set_state(self.state)
        arr = self.viewer.render(return_rgb_array = mode=='rgb_array')
        self.state.attack_events = []
        self.state.defense_events = []
        return arr

    def close(self) -> None:
        """
        Closes the viewer (cleanup)

        :return: None
        """
        if self.viewer:
            self.viewer.close()
            self.viewer = None
            self.idsgame_config.render_config.new_window()

    def get_attacker_node_from_observation(self, observation: np.ndarray) -> int:
        """
        Extracts which node the attacker is currently at from the observation representation

        :param observation: the observation representation emitted from the environment
        :return: the id of the node that the attacker is in
        """
        return self.state.get_attacker_node_from_observation(observation)

    def hack_probability(self) -> float:
        """
        :return: the hack-probabiltiy according to the game history
        """
        hack_probability = 0.0
        if self.state.num_hacks > 0:
            hack_probability = float(self.state.num_hacks) / float(self.state.num_games)
        return hack_probability

    def is_attack_legal(self, attack_action:int) -> bool:
        """
        Check if a given attack is legal or not.

        :param attack_action: the attack to verify
        :return: True if legal otherwise False
        """
        return util.is_attack_id_legal(attack_action, self.idsgame_config.game_config, self.state.attacker_pos)

    def is_defense_legal(self, defense_action: int) -> bool:
        """
        Check if a given defense is legal or not.

        :param defense_action: the defense action to verify
        :return: True if legal otherwise False
        """
        return util.is_defense_id_legal(defense_action, self.idsgame_config.game_config)

    @abstractmethod
    def get_attacker_action(self, action) -> Union[int, Union[int, int], int]:
        pass

    @abstractmethod
    def get_defender_action(self, action) -> Union[Union[int, int], int, int]:
        pass

    @abstractmethod
    def get_observation(self) -> np.ndarray:
        pass

    @abstractmethod
    def get_hack_reward(self) -> np.ndarray:
        pass

    @abstractmethod
    def get_detect_reward(self) -> np.ndarray:
        pass

    # -------- Private methods ------------

    def __setup_viewer(self):
        """
        Setup for the viewer to use for rendering
        :return: None
        """
        from gym_idsgame.envs.rendering.viewer import Viewer
        script_dir = os.path.dirname(__file__)
        resource_path = os.path.join(script_dir, './rendering/', constants.RENDERING.RESOURCES_DIR)
        self.idsgame_config.render_config.resources_dir = resource_path
        self.viewer = Viewer(idsgame_config=self.idsgame_config)
        self.viewer.agent_start()

class AttackerEnv(IdsGameEnv, ABC):
    """
    Abstract AttackerEnv of the IdsGameEnv.

    Environments where the defender is part of the environment and the environment is designed to be used by an
    attacker-agent should inherit this class
    """

    def __init__(self, idsgame_config: IdsGameConfig):
        if idsgame_config is None:
            raise ValueError("Cannot instantiate env without configuration")
        if idsgame_config.defender_agent is None:
            raise ValueError("Cannot instantiate attacker-env without a defender agent")
        super().__init__(idsgame_config=idsgame_config)
        self.observation_space = self.idsgame_config.game_config.get_attacker_observation_space()

    def get_attacker_action(self, action) -> Union[int, Union[int, int], int]:
        return util.interpret_action(action, self.idsgame_config.game_config)

    def get_defender_action(self, action) -> Union[Union[int, int], int, int]:
        defend_id = self.idsgame_config.defender_agent.action(self.state)
        defend_node_id, defend_node_pos, defend_type = util.interpret_action(
            defend_id, self.idsgame_config.game_config)
        return defend_node_id, defend_node_pos, defend_type

    def get_observation(self) -> np.ndarray:
        return self.state.get_attacker_observation(self.idsgame_config.game_config.network_config)

    def get_hack_reward(self) -> int:
        return constants.GAME_CONFIG.POSITIVE_REWARD

    def get_detect_reward(self) -> int:
        return -constants.GAME_CONFIG.POSITIVE_REWARD

class DefenderEnv(IdsGameEnv, ABC):
    """
    Abstract DefenderEnv of the IdsGameEnv.

    Environments where the attacker is part of the environment and the environment is designed to be used by a
    defender-agent should inherit this class
    """
    def __init__(self, idsgame_config: IdsGameConfig):
        if idsgame_config is None:
            raise ValueError("Cannot instantiate env without configuration")
        if idsgame_config.attacker_agent is None:
            raise ValueError("Cannot instantiate defender-env without an attacker agent")
        super().__init__(idsgame_config=idsgame_config)
        self.observation_space = self.idsgame_config.game_config.get_defender_observation_space()

    def get_defender_action(self, action) -> Union[int, Union[int, int], int]:
        return util.interpret_action(action, self.idsgame_config.game_config)

    def get_attacker_action(self, action) -> Union[Union[int, int], int, int]:
        attack_id = self.idsgame_config.attacker_agent.action(self.state)
        attack_node_id, attack_node_pos, attack_type = util.interpret_action(attack_id, self.idsgame_config.game_config)
        return attack_node_id, attack_node_pos, attack_type

    def get_observation(self) -> np.ndarray:
        return self.state.get_defender_observation(self.idsgame_config.game_config.network_config)

    def get_hack_reward(self) -> int:
        return -constants.GAME_CONFIG.POSITIVE_REWARD

    def get_detect_reward(self) -> int:
        return constants.GAME_CONFIG.POSITIVE_REWARD


class AttackDefenseEnv(IdsGameEnv, ABC):
    """
    Abstract AttacKDefenseEnv of the IdsGameEnv.

    Environments where both the attacker and defender are external to the environment should inherit this class.
    """
    def __init__(self, idsgame_config: IdsGameConfig):
        if idsgame_config is None:
            raise ValueError("Cannot instantiate env without configuration")
        super().__init__(idsgame_config=idsgame_config)

    def get_defender_action(self, action: Union[int, int]) -> Union[int, Union[int, int], int]:
        _, defender_action = action
        return util.interpret_action(defender_action, self.idsgame_config.game_config)

    def get_attacker_action(self, action: Union[int, int]) -> Union[Union[int, int], int, int]:
        attacker_action, _ = action
        return util.interpret_action(attacker_action, self.idsgame_config.game_config)

    def get_observation(self) -> Union[np.ndarray, np.ndarray]:
        attacker_obs = self.state.get_attacker_observation(self.idsgame_config.game_config.network_config)
        defender_obs = self.state.get_defender_observation(self.idsgame_config.game_config.network_config)
        return attacker_obs, defender_obs

    def get_hack_reward(self) -> Union[int, int]:
        return constants.GAME_CONFIG.POSITIVE_REWARD, -constants.GAME_CONFIG.POSITIVE_REWARD

    def get_detect_reward(self) -> Union[int, int]:
        return -constants.GAME_CONFIG.POSITIVE_REWARD, constants.GAME_CONFIG.POSITIVE_REWARD

# -------- Concrete envs ------------

# -------- Version 0 ------------


class IdsGameRandomDefenseV0Env(AttackerEnv):
    """
    [AttackerEnv] 1 layer, 1 server per layer, 10 attack-defense-values, random defender
    [Initial State] Defense: 2, Attack:0, Num vulnerabilities: 1, Det: 2, Vulnerability value: 0
    [Version] 0
    """
    def __init__(self):
        game_config = GameConfig(num_layers=1, num_servers_per_layer=1, num_attack_types=10, max_value=9)
        game_config.set_initial_state(defense_val=2, attack_val=0, num_vulnerabilities_per_node=1, det_val=2,
                                      vulnerability_val=0)
        defender_agent = RandomDefenseBotAgent(game_config)
        idsgame_config = IdsGameConfig(game_config=game_config, defender_agent=defender_agent)
        idsgame_config.render_config.caption = "idsgame-random_defense-v0"
        super().__init__(idsgame_config=idsgame_config)


class IdsGameMinimalDefenseV0Env(AttackerEnv):
    """
    [AttackerEnv] 1 layer, 1 server per layer, 10 attack-defense-values, defender following the "defend minimal strategy"
    [Initial State] Defense: 2, Attack:0, Num vulnerabilities: 1, Det: 2, Vulnerability value: 0
    [Version] 0
    """
    def __init__(self):
        game_config = GameConfig(num_layers=1, num_servers_per_layer=1, num_attack_types=10, max_value=9)
        game_config.set_initial_state(defense_val=2, attack_val=0, num_vulnerabilities_per_node=1, det_val=2,
                                      vulnerability_val=0)
        defender_agent = DefendMinimalValueBotAgent(game_config)
        idsgame_config = IdsGameConfig(game_config=game_config, defender_agent=defender_agent)
        idsgame_config.render_config.caption = "idsgame-minimal_defense-v0"
        super().__init__(idsgame_config=idsgame_config)


class IdsGameRandomAttackV0Env(DefenderEnv):
    """
    [DefenderEnv] 1 layer, 1 server per layer, 10 attack-defense-values, random attacker
    [Initial State] Defense: 2, Attack:0, Num vulnerabilities: 1, Det: 2, Vulnerability value: 0
    [Version] 0
    """
    def __init__(self):
        game_config = GameConfig(num_layers=1, num_servers_per_layer=1, num_attack_types=10, max_value=9)
        game_config.set_initial_state(defense_val=2, attack_val=0, num_vulnerabilities_per_node=1, det_val=2,
                                      vulnerability_val=0)
        attacker_agent = RandomAttackBotAgent(game_config)
        idsgame_config = IdsGameConfig(game_config=game_config, attacker_agent=attacker_agent)
        idsgame_config.render_config.caption = "idsgame-random_attack-v0"
        super().__init__(idsgame_config=idsgame_config)


class IdsGameMaximalAttackV0Env(DefenderEnv):
    """
    [DefenderEnv] 1 layer, 1 server per layer, 10 attack-defense-values, attacker following the "attack maximal strategy"
    [Initial State] Defense: 2, Attack:0, Num vulnerabilities: 1, Det: 2, Vulnerability value: 0
    [Version] 0
    """
    def __init__(self):
        game_config = GameConfig(num_layers=1, num_servers_per_layer=1, num_attack_types=10, max_value=9)
        game_config.set_initial_state(defense_val=2, attack_val=0, num_vulnerabilities_per_node=1, det_val=2,
                                      vulnerability_val=0)
        attacker_agent = AttackMaximalValueBotAgent(game_config)
        idsgame_config = IdsGameConfig(game_config=game_config, attacker_agent=attacker_agent)
        idsgame_config.render_config.caption = "idsgame-maximal_attack-v0"
        super().__init__(idsgame_config=idsgame_config)


class IdsGameV0Env(AttackDefenseEnv):
    """
    [AttackDefenseEnv] 1 layer, 1 server per layer, 10 attack-defense-values
    [Initial State] Defense: 2, Attack:0, Num vulnerabilities: 1, Det: 2, Vulnerability value: 0
    [Version] 0
    """
    def __init__(self):
        game_config = GameConfig(num_layers=1, num_servers_per_layer=1, num_attack_types=10, max_value=9)
        game_config.set_initial_state(defense_val=2, attack_val=0, num_vulnerabilities_per_node=1, det_val=2,
                                      vulnerability_val=0)
        idsgame_config = IdsGameConfig(game_config=game_config)
        idsgame_config.render_config.caption = "idsgame-v0"
        super().__init__(idsgame_config=idsgame_config)


# -------- Version 1 ------------


class IdsGameRandomDefenseV1Env(AttackerEnv):
    """
    [AttackerEnv] 1 layer, 1 server per layer, 10 attack-defense-values, random defender
    [Initial State] Defense: 4, Attack:0, Num vulnerabilities: 4, Det: 3, Vulnerability value: 0
    [Version] 1
    """
    def __init__(self):
        game_config = GameConfig(num_layers=1, num_servers_per_layer=1, num_attack_types=10, max_value=9)
        game_config.set_initial_state(defense_val=4, attack_val=0, num_vulnerabilities_per_node=4, det_val=3,
                                      vulnerability_val=0)
        defender_agent = RandomDefenseBotAgent(game_config)
        idsgame_config = IdsGameConfig(game_config=game_config, defender_agent=defender_agent)
        idsgame_config.render_config.caption = "idsgame-random_defense-v1"
        super().__init__(idsgame_config=idsgame_config)


class IdsGameMinimalDefenseV1Env(AttackerEnv):
    """
    [AttackerEnv] 1 layer, 1 server per layer, 10 attack-defense-values, defender following the "defend minimal strategy"
    [Initial State] Defense: 4, Attack:0, Num vulnerabilities: 4, Det: 3, Vulnerability value: 0
    [Version] 1
    """
    def __init__(self):
        game_config = GameConfig(num_layers=1, num_servers_per_layer=1, num_attack_types=10, max_value=9)
        game_config.set_initial_state(defense_val=4, attack_val=0, num_vulnerabilities_per_node=4, det_val=3,
                                      vulnerability_val=0)
        defender_agent = DefendMinimalValueBotAgent(game_config)
        idsgame_config = IdsGameConfig(game_config=game_config, defender_agent=defender_agent)
        idsgame_config.render_config.caption = "idsgame-random_defense-v1"
        super().__init__(idsgame_config=idsgame_config)


class IdsGameRandomAttackV1Env(DefenderEnv):
    """
    [DefenderEnv] 1 layer, 1 server per layer, 10 attack-defense-values, random attacker
    [Initial State] Defense: 4, Attack:0, Num vulnerabilities: 4, Det: 3, Vulnerability value: 0
    [Version] 1
    """
    def __init__(self):
        game_config = GameConfig(num_layers=1, num_servers_per_layer=1, num_attack_types=10, max_value=9)
        game_config.set_initial_state(defense_val=4, attack_val=0, num_vulnerabilities_per_node=4, det_val=3,
                                      vulnerability_val=0)
        attacker_agent = RandomAttackBotAgent(game_config)
        idsgame_config = IdsGameConfig(game_config=game_config, attacker_agent=attacker_agent)
        idsgame_config.render_config.caption = "idsgame-random_attack-v1"
        super().__init__(idsgame_config=idsgame_config)


class IdsGameMaximalAttackV1Env(DefenderEnv):
    """
    [DefenderEnv] 1 layer, 1 server per layer, 10 attack-defense-values, attacker following the "attack maximal strategy"
    [Initial State] Defense: 4, Attack:0, Num vulnerabilities: 4, Det: 3, Vulnerability value: 0
    [Version] 1
    """
    def __init__(self):
        game_config = GameConfig(num_layers=1, num_servers_per_layer=1, num_attack_types=10, max_value=9)
        game_config.set_initial_state(defense_val=4, attack_val=0, num_vulnerabilities_per_node=4, det_val=3,
                                      vulnerability_val=0)
        attacker_agent = AttackMaximalValueBotAgent(game_config)
        idsgame_config = IdsGameConfig(game_config=game_config, attacker_agent=attacker_agent)
        idsgame_config.render_config.caption = "idsgame-maximal_attack-v1"
        super().__init__(idsgame_config=idsgame_config)


class IdsGameV1Env(AttackDefenseEnv):
    """
    [AttackDefenseEnv] 1 layer, 1 server per layer, 10 attack-defense-values
    [Initial State] Defense: 4, Attack:0, Num vulnerabilities: 4, Det: 3, Vulnerability value: 0
    [Version] 1
    """
    def __init__(self):
        game_config = GameConfig(num_layers=1, num_servers_per_layer=1, num_attack_types=10, max_value=9)
        game_config.set_initial_state(defense_val=4, attack_val=0, num_vulnerabilities_per_node=4, det_val=3,
                                      vulnerability_val=0)
        idsgame_config = IdsGameConfig(game_config=game_config)
        idsgame_config.render_config.caption = "idsgame-v1"
        super().__init__(idsgame_config=idsgame_config)


# -------- Version 2 ------------

class IdsGameRandomDefenseV2Env(AttackerEnv):
    """
    [AttackerEnv] 1 layer, 2 servers per layer, 10 attack-defense-values, random defender
    [Initial State] Defense: 2, Attack:0, Num vulnerabilities: 1, Det: 2, Vulnerability value: 0
    [Version] 0
    """
    def __init__(self):
        game_config = GameConfig(num_layers=1, num_servers_per_layer=2, num_attack_types=10, max_value=9)
        game_config.set_initial_state(defense_val=2, attack_val=0, num_vulnerabilities_per_node=1, det_val=2,
                                      vulnerability_val=0)
        defender_agent = RandomDefenseBotAgent(game_config)
        idsgame_config = IdsGameConfig(game_config=game_config, defender_agent=defender_agent)
        idsgame_config.render_config.caption = "idsgame-random_defense-v2"
        super().__init__(idsgame_config=idsgame_config)


class IdsGameMinimalDefenseV2Env(AttackerEnv):
    """
    [AttackerEnv] 1 layer, 2 servers per layer, 10 attack-defense-values, defender following the "defend minimal strategy"
    [Initial State] Defense: 2, Attack:0, Num vulnerabilities: 1, Det: 2, Vulnerability value: 0
    [Version] 0
    """
    def __init__(self):
        game_config = GameConfig(num_layers=1, num_servers_per_layer=2, num_attack_types=10, max_value=9)
        game_config.set_initial_state(defense_val=2, attack_val=0, num_vulnerabilities_per_node=1, det_val=2,
                                      vulnerability_val=0)
        defender_agent = DefendMinimalValueBotAgent(game_config)
        idsgame_config = IdsGameConfig(game_config=game_config, defender_agent=defender_agent)
        idsgame_config.render_config.caption = "idsgame-minimal_defense-v2"
        super().__init__(idsgame_config=idsgame_config)


class IdsGameRandomAttackV2Env(DefenderEnv):
    """
    [DefenderEnv] 1 layer, 2 servers per layer, 10 attack-defense-values, random attacker
    [Initial State] Defense: 2, Attack:0, Num vulnerabilities: 1, Det: 2, Vulnerability value: 0
    [Version] 0
    """
    def __init__(self):
        game_config = GameConfig(num_layers=1, num_servers_per_layer=2, num_attack_types=10, max_value=9)
        game_config.set_initial_state(defense_val=2, attack_val=0, num_vulnerabilities_per_node=1, det_val=2,
                                      vulnerability_val=0)
        attacker_agent = RandomAttackBotAgent(game_config)
        idsgame_config = IdsGameConfig(game_config=game_config, attacker_agent=attacker_agent)
        idsgame_config.render_config.caption = "idsgame-random_attack-v2"
        super().__init__(idsgame_config=idsgame_config)


class IdsGameMaximalAttackV2Env(DefenderEnv):
    """
    [DefenderEnv] 1 layer, 2 servers per layer, 10 attack-defense-values, attacker following the "attack maximal strategy"
    [Initial State] Defense: 2, Attack:0, Num vulnerabilities: 1, Det: 2, Vulnerability value: 0
    [Version] 0
    """
    def __init__(self):
        game_config = GameConfig(num_layers=1, num_servers_per_layer=2, num_attack_types=10, max_value=9)
        game_config.set_initial_state(defense_val=2, attack_val=0, num_vulnerabilities_per_node=1, det_val=2,
                                      vulnerability_val=0)
        attacker_agent = AttackMaximalValueBotAgent(game_config)
        idsgame_config = IdsGameConfig(game_config=game_config, attacker_agent=attacker_agent)
        idsgame_config.render_config.caption = "idsgame-maximal_attack-v2"
        super().__init__(idsgame_config=idsgame_config)


class IdsGameV2Env(AttackDefenseEnv):
    """
    [AttackDefenseEnv] 1 layer, 2 servers per layer, 10 attack-defense-values
    [Initial State] Defense: 2, Attack:0, Num vulnerabilities: 1, Det: 2, Vulnerability value: 0
    [Version] 0
    """
    def __init__(self):
        game_config = GameConfig(num_layers=1, num_servers_per_layer=2, num_attack_types=10, max_value=9)
        game_config.set_initial_state(defense_val=2, attack_val=0, num_vulnerabilities_per_node=1, det_val=2,
                                      vulnerability_val=0)
        idsgame_config = IdsGameConfig(game_config=game_config)
        idsgame_config.render_config.caption = "idsgame-v2"
        super().__init__(idsgame_config=idsgame_config)
"""
RL environment for an intrusion detection Markov game
"""
from typing import Union
import numpy as np
import gym
import os
import time
import pickle
import csv
import math
import itertools
from abc import ABC, abstractmethod
from gym_idsgame.envs.dao.game_config import GameConfig
from gym_idsgame.agents.bot_agents.random_defense_bot_agent import RandomDefenseBotAgent
from gym_idsgame.agents.bot_agents.random_attack_bot_agent import RandomAttackBotAgent
from gym_idsgame.agents.bot_agents.defend_minimal_value_bot_agent import DefendMinimalValueBotAgent
from gym_idsgame.agents.bot_agents.attack_maximal_value_bot_agent import AttackMaximalValueBotAgent
from gym_idsgame.envs.dao.game_state import GameState
from gym_idsgame.envs.dao.idsgame_config import IdsGameConfig
from gym_idsgame.envs.dao.network_config import NetworkConfig
import gym_idsgame.envs.util.idsgame_util as util
from gym_idsgame.envs.constants import constants

class IdsGameEnv(gym.Env, ABC):
    """
    Implementation of the RL environment from the paper
    "Adversarial Reinforcement Learning in a Cyber Security Simulation" by Elderman et. al.

    It is an abstract very simple cyber security simulation where there is an attacker agent that tries to penetrate
    a network and a defender agent that tries to defend the network.
    """

    def __init__(self, idsgame_config: IdsGameConfig = None, save_dir: str = None, initial_state_path: str = None):
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
        :param save_dir: directory to save outputs, e.g. initial state
        :param initial_state_path: path to the initial state (if none, use default)
        """
        if idsgame_config is None:
            idsgame_config = IdsGameConfig(initial_state_path=initial_state_path)
        self.save_dir = save_dir
        util.validate_config(idsgame_config)
        self.idsgame_config: IdsGameConfig = idsgame_config
        self.state: GameState = self.idsgame_config.game_config.initial_state.copy()
        self.observation_space = self.idsgame_config.game_config.get_attacker_observation_space()
        self.attacker_action_space = self.idsgame_config.game_config.get_action_space(defender=False)
        self.defender_action_space = self.idsgame_config.game_config.get_action_space(defender=True)
        self.viewer = None
        self.steps_beyond_done = None
        self.metadata = {
         'render.modes': ['human', 'rgb_array'],
         'video.frames_per_second' : 50 # Video rendering speed
        }
        self.reward_range = (float(constants.GAME_CONFIG.NEGATIVE_REWARD), float(constants.GAME_CONFIG.POSITIVE_REWARD))
        self.num_states = self.idsgame_config.game_config.num_nodes
        self.num_states_full = int(math.pow(self.idsgame_config.game_config.max_value+1,
                                        self.idsgame_config.game_config.num_nodes*
                                        (self.idsgame_config.game_config.num_attack_types+1)))
        if self.idsgame_config.game_config.network_config.fully_observed:
            self.num_states_full = int(math.pow(self.idsgame_config.game_config.max_value+1,
                                            self.idsgame_config.game_config.num_nodes *
                                            (self.idsgame_config.game_config.num_attack_types+1)* 2))
        self.num_attack_actions = self.idsgame_config.game_config.num_attack_actions
        self.num_defense_actions = self.idsgame_config.game_config.num_defense_actions
        self.past_moves = []
        self.past_positions = []
        self.past_positions.append(self.state.attacker_pos)
        self.past_reconnaissance_activities = []
        self.save_initial_state()
        self.furthest_hack = self.idsgame_config.game_config.network_config.num_rows-1
        self.a_cumulative_reward = 0
        self.d_cumulative_reward = 0
        self.game_trajectories = []
        self.game_trajectory = []
        self.attack_detections = []
        self.total_attacks = []
        self.defenses = []
        self.attacks = []

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
        trajectory = []
        trajectory.append(self.state)
        reward = (0,0)
        info = {}
        self.state.attack_events = []
        self.state.defense_events = []

        if self.state.game_step > constants.GAME_CONFIG.MAX_GAME_STEPS:
            return self.get_observation(), (100*constants.GAME_CONFIG.NEGATIVE_REWARD,
                                            100*constants.GAME_CONFIG.NEGATIVE_REWARD), True, info

        attack_action, defense_action = action

        # 1. Interpret attacker action
        attacker_pos = self.state.attacker_pos
        if attack_action != -1:
            target_node_id, target_pos, attack_type, reconnaissance = self.get_attacker_action(action)
            trajectory.append([target_node_id, target_pos, attack_type, reconnaissance])

        # 2. Interpret defense action
        defense_node_id, defense_pos, defense_type,  = self.get_defender_action(action)
        trajectory.append([defense_node_id, defense_pos, defense_type])

        # 3. Defend
        detect = defense_type == self.idsgame_config.game_config.num_attack_types
        defense_successful = self.state.defend(defense_node_id, defense_type, self.idsgame_config.game_config.max_value,
                          self.idsgame_config.game_config.network_config, detect=detect)
        if defense_successful:
            self.defenses.append((defense_node_id, defense_type, detect, self.state.game_step))
        self.state.add_defense_event(defense_pos, defense_type)

        if attack_action != -1 and util.is_attack_legal(target_pos, attacker_pos, self.idsgame_config.game_config.network_config,
                                past_positions=self.past_positions):
            self.past_moves.append(target_node_id)
            if not reconnaissance:
                # 4. Attack
                self.state.attack(target_node_id, attack_type, self.idsgame_config.game_config.max_value,
                                  self.idsgame_config.game_config.network_config)
            else:
                rec_reward = self.state.reconnaissance(target_node_id, attack_type)
                self.past_reconnaissance_activities.append((target_node_id, attack_type))
                reward = (rec_reward, 0)

            self.state.add_attack_event(target_pos, attack_type, self.state.attacker_pos, reconnaissance)
            self.attacks.append((target_node_id, attack_type, self.state.game_step, reconnaissance))

            attack_successful = False
            if not reconnaissance:
                # 5. Simulate attack outcome
                attack_successful = self.state.simulate_attack(target_node_id, attack_type,
                                                               self.idsgame_config.game_config.network_config)
            if self.idsgame_config.save_attack_stats:
                self.total_attacks.append([target_node_id, attack_successful, reconnaissance])

            # 6. Update state based on attack outcome
            if attack_successful:
                if not reconnaissance:
                    self.past_positions.append(target_pos)
                    self.state.attacker_pos = target_pos
                    if target_pos == self.idsgame_config.game_config.network_config.data_pos:
                        self.state.done = True
                        self.state.hacked = True
                        reward = self.get_hack_reward()
                    else:
                        reward = self.get_successful_attack_reward(attack_type)
            else:
                self.past_positions.append(self.state.attacker_pos)
                detected = self.state.simulate_detection(target_node_id, reconnaissance=reconnaissance)
                if detected:
                    self.state.done = True
                    self.state.detected = True
                    reward = self.get_detect_reward(target_node_id,  attack_type, self.state.defense_det[target_node_id])
                else:
                    if not reconnaissance:
                        reward = self.get_blocked_attack_reward(target_node_id, attack_type)
                if self.idsgame_config.save_attack_stats:
                    self.attack_detections.append([target_node_id, detected, self.state.defense_det[target_node_id]])
        else:
            reward = -100*constants.GAME_CONFIG.POSITIVE_REWARD, 0
            self.state.done = True
            self.state.detected = True

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
        self.a_cumulative_reward += reward[0]
        self.d_cumulative_reward += reward[1]
        trajectory.append(reward[0])
        trajectory.append(reward[1])
        trajectory.append(self.state)
        if self.idsgame_config.save_trajectories:
            self.game_trajectories.append(trajectory)
        return observation, reward, self.state.done, info

    def reset(self, update_stats = False) -> np.ndarray:
        """
        Resets the environment and returns the initial state

        :param update_stats: whether the game count should be incremented or not
        :return: the initial state
        """
        self.past_moves = []
        self.past_positions = []
        self.past_reconnaissance_activities = []
        self.furthest_hack = self.idsgame_config.game_config.network_config.num_rows-1
        self.steps_beyond_done = None
        self.state.new_game(self.idsgame_config.game_config.initial_state, self.a_cumulative_reward,
                            self.d_cumulative_reward, update_stats=update_stats,
                            randomize_state=self.idsgame_config.randomize_env,
                            network_config=self.idsgame_config.game_config.network_config,
                            num_attack_types=self.idsgame_config.game_config.num_attack_types,
                            defense_val = self.idsgame_config.game_config.defense_val,
                            attack_val = self.idsgame_config.game_config.attack_val,
                            det_val = self.idsgame_config.game_config.det_val,
                            vulnerability_val = self.idsgame_config.game_config.vulnerabilitiy_val,
                            num_vulnerabilities_per_layer=self.idsgame_config.game_config.num_vulnerabilities_per_layer,
                            num_vulnerabilities_per_node=self.idsgame_config.game_config.num_vulnerabilities_per_node)
        self.a_cumulative_reward = 0
        self.d_cumulative_reward = 0
        if self.idsgame_config.randomize_starting_position:
            self.state.randomize_attacker_position(self.idsgame_config.game_config.network_config)
        if self.viewer is not None:
            self.viewer.gameframe.reset()
        observation = self.get_observation()
        self.past_positions.append(self.state.attacker_pos)
        self.defenses = []
        self.attacks = []
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
        return self.state.get_attacker_node_from_observation(
            observation, reconnaissance=self.idsgame_config.game_config.reconnaissance_actions)

    def hack_probability(self) -> float:
        """
        :return: the cumulative hack-probabiltiy according to the game history
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
        return util.is_attack_id_legal(attack_action, self.idsgame_config.game_config, self.state.attacker_pos,
                                       self.state, self.past_positions,
                                       past_reconnaissance_activities = self.past_reconnaissance_activities)

    def is_defense_legal(self, defense_action: int) -> bool:
        """
        Check if a given defense is legal or not.

        :param defense_action: the defense action to verify
        :return: True if legal otherwise False
        """
        return util.is_defense_id_legal(defense_action, self.idsgame_config.game_config, self.state)

    def save_initial_state(self) -> None:
        """
        Saves initial state to disk in binary npy format

        :return: None
        """
        if self.save_dir is not None and os.path.exists(self.save_dir):
            GameState.save(self.save_dir, self.state)

    def save_trajectories(self, checkpoint = True) -> None:
        """
        Saves the current list of game trajectories to disk

        :param checkpoint: boolean flag that indicates whether this is a checkpoint save or final save
        :return: None
        """
        suffix = ".pkl"
        if checkpoint:
            suffix = "_checkpoint.pkl"
        if self.idsgame_config.save_trajectories:
            path = self.save_dir
            time_str = str(time.time())
            filehandler = open(path + "/trajectories_" + time_str + suffix, 'wb')
            pickle.dump(self.game_trajectories, filehandler)
        else:
            self.game_trajectories = []

    def save_attack_data(self, checkpoint = True) -> None:
        """
        Saves the attack statistics to disk

        :param checkpoint: boolean flag that indicates whether this is a checkpoint save or final save
        :return: None
        """
        suffix = ".csv"
        if checkpoint:
            suffix = "_checkpoint.csv"
        if self.idsgame_config.save_attack_stats:
            time_str = str(time.time())
            with open(self.save_dir + "/attack_detections_stats_" + time_str + suffix, "w") as f:
                writer = csv.writer(f)
                writer.writerow(["target_node", "detected", "detection_val"])
                for row in self.attack_detections:
                    writer.writerow(row)
            with open(self.save_dir + "/attack_stats_" + time_str + suffix, "w") as f:
                writer = csv.writer(f)
                writer.writerow(["target_node", "attack_outcome"])
                for row in self.total_attacks:
                    writer.writerow(row)
        else:
            self.attack_detections = []
            self.total_attacks = []

    def get_hack_reward(self) -> Union[int, int]:
        """
        Returns the attacker and defender reward in the case when the hacker manages to reach the target node

        :return: (attacker_reward, defender_reward)
        """
        if not self.idsgame_config.game_config.dense_rewards and not self.idsgame_config.game_config.dense_rewards_v2:
            return constants.GAME_CONFIG.POSITIVE_REWARD, -constants.GAME_CONFIG.POSITIVE_REWARD
        elif self.idsgame_config.game_config.dense_rewards and not self.idsgame_config.game_config.dense_rewards_v2:
            return 100*constants.GAME_CONFIG.POSITIVE_REWARD, -100*constants.GAME_CONFIG.POSITIVE_REWARD
        else:
            # detection_actions = 0
            # for defense in self.defenses:
            #     if defense[2]:
            #         detection_actions += 1
            # unblocked_attacks = 0
            # for defense in self.defenses:
            #     blocked = False
            #     for attack in self.attacks:
            #         if defense[0] == attack[0] and defense[1] == attack[1] and defense[3] < attack[2]:
            #             blocked = True
            #     if not blocked:
            #         unblocked_attacks += 1
            # norm_factor = self.state.game_step if self.state.game_step > 0 else 1
            # reward = -((unblocked_attacks)/norm_factor)
            # #print("reward:{}".format(reward))
            extra_reward = 0
            if self.idsgame_config.extra_reconnaisasnce_reward:
                for rec_act in self.past_reconnaissance_activities:
                    node_id, rec_type = rec_act
                    server_id = self.idsgame_config.game_config.network_config.get_node_id(self.state.attacker_pos)
                    if node_id == server_id:
                        extra_reward = 1
            return extra_reward + 100 * constants.GAME_CONFIG.POSITIVE_REWARD, 0

    def get_detect_reward(self, target_node_id : int, attack_type : int, detection_value) -> Union[int, int]:
        """
        Returns the attacker and defender reward in the case when the attacker was detected.

        :return: (attacker_reward, defender_reward)
        """
        if not self.idsgame_config.game_config.dense_rewards and not self.idsgame_config.game_config.dense_rewards_v2:
            return -constants.GAME_CONFIG.POSITIVE_REWARD, constants.GAME_CONFIG.POSITIVE_REWARD
        elif self.idsgame_config.game_config.dense_rewards and not self.idsgame_config.game_config.dense_rewards_v2:
            return -100*constants.GAME_CONFIG.POSITIVE_REWARD, 100*constants.GAME_CONFIG.POSITIVE_REWARD
        else:
            added_detection = 0
            for defense in self.defenses:
                if defense[2] and defense[0] == target_node_id:
                    added_detection += 1
            # detection_ratio = added_detection/((detection_value+1))
            # blocked_attacks = 0
            # for defense in self.defenses:
            #     blocked = False
            #     for attack in self.attacks:
            #         if defense[0] == attack[0] and defense[1] == attack[1] and defense[3] < attack[2]:
            #             blocked = True
            #     if blocked:
            #         blocked_attacks += 1
            # norm_factor = self.state.game_step if self.state.game_step > 0 else 1
            # reward = (blocked_attacks)/norm_factor
            #return 0*constants.GAME_CONFIG.POSITIVE_REWARD, added_detection
            return -100*constants.GAME_CONFIG.POSITIVE_REWARD, added_detection

    def get_successful_attack_reward(self, attack_type : int) -> Union[int, int]:
        """
        Returns the reward for the attacker and defender after a successful attack on some server in
        the network

        :return:(attacker_reward, defender_reward)
        """
        if not self.idsgame_config.game_config.dense_rewards and not self.idsgame_config.game_config.dense_rewards_v2:
            return 0, 0
        elif self.idsgame_config.game_config.dense_rewards and not self.idsgame_config.game_config.dense_rewards_v2:
            attack_row, attack_col = self.state.attacker_pos
            if attack_row < self.furthest_hack:
                self.furthest_hack = attack_row
                return constants.GAME_CONFIG.POSITIVE_REWARD, -constants.GAME_CONFIG.POSITIVE_REWARD
            elif attack_row > self.furthest_hack:
                return -constants.GAME_CONFIG.POSITIVE_REWARD, constants.GAME_CONFIG.POSITIVE_REWARD
            return 0,0
        else:
            attack_row, attack_col = self.state.attacker_pos
            if attack_row < self.furthest_hack:
                self.furthest_hack = attack_row
                extra_reward = 0
                if self.idsgame_config.extra_reconnaisasnce_reward:
                    for rec_act in self.past_reconnaissance_activities:
                        node_id, rec_type = rec_act
                        server_id = self.idsgame_config.game_config.network_config.get_node_id(self.state.attacker_pos)
                        if node_id == server_id:
                            extra_reward = 1
                return extra_reward + 100*constants.GAME_CONFIG.POSITIVE_REWARD, 0
            elif attack_row > self.furthest_hack:
                return -100*constants.GAME_CONFIG.POSITIVE_REWARD, 0
            return 0, 0

    def get_blocked_attack_reward(self, target_node_id : int, attack_type : int) -> Union[int, int]:
        """
        Returns the reward for the attacker and defender after a blocked attack on some server in
        the network

        :return:(attacker_reward, defender_reward)
        """
        if not self.idsgame_config.game_config.dense_rewards and not self.idsgame_config.game_config.dense_rewards_v2:
            return 0, 0
        elif self.idsgame_config.game_config.dense_rewards and not self.idsgame_config.game_config.dense_rewards_v2:
            return 0, 0
        else:
            upd_defenses = []
            match = False
            for defense in self.defenses:
                if ((defense[0] == target_node_id and defense[1] == attack_type)) \
                        and not match:
                   match = True
                else:
                    upd_defenses.append(defense)
            self.defenses = upd_defenses
            if match:
                return 0, constants.GAME_CONFIG.POSITIVE_REWARD
            return 0, 0

    def get_observation(self) -> Union[np.ndarray, np.ndarray]:
        """
        Returns an observation of the state

        :return: (attacker_obs, defender_obs)
        """
        attacker_obs = self.state.get_attacker_observation(
            self.idsgame_config.game_config.network_config, local_view=self.idsgame_config.local_view_observations,
            reconnaissance=self.idsgame_config.game_config.reconnaissance_actions,
        reconnaissance_bool_features=self.idsgame_config.reconnaissance_bool_features)
        defender_obs = self.state.get_defender_observation(self.idsgame_config.game_config.network_config)
        return attacker_obs, defender_obs

    def fully_observed(self) -> bool:
        """
        Boolean function to check whether the environment is configured to be fully observed or not

        :return: True if the environment is fully observed, otherwise false
        """
        return self.idsgame_config.game_config.network_config.fully_observed

    def local_view_features(self) -> bool:
        """
        Boolean function to check whether the environment uses local view observations of the attacker

        :return: True if the environment uses local view observations
        """
        return self.idsgame_config.local_view_observations

    @abstractmethod
    def get_attacker_action(self, action) -> Union[int, Union[int, int], int]:
        pass

    @abstractmethod
    def get_defender_action(self, action) -> Union[Union[int, int], int, int]:
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

    def _build_state_to_idx_map(self):
        """
        Builds a map that maps states to index (useful when constructing Q-tables for example)

        :return: the lookup map
        """
        n_state_elems = self.idsgame_config.game_config.num_nodes * \
                        (self.idsgame_config.game_config.num_attack_types + 1)
        if self.idsgame_config.game_config.network_config.fully_observed:
            n_state_elems = self.idsgame_config.game_config.num_nodes * \
                            (self.idsgame_config.game_config.num_attack_types + 1) * 2
        states = list(
            itertools.product(list(range(self.idsgame_config.game_config.max_value + 1)), repeat=n_state_elems))
        assert int(len(states)) == int(math.pow(self.idsgame_config.game_config.max_value + 1, n_state_elems))
        state_to_idx = {}
        for idx, s in enumerate(states):
            state_to_idx[s] = idx
        return state_to_idx


class AttackerEnv(IdsGameEnv, ABC):
    """
    Abstract AttackerEnv of the IdsGameEnv.

    Environments where the defender is part of the environment and the environment is designed to be used by an
    attacker-agent should inherit this class
    """

    def __init__(self, idsgame_config: IdsGameConfig, save_dir: str = None, initial_state_path: str = None):
        """
        Initialization of the environment

        :param save_dir: directory to save outputs of the env
        :param initial_state_path: path to the initial state (if none, use default)
        :param idsgame_config: configuration of the environment (if not specified a default config is used)
        """
        if idsgame_config is None:
            raise ValueError("Cannot instantiate env without configuration")
        if idsgame_config.defender_agent is None:
            raise ValueError("Cannot instantiate attacker-env without a defender agent")
        super().__init__(idsgame_config=idsgame_config, save_dir=save_dir, initial_state_path=initial_state_path)
        self.observation_space = self.idsgame_config.game_config.get_attacker_observation_space()

    def get_attacker_action(self, action) -> Union[int, Union[int, int], int]:
        attacker_action, _ = action
        return util.interpret_attack_action(attacker_action, self.idsgame_config.game_config)

    def get_defender_action(self, action) -> Union[Union[int, int], int, int]:
        defend_id = self.idsgame_config.defender_agent.action(self.state)
        defend_node_id, defend_node_pos, defend_type = util.interpret_defense_action(
            defend_id, self.idsgame_config.game_config)
        return defend_node_id, defend_node_pos, defend_type


class DefenderEnv(IdsGameEnv, ABC):
    """
    Abstract DefenderEnv of the IdsGameEnv.

    Environments where the attacker is part of the environment and the environment is designed to be used by a
    defender-agent should inherit this class
    """
    def __init__(self, idsgame_config: IdsGameConfig, save_dir: str = None, initial_state_path: str = None):
        """
        Initialization of the environment

        :param save_dir: directory to save outputs of the env
        :param initial_state_path: path to the initial state (if none, use default)
        :param idsgame_config: configuration of the environment (if not specified a default config is used)
        """
        if idsgame_config is None:
            raise ValueError("Cannot instantiate env without configuration")
        if idsgame_config.attacker_agent is None:
            raise ValueError("Cannot instantiate defender-env without an attacker agent")
        super().__init__(idsgame_config=idsgame_config, save_dir=save_dir, initial_state_path=initial_state_path)
        self.observation_space = self.idsgame_config.game_config.get_defender_observation_space()

    def get_defender_action(self, action) -> Union[int, Union[int, int], int]:
        _, defender_action = action
        return util.interpret_defense_action(defender_action, self.idsgame_config.game_config)

    def get_attacker_action(self, action) -> Union[Union[int, int], int, int, bool]:
        attack_id = self.idsgame_config.attacker_agent.action(self.state)
        attack_node_id, attack_node_pos, attack_type, reconnaissance = util.interpret_attack_action(attack_id, self.idsgame_config.game_config)
        return attack_node_id, attack_node_pos, attack_type, reconnaissance


class AttackDefenseEnv(IdsGameEnv, ABC):
    """
    Abstract AttackDefenseEnv of the IdsGameEnv.

    Environments where both the attacker and defender are external to the environment should inherit this class.
    """
    def __init__(self, idsgame_config: IdsGameConfig, save_dir: str = None, initial_state_path: str = None):
        """
        Initialization of the environment

        :param save_dir: directory to save outputs of the env
        :param initial_state_path: path to the initial state (if none, use default)
        :param idsgame_config: configuration of the environment (if not specified a default config is used)
        """
        if idsgame_config is None:
            raise ValueError("Cannot instantiate env without configuration")
        super().__init__(idsgame_config=idsgame_config, save_dir=save_dir, initial_state_path=initial_state_path)

    def get_defender_action(self, action: Union[int, int]) -> Union[int, Union[int, int], int]:
        _, defender_action = action
        return util.interpret_defense_action(defender_action, self.idsgame_config.game_config)

    def get_attacker_action(self, action: Union[int, int]) -> Union[Union[int, int], int, int]:
        attacker_action, _ = action
        return util.interpret_attack_action(attacker_action, self.idsgame_config.game_config)

# -------- Concrete envs ------------

# -------- Version 0 ------------

class IdsGameRandomDefenseV0Env(AttackerEnv):
    """
    [AttackerEnv] 1 layer, 1 server per layer, 10 attack-defense-values, random defender
    [Initial State] Defense: 2, Attack:0, Num vulnerabilities: 1, Det: 2, Vulnerability value: 0
    [Rewards] Sparse
    [Version] 0
    [Observations] partially observed
    [Environment] Deterministic
    [Attacker Starting Position] Start node
    [Reconnaissance activities] disabled
    [Reconnaissance bool features] No
    """
    def __init__(self, idsgame_config: IdsGameConfig = None, save_dir: str = None, initial_state_path: str = None):
        """
        Initialization of the environment

        :param save_dir: directory to save outputs of the env
        :param initial_state_path: path to the initial state (if none, use default)
        :param idsgame_config: configuration of the environment (if not specified a default config is used)
        """
        if idsgame_config is None:
            game_config = GameConfig(num_layers=1, num_servers_per_layer=1, num_attack_types=10, max_value=9)
            game_config.set_initial_state(defense_val=2, attack_val=0, num_vulnerabilities_per_node=1, det_val=2,
                                          vulnerability_val=0, num_vulnerabilities_per_layer=1)
            if initial_state_path is not None:
                game_config.set_load_initial_state(initial_state_path)
            defender_agent = RandomDefenseBotAgent(game_config)
            idsgame_config = IdsGameConfig(game_config=game_config, defender_agent=defender_agent)
            idsgame_config.render_config.caption = "idsgame-random_defense-v0"
        super().__init__(idsgame_config=idsgame_config, save_dir=save_dir)


class IdsGameMinimalDefenseV0Env(AttackerEnv):
    """
    [AttackerEnv] 1 layer, 1 server per layer, 10 attack-defense-values, defender following the "defend minimal strategy"
    [Initial State] Defense: 2, Attack:0, Num vulnerabilities: 1, Det: 2, Vulnerability value: 0
    [Rewards] Sparse
    [Version] 0
    [Observations] partially observed
    [Environment] Deterministic
    [Attacker Starting Position] Start node
    [Reconnaissance activities] disabled
    [Reconnaissance bool features] No
    """
    def __init__(self, idsgame_config: IdsGameConfig = None, save_dir: str = None, initial_state_path: str = None):
        """
        Initialization of the environment

        :param save_dir: directory to save outputs of the env
        :param initial_state_path: path to the initial state (if none, use default)
        :param idsgame_config: configuration of the environment (if not specified a default config is used)
        """
        if idsgame_config is None:
            game_config = GameConfig(num_layers=1, num_servers_per_layer=1, num_attack_types=10, max_value=9)
            game_config.set_initial_state(defense_val=2, attack_val=0, num_vulnerabilities_per_node=1, det_val=2,
                                          vulnerability_val=0, num_vulnerabilities_per_layer=1)
            if initial_state_path is not None:
                game_config.set_load_initial_state(initial_state_path)
            defender_agent = DefendMinimalValueBotAgent(game_config)
            idsgame_config = IdsGameConfig(game_config=game_config, defender_agent=defender_agent)
            idsgame_config.render_config.caption = "idsgame-minimal_defense-v0"
        super().__init__(idsgame_config=idsgame_config, save_dir=save_dir)


class IdsGameRandomAttackV0Env(DefenderEnv):
    """
    [DefenderEnv] 1 layer, 1 server per layer, 10 attack-defense-values, random attacker
    [Initial State] Defense: 2, Attack:0, Num vulnerabilities: 1, Det: 2, Vulnerability value: 0
    [Rewards] Sparse
    [Version] 0
    [Observations] partially observed
    [Environment] Deterministic
    [Attacker Starting Position] Start node
    [Reconnaissance activities] disabled
    [Reconnaissance bool features] No
    """
    def __init__(self, idsgame_config: IdsGameConfig = None, save_dir: str = None, initial_state_path: str = None):
        """
        Initialization of the environment

        :param save_dir: directory to save outputs of the env
        :param initial_state_path: path to the initial state (if none, use default)
        :param idsgame_config: configuration of the environment (if not specified a default config is used)
        """
        if idsgame_config is None:
            game_config = GameConfig(num_layers=1, num_servers_per_layer=1, num_attack_types=10, max_value=9)
            game_config.set_initial_state(defense_val=2, attack_val=0, num_vulnerabilities_per_node=1, det_val=2,
                                          vulnerability_val=0, num_vulnerabilities_per_layer=1)
            if initial_state_path is not None:
                game_config.set_load_initial_state(initial_state_path)
            attacker_agent = RandomAttackBotAgent(game_config)
            idsgame_config = IdsGameConfig(game_config=game_config, attacker_agent=attacker_agent)
            idsgame_config.render_config.caption = "idsgame-random_attack-v0"
        super().__init__(idsgame_config=idsgame_config, save_dir=save_dir)


class IdsGameMaximalAttackV0Env(DefenderEnv):
    """
    [DefenderEnv] 1 layer, 1 server per layer, 10 attack-defense-values, attacker following the "attack maximal strategy"
    [Initial State] Defense: 2, Attack:0, Num vulnerabilities: 1, Det: 2, Vulnerability value: 0
    [Rewards] Sparse
    [Version] 0
    [Observations] partially observed
    [Environment] Deterministic
    [Attacker Starting Position] Start node
    [Reconnaissance activities] disabled
    [Reconnaissance bool features] No
    """
    def __init__(self, idsgame_config: IdsGameConfig = None, save_dir: str = None, initial_state_path: str = None):
        """
        Initialization of the environment

        :param save_dir: directory to save outputs of the env
        :param initial_state_path: path to the initial state (if none, use default)
        :param idsgame_config: configuration of the environment (if not specified a default config is used)
        """
        if idsgame_config is None:
            game_config = GameConfig(num_layers=1, num_servers_per_layer=1, num_attack_types=10, max_value=9)
            game_config.set_initial_state(defense_val=2, attack_val=0, num_vulnerabilities_per_node=1, det_val=2,
                                          vulnerability_val=0, num_vulnerabilities_per_layer=1)
            if initial_state_path is not None:
                game_config.set_load_initial_state(initial_state_path)
            attacker_agent = AttackMaximalValueBotAgent(game_config)
            idsgame_config = IdsGameConfig(game_config=game_config, attacker_agent=attacker_agent)
            idsgame_config.render_config.caption = "idsgame-maximal_attack-v0"
        super().__init__(idsgame_config=idsgame_config, save_dir=save_dir)


class IdsGameV0Env(AttackDefenseEnv):
    """
    [AttackDefenseEnv] 1 layer, 1 server per layer, 10 attack-defense-values
    [Initial State] Defense: 2, Attack:0, Num vulnerabilities: 1, Det: 2, Vulnerability value: 0
    [Rewards] Sparse
    [Version] 0
    [Observations] partially observed
    [Environment] Deterministic
    [Attacker Starting Position] Start node
    [Reconnaissance activities] disabled
    [Reconnaissance bool features] No
    """
    def __init__(self, idsgame_config: IdsGameConfig = None, save_dir: str = None, initial_state_path: str = None):
        """
        Initialization of the environment

        :param save_dir: directory to save outputs of the env
        :param initial_state_path: path to the initial state (if none, use default)
        :param idsgame_config: configuration of the environment (if not specified a default config is used)
        """
        if idsgame_config is None:
            game_config = GameConfig(num_layers=1, num_servers_per_layer=1, num_attack_types=10, max_value=9)
            game_config.set_initial_state(defense_val=2, attack_val=0, num_vulnerabilities_per_node=1, det_val=2,
                                          vulnerability_val=0, num_vulnerabilities_per_layer=1)
            if initial_state_path is not None:
                game_config.set_load_initial_state(initial_state_path)
            idsgame_config = IdsGameConfig(game_config=game_config)
            idsgame_config.render_config.caption = "idsgame-v0"
        super().__init__(idsgame_config=idsgame_config, save_dir=save_dir)


# -------- Version 1 ------------


class IdsGameRandomDefenseV1Env(AttackerEnv):
    """
    [AttackerEnv] 1 layer, 1 server per layer, 10 attack-defense-values, random defender
    [Initial State] Defense: 4, Attack:0, Num vulnerabilities: 4, Det: 3, Vulnerability value: 0
    [Rewards] Sparse
    [Version] 1
    [Observations] partially observed
    [Environment] Deterministic
    [Attacker Starting Position] Start node
    [Reconnaissance activities] disabled
    [Reconnaissance bool features] No
    """
    def __init__(self, idsgame_config: IdsGameConfig = None, save_dir: str = None, initial_state_path: str = None):
        """
        Initialization of the environment

        :param save_dir: directory to save outputs of the env
        :param initial_state_path: path to the initial state (if none, use default)
        :param idsgame_config: configuration of the environment (if not specified a default config is used)
        """
        if idsgame_config is None:
            game_config = GameConfig(num_layers=1, num_servers_per_layer=1, num_attack_types=10, max_value=9)
            game_config.set_initial_state(defense_val=4, attack_val=0, num_vulnerabilities_per_node=4, det_val=3,
                                          vulnerability_val=0, num_vulnerabilities_per_layer=1)
            if initial_state_path is not None:
                game_config.set_load_initial_state(initial_state_path)
            defender_agent = RandomDefenseBotAgent(game_config)
            idsgame_config = IdsGameConfig(game_config=game_config, defender_agent=defender_agent)
            idsgame_config.render_config.caption = "idsgame-random_defense-v1"
        super().__init__(idsgame_config=idsgame_config, save_dir=save_dir)


class IdsGameMinimalDefenseV1Env(AttackerEnv):
    """
    [AttackerEnv] 1 layer, 1 server per layer, 10 attack-defense-values, defender following the "defend minimal strategy"
    [Initial State] Defense: 4, Attack:0, Num vulnerabilities: 4, Det: 3, Vulnerability value: 0
    [Rewards] Sparse
    [Version] 1
    [Observations] partially observed
    [Environment] Deterministic
    [Attacker Starting Position] Start node
    [Reconnaissance activities] disabled
    [Reconnaissance bool features] No
    """
    def __init__(self, idsgame_config: IdsGameConfig = None, save_dir: str = None, initial_state_path: str = None):
        """
        Initialization of the environment

        :param save_dir: directory to save outputs of the env
        :param initial_state_path: path to the initial state (if none, use default)
        :param idsgame_config: configuration of the environment (if not specified a default config is used)
        """
        if idsgame_config is None:
            game_config = GameConfig(num_layers=1, num_servers_per_layer=1, num_attack_types=10, max_value=9)
            game_config.set_initial_state(defense_val=4, attack_val=0, num_vulnerabilities_per_node=4, det_val=3,
                                          vulnerability_val=0, num_vulnerabilities_per_layer=1)
            if initial_state_path is not None:
                game_config.set_load_initial_state(initial_state_path)
            defender_agent = DefendMinimalValueBotAgent(game_config)
            idsgame_config = IdsGameConfig(game_config=game_config, defender_agent=defender_agent)
            idsgame_config.render_config.caption = "idsgame-random_defense-v1"
        super().__init__(idsgame_config=idsgame_config, save_dir=save_dir)


class IdsGameRandomAttackV1Env(DefenderEnv):
    """
    [DefenderEnv] 1 layer, 1 server per layer, 10 attack-defense-values, random attacker
    [Initial State] Defense: 4, Attack:0, Num vulnerabilities: 4, Det: 3, Vulnerability value: 0
    [Rewards] Sparse
    [Version] 1
    [Observations] partially observed
    [Environment] Deterministic
    [Attacker Starting Position] Start node
    [Reconnaissance activities] disabled
    [Reconnaissance bool features] No
    """
    def __init__(self, idsgame_config: IdsGameConfig = None, save_dir: str = None, initial_state_path: str = None):
        """
        Initialization of the environment

        :param save_dir: directory to save outputs of the env
        :param initial_state_path: path to the initial state (if none, use default)
        :param idsgame_config: configuration of the environment (if not specified a default config is used)
        """
        if idsgame_config is None:
            game_config = GameConfig(num_layers=1, num_servers_per_layer=1, num_attack_types=10, max_value=9)
            game_config.set_initial_state(defense_val=4, attack_val=0, num_vulnerabilities_per_node=4, det_val=3,
                                          vulnerability_val=0, num_vulnerabilities_per_layer=1)
            if initial_state_path is not None:
                game_config.set_load_initial_state(initial_state_path)
            attacker_agent = RandomAttackBotAgent(game_config)
            idsgame_config = IdsGameConfig(game_config=game_config, attacker_agent=attacker_agent)
            idsgame_config.render_config.caption = "idsgame-random_attack-v1"
        super().__init__(idsgame_config=idsgame_config, save_dir=save_dir)


class IdsGameMaximalAttackV1Env(DefenderEnv):
    """
    [DefenderEnv] 1 layer, 1 server per layer, 10 attack-defense-values, attacker following the "attack maximal strategy"
    [Initial State] Defense: 4, Attack:0, Num vulnerabilities: 4, Det: 3, Vulnerability value: 0
    [Rewards] Sparse
    [Version] 1
    [Observations] partially observed
    [Environment] Deterministic
    [Attacker Starting Position] Start node
    [Reconnaissance activities] disabled
    [Reconnaissance bool features] No
    """
    def __init__(self, idsgame_config: IdsGameConfig = None, save_dir: str = None, initial_state_path: str = None):
        """
        Initialization of the environment

        :param save_dir: directory to save outputs of the env
        :param initial_state_path: path to the initial state (if none, use default)
        :param idsgame_config: configuration of the environment (if not specified a default config is used)
        """
        if idsgame_config is None:
            game_config = GameConfig(num_layers=1, num_servers_per_layer=1, num_attack_types=10, max_value=9)
            game_config.set_initial_state(defense_val=4, attack_val=0, num_vulnerabilities_per_node=4, det_val=3,
                                          vulnerability_val=0, num_vulnerabilities_per_layer=1)
            if initial_state_path is not None:
                game_config.set_load_initial_state(initial_state_path)
            attacker_agent = AttackMaximalValueBotAgent(game_config)
            idsgame_config = IdsGameConfig(game_config=game_config, attacker_agent=attacker_agent)
            idsgame_config.render_config.caption = "idsgame-maximal_attack-v1"
        super().__init__(idsgame_config=idsgame_config, save_dir=save_dir)


class IdsGameV1Env(AttackDefenseEnv):
    """
    [AttackDefenseEnv] 1 layer, 1 server per layer, 10 attack-defense-values
    [Initial State] Defense: 4, Attack:0, Num vulnerabilities: 4, Det: 3, Vulnerability value: 0
    [Rewards] Sparse
    [Version] 1
    [Observations] partially observed
    [Environment] Deterministic
    [Attacker Starting Position] Start node
    [Reconnaissance activities] disabled
    [Reconnaissance bool features] No
    """
    def __init__(self, idsgame_config: IdsGameConfig = None, save_dir: str = None, initial_state_path: str = None):
        """
        Initialization of the environment

        :param save_dir: directory to save outputs of the env
        :param initial_state_path: path to the initial state (if none, use default)
        :param idsgame_config: configuration of the environment (if not specified a default config is used)
        """
        if idsgame_config is None:
            game_config = GameConfig(num_layers=1, num_servers_per_layer=1, num_attack_types=10, max_value=9)
            game_config.set_initial_state(defense_val=4, attack_val=0, num_vulnerabilities_per_node=4, det_val=3,
                                          vulnerability_val=0, num_vulnerabilities_per_layer=1)
            if initial_state_path is not None:
                game_config.set_load_initial_state(initial_state_path)
            idsgame_config = IdsGameConfig(game_config=game_config)
            idsgame_config.render_config.caption = "idsgame-v1"
        super().__init__(idsgame_config=idsgame_config, save_dir=save_dir)


# -------- Version 2 ------------

class IdsGameRandomDefenseV2Env(AttackerEnv):
    """
    [AttackerEnv] 1 layer, 2 servers per layer, 10 attack-defense-values, random defender
    [Initial State] Defense: 2, Attack:0, Num vulnerabilities: 1, Det: 2, Vulnerability value: 0
    [Rewards] Sparse
    [Version] 2
    [Observations] partially observed
    [Environment] Deterministic
    [Attacker Starting Position] Start node
    [Reconnaissance activities] disabled
    [Reconnaissance bool features] No
    """
    def __init__(self, idsgame_config: IdsGameConfig = None, save_dir: str = None, initial_state_path: str = None):
        """
        Initialization of the environment

        :param save_dir: directory to save outputs of the env
        :param initial_state_path: path to the initial state (if none, use default)
        :param idsgame_config: configuration of the environment (if not specified a default config is used)
        """
        if idsgame_config is None:
            game_config = GameConfig(num_layers=1, num_servers_per_layer=2, num_attack_types=10, max_value=9)
            game_config.set_initial_state(defense_val=2, attack_val=0, num_vulnerabilities_per_node=1, det_val=2,
                                          vulnerability_val=0, num_vulnerabilities_per_layer=2)
            if initial_state_path is not None:
                game_config.set_load_initial_state(initial_state_path)
            defender_agent = RandomDefenseBotAgent(game_config)
            idsgame_config = IdsGameConfig(game_config=game_config, defender_agent=defender_agent)
            idsgame_config.render_config.caption = "idsgame-random_defense-v2"
        super().__init__(idsgame_config=idsgame_config, save_dir=save_dir)


class IdsGameMinimalDefenseV2Env(AttackerEnv):
    """
    [AttackerEnv] 1 layer, 2 servers per layer, 10 attack-defense-values, defender following the "defend minimal strategy"
    [Initial State] Defense: 2, Attack:0, Num vulnerabilities: 1, Det: 2, Vulnerability value: 0
    [Rewards] Sparse
    [Version] 2
    [Observations] partially observed
    [Environment] Deterministic
    [Attacker Starting Position] Start node
    [Reconnaissance activities] disabled
    [Reconnaissance bool features] No
    """
    def __init__(self, idsgame_config: IdsGameConfig = None, save_dir: str = None, initial_state_path: str = None):
        """
        Initialization of the environment

        :param save_dir: directory to save outputs of the env
        :param initial_state_path: path to the initial state (if none, use default)
        :param idsgame_config: configuration of the environment (if not specified a default config is used)
        """
        if idsgame_config is None:
            game_config = GameConfig(num_layers=1, num_servers_per_layer=2, num_attack_types=10, max_value=9)
            game_config.set_initial_state(defense_val=2, attack_val=0, num_vulnerabilities_per_node=1, det_val=2,
                                          vulnerability_val=0, num_vulnerabilities_per_layer=2)
            if initial_state_path is not None:
                game_config.set_load_initial_state(initial_state_path)
            defender_agent = DefendMinimalValueBotAgent(game_config)
            idsgame_config = IdsGameConfig(game_config=game_config, defender_agent=defender_agent)
            idsgame_config.render_config.caption = "idsgame-minimal_defense-v2"
        super().__init__(idsgame_config=idsgame_config, save_dir=save_dir)


class IdsGameRandomAttackV2Env(DefenderEnv):
    """
    [DefenderEnv] 1 layer, 2 servers per layer, 10 attack-defense-values, random attacker
    [Initial State] Defense: 2, Attack:0, Num vulnerabilities: 1, Det: 2, Vulnerability value: 0
    [Rewards] Sparse
    [Version] 2
    [Observations] partially observed
    [Environment] Deterministic
    [Attacker Starting Position] Start node
    [Reconnaissance activities] disabled
    [Reconnaissance bool features] No
    """
    def __init__(self, idsgame_config: IdsGameConfig = None, save_dir: str = None, initial_state_path: str = None):
        """
        Initialization of the environment

        :param save_dir: directory to save outputs of the env
        :param initial_state_path: path to the initial state (if none, use default)
        :param idsgame_config: configuration of the environment (if not specified a default config is used)
        """
        if idsgame_config is None:
            game_config = GameConfig(num_layers=1, num_servers_per_layer=2, num_attack_types=10, max_value=9)
            game_config.set_initial_state(defense_val=2, attack_val=0, num_vulnerabilities_per_node=1, det_val=2,
                                          vulnerability_val=0, num_vulnerabilities_per_layer=2)
            if initial_state_path is not None:
                game_config.set_load_initial_state(initial_state_path)
            attacker_agent = RandomAttackBotAgent(game_config)
            idsgame_config = IdsGameConfig(game_config=game_config, attacker_agent=attacker_agent)
            idsgame_config.render_config.caption = "idsgame-random_attack-v2"
        super().__init__(idsgame_config=idsgame_config, save_dir=save_dir)


class IdsGameMaximalAttackV2Env(DefenderEnv):
    """
    [DefenderEnv] 1 layer, 2 servers per layer, 10 attack-defense-values, attacker following the "attack maximal strategy"
    [Initial State] Defense: 2, Attack:0, Num vulnerabilities: 1, Det: 2, Vulnerability value: 0
    [Rewards] Sparse
    [Version] 2
    [Observations] partially observed
    [Environment] Deterministic
    [Attacker Starting Position] Start node
    [Reconnaissance activities] disabled
    [Reconnaissance bool features] No
    """
    def __init__(self, idsgame_config: IdsGameConfig = None, save_dir: str = None, initial_state_path: str = None):
        """
        Initialization of the environment

        :param save_dir: directory to save outputs of the env
        :param initial_state_path: path to the initial state (if none, use default)
        :param idsgame_config: configuration of the environment (if not specified a default config is used)
        """
        if idsgame_config is None:
            game_config = GameConfig(num_layers=1, num_servers_per_layer=2, num_attack_types=10, max_value=9)
            game_config.set_initial_state(defense_val=2, attack_val=0, num_vulnerabilities_per_node=1, det_val=2,
                                          vulnerability_val=0, num_vulnerabilities_per_layer=2)
            if initial_state_path is not None:
                game_config.set_load_initial_state(initial_state_path)
            attacker_agent = AttackMaximalValueBotAgent(game_config)
            idsgame_config = IdsGameConfig(game_config=game_config, attacker_agent=attacker_agent)
            idsgame_config.render_config.caption = "idsgame-maximal_attack-v2"
        super().__init__(idsgame_config=idsgame_config, save_dir=save_dir)


class IdsGameV2Env(AttackDefenseEnv):
    """
    [AttackDefenseEnv] 1 layer, 2 servers per layer, 10 attack-defense-values
    [Initial State] Defense: 2, Attack:0, Num vulnerabilities: 1, Det: 2, Vulnerability value: 0
    [Rewards] Sparse
    [Version] 2
    [Observations] partially observed
    [Environment] Deterministic
    [Attacker Starting Position] Start node
    [Reconnaissance activities] disabled
    [Reconnaissance bool features] No
    """
    def __init__(self, idsgame_config: IdsGameConfig = None, save_dir: str = None, initial_state_path: str = None):
        """
        Initialization of the environment

        :param save_dir: directory to save outputs of the env
        :param initial_state_path: path to the initial state (if none, use default)
        :param idsgame_config: configuration of the environment (if not specified a default config is used)
        """
        if idsgame_config is None:
            game_config = GameConfig(num_layers=1, num_servers_per_layer=2, num_attack_types=10, max_value=9)
            game_config.set_initial_state(defense_val=2, attack_val=0, num_vulnerabilities_per_node=1, det_val=2,
                                          vulnerability_val=0, num_vulnerabilities_per_layer=2)
            if initial_state_path is not None:
                game_config.set_load_initial_state(initial_state_path)
            idsgame_config = IdsGameConfig(game_config=game_config)
            idsgame_config.render_config.caption = "idsgame-v2"
        super().__init__(idsgame_config=idsgame_config, save_dir=save_dir)


# -------- Version 3 ------------

class IdsGameRandomDefenseV3Env(AttackerEnv):
    """
    [AttackerEnv] 2 layers, 3 servers per layer, 10 attack-defense-values, random defender
    [Initial State] Defense: 2, Attack:0, Num vulnerabilities: 1, Det: 2, Vulnerability value: 0
    [Rewards] Sparse
    [Version] 3
    [Observations] partially observed
    [Environment] Deterministic
    [Attacker Starting Position] Start node
    [Reconnaissance activities] disabled
    [Reconnaissance bool features] No
    """
    def __init__(self, idsgame_config: IdsGameConfig = None, save_dir: str = None, initial_state_path: str = None):
        """
        Initialization of the environment

        :param save_dir: directory to save outputs of the env
        :param initial_state_path: path to the initial state (if none, use default)
        :param idsgame_config: configuration of the environment (if not specified a default config is used)
        """
        if idsgame_config is None:
            game_config = GameConfig(num_layers=2, num_servers_per_layer=3, num_attack_types=10, max_value=9)
            game_config.set_initial_state(defense_val=2, attack_val=0, num_vulnerabilities_per_node=1, det_val=2,
                                          vulnerability_val=0, num_vulnerabilities_per_layer=3)
            if initial_state_path is not None:
                game_config.set_load_initial_state(initial_state_path)
            defender_agent = RandomDefenseBotAgent(game_config)
            idsgame_config = IdsGameConfig(game_config=game_config, defender_agent=defender_agent)
            idsgame_config.render_config.caption = "idsgame-random_defense-v3"
        super().__init__(idsgame_config=idsgame_config, save_dir=save_dir)


class IdsGameMinimalDefenseV3Env(AttackerEnv):
    """
    [AttackerEnv] 2 layers, 3 servers per layer, 10 attack-defense-values, defender following the "defend minimal strategy"
    [Initial State] Defense: 2, Attack:0, Num vulnerabilities: 1, Det: 2, Vulnerability value: 0
    [Rewards] Sparse
    [Version] 3
    [Observations] partially observed
    [Environment] Deterministic
    [Attacker Starting Position] Start node
    [Reconnaissance activities] disabled
    [Reconnaissance bool features] No
    """
    def __init__(self, idsgame_config: IdsGameConfig = None, save_dir: str = None, initial_state_path: str = None):
        """
        Initialization of the environment

        :param save_dir: directory to save outputs of the env
        :param initial_state_path: path to the initial state (if none, use default)
        :param idsgame_config: configuration of the environment (if not specified a default config is used)
        """
        if idsgame_config is None:
            game_config = GameConfig(num_layers=2, num_servers_per_layer=3, num_attack_types=10, max_value=9)
            game_config.set_initial_state(defense_val=2, attack_val=0, num_vulnerabilities_per_node=1, det_val=2,
                                          vulnerability_val=0, num_vulnerabilities_per_layer=3)
            if initial_state_path is not None:
                game_config.set_load_initial_state(initial_state_path)
            defender_agent = DefendMinimalValueBotAgent(game_config)
            idsgame_config = IdsGameConfig(game_config=game_config, defender_agent=defender_agent)
            idsgame_config.render_config.caption = "idsgame-minimal_defense-v3"
        super().__init__(idsgame_config=idsgame_config, save_dir=save_dir)


class IdsGameRandomAttackV3Env(DefenderEnv):
    """
    [DefenderEnv] 2 layers, 3 servers per layer, 10 attack-defense-values, random attacker
    [Initial State] Defense: 2, Attack:0, Num vulnerabilities: 1, Det: 2, Vulnerability value: 0
    [Rewards] Sparse
    [Version] 3
    [Observations] partially observed
    [Environment] Deterministic
    [Attacker Starting Position] Start node
    [Reconnaissance activities] disabled
    [Reconnaissance bool features] No
    """
    def __init__(self, idsgame_config: IdsGameConfig = None, save_dir: str = None, initial_state_path: str = None):
        """
        Initialization of the environment

        :param save_dir: directory to save outputs of the env
        :param initial_state_path: path to the initial state (if none, use default)
        :param idsgame_config: configuration of the environment (if not specified a default config is used)
        """
        if idsgame_config is None:
            game_config = GameConfig(num_layers=2, num_servers_per_layer=3, num_attack_types=10, max_value=9)
            game_config.set_initial_state(defense_val=2, attack_val=0, num_vulnerabilities_per_node=1, det_val=2,
                                          vulnerability_val=0, num_vulnerabilities_per_layer=3)
            if initial_state_path is not None:
                game_config.set_load_initial_state(initial_state_path)
            attacker_agent = RandomAttackBotAgent(game_config)
            idsgame_config = IdsGameConfig(game_config=game_config, attacker_agent=attacker_agent)
            idsgame_config.render_config.caption = "idsgame-random_attack-v3"
        super().__init__(idsgame_config=idsgame_config, save_dir=save_dir)


class IdsGameMaximalAttackV3Env(DefenderEnv):
    """
    [DefenderEnv] 2 layers, 3 servers per layer, 10 attack-defense-values, attacker following the "attack maximal strategy"
    [Initial State] Defense: 2, Attack:0, Num vulnerabilities: 1, Det: 2, Vulnerability value: 0
    [Rewards] Sparse
    [Version] 3
    [Observations] partially observed
    [Environment] Deterministic
    [Attacker Starting Position] Start node
    [Reconnaissance activities] disabled
    [Reconnaissance bool features] No
    """
    def __init__(self, idsgame_config: IdsGameConfig = None, save_dir: str = None, initial_state_path: str = None):
        """
        Initialization of the environment

        :param save_dir: directory to save outputs of the env
        :param initial_state_path: path to the initial state (if none, use default)
        :param idsgame_config: configuration of the environment (if not specified a default config is used)
        """
        if idsgame_config is None:
            game_config = GameConfig(num_layers=2, num_servers_per_layer=3, num_attack_types=10, max_value=9)
            game_config.set_initial_state(defense_val=2, attack_val=0, num_vulnerabilities_per_node=1, det_val=2,
                                          vulnerability_val=0, num_vulnerabilities_per_layer=3)
            if initial_state_path is not None:
                game_config.set_load_initial_state(initial_state_path)
            attacker_agent = AttackMaximalValueBotAgent(game_config)
            idsgame_config = IdsGameConfig(game_config=game_config, attacker_agent=attacker_agent)
            idsgame_config.render_config.caption = "idsgame-maximal_attack-v3"
        super().__init__(idsgame_config=idsgame_config, save_dir=save_dir)


class IdsGameV3Env(AttackDefenseEnv):
    """
    [AttackDefenseEnv] 2 layer, 3 servers per layer, 10 attack-defense-values
    [Initial State] Defense: 2, Attack:0, Num vulnerabilities: 1, Det: 2, Vulnerability value: 0
    [Rewards] Sparse
    [Version] 3
    [Observations] partially observed
    [Environment] Deterministic
    [Attacker Starting Position] Start node
    [Reconnaissance activities] disabled
    [Reconnaissance bool features] No
    """
    def __init__(self, idsgame_config: IdsGameConfig = None, save_dir: str = None, initial_state_path: str = None):
        """
        Initialization of the environment

        :param save_dir: directory to save outputs of the env
        :param initial_state_path: path to the initial state (if none, use default)
        :param idsgame_config: configuration of the environment (if not specified a default config is used)
        """
        if idsgame_config is None:
            game_config = GameConfig(num_layers=2, num_servers_per_layer=3, num_attack_types=10, max_value=9)
            game_config.set_initial_state(defense_val=2, attack_val=0, num_vulnerabilities_per_node=1, det_val=2,
                                          vulnerability_val=0, num_vulnerabilities_per_layer=3)
            if initial_state_path is not None:
                game_config.set_load_initial_state(initial_state_path)
            idsgame_config = IdsGameConfig(game_config=game_config)
            idsgame_config.render_config.caption = "idsgame-v3"
        super().__init__(idsgame_config=idsgame_config, save_dir=save_dir)


# -------- Version 4 ------------

class IdsGameRandomDefenseV4Env(AttackerEnv):
    """
    [AttackerEnv] 4 layers, 5 servers per layer, 10 attack-defense-values, random defender
    [Initial State] Defense: 2, Attack:0, Num vulnerabilities: 1, Det: 2, Vulnerability value: 0
    [Rewards] Sparse
    [Version] 4
    [Observations] partially observed
    [Environment] Deterministic
    [Attacker Starting Position] Start node
    [Reconnaissance activities] disabled
    [Reconnaissance bool features] No
    """
    def __init__(self, idsgame_config: IdsGameConfig = None, save_dir: str = None, initial_state_path: str = None):
        """
        Initialization of the environment

        :param save_dir: directory to save outputs of the env
        :param initial_state_path: path to the initial state (if none, use default)
        :param idsgame_config: configuration of the environment (if not specified a default config is used)
        """
        if idsgame_config is None:
            game_config = GameConfig(num_layers=4, num_servers_per_layer=5, num_attack_types=10, max_value=9)
            game_config.set_initial_state(defense_val=2, attack_val=0, num_vulnerabilities_per_node=1, det_val=2,
                                          vulnerability_val=0, num_vulnerabilities_per_layer=5)
            if initial_state_path is not None:
                game_config.set_load_initial_state(initial_state_path)
            defender_agent = RandomDefenseBotAgent(game_config)
            idsgame_config = IdsGameConfig(game_config=game_config, defender_agent=defender_agent)
            idsgame_config.render_config.caption = "idsgame-random_defense-v4"
        super().__init__(idsgame_config=idsgame_config, save_dir=save_dir)


class IdsGameMinimalDefenseV4Env(AttackerEnv):
    """
    [AttackerEnv] 4 layers, 5 servers per layer, 10 attack-defense-values, defender following the "defend minimal strategy"
    [Initial State] Defense: 2, Attack:0, Num vulnerabilities: 1, Det: 2, Vulnerability value: 0
    [Rewards] Sparse
    [Version] 4
    [Observations] partially observed
    [Environment] Deterministic
    [Attacker Starting Position] Start node
    [Reconnaissance activities] disabled
    [Reconnaissance bool features] No
    """
    def __init__(self, idsgame_config: IdsGameConfig = None, save_dir: str = None, initial_state_path: str = None):
        """
        Initialization of the environment

        :param save_dir: directory to save outputs of the env
        :param initial_state_path: path to the initial state (if none, use default)
        :param idsgame_config: configuration of the environment (if not specified a default config is used)
        """
        if idsgame_config is None:
            game_config = GameConfig(num_layers=4, num_servers_per_layer=5, num_attack_types=10, max_value=9)
            game_config.set_initial_state(defense_val=2, attack_val=0, num_vulnerabilities_per_node=1, det_val=2,
                                          vulnerability_val=0, num_vulnerabilities_per_layer=5)
            if initial_state_path is not None:
                game_config.set_load_initial_state(initial_state_path)
            defender_agent = DefendMinimalValueBotAgent(game_config)
            idsgame_config = IdsGameConfig(game_config=game_config, defender_agent=defender_agent)
            idsgame_config.render_config.caption = "idsgame-minimal_defense-v4"
        super().__init__(idsgame_config=idsgame_config, save_dir=save_dir)


class IdsGameRandomAttackV4Env(DefenderEnv):
    """
    [DefenderEnv] 4 layers, 5 servers per layer, 10 attack-defense-values, random attacker
    [Initial State] Defense: 2, Attack:0, Num vulnerabilities: 1, Det: 2, Vulnerability value: 0
    [Rewards] Sparse
    [Version] 4
    [Observations] partially observed
    [Environment] Deterministic
    [Attacker Starting Position] Start node
    [Reconnaissance activities] disabled
    [Reconnaissance bool features] No
    """
    def __init__(self, idsgame_config: IdsGameConfig = None, save_dir: str = None, initial_state_path: str = None):
        """
        Initialization of the environment

        :param save_dir: directory to save outputs of the env
        :param initial_state_path: path to the initial state (if none, use default)
        :param idsgame_config: configuration of the environment (if not specified a default config is used)
        """
        if idsgame_config is None:
            game_config = GameConfig(num_layers=4, num_servers_per_layer=5, num_attack_types=10, max_value=9)
            game_config.set_initial_state(defense_val=2, attack_val=0, num_vulnerabilities_per_node=1, det_val=2,
                                          vulnerability_val=0, num_vulnerabilities_per_layer=5)
            if initial_state_path is not None:
                game_config.set_load_initial_state(initial_state_path)
            attacker_agent = RandomAttackBotAgent(game_config)
            idsgame_config = IdsGameConfig(game_config=game_config, attacker_agent=attacker_agent)
            idsgame_config.render_config.caption = "idsgame-random_attack-v4"
        super().__init__(idsgame_config=idsgame_config, save_dir=save_dir)


class IdsGameMaximalAttackV4Env(DefenderEnv):
    """
    [DefenderEnv] 4 layers, 5 servers per layer, 10 attack-defense-values, attacker following the "attack maximal strategy"
    [Initial State] Defense: 2, Attack:0, Num vulnerabilities: 1, Det: 2, Vulnerability value: 0
    [Rewards] Sparse
    [Version] 4
    [Observations] partially observed
    [Environment] Deterministic
    [Attacker Starting Position] Start node
    [Reconnaissance activities] disabled
    [Reconnaissance bool features] No
    """
    def __init__(self, idsgame_config: IdsGameConfig = None, save_dir: str = None, initial_state_path: str = None):
        """
        Initialization of the environment

        :param save_dir: directory to save outputs of the env
        :param initial_state_path: path to the initial state (if none, use default)
        :param idsgame_config: configuration of the environment (if not specified a default config is used)
        """
        if idsgame_config is None:
            game_config = GameConfig(num_layers=4, num_servers_per_layer=5, num_attack_types=10, max_value=9)
            game_config.set_initial_state(defense_val=2, attack_val=0, num_vulnerabilities_per_node=1, det_val=2,
                                          vulnerability_val=0, num_vulnerabilities_per_layer=5)
            if initial_state_path is not None:
                game_config.set_load_initial_state(initial_state_path)
            attacker_agent = AttackMaximalValueBotAgent(game_config)
            idsgame_config = IdsGameConfig(game_config=game_config, attacker_agent=attacker_agent)
            idsgame_config.render_config.caption = "idsgame-maximal_attack-v4"
        super().__init__(idsgame_config=idsgame_config, save_dir=save_dir)


class IdsGameV4Env(AttackDefenseEnv):
    """
    [AttackDefenseEnv] 4 layer, 5 servers per layer, 10 attack-defense-values
    [Initial State] Defense: 2, Attack:0, Num vulnerabilities: 1, Det: 2, Vulnerability value: 0
    [Rewards] Sparse
    [Version] 4
    [Observations] partially observed
    [Environment] Deterministic
    [Attacker Starting Position] Start node
    [Reconnaissance activities] disabled
    [Reconnaissance bool features] No
    """
    def __init__(self, idsgame_config: IdsGameConfig = None, save_dir: str = None, initial_state_path: str = None):
        """
        Initialization of the environment

        :param save_dir: directory to save outputs of the env
        :param initial_state_path: path to the initial state (if none, use default)
        :param idsgame_config: configuration of the environment (if not specified a default config is used)
        """
        if idsgame_config is None:
            game_config = GameConfig(num_layers=4, num_servers_per_layer=5, num_attack_types=10, max_value=9)
            game_config.set_initial_state(defense_val=2, attack_val=0, num_vulnerabilities_per_node=1, det_val=2,
                                          vulnerability_val=0, num_vulnerabilities_per_layer=5)
            if initial_state_path is not None:
                game_config.set_load_initial_state(initial_state_path)
            idsgame_config = IdsGameConfig(game_config=game_config)
            idsgame_config.render_config.caption = "idsgame-v4"
        super().__init__(idsgame_config=idsgame_config, save_dir=save_dir)

# -------- Version 5 ------------

class IdsGameRandomDefenseV5Env(AttackerEnv):
    """
    [AttackerEnv] 4 layers, 5 servers per layer, 10 attack-defense-values, random defender, connected layers
    [Initial State] Defense: 2, Attack:0, Num vulnerabilities: 1, Det: 2, Vulnerability value: 0
    [Rewards] Sparse
    [Version] 5
    [Observations] partially observed
    [Environment] Deterministic
    [Attacker Starting Position] Start node
    [Reconnaissance activities] disabled
    [Reconnaissance bool features] No
    """
    def __init__(self, idsgame_config: IdsGameConfig = None, save_dir: str = None, initial_state_path: str = None):
        """
        Initialization of the environment

        :param save_dir: directory to save outputs of the env
        :param initial_state_path: path to the initial state (if none, use default)
        :param idsgame_config: configuration of the environment (if not specified a default config is used)
        """
        if idsgame_config is None:
            game_config = GameConfig(num_layers=4, num_servers_per_layer=5, num_attack_types=10, max_value=9)
            game_config.set_initial_state(defense_val=2, attack_val=0, num_vulnerabilities_per_node=1, det_val=2,
                                          vulnerability_val=0, num_vulnerabilities_per_layer=2)
            game_config.network_config = NetworkConfig(game_config.num_rows, game_config.num_cols,
                                                       connected_layers=True)
            if initial_state_path is not None:
                game_config.set_load_initial_state(initial_state_path)
            defender_agent = RandomDefenseBotAgent(game_config)
            idsgame_config = IdsGameConfig(game_config=game_config, defender_agent=defender_agent)
            idsgame_config.render_config.caption = "idsgame-random_defense-v5"
        super().__init__(idsgame_config=idsgame_config, save_dir=save_dir)


class IdsGameMinimalDefenseV5Env(AttackerEnv):
    """
    [AttackerEnv] 4 layers, 5 servers per layer, 10 attack-defense-values,
    defender following the "defend minimal strategy", connected layers
    [Initial State] Defense: 2, Attack:0, Num vulnerabilities: 1, Det: 2, Vulnerability value: 0
    [Rewards] Sparse
    [Version] 5
    [Observations] partially observed
    [Environment] Deterministic
    [Attacker Starting Position] Start node
    [Reconnaissance activities] disabled
    [Reconnaissance bool features] No
    """
    def __init__(self, idsgame_config: IdsGameConfig = None, save_dir: str = None, initial_state_path: str = None):
        """
        Initialization of the environment

        :param save_dir: directory to save outputs of the env
        :param initial_state_path: path to the initial state (if none, use default)
        :param idsgame_config: configuration of the environment (if not specified a default config is used)
        """
        if idsgame_config is None:
            game_config = GameConfig(num_layers=4, num_servers_per_layer=5, num_attack_types=10, max_value=9)
            game_config.set_initial_state(defense_val=2, attack_val=0, num_vulnerabilities_per_node=1, det_val=2,
                                          vulnerability_val=0, num_vulnerabilities_per_layer=2)
            game_config.network_config = NetworkConfig(game_config.num_rows, game_config.num_cols,
                                                       connected_layers=True)
            if initial_state_path is not None:
                game_config.set_load_initial_state(initial_state_path)
            defender_agent = DefendMinimalValueBotAgent(game_config)
            idsgame_config = IdsGameConfig(game_config=game_config, defender_agent=defender_agent)
            idsgame_config.render_config.caption = "idsgame-minimal_defense-v5"
        super().__init__(idsgame_config=idsgame_config, save_dir=save_dir)


class IdsGameRandomAttackV5Env(DefenderEnv):
    """
    [DefenderEnv] 4 layers, 5 servers per layer, 10 attack-defense-values, random attacker, connected layers
    [Initial State] Defense: 2, Attack:0, Num vulnerabilities: 1, Det: 2, Vulnerability value: 0
    [Rewards] Sparse
    [Version] 5
    [Observations] partially observed
    [Environment] Deterministic
    [Attacker Starting Position] Start node
    [Reconnaissance activities] disabled
    [Reconnaissance bool features] No
    """
    def __init__(self, idsgame_config: IdsGameConfig = None, save_dir: str = None, initial_state_path: str = None):
        """
        Initialization of the environment

        :param save_dir: directory to save outputs of the env
        :param initial_state_path: path to the initial state (if none, use default)
        :param idsgame_config: configuration of the environment (if not specified a default config is used)
        """
        if idsgame_config is None:
            game_config = GameConfig(num_layers=4, num_servers_per_layer=5, num_attack_types=10, max_value=9)
            game_config.set_initial_state(defense_val=2, attack_val=0, num_vulnerabilities_per_node=1, det_val=2,
                                          vulnerability_val=0, num_vulnerabilities_per_layer=2)
            game_config.network_config = NetworkConfig(game_config.num_rows, game_config.num_cols,
                                                       connected_layers=True)
            if initial_state_path is not None:
                game_config.set_load_initial_state(initial_state_path)
            attacker_agent = RandomAttackBotAgent(game_config)
            idsgame_config = IdsGameConfig(game_config=game_config, attacker_agent=attacker_agent)
            idsgame_config.render_config.caption = "idsgame-random_attack-v5"
        super().__init__(idsgame_config=idsgame_config, save_dir=save_dir)


class IdsGameMaximalAttackV5Env(DefenderEnv):
    """
    [DefenderEnv] 4 layers, 5 servers per layer, 10 attack-defense-values,
    attacker following the "attack maximal strategy", connected layers
    [Initial State] Defense: 2, Attack:0, Num vulnerabilities: 1, Det: 2, Vulnerability value: 0
    [Rewards] Sparse
    [Version] 5
    [Observations] partially observed
    [Environment] Deterministic
    [Attacker Starting Position] Start node
    [Reconnaissance activities] disabled
    [Reconnaissance bool features] No
    """
    def __init__(self, idsgame_config: IdsGameConfig = None, save_dir: str = None, initial_state_path: str = None):
        """
        Initialization of the environment

        :param save_dir: directory to save outputs of the env
        :param initial_state_path: path to the initial state (if none, use default)
        :param idsgame_config: configuration of the environment (if not specified a default config is used)
        """
        if idsgame_config is None:
            game_config = GameConfig(num_layers=4, num_servers_per_layer=5, num_attack_types=10, max_value=9)
            game_config.set_initial_state(defense_val=2, attack_val=0, num_vulnerabilities_per_node=1, det_val=2,
                                          vulnerability_val=0, num_vulnerabilities_per_layer=2)
            game_config.network_config = NetworkConfig(game_config.num_rows, game_config.num_cols,
                                                       connected_layers=True)
            if initial_state_path is not None:
                game_config.set_load_initial_state(initial_state_path)
            attacker_agent = AttackMaximalValueBotAgent(game_config)
            idsgame_config = IdsGameConfig(game_config=game_config, attacker_agent=attacker_agent)
            idsgame_config.render_config.caption = "idsgame-maximal_attack-v5"
        super().__init__(idsgame_config=idsgame_config, save_dir=save_dir)


class IdsGameV5Env(AttackDefenseEnv):
    """
    [AttackDefenseEnv] 4 layer, 5 servers per layer, 10 attack-defense-values, connected layers
    [Initial State] Defense: 2, Attack:0, Num vulnerabilities: 1, Det: 2, Vulnerability value: 0
    [Rewards] Sparse
    [Version] 5
    [Observations] partially observed
    [Environment] Deterministic
    [Attacker Starting Position] Start node
    [Reconnaissance activities] disabled
    [Reconnaissance bool features] No
    """
    def __init__(self, idsgame_config: IdsGameConfig = None, save_dir: str = None, initial_state_path: str = None):
        """
        Initialization of the environment

        :param save_dir: directory to save outputs of the env
        :param initial_state_path: path to the initial state (if none, use default)
        :param idsgame_config: configuration of the environment (if not specified a default config is used)
        """
        if idsgame_config is None:
            game_config = GameConfig(num_layers=4, num_servers_per_layer=5, num_attack_types=10, max_value=9)
            game_config.set_initial_state(defense_val=2, attack_val=0, num_vulnerabilities_per_node=1, det_val=2,
                                          vulnerability_val=0, num_vulnerabilities_per_layer=2)
            game_config.network_config = NetworkConfig(game_config.num_rows, game_config.num_cols,
                                                       connected_layers=True)
            if initial_state_path is not None:
                game_config.set_load_initial_state(initial_state_path)
            idsgame_config = IdsGameConfig(game_config=game_config)
            idsgame_config.render_config.caption = "idsgame-v5"
        super().__init__(idsgame_config=idsgame_config, save_dir=save_dir)

# -------- Version 6 ------------

class IdsGameRandomDefenseV6Env(AttackerEnv):
    """
    [AttackerEnv] 4 layers, 5 servers per layer, 10 attack-defense-values, random defender, connected layers
    [Initial State] Defense: 2, Attack:0, Num vulnerabilities: 1, Det: 2, Vulnerability value: 0
    [Rewards]: Dense
    [Version] 6
    [Observations] partially observed
    [Environment] Deterministic
    [Attacker Starting Position] Start node
    [Reconnaissance activities] disabled
    [Reconnaissance bool features] No
    """
    def __init__(self, idsgame_config: IdsGameConfig = None, save_dir: str = None, initial_state_path: str = None):
        """
        Initialization of the environment

        :param save_dir: directory to save outputs of the env
        :param initial_state_path: path to the initial state (if none, use default)
        :param idsgame_config: configuration of the environment (if not specified a default config is used)
        """
        if idsgame_config is None:
            game_config = GameConfig(num_layers=4, num_servers_per_layer=5, num_attack_types=10, max_value=9)
            game_config.set_initial_state(defense_val=2, attack_val=0, num_vulnerabilities_per_node=1, det_val=2,
                                          vulnerability_val=0, num_vulnerabilities_per_layer=2)
            game_config.network_config = NetworkConfig(game_config.num_rows, game_config.num_cols,
                                                       connected_layers=True)
            game_config.dense_rewards = True
            if initial_state_path is not None:
                game_config.set_load_initial_state(initial_state_path)
            defender_agent = RandomDefenseBotAgent(game_config)
            idsgame_config = IdsGameConfig(game_config=game_config, defender_agent=defender_agent)
            idsgame_config.render_config.caption = "idsgame-random_defense-v6"
        super().__init__(idsgame_config=idsgame_config, save_dir=save_dir)


class IdsGameMinimalDefenseV6Env(AttackerEnv):
    """
    [AttackerEnv] 4 layers, 5 servers per layer, 10 attack-defense-values,
    defender following the "defend minimal strategy", connected layers
    [Initial State] Defense: 2, Attack:0, Num vulnerabilities: 1, Det: 2, Vulnerability value: 0
    [Rewards]: Dense
    [Version] 6
    [Observations] partially observed
    [Environment] Deterministic
    [Attacker Starting Position] Start node
    [Reconnaissance activities] disabled
    [Reconnaissance bool features] No
    """
    def __init__(self, idsgame_config: IdsGameConfig = None, save_dir: str = None, initial_state_path: str = None):
        """
        Initialization of the environment

        :param save_dir: directory to save outputs of the env
        :param initial_state_path: path to the initial state (if none, use default)
        :param idsgame_config: configuration of the environment (if not specified a default config is used)
        """
        if idsgame_config is None:
            game_config = GameConfig(num_layers=4, num_servers_per_layer=5, num_attack_types=10, max_value=9)
            game_config.set_initial_state(defense_val=2, attack_val=0, num_vulnerabilities_per_node=1, det_val=2,
                                          vulnerability_val=0, num_vulnerabilities_per_layer=2)
            game_config.network_config = NetworkConfig(game_config.num_rows, game_config.num_cols,
                                                       connected_layers=True)
            game_config.dense_rewards = True
            if initial_state_path is not None:
                game_config.set_load_initial_state(initial_state_path)
            defender_agent = DefendMinimalValueBotAgent(game_config)
            idsgame_config = IdsGameConfig(game_config=game_config, defender_agent=defender_agent)
            idsgame_config.render_config.caption = "idsgame-minimal_defense-v6"
        super().__init__(idsgame_config=idsgame_config, save_dir=save_dir)


class IdsGameRandomAttackV6Env(DefenderEnv):
    """
    [DefenderEnv] 4 layers, 5 servers per layer, 10 attack-defense-values, random attacker, connected layers
    [Initial State] Defense: 2, Attack:0, Num vulnerabilities: 1, Det: 2, Vulnerability value: 0
    [Rewards] Dense
    [Version] 6
    [Observations] partially observed
    [Environment] Deterministic
    [Attacker Starting Position] Start node
    [Reconnaissance activities] disabled
    [Reconnaissance bool features] No
    """
    def __init__(self, idsgame_config: IdsGameConfig = None, save_dir: str = None, initial_state_path: str = None):
        """
        Initialization of the environment

        :param save_dir: directory to save outputs of the env
        :param initial_state_path: path to the initial state (if none, use default)
        :param idsgame_config: configuration of the environment (if not specified a default config is used)
        """
        if idsgame_config is None:
            game_config = GameConfig(num_layers=4, num_servers_per_layer=5, num_attack_types=10, max_value=9)
            game_config.set_initial_state(defense_val=2, attack_val=0, num_vulnerabilities_per_node=1, det_val=2,
                                          vulnerability_val=0, num_vulnerabilities_per_layer=2)
            game_config.network_config = NetworkConfig(game_config.num_rows, game_config.num_cols,
                                                       connected_layers=True)
            game_config.dense_rewards = True
            if initial_state_path is not None:
                game_config.set_load_initial_state(initial_state_path)
            attacker_agent = RandomAttackBotAgent(game_config)
            idsgame_config = IdsGameConfig(game_config=game_config, attacker_agent=attacker_agent)
            idsgame_config.render_config.caption = "idsgame-random_attack-v6"
        super().__init__(idsgame_config=idsgame_config, save_dir=save_dir)


class IdsGameMaximalAttackV6Env(DefenderEnv):
    """
    [DefenderEnv] 4 layers, 5 servers per layer, 10 attack-defense-values,
    attacker following the "attack maximal strategy", connected layers
    [Initial State] Defense: 2, Attack:0, Num vulnerabilities: 1, Det: 2, Vulnerability value: 0
    [Rewards]: Dense
    [Version] 6
    [Observations] partially observed
    [Environment] Deterministic
    [Attacker Starting Position] Start node
    [Reconnaissance activities] disabled
    [Reconnaissance bool features] No
    """
    def __init__(self, idsgame_config: IdsGameConfig = None, save_dir: str = None, initial_state_path: str = None):
        """
        Initialization of the environment

        :param save_dir: directory to save outputs of the env
        :param initial_state_path: path to the initial state (if none, use default)
        :param idsgame_config: configuration of the environment (if not specified a default config is used)
        """
        if idsgame_config is None:
            game_config = GameConfig(num_layers=4, num_servers_per_layer=5, num_attack_types=10, max_value=9)
            game_config.set_initial_state(defense_val=2, attack_val=0, num_vulnerabilities_per_node=1, det_val=2,
                                          vulnerability_val=0, num_vulnerabilities_per_layer=2)
            game_config.network_config = NetworkConfig(game_config.num_rows, game_config.num_cols,
                                                       connected_layers=True)
            game_config.dense_rewards = True
            if initial_state_path is not None:
                game_config.set_load_initial_state(initial_state_path)
            attacker_agent = AttackMaximalValueBotAgent(game_config)
            idsgame_config = IdsGameConfig(game_config=game_config, attacker_agent=attacker_agent)
            idsgame_config.render_config.caption = "idsgame-maximal_attack-v6"
        super().__init__(idsgame_config=idsgame_config, save_dir=save_dir)


class IdsGameV6Env(AttackDefenseEnv):
    """
    [AttackDefenseEnv] 4 layer, 5 servers per layer, 10 attack-defense-values, connected layers
    [Initial State] Defense: 2, Attack:0, Num vulnerabilities: 1, Det: 2, Vulnerability value: 0
    [Rewards] Dense
    [Version] 6
    [Observations] partially observed
    [Environment] Deterministic
    [Attacker Starting Position] Start node
    [Reconnaissance activities] disabled
    [Reconnaissance bool features] No
    """
    def __init__(self, idsgame_config: IdsGameConfig = None, save_dir: str = None, initial_state_path: str = None):
        """
        Initialization of the environment

        :param save_dir: directory to save outputs of the env
        :param initial_state_path: path to the initial state (if none, use default)
        :param idsgame_config: configuration of the environment (if not specified a default config is used)
        """
        if idsgame_config is None:
            game_config = GameConfig(num_layers=4, num_servers_per_layer=5, num_attack_types=10, max_value=9)
            game_config.set_initial_state(defense_val=2, attack_val=0, num_vulnerabilities_per_node=1, det_val=2,
                                          vulnerability_val=0, num_vulnerabilities_per_layer=2)
            game_config.network_config = NetworkConfig(game_config.num_rows, game_config.num_cols,
                                                       connected_layers=True)
            game_config.dense_rewards = True
            if initial_state_path is not None:
                game_config.set_load_initial_state(initial_state_path)
            idsgame_config = IdsGameConfig(game_config=game_config)
            idsgame_config.render_config.caption = "idsgame-v6"
        super().__init__(idsgame_config=idsgame_config, save_dir=save_dir)


# -------- Version 7 ------------

class IdsGameRandomDefenseV7Env(AttackerEnv):
    """
    [AttackerEnv] 2 layers, 3 servers per layer, 10 attack-defense-values, random defender
    [Initial State] Defense: 2, Attack:0, Num vulnerabilities: 1, Det: 2, Vulnerability value: 0
    [Rewards] Dense
    [Version] 7
    [Observations] partially observed
    [Environment] Deterministic
    [Attacker Starting Position] Start node
    [Reconnaissance activities] disabled
    [Reconnaissance bool features] No
    """
    def __init__(self, idsgame_config: IdsGameConfig = None, save_dir: str = None, initial_state_path: str = None):
        """
        Initialization of the environment

        :param save_dir: directory to save outputs of the env
        :param initial_state_path: path to the initial state (if none, use default)
        :param idsgame_config: configuration of the environment (if not specified a default config is used)
        """
        if idsgame_config is None:
            game_config = GameConfig(num_layers=2, num_servers_per_layer=3, num_attack_types=10, max_value=9)
            game_config.set_initial_state(defense_val=2, attack_val=0, num_vulnerabilities_per_node=1, det_val=2,
                                          vulnerability_val=0, num_vulnerabilities_per_layer=3)
            game_config.dense_rewards = True
            if initial_state_path is not None:
                game_config.set_load_initial_state(initial_state_path)
            defender_agent = RandomDefenseBotAgent(game_config)
            idsgame_config = IdsGameConfig(game_config=game_config, defender_agent=defender_agent)
            idsgame_config.render_config.caption = "idsgame-random_defense-v7"
        super().__init__(idsgame_config=idsgame_config, save_dir=save_dir)


class IdsGameMinimalDefenseV7Env(AttackerEnv):
    """
    [AttackerEnv] 2 layers, 3 servers per layer, 10 attack-defense-values, defender following the "defend minimal strategy"
    [Initial State] Defense: 2, Attack:0, Num vulnerabilities: 1, Det: 2, Vulnerability value: 0
    [Rewards] Dense
    [Version] 7
    [Observations] partially observed
    [Environment] Deterministic
    [Attacker Starting Position] Start node
    [Reconnaissance activities] disabled
    [Reconnaissance bool features] No
    """
    def __init__(self, idsgame_config: IdsGameConfig = None, save_dir: str = None, initial_state_path: str = None):
        """
        Initialization of the environment

        :param save_dir: directory to save outputs of the env
        :param initial_state_path: path to the initial state (if none, use default)
        :param idsgame_config: configuration of the environment (if not specified a default config is used)
        """
        if idsgame_config is None:
            game_config = GameConfig(num_layers=2, num_servers_per_layer=3, num_attack_types=10, max_value=9)
            game_config.set_initial_state(defense_val=2, attack_val=0, num_vulnerabilities_per_node=1, det_val=2,
                                          vulnerability_val=0, num_vulnerabilities_per_layer=3)
            game_config.dense_rewards = True
            if initial_state_path is not None:
                game_config.set_load_initial_state(initial_state_path)
            defender_agent = DefendMinimalValueBotAgent(game_config)
            idsgame_config = IdsGameConfig(game_config=game_config, defender_agent=defender_agent)
            idsgame_config.render_config.caption = "idsgame-minimal_defense-v7"
        super().__init__(idsgame_config=idsgame_config, save_dir=save_dir)


class IdsGameRandomAttackV7Env(DefenderEnv):
    """
    [DefenderEnv] 2 layers, 3 servers per layer, 10 attack-defense-values, random attacker
    [Initial State] Defense: 2, Attack:0, Num vulnerabilities: 1, Det: 2, Vulnerability value: 0
    [Rewards] Dense
    [Version] 7
    [Observations] partially observed
    [Environment] Deterministic
    [Attacker Starting Position] Start node
    [Reconnaissance activities] disabled
    [Reconnaissance bool features] No
    """
    def __init__(self, idsgame_config: IdsGameConfig = None, save_dir: str = None, initial_state_path: str = None):
        """
        Initialization of the environment

        :param save_dir: directory to save outputs of the env
        :param initial_state_path: path to the initial state (if none, use default)
        :param idsgame_config: configuration of the environment (if not specified a default config is used)
        """
        if idsgame_config is None:
            game_config = GameConfig(num_layers=2, num_servers_per_layer=3, num_attack_types=10, max_value=9)
            game_config.set_initial_state(defense_val=2, attack_val=0, num_vulnerabilities_per_node=1, det_val=2,
                                          vulnerability_val=0, num_vulnerabilities_per_layer=3)
            game_config.dense_rewards = True
            if initial_state_path is not None:
                game_config.set_load_initial_state(initial_state_path)
            attacker_agent = RandomAttackBotAgent(game_config)
            idsgame_config = IdsGameConfig(game_config=game_config, attacker_agent=attacker_agent)
            idsgame_config.render_config.caption = "idsgame-random_attack-v7"
        super().__init__(idsgame_config=idsgame_config, save_dir=save_dir)


class IdsGameMaximalAttackV7Env(DefenderEnv):
    """
    [DefenderEnv] 2 layers, 3 servers per layer, 10 attack-defense-values, attacker following the "attack maximal strategy"
    [Initial State] Defense: 2, Attack:0, Num vulnerabilities: 1, Det: 2, Vulnerability value: 0
    [Rewards] Dense
    [Version] 7
    [Observations] partially observed
    [Environment] Deterministic
    [Attacker Starting Position] Start node
    [Reconnaissance activities] disabled
    [Reconnaissance bool features] No
    """
    def __init__(self, idsgame_config: IdsGameConfig = None, save_dir: str = None, initial_state_path: str = None):
        """
        Initialization of the environment

        :param save_dir: directory to save outputs of the env
        :param initial_state_path: path to the initial state (if none, use default)
        :param idsgame_config: configuration of the environment (if not specified a default config is used)
        """
        if idsgame_config is None:
            game_config = GameConfig(num_layers=2, num_servers_per_layer=3, num_attack_types=10, max_value=9)
            game_config.set_initial_state(defense_val=2, attack_val=0, num_vulnerabilities_per_node=1, det_val=2,
                                          vulnerability_val=0, num_vulnerabilities_per_layer=3)
            game_config.dense_rewards = True
            if initial_state_path is not None:
                game_config.set_load_initial_state(initial_state_path)
            attacker_agent = AttackMaximalValueBotAgent(game_config)
            idsgame_config = IdsGameConfig(game_config=game_config, attacker_agent=attacker_agent)
            idsgame_config.render_config.caption = "idsgame-maximal_attack-v7"
        super().__init__(idsgame_config=idsgame_config, save_dir=save_dir)


class IdsGameV7Env(AttackDefenseEnv):
    """
    [AttackDefenseEnv] 2 layer, 3 servers per layer, 10 attack-defense-values
    [Initial State] Defense: 2, Attack:0, Num vulnerabilities: 1, Det: 2, Vulnerability value: 0
    [Rewards] Dense
    [Version] 7
    [Observations] partially observed
    [Environment] Deterministic
    [Attacker Starting Position] Start node
    [Reconnaissance activities] disabled
    [Reconnaissance bool features] No
    """
    def __init__(self, idsgame_config: IdsGameConfig = None, save_dir: str = None, initial_state_path: str = None):
        """
        Initialization of the environment

        :param save_dir: directory to save outputs of the env
        :param initial_state_path: path to the initial state (if none, use default)
        :param idsgame_config: configuration of the environment (if not specified a default config is used)
        """
        if idsgame_config is None:
            game_config = GameConfig(num_layers=2, num_servers_per_layer=3, num_attack_types=10, max_value=9)
            game_config.set_initial_state(defense_val=2, attack_val=0, num_vulnerabilities_per_node=1, det_val=2,
                                          vulnerability_val=0, num_vulnerabilities_per_layer=3)
            game_config.dense_rewards = True
            if initial_state_path is not None:
                game_config.set_load_initial_state(initial_state_path)
            idsgame_config = IdsGameConfig(game_config=game_config)
            idsgame_config.render_config.caption = "idsgame-v7"
        super().__init__(idsgame_config=idsgame_config, save_dir=save_dir)


# -------- Version 8 ------------

class IdsGameRandomDefenseV8Env(AttackerEnv):
    """
    [AttackerEnv] 1 layer, 1 server per layer, 10 attack-defense-values, random defender
    [Initial State] Defense: 2, Attack:0, Num vulnerabilities: 1, Det: 2, Vulnerability value: 0
    [Rewards] Dense
    [Version] 8
    [Observations] partially observed
    [Environment] Deterministic
    [Attacker Starting Position] Start node
    [Reconnaissance activities] disabled
    [Reconnaissance bool features] No
    """
    def __init__(self, idsgame_config: IdsGameConfig = None, save_dir: str = None, initial_state_path: str = None):
        """
        Initialization of the environment

        :param save_dir: directory to save outputs of the env
        :param initial_state_path: path to the initial state (if none, use default)
        :param idsgame_config: configuration of the environment (if not specified a default config is used)
        """
        if idsgame_config is None:
            game_config = GameConfig(num_layers=1, num_servers_per_layer=1, num_attack_types=10, max_value=9)
            game_config.set_initial_state(defense_val=2, attack_val=0, num_vulnerabilities_per_node=1, det_val=2,
                                          vulnerability_val=0, num_vulnerabilities_per_layer=1)
            game_config.dense_rewards = True
            if initial_state_path is not None:
                game_config.set_load_initial_state(initial_state_path)
            defender_agent = RandomDefenseBotAgent(game_config)
            idsgame_config = IdsGameConfig(game_config=game_config, defender_agent=defender_agent)
            idsgame_config.render_config.caption = "idsgame-random_defense-v8"
        super().__init__(idsgame_config=idsgame_config, save_dir=save_dir)


class IdsGameMinimalDefenseV8Env(AttackerEnv):
    """
    [AttackerEnv] 1 layer, 1 server per layer, 10 attack-defense-values, defender following the "defend minimal strategy"
    [Initial State] Defense: 2, Attack:0, Num vulnerabilities: 1, Det: 2, Vulnerability value: 0
    [Rewards] Dense
    [Version] 8
    [Observations] partially observed
    [Environment] Deterministic
    [Attacker Starting Position] Start node
    [Reconnaissance activities] disabled
    [Reconnaissance bool features] No
    """
    def __init__(self, idsgame_config: IdsGameConfig = None, save_dir: str = None, initial_state_path: str = None):
        """
        Initialization of the environment

        :param save_dir: directory to save outputs of the env
        :param initial_state_path: path to the initial state (if none, use default)
        :param idsgame_config: configuration of the environment (if not specified a default config is used)
        """
        if idsgame_config is None:
            game_config = GameConfig(num_layers=1, num_servers_per_layer=1, num_attack_types=10, max_value=9)
            game_config.set_initial_state(defense_val=2, attack_val=0, num_vulnerabilities_per_node=1, det_val=2,
                                          vulnerability_val=0, num_vulnerabilities_per_layer=1)
            game_config.dense_rewards = True
            if initial_state_path is not None:
                game_config.set_load_initial_state(initial_state_path)
            defender_agent = DefendMinimalValueBotAgent(game_config)
            idsgame_config = IdsGameConfig(game_config=game_config, defender_agent=defender_agent)
            idsgame_config.render_config.caption = "idsgame-minimal_defense-v8"
        super().__init__(idsgame_config=idsgame_config, save_dir=save_dir)


class IdsGameRandomAttackV8Env(DefenderEnv):
    """
    [DefenderEnv] 1 layer, 1 server per layer, 10 attack-defense-values, random attacker
    [Initial State] Defense: 2, Attack:0, Num vulnerabilities: 1, Det: 2, Vulnerability value: 0
    [Rewards] Dense
    [Version] 8
    [Observations] partially observed
    [Environment] Deterministic
    [Attacker Starting Position] Start node
    [Reconnaissance activities] disabled
    [Reconnaissance bool features] No
    """
    def __init__(self, idsgame_config: IdsGameConfig = None, save_dir: str = None, initial_state_path: str = None):
        """
        Initialization of the environment

        :param save_dir: directory to save outputs of the env
        :param initial_state_path: path to the initial state (if none, use default)
        :param idsgame_config: configuration of the environment (if not specified a default config is used)
        """
        if idsgame_config is None:
            game_config = GameConfig(num_layers=1, num_servers_per_layer=1, num_attack_types=10, max_value=9)
            game_config.set_initial_state(defense_val=2, attack_val=0, num_vulnerabilities_per_node=1, det_val=2,
                                          vulnerability_val=0, num_vulnerabilities_per_layer=1)
            game_config.dense_rewards = True
            if initial_state_path is not None:
                game_config.set_load_initial_state(initial_state_path)
            attacker_agent = RandomAttackBotAgent(game_config)
            idsgame_config = IdsGameConfig(game_config=game_config, attacker_agent=attacker_agent)
            idsgame_config.render_config.caption = "idsgame-random_attack-v8"
        super().__init__(idsgame_config=idsgame_config, save_dir=save_dir)


class IdsGameMaximalAttackV8Env(DefenderEnv):
    """
    [DefenderEnv] 1 layer, 1 server per layer, 10 attack-defense-values, attacker following the "attack maximal strategy"
    [Initial State] Defense: 2, Attack:0, Num vulnerabilities: 1, Det: 2, Vulnerability value: 0
    [Rewards] Dense
    [Version] 8
    [Observations] partially observed
    [Environment] Deterministic
    [Attacker Starting Position] Start node
    [Reconnaissance activities] disabled
    [Reconnaissance bool features] No
    """
    def __init__(self, idsgame_config: IdsGameConfig = None, save_dir: str = None, initial_state_path: str = None):
        """
        Initialization of the environment

        :param save_dir: directory to save outputs of the env
        :param initial_state_path: path to the initial state (if none, use default)
        :param idsgame_config: configuration of the environment (if not specified a default config is used)
        """
        if idsgame_config is None:
            game_config = GameConfig(num_layers=1, num_servers_per_layer=1, num_attack_types=10, max_value=9)
            game_config.set_initial_state(defense_val=2, attack_val=0, num_vulnerabilities_per_node=1, det_val=2,
                                          vulnerability_val=0, num_vulnerabilities_per_layer=1)
            game_config.dense_rewards = True
            if initial_state_path is not None:
                game_config.set_load_initial_state(initial_state_path)
            attacker_agent = AttackMaximalValueBotAgent(game_config)
            idsgame_config = IdsGameConfig(game_config=game_config, attacker_agent=attacker_agent)
            idsgame_config.render_config.caption = "idsgame-maximal_attack-v8"
        super().__init__(idsgame_config=idsgame_config, save_dir=save_dir)


class IdsGameV8Env(AttackDefenseEnv):
    """
    [AttackDefenseEnv] 1 layer, 1 server per layer, 10 attack-defense-values
    [Initial State] Defense: 2, Attack:0, Num vulnerabilities: 1, Det: 2, Vulnerability value: 0
    [Rewards] Dense
    [Version] 8
    [Observations] partially observed
    [Environment] Deterministic
    [Attacker Starting Position] Start node
    [Reconnaissance activities] disabled
    [Reconnaissance bool features] No
    """
    def __init__(self, idsgame_config: IdsGameConfig = None, save_dir: str = None, initial_state_path: str = None):
        """
        Initialization of the environment

        :param save_dir: directory to save outputs of the env
        :param initial_state_path: path to the initial state (if none, use default)
        :param idsgame_config: configuration of the environment (if not specified a default config is used)
        """
        if idsgame_config is None:
            game_config = GameConfig(num_layers=1, num_servers_per_layer=1, num_attack_types=10, max_value=9)
            game_config.set_initial_state(defense_val=2, attack_val=0, num_vulnerabilities_per_node=1, det_val=2,
                                          vulnerability_val=0, num_vulnerabilities_per_layer=1)
            game_config.dense_rewards = True
            if initial_state_path is not None:
                game_config.set_load_initial_state(initial_state_path)
            idsgame_config = IdsGameConfig(game_config=game_config)
            idsgame_config.render_config.caption = "idsgame-v8"
        super().__init__(idsgame_config=idsgame_config, save_dir=save_dir)


# -------- Version 9 ------------

class IdsGameRandomDefenseV9Env(AttackerEnv):
    """
    [AttackerEnv] 1 layer, 2 servers per layer, 10 attack-defense-values, random defender
    [Initial State] Defense: 2, Attack:0, Num vulnerabilities: 1, Det: 2, Vulnerability value: 0
    [Rewards] Dense
    [Version] 9
    [Observations] partially observed
    [Environment] Deterministic
    [Attacker Starting Position] Start node
    [Reconnaissance activities] disabled
    [Reconnaissance bool features] No
    """
    def __init__(self, idsgame_config: IdsGameConfig = None, save_dir: str = None, initial_state_path: str = None):
        """
        Initialization of the environment

        :param save_dir: directory to save outputs of the env
        :param initial_state_path: path to the initial state (if none, use default)
        :param idsgame_config: configuration of the environment (if not specified a default config is used)
        """
        if idsgame_config is None:
            game_config = GameConfig(num_layers=1, num_servers_per_layer=2, num_attack_types=10, max_value=9)
            game_config.set_initial_state(defense_val=2, attack_val=0, num_vulnerabilities_per_node=1, det_val=2,
                                          vulnerability_val=0, num_vulnerabilities_per_layer=2)
            game_config.dense_rewards = True
            if initial_state_path is not None:
                game_config.set_load_initial_state(initial_state_path)
            defender_agent = RandomDefenseBotAgent(game_config)
            idsgame_config = IdsGameConfig(game_config=game_config, defender_agent=defender_agent)
            idsgame_config.render_config.caption = "idsgame-random_defense-v9"
        super().__init__(idsgame_config=idsgame_config, save_dir=save_dir)


class IdsGameMinimalDefenseV9Env(AttackerEnv):
    """
    [AttackerEnv] 1 layer, 2 servers per layer, 10 attack-defense-values, defender following the "defend minimal strategy"
    [Initial State] Defense: 2, Attack:0, Num vulnerabilities: 1, Det: 2, Vulnerability value: 0
    [Rewards] Dense
    [Version] 9
    [Observations] partially observed
    [Environment] Deterministic
    [Attacker Starting Position] Start node
    [Reconnaissance activities] disabled
    [Reconnaissance bool features] No
    """
    def __init__(self, idsgame_config: IdsGameConfig = None, save_dir: str = None, initial_state_path: str = None):
        """
        Initialization of the environment

        :param save_dir: directory to save outputs of the env
        :param initial_state_path: path to the initial state (if none, use default)
        :param idsgame_config: configuration of the environment (if not specified a default config is used)
        """
        if idsgame_config is None:
            game_config = GameConfig(num_layers=1, num_servers_per_layer=2, num_attack_types=10, max_value=9)
            game_config.set_initial_state(defense_val=2, attack_val=0, num_vulnerabilities_per_node=1, det_val=2,
                                          vulnerability_val=0, num_vulnerabilities_per_layer=2)
            game_config.dense_rewards = True
            if initial_state_path is not None:
                game_config.set_load_initial_state(initial_state_path)
            defender_agent = DefendMinimalValueBotAgent(game_config)
            idsgame_config = IdsGameConfig(game_config=game_config, defender_agent=defender_agent)
            idsgame_config.render_config.caption = "idsgame-minimal_defense-v9"
        super().__init__(idsgame_config=idsgame_config, save_dir=save_dir)


class IdsGameRandomAttackV9Env(DefenderEnv):
    """
    [DefenderEnv] 1 layer, 2 servers per layer, 10 attack-defense-values, random attacker
    [Initial State] Defense: 2, Attack:0, Num vulnerabilities: 1, Det: 2, Vulnerability value: 0
    [Rewards] Dense
    [Version] 9
    [Observations] partially observed
    [Environment] Deterministic
    [Attacker Starting Position] Start node
    [Reconnaissance activities] disabled
    [Reconnaissance bool features] No
    """
    def __init__(self, idsgame_config: IdsGameConfig = None, save_dir: str = None, initial_state_path: str = None):
        """
        Initialization of the environment

        :param save_dir: directory to save outputs of the env
        :param initial_state_path: path to the initial state (if none, use default)
        :param idsgame_config: configuration of the environment (if not specified a default config is used)
        """
        if idsgame_config is None:
            game_config = GameConfig(num_layers=1, num_servers_per_layer=2, num_attack_types=10, max_value=9)
            game_config.set_initial_state(defense_val=2, attack_val=0, num_vulnerabilities_per_node=1, det_val=2,
                                          vulnerability_val=0, num_vulnerabilities_per_layer=2)
            game_config.dense_rewards = True
            if initial_state_path is not None:
                game_config.set_load_initial_state(initial_state_path)
            attacker_agent = RandomAttackBotAgent(game_config)
            idsgame_config = IdsGameConfig(game_config=game_config, attacker_agent=attacker_agent)
            idsgame_config.render_config.caption = "idsgame-random_attack-v9"
        super().__init__(idsgame_config=idsgame_config, save_dir=save_dir)


class IdsGameMaximalAttackV9Env(DefenderEnv):
    """
    [DefenderEnv] 1 layer, 2 servers per layer, 10 attack-defense-values, attacker following the "attack maximal strategy"
    [Initial State] Defense: 2, Attack:0, Num vulnerabilities: 1, Det: 2, Vulnerability value: 0
    [Rewards] Dense
    [Version] 2
    [Observations] partially observed
    [Environment] Deterministic
    [Attacker Starting Position] Start node
    [Reconnaissance activities] disabled
    [Reconnaissance bool features] No
    """
    def __init__(self, idsgame_config: IdsGameConfig = None, save_dir: str = None, initial_state_path: str = None):
        """
        Initialization of the environment

        :param save_dir: directory to save outputs of the env
        :param initial_state_path: path to the initial state (if none, use default)
        :param idsgame_config: configuration of the environment (if not specified a default config is used)
        """
        if idsgame_config is None:
            game_config = GameConfig(num_layers=1, num_servers_per_layer=2, num_attack_types=10, max_value=9)
            game_config.set_initial_state(defense_val=2, attack_val=0, num_vulnerabilities_per_node=1, det_val=2,
                                          vulnerability_val=0, num_vulnerabilities_per_layer=2)
            game_config.dense_rewards = True
            if initial_state_path is not None:
                game_config.set_load_initial_state(initial_state_path)
            attacker_agent = AttackMaximalValueBotAgent(game_config)
            idsgame_config = IdsGameConfig(game_config=game_config, attacker_agent=attacker_agent)
            idsgame_config.render_config.caption = "idsgame-maximal_attack-v9"
        super().__init__(idsgame_config=idsgame_config, save_dir=save_dir)


class IdsGameV9Env(AttackDefenseEnv):
    """
    [AttackDefenseEnv] 1 layer, 2 servers per layer, 10 attack-defense-values
    [Initial State] Defense: 2, Attack:0, Num vulnerabilities: 1, Det: 2, Vulnerability value: 0
    [Rewards] Dense
    [Version] 9
    [Observations] partially observed
    [Environment] Deterministic
    [Attacker Starting Position] Start node
    [Reconnaissance activities] disabled
    [Reconnaissance bool features] No
    """
    def __init__(self, idsgame_config: IdsGameConfig = None, save_dir: str = None, initial_state_path: str = None):
        """
        Initialization of the environment

        :param save_dir: directory to save outputs of the env
        :param initial_state_path: path to the initial state (if none, use default)
        :param idsgame_config: configuration of the environment (if not specified a default config is used)
        """
        if idsgame_config is None:
            game_config = GameConfig(num_layers=1, num_servers_per_layer=2, num_attack_types=10, max_value=9)
            game_config.set_initial_state(defense_val=2, attack_val=0, num_vulnerabilities_per_node=1, det_val=2,
                                          vulnerability_val=0, num_vulnerabilities_per_layer=2)
            game_config.dense_rewards = True
            if initial_state_path is not None:
                game_config.set_load_initial_state(initial_state_path)
            idsgame_config = IdsGameConfig(game_config=game_config)
            idsgame_config.render_config.caption = "idsgame-v9"
        super().__init__(idsgame_config=idsgame_config, save_dir=save_dir)


# -------- Version 10 ------------

class IdsGameRandomDefenseV10Env(AttackerEnv):
    """
    [AttackerEnv] 1 layer, 1 server per layer, 10 attack-defense-values, random defender
    [Initial State] Defense: 2, Attack:0, Num vulnerabilities: 1, Det: 2, Vulnerability value: 0
    [Rewards] Dense
    [Version] 10
    [Observations] fully observed
    [Environment] Deterministic
    [Attacker Starting Position] Start node
    [Reconnaissance activities] disabled
    [Reconnaissance bool features] No
    """
    def __init__(self, idsgame_config: IdsGameConfig = None, save_dir: str = None, initial_state_path: str = None):
        """
        Initialization of the environment

        :param save_dir: directory to save outputs of the env
        :param initial_state_path: path to the initial state (if none, use default)
        :param idsgame_config: configuration of the environment (if not specified a default config is used)
        """
        if idsgame_config is None:
            game_config = GameConfig(num_layers=1, num_servers_per_layer=1, num_attack_types=10, max_value=9)
            game_config.set_initial_state(defense_val=2, attack_val=0, num_vulnerabilities_per_node=1, det_val=2,
                                          vulnerability_val=0, num_vulnerabilities_per_layer=1)
            game_config.dense_rewards = True
            game_config.network_config.fully_observed = True
            if initial_state_path is not None:
                game_config.set_load_initial_state(initial_state_path)
            defender_agent = RandomDefenseBotAgent(game_config)
            idsgame_config = IdsGameConfig(game_config=game_config, defender_agent=defender_agent)
            idsgame_config.render_config.caption = "idsgame-random_defense-v10"
        super().__init__(idsgame_config=idsgame_config, save_dir=save_dir)


class IdsGameMinimalDefenseV10Env(AttackerEnv):
    """
    [AttackerEnv] 1 layer, 1 server per layer, 10 attack-defense-values, defender following the "defend minimal strategy"
    [Initial State] Defense: 2, Attack:0, Num vulnerabilities: 1, Det: 2, Vulnerability value: 0
    [Rewards] Dense
    [Version] 10
    [Observations] fully observed
    [Environment] Deterministic
    [Attacker Starting Position] Start node
    [Reconnaissance activities] disabled
    [Reconnaissance bool features] No
    """
    def __init__(self, idsgame_config: IdsGameConfig = None, save_dir: str = None, initial_state_path: str = None):
        """
        Initialization of the environment

        :param save_dir: directory to save outputs of the env
        :param initial_state_path: path to the initial state (if none, use default)
        :param idsgame_config: configuration of the environment (if not specified a default config is used)
        """
        if idsgame_config is None:
            game_config = GameConfig(num_layers=1, num_servers_per_layer=1, num_attack_types=10, max_value=9)
            game_config.set_initial_state(defense_val=2, attack_val=0, num_vulnerabilities_per_node=1, det_val=2,
                                          vulnerability_val=0, num_vulnerabilities_per_layer=1)
            game_config.dense_rewards = True
            game_config.network_config.fully_observed = True
            if initial_state_path is not None:
                game_config.set_load_initial_state(initial_state_path)
            defender_agent = DefendMinimalValueBotAgent(game_config)
            idsgame_config = IdsGameConfig(game_config=game_config, defender_agent=defender_agent)
            idsgame_config.render_config.caption = "idsgame-minimal_defense-v10"
        super().__init__(idsgame_config=idsgame_config, save_dir=save_dir)


class IdsGameRandomAttackV10Env(DefenderEnv):
    """
    [DefenderEnv] 1 layer, 1 server per layer, 10 attack-defense-values, random attacker
    [Initial State] Defense: 2, Attack:0, Num vulnerabilities: 1, Det: 2, Vulnerability value: 0
    [Rewards] Dense
    [Version] 10
    [Observations] fully observed
    [Environment] Deterministic
    [Attacker Starting Position] Start node
    [Reconnaissance activities] disabled
    [Reconnaissance bool features] No
    """
    def __init__(self, idsgame_config: IdsGameConfig = None, save_dir: str = None, initial_state_path: str = None):
        """
        Initialization of the environment

        :param save_dir: directory to save outputs of the env
        :param initial_state_path: path to the initial state (if none, use default)
        :param idsgame_config: configuration of the environment (if not specified a default config is used)
        """
        if idsgame_config is None:
            game_config = GameConfig(num_layers=1, num_servers_per_layer=1, num_attack_types=10, max_value=9)
            game_config.set_initial_state(defense_val=2, attack_val=0, num_vulnerabilities_per_node=1, det_val=2,
                                          vulnerability_val=0, num_vulnerabilities_per_layer=1)
            game_config.dense_rewards = True
            game_config.network_config.fully_observed = True
            if initial_state_path is not None:
                game_config.set_load_initial_state(initial_state_path)
            attacker_agent = RandomAttackBotAgent(game_config)
            idsgame_config = IdsGameConfig(game_config=game_config, attacker_agent=attacker_agent)
            idsgame_config.render_config.caption = "idsgame-random_attack-v10"
        super().__init__(idsgame_config=idsgame_config, save_dir=save_dir)


class IdsGameMaximalAttackV10Env(DefenderEnv):
    """
    [DefenderEnv] 1 layer, 1 server per layer, 10 attack-defense-values, attacker following the "attack maximal strategy"
    [Initial State] Defense: 2, Attack:0, Num vulnerabilities: 1, Det: 2, Vulnerability value: 0
    [Rewards] Dense
    [Version] 10
    [Observations] fully observed
    [Environment] Deterministic
    [Attacker Starting Position] Start node
    [Reconnaissance activities] disabled
    [Reconnaissance bool features] No
    """
    def __init__(self, idsgame_config: IdsGameConfig = None, save_dir: str = None, initial_state_path: str = None):
        """
        Initialization of the environment

        :param save_dir: directory to save outputs of the env
        :param initial_state_path: path to the initial state (if none, use default)
        :param idsgame_config: configuration of the environment (if not specified a default config is used)
        """
        if idsgame_config is None:
            game_config = GameConfig(num_layers=1, num_servers_per_layer=1, num_attack_types=10, max_value=9)
            game_config.set_initial_state(defense_val=2, attack_val=0, num_vulnerabilities_per_node=1, det_val=2,
                                          vulnerability_val=0, num_vulnerabilities_per_layer=1)
            game_config.dense_rewards = True
            game_config.network_config.fully_observed = True
            if initial_state_path is not None:
                game_config.set_load_initial_state(initial_state_path)
            attacker_agent = AttackMaximalValueBotAgent(game_config)
            idsgame_config = IdsGameConfig(game_config=game_config, attacker_agent=attacker_agent)
            idsgame_config.render_config.caption = "idsgame-maximal_attack-v10"
        super().__init__(idsgame_config=idsgame_config, save_dir=save_dir)


class IdsGameV10Env(AttackDefenseEnv):
    """
    [AttackDefenseEnv] 1 layer, 1 server per layer, 10 attack-defense-values
    [Initial State] Defense: 2, Attack:0, Num vulnerabilities: 1, Det: 2, Vulnerability value: 0
    [Rewards] Dense
    [Version] 10
    [Observations] fully observed
    [Environment] Deterministic
    [Attacker Starting Position] Start node
    [Reconnaissance activities] disabled
    [Reconnaissance bool features] No
    """
    def __init__(self, idsgame_config: IdsGameConfig = None, save_dir: str = None, initial_state_path: str = None):
        """
        Initialization of the environment

        :param save_dir: directory to save outputs of the env
        :param initial_state_path: path to the initial state (if none, use default)
        :param idsgame_config: configuration of the environment (if not specified a default config is used)
        """
        if idsgame_config is None:
            game_config = GameConfig(num_layers=1, num_servers_per_layer=1, num_attack_types=10, max_value=9)
            game_config.set_initial_state(defense_val=2, attack_val=0, num_vulnerabilities_per_node=1, det_val=2,
                                          vulnerability_val=0, num_vulnerabilities_per_layer=1)
            game_config.dense_rewards = True
            game_config.network_config.fully_observed = True
            if initial_state_path is not None:
                game_config.set_load_initial_state(initial_state_path)
            idsgame_config = IdsGameConfig(game_config=game_config)
            idsgame_config.render_config.caption = "idsgame-v10"
        super().__init__(idsgame_config=idsgame_config, save_dir=save_dir)

# -------- Version 11 ------------

class IdsGameRandomDefenseV11Env(AttackerEnv):
    """
    [AttackerEnv] 0 layers, 1 server per layer, 2 attack-defense-values
    [Initial State] Defense: 0, Attack:0, Num vulnerabilities: 0, Det: 1, Vulnerability value: 0
    [Rewards] Dense
    [Version] 11
    [Observations] fully observed
    [Environment] Deterministic
    [Attacker Starting Position] Start node
    [Reconnaissance activities] disabled
    [Reconnaissance bool features] No
    """
    def __init__(self, idsgame_config: IdsGameConfig = None, save_dir: str = None, initial_state_path: str = None):
        """
        Initialization of the environment

        :param save_dir: directory to save outputs of the env
        :param initial_state_path: path to the initial state (if none, use default)
        :param idsgame_config: configuration of the environment (if not specified a default config is used)
        """
        if idsgame_config is None:
            game_config = GameConfig(num_layers=0, num_servers_per_layer=1, num_attack_types=2, max_value=2)
            game_config.set_initial_state(defense_val=0, attack_val=0, num_vulnerabilities_per_node=0, det_val=1,
                                          vulnerability_val=0, num_vulnerabilities_per_layer=0)
            game_config.dense_rewards = True
            game_config.network_config.fully_observed = True
            if initial_state_path is not None:
                game_config.set_load_initial_state(initial_state_path)
            defender_agent = RandomDefenseBotAgent(game_config)
            idsgame_config = IdsGameConfig(game_config=game_config, defender_agent=defender_agent)
            idsgame_config.render_config.caption = "idsgame-random_defense-v11"
        super().__init__(idsgame_config=idsgame_config, save_dir=save_dir)


class IdsGameMinimalDefenseV11Env(AttackerEnv):
    """
    [AttackerEnv] 0 layers, 1 server per layer, 2 attack-defense-values
    [Initial State] Defense: 0, Attack:0, Num vulnerabilities: 0, Det: 1, Vulnerability value: 0
    [Rewards] Dense
    [Version] 11
    [Observations] fully observed
    [Environment] Deterministic
    [Attacker Starting Position] Start node
    [Reconnaissance activities] disabled
    [Reconnaissance bool features] No
    """
    def __init__(self, idsgame_config: IdsGameConfig = None, save_dir: str = None, initial_state_path: str = None):
        """
        Initialization of the environment

        :param save_dir: directory to save outputs of the env
        :param initial_state_path: path to the initial state (if none, use default)
        :param idsgame_config: configuration of the environment (if not specified a default config is used)
        """
        if idsgame_config is None:
            game_config = GameConfig(num_layers=0, num_servers_per_layer=1, num_attack_types=2, max_value=2)
            game_config.set_initial_state(defense_val=0, attack_val=0, num_vulnerabilities_per_node=0, det_val=1,
                                          vulnerability_val=0, num_vulnerabilities_per_layer=0)
            game_config.dense_rewards = True
            game_config.network_config.fully_observed = True
            if initial_state_path is not None:
                game_config.set_load_initial_state(initial_state_path)
            defender_agent = DefendMinimalValueBotAgent(game_config)
            idsgame_config = IdsGameConfig(game_config=game_config, defender_agent=defender_agent)
            idsgame_config.render_config.caption = "idsgame-minimal_defense-v11"
        super().__init__(idsgame_config=idsgame_config, save_dir=save_dir)


class IdsGameRandomAttackV11Env(DefenderEnv):
    """
    [DefenseEnv] 0 layers, 1 server per layer, 2 attack-defense-values
    [Initial State] Defense: 0, Attack:0, Num vulnerabilities: 0, Det: 1, Vulnerability value: 0
    [Rewards] Dense
    [Version] 11
    [Observations] fully observed
    [Environment] Deterministic
    [Attacker Starting Position] Start node
    [Reconnaissance activities] disabled
    [Reconnaissance bool features] No
    """
    def __init__(self, idsgame_config: IdsGameConfig = None, save_dir: str = None, initial_state_path: str = None):
        """
        Initialization of the environment

        :param save_dir: directory to save outputs of the env
        :param initial_state_path: path to the initial state (if none, use default)
        :param idsgame_config: configuration of the environment (if not specified a default config is used)
        """
        if idsgame_config is None:
            game_config = GameConfig(num_layers=0, num_servers_per_layer=1, num_attack_types=2, max_value=2)
            game_config.set_initial_state(defense_val=0, attack_val=0, num_vulnerabilities_per_node=0, det_val=1,
                                          vulnerability_val=0, num_vulnerabilities_per_layer=0)
            game_config.dense_rewards = True
            game_config.network_config.fully_observed = True
            if initial_state_path is not None:
                game_config.set_load_initial_state(initial_state_path)
            attacker_agent = RandomAttackBotAgent(game_config)
            idsgame_config = IdsGameConfig(game_config=game_config, attacker_agent=attacker_agent)
            idsgame_config.render_config.caption = "idsgame-random_attack-v11"
        super().__init__(idsgame_config=idsgame_config, save_dir=save_dir)


class IdsGameMaximalAttackV11Env(DefenderEnv):
    """
    [DefenseEnv] 0 layers, 1 server per layer, 2 attack-defense-values
    [Initial State] Defense: 0, Attack:0, Num vulnerabilities: 0, Det: 1, Vulnerability value: 0
    [Rewards] Dense
    [Version] 11
    [Observations] fully observed
    [Environment] Deterministic
    [Attacker Starting Position] Start node
    [Reconnaissance activities] disabled
    [Reconnaissance bool features] No
    """
    def __init__(self, idsgame_config: IdsGameConfig = None, save_dir: str = None, initial_state_path: str = None):
        """
        Initialization of the environment

        :param save_dir: directory to save outputs of the env
        :param initial_state_path: path to the initial state (if none, use default)
        :param idsgame_config: configuration of the environment (if not specified a default config is used)
        """
        if idsgame_config is None:
            game_config = GameConfig(num_layers=0, num_servers_per_layer=1, num_attack_types=2, max_value=2)
            game_config.set_initial_state(defense_val=0, attack_val=0, num_vulnerabilities_per_node=0, det_val=1,
                                          vulnerability_val=0, num_vulnerabilities_per_layer=0)
            game_config.dense_rewards = True
            game_config.network_config.fully_observed = True
            if initial_state_path is not None:
                game_config.set_load_initial_state(initial_state_path)
            attacker_agent = AttackMaximalValueBotAgent(game_config)
            idsgame_config = IdsGameConfig(game_config=game_config, attacker_agent=attacker_agent)
            idsgame_config.render_config.caption = "idsgame-maximal_attack-v11"
        super().__init__(idsgame_config=idsgame_config, save_dir=save_dir)


class IdsGameV11Env(AttackDefenseEnv):
    """
    [AttackDefenseEnv] 0 layers, 1 server per layer, 2 attack-defense-values
    [Initial State] Defense: 0, Attack:0, Num vulnerabilities: 0, Det: 1, Vulnerability value: 0
    [Rewards] Dense
    [Version] 11
    [Observations] fully observed
    [Environment] Deterministic
    [Attacker Starting Position] Start node
    [Reconnaissance activities] disabled
    [Reconnaissance bool features] No
    """
    def __init__(self, idsgame_config: IdsGameConfig = None, save_dir: str = None, initial_state_path: str = None):
        """
        Initialization of the environment

        :param save_dir: directory to save outputs of the env
        :param initial_state_path: path to the initial state (if none, use default)
        :param idsgame_config: configuration of the environment (if not specified a default config is used)
        """
        if idsgame_config is None:
            game_config = GameConfig(num_layers=0, num_servers_per_layer=1, num_attack_types=2, max_value=2)
            game_config.set_initial_state(defense_val=0, attack_val=0, num_vulnerabilities_per_node=0, det_val=1,
                                          vulnerability_val=0, num_vulnerabilities_per_layer=0)
            game_config.dense_rewards = True
            game_config.network_config.fully_observed = True
            if initial_state_path is not None:
                game_config.set_load_initial_state(initial_state_path)
            idsgame_config = IdsGameConfig(game_config=game_config)
            idsgame_config.render_config.caption = "idsgame-v11"
        super().__init__(idsgame_config=idsgame_config, save_dir=save_dir)

# -------- Version 12 ------------

class IdsGameRandomDefenseV12Env(AttackerEnv):
    """
    [AttackerEnv] 0 layers, 1 server per layer, 2 attack-defense-values
    [Initial State] Defense: (0-1) (random), Attack: (0-1) (random), Num vulnerabilities: 0, Det: (0-1) (random),
    Vulnerability value: 0
    [Rewards] Dense
    [Version] 12
    [Observations] fully observed
    [Environment] Random
    [Attacker Starting Position] Start node
    [Reconnaissance activities] disabled
    [Reconnaissance bool features] No
    """
    def __init__(self, idsgame_config: IdsGameConfig = None, save_dir: str = None, initial_state_path: str = None):
        """
        Initialization of the environment

        :param save_dir: directory to save outputs of the env
        :param initial_state_path: path to the initial state (if none, use default)
        :param idsgame_config: configuration of the environment (if not specified a default config is used)
        """
        if idsgame_config is None:
            game_config = GameConfig(num_layers=0, num_servers_per_layer=1, num_attack_types=2, max_value=2)
            game_config.set_initial_state(defense_val=1, attack_val=1, num_vulnerabilities_per_node=0, det_val=1,
                                          vulnerability_val=0, num_vulnerabilities_per_layer=0)
            game_config.dense_rewards = True
            game_config.network_config.fully_observed = True
            if initial_state_path is not None:
                game_config.set_load_initial_state(initial_state_path)
            defender_agent = RandomDefenseBotAgent(game_config)
            idsgame_config = IdsGameConfig(game_config=game_config, defender_agent=defender_agent)
            idsgame_config.render_config.caption = "idsgame-random_defense-v12"
            idsgame_config.randomize_env = True
        super().__init__(idsgame_config=idsgame_config, save_dir=save_dir)


class IdsGameMinimalDefenseV12Env(AttackerEnv):
    """
    [AttackerEnv] 0 layers, 1 server per layer, 2 attack-defense-values
    [Initial State] Defense: (0-1) (random), Attack: (0-1) (random), Num vulnerabilities: 0, Det: (0-1) (random),
    Vulnerability value: 0
    [Rewards] Dense
    [Version] 12
    [Observations] fully observed
    [Environment] Random
    [Attacker Starting Position] Start node
    [Reconnaissance activities] disabled
    [Reconnaissance bool features] No
    """
    def __init__(self, idsgame_config: IdsGameConfig = None, save_dir: str = None, initial_state_path: str = None):
        """
        Initialization of the environment

        :param save_dir: directory to save outputs of the env
        :param initial_state_path: path to the initial state (if none, use default)
        :param idsgame_config: configuration of the environment (if not specified a default config is used)
        """
        if idsgame_config is None:
            game_config = GameConfig(num_layers=0, num_servers_per_layer=1, num_attack_types=2, max_value=2)
            game_config.set_initial_state(defense_val=1, attack_val=1, num_vulnerabilities_per_node=0, det_val=1,
                                          vulnerability_val=0, num_vulnerabilities_per_layer=0)
            game_config.dense_rewards = True
            game_config.network_config.fully_observed = True
            if initial_state_path is not None:
                game_config.set_load_initial_state(initial_state_path)
            defender_agent = DefendMinimalValueBotAgent(game_config)
            idsgame_config = IdsGameConfig(game_config=game_config, defender_agent=defender_agent)
            idsgame_config.render_config.caption = "idsgame-minimal_defense-v12"
            idsgame_config.randomize_env = True
        super().__init__(idsgame_config=idsgame_config, save_dir=save_dir)


class IdsGameRandomAttackV12Env(DefenderEnv):
    """
    [DefenseEnv] 0 layers, 1 server per layer, 2 attack-defense-values
    [Initial State] Defense: (0-1) (random), Attack: (0-1) (random), Num vulnerabilities: 0, Det: (0-1) (random),
    Vulnerability value: 0
    [Rewards] Dense
    [Version] 12
    [Observations] fully observed
    [Environment] Random
    [Attacker Starting Position] Start node
    [Reconnaissance activities] disabled
    [Reconnaissance bool features] No
    """
    def __init__(self, idsgame_config: IdsGameConfig = None, save_dir: str = None, initial_state_path: str = None):
        """
        Initialization of the environment

        :param save_dir: directory to save outputs of the env
        :param initial_state_path: path to the initial state (if none, use default)
        :param idsgame_config: configuration of the environment (if not specified a default config is used)
        """
        if idsgame_config is None:
            game_config = GameConfig(num_layers=0, num_servers_per_layer=1, num_attack_types=2, max_value=2)
            game_config.set_initial_state(defense_val=1, attack_val=1, num_vulnerabilities_per_node=0, det_val=1,
                                          vulnerability_val=0, num_vulnerabilities_per_layer=0)
            game_config.dense_rewards = True
            game_config.network_config.fully_observed = True
            if initial_state_path is not None:
                game_config.set_load_initial_state(initial_state_path)
            attacker_agent = RandomAttackBotAgent(game_config)
            idsgame_config = IdsGameConfig(game_config=game_config, attacker_agent=attacker_agent)
            idsgame_config.render_config.caption = "idsgame-random_attack-v12"
            idsgame_config.randomize_env = True
        super().__init__(idsgame_config=idsgame_config, save_dir=save_dir)


class IdsGameMaximalAttackV12Env(DefenderEnv):
    """
    [DefenseEnv] 0 layers, 1 server per layer, 2 attack-defense-values
    [Initial State] Defense: (0-1) (random), Attack: (0-1) (random), Num vulnerabilities: 0, Det: (0-1) (random),
    Vulnerability value: 0
    [Rewards] Dense
    [Version] 12
    [Observations] fully observed
    [Environment] Random
    [Attacker Starting Position] Start node
    [Reconnaissance activities] disabled
    [Reconnaissance bool features] No
    """
    def __init__(self, idsgame_config: IdsGameConfig = None, save_dir: str = None, initial_state_path: str = None):
        """
        Initialization of the environment

        :param save_dir: directory to save outputs of the env
        :param initial_state_path: path to the initial state (if none, use default)
        :param idsgame_config: configuration of the environment (if not specified a default config is used)
        """
        if idsgame_config is None:
            game_config = GameConfig(num_layers=0, num_servers_per_layer=1, num_attack_types=2, max_value=2)
            game_config.set_initial_state(defense_val=1, attack_val=1, num_vulnerabilities_per_node=0, det_val=1,
                                          vulnerability_val=0, num_vulnerabilities_per_layer=0)
            game_config.dense_rewards = True
            game_config.network_config.fully_observed = True
            if initial_state_path is not None:
                game_config.set_load_initial_state(initial_state_path)
            attacker_agent = AttackMaximalValueBotAgent(game_config)
            idsgame_config = IdsGameConfig(game_config=game_config, attacker_agent=attacker_agent)
            idsgame_config.render_config.caption = "idsgame-maximal_attack-v12"
            idsgame_config.randomize_env = True
        super().__init__(idsgame_config=idsgame_config, save_dir=save_dir)


class IdsGameV12Env(AttackDefenseEnv):
    """
    [AttackDefenseEnv] 0 layers, 1 server per layer, 2 attack-defense-values
    [Initial State] Defense: (0-1) (random), Attack: (0-1) (random), Num vulnerabilities: 0, Det: (0-1) (random),
    Vulnerability value: 0
    [Rewards] Dense
    [Version] 12
    [Observations] fully observed
    [Environment] Random
    [Attacker Starting Position] Start node
    [Reconnaissance activities] disabled
    [Reconnaissance bool features] No
    """
    def __init__(self, idsgame_config: IdsGameConfig = None, save_dir: str = None, initial_state_path: str = None):
        """
        Initialization of the environment

        :param save_dir: directory to save outputs of the env
        :param initial_state_path: path to the initial state (if none, use default)
        :param idsgame_config: configuration of the environment (if not specified a default config is used)
        """
        if idsgame_config is None:
            game_config = GameConfig(num_layers=0, num_servers_per_layer=1, num_attack_types=2, max_value=2)
            game_config.set_initial_state(defense_val=1, attack_val=1, num_vulnerabilities_per_node=0, det_val=1,
                                          vulnerability_val=0, num_vulnerabilities_per_layer=0)
            game_config.dense_rewards = True
            game_config.network_config.fully_observed = True
            if initial_state_path is not None:
                game_config.set_load_initial_state(initial_state_path)
            idsgame_config = IdsGameConfig(game_config=game_config)
            idsgame_config.render_config.caption = "idsgame-v12"
            idsgame_config.randomize_env = True
        super().__init__(idsgame_config=idsgame_config, save_dir=save_dir)


# -------- Version 13 ------------

class IdsGameRandomDefenseV13Env(AttackerEnv):
    """
    [AttackerEnv] 0 layers, 1 server per layer, 2 attack-defense-values
    [Initial State] Defense: 0, Attack:0, Num vulnerabilities: 0, Det: 10, Vulnerability value: 0
    [Rewards] Dense
    [Version] 13
    [Observations] fully observed
    [Environment] Deterministic
    [Attacker Starting Position] Start node
    [Reconnaissance activities] disabled
    [Reconnaissance bool features] No
    """
    def __init__(self, idsgame_config: IdsGameConfig = None, save_dir: str = None, initial_state_path: str = None):
        """
        Initialization of the environment

        :param save_dir: directory to save outputs of the env
        :param initial_state_path: path to the initial state (if none, use default)
        :param idsgame_config: configuration of the environment (if not specified a default config is used)
        """
        if idsgame_config is None:
            game_config = GameConfig(num_layers=0, num_servers_per_layer=1, num_attack_types=2, max_value=1)
            game_config.set_initial_state(defense_val=0, attack_val=0, num_vulnerabilities_per_node=0, det_val=10,
                                          vulnerability_val=0, num_vulnerabilities_per_layer=0)
            game_config.dense_rewards = True
            game_config.network_config.fully_observed = True
            if initial_state_path is not None:
                game_config.set_load_initial_state(initial_state_path)
            defender_agent = RandomDefenseBotAgent(game_config)
            idsgame_config = IdsGameConfig(game_config=game_config, defender_agent=defender_agent)
            idsgame_config.render_config.caption = "idsgame-random_defense-v13"
        super().__init__(idsgame_config=idsgame_config, save_dir=save_dir)


class IdsGameMinimalDefenseV13Env(AttackerEnv):
    """
    [AttackerEnv] 0 layers, 1 server per layer, 2 attack-defense-values
    [Initial State] Defense: 0, Attack:0, Num vulnerabilities: 0, Det: 10, Vulnerability value: 0
    [Rewards] Dense
    [Version] 13
    [Observations] fully observed
    [Environment] Deterministic
    [Attacker Starting Position] Start node
    [Reconnaissance activities] disabled
    [Reconnaissance bool features] No
    """
    def __init__(self, idsgame_config: IdsGameConfig = None, save_dir: str = None, initial_state_path: str = None):
        """
        Initialization of the environment

        :param save_dir: directory to save outputs of the env
        :param initial_state_path: path to the initial state (if none, use default)
        :param idsgame_config: configuration of the environment (if not specified a default config is used)
        """
        if idsgame_config is None:
            game_config = GameConfig(num_layers=0, num_servers_per_layer=1, num_attack_types=2, max_value=1)
            game_config.set_initial_state(defense_val=0, attack_val=0, num_vulnerabilities_per_node=0, det_val=10,
                                          vulnerability_val=0, num_vulnerabilities_per_layer=0)
            game_config.dense_rewards = True
            game_config.network_config.fully_observed = True
            if initial_state_path is not None:
                game_config.set_load_initial_state(initial_state_path)
            defender_agent = DefendMinimalValueBotAgent(game_config)
            idsgame_config = IdsGameConfig(game_config=game_config, defender_agent=defender_agent)
            idsgame_config.render_config.caption = "idsgame-minimal_defense-v13"
        super().__init__(idsgame_config=idsgame_config, save_dir=save_dir)


class IdsGameRandomAttackV13Env(DefenderEnv):
    """
    [DefenseEnv] 0 layers, 1 server per layer, 2 attack-defense-values
    [Initial State] Defense: 0, Attack:0, Num vulnerabilities: 0, Det: 10, Vulnerability value: 0
    [Rewards] Dense
    [Version] 13
    [Observations] fully observed
    [Environment] Deterministic
    [Attacker Starting Position] Start node
    [Reconnaissance activities] disabled
    [Reconnaissance bool features] No
    """
    def __init__(self, idsgame_config: IdsGameConfig = None, save_dir: str = None, initial_state_path: str = None):
        """
        Initialization of the environment

        :param save_dir: directory to save outputs of the env
        :param initial_state_path: path to the initial state (if none, use default)
        :param idsgame_config: configuration of the environment (if not specified a default config is used)
        """
        if idsgame_config is None:
            game_config = GameConfig(num_layers=0, num_servers_per_layer=1, num_attack_types=2, max_value=1)
            game_config.set_initial_state(defense_val=0, attack_val=0, num_vulnerabilities_per_node=0, det_val=10,
                                          vulnerability_val=0, num_vulnerabilities_per_layer=0)
            game_config.dense_rewards = True
            game_config.network_config.fully_observed = True
            if initial_state_path is not None:
                game_config.set_load_initial_state(initial_state_path)
            attacker_agent = RandomAttackBotAgent(game_config)
            idsgame_config = IdsGameConfig(game_config=game_config, attacker_agent=attacker_agent)
            idsgame_config.render_config.caption = "idsgame-random_attack-v13"
        super().__init__(idsgame_config=idsgame_config, save_dir=save_dir)


class IdsGameMaximalAttackV13Env(DefenderEnv):
    """
    [DefenseEnv] 0 layers, 1 server per layer, 2 attack-defense-values
    [Initial State] Defense: 0, Attack:0, Num vulnerabilities: 0, Det: 10, Vulnerability value: 0
    [Rewards] Dense
    [Version] 13
    [Observations] fully observed
    [Environment] Deterministic
    [Attacker Starting Position] Start node
    [Reconnaissance activities] disabled
    [Reconnaissance bool features] No
    """
    def __init__(self, idsgame_config: IdsGameConfig = None, save_dir: str = None, initial_state_path: str = None):
        """
        Initialization of the environment

        :param save_dir: directory to save outputs of the env
        :param initial_state_path: path to the initial state (if none, use default)
        :param idsgame_config: configuration of the environment (if not specified a default config is used)
        """
        if idsgame_config is None:
            game_config = GameConfig(num_layers=0, num_servers_per_layer=1, num_attack_types=2, max_value=1)
            game_config.set_initial_state(defense_val=0, attack_val=0, num_vulnerabilities_per_node=0, det_val=10,
                                          vulnerability_val=0, num_vulnerabilities_per_layer=0)
            game_config.dense_rewards = True
            game_config.network_config.fully_observed = True
            if initial_state_path is not None:
                game_config.set_load_initial_state(initial_state_path)
            attacker_agent = AttackMaximalValueBotAgent(game_config)
            idsgame_config = IdsGameConfig(game_config=game_config, attacker_agent=attacker_agent)
            idsgame_config.render_config.caption = "idsgame-maximal_attack-v13"
        super().__init__(idsgame_config=idsgame_config, save_dir=save_dir)


class IdsGameV13Env(AttackDefenseEnv):
    """
    [AttackDefenseEnv] 0 layers, 1 server per layer, 2 attack-defense-values
    [Initial State] Defense: 0, Attack:0, Num vulnerabilities: 0, Det: 10, Vulnerability value: 0
    [Rewards] Dense
    [Version] 13
    [Observations] fully observed
    [Environment] Deterministic
    [Attacker Starting Position] Start node
    [Reconnaissance activities] disabled
    [Reconnaissance bool features] No
    """
    def __init__(self, idsgame_config: IdsGameConfig = None, save_dir: str = None, initial_state_path: str = None):
        """
        Initialization of the environment

        :param save_dir: directory to save outputs of the env
        :param initial_state_path: path to the initial state (if none, use default)
        :param idsgame_config: configuration of the environment (if not specified a default config is used)
        """
        if idsgame_config is None:
            game_config = GameConfig(num_layers=0, num_servers_per_layer=1, num_attack_types=2, max_value=1)
            game_config.set_initial_state(defense_val=0, attack_val=0, num_vulnerabilities_per_node=0, det_val=10,
                                          vulnerability_val=0, num_vulnerabilities_per_layer=0)
            game_config.dense_rewards = True
            game_config.network_config.fully_observed = True
            if initial_state_path is not None:
                game_config.set_load_initial_state(initial_state_path)
            idsgame_config = IdsGameConfig(game_config=game_config)
            idsgame_config.render_config.caption = "idsgame-v13"
        super().__init__(idsgame_config=idsgame_config, save_dir=save_dir)


# -------- Version 14 ------------

class IdsGameRandomDefenseV14Env(AttackerEnv):
    """
    [AttackerEnv] 1 layer, 1 server per layer, 4 attack-defense-values, random defender
    [Initial State] Defense: 2, Attack:0, Num vulnerabilities: 1, Det: 1, Vulnerability value: 0
    [Rewards] Dense
    [Version] 14
    [Observations] fully observed
    [Environment] Random
    [Attacker Starting Position] Start node
    [Local View] Yes
    [Reconnaissance activities] disabled
    [Reconnaissance bool features] No
    """
    def __init__(self, idsgame_config: IdsGameConfig = None, save_dir: str = None, initial_state_path: str = None):
        """
        Initialization of the environment

        :param save_dir: directory to save outputs of the env
        :param initial_state_path: path to the initial state (if none, use default)
        :param idsgame_config: configuration of the environment (if not specified a default config is used)
        """
        if idsgame_config is None:
            game_config = GameConfig(num_layers=1, num_servers_per_layer=1, num_attack_types=4, max_value=4,
                                     min_random_a_val=0, min_random_d_val=3, min_random_det_val=1)
            game_config.set_initial_state(defense_val=4, attack_val=0, num_vulnerabilities_per_node=1, det_val=1,
                                          vulnerability_val=0, num_vulnerabilities_per_layer=1)
            game_config.dense_rewards = True
            game_config.network_config.fully_observed = True
            game_config.network_config.relative_neighbor_positions = [(-1, 0), (1, 0)]
            if initial_state_path is not None:
                game_config.set_load_initial_state(initial_state_path)
            defender_agent = RandomDefenseBotAgent(game_config)
            idsgame_config = IdsGameConfig(game_config=game_config, defender_agent=defender_agent)
            idsgame_config.render_config.caption = "idsgame-random_defense-v14"
            idsgame_config.randomize_env = True
            idsgame_config.local_view_observations = True
        super().__init__(idsgame_config=idsgame_config, save_dir=save_dir)


class IdsGameMinimalDefenseV14Env(AttackerEnv):
    """
    [AttackerEnv] 1 layer, 1 server per layer, 4 attack-defense-values, defender following the "defend minimal strategy"
    [Initial State] Defense: 2, Attack:0, Num vulnerabilities: 1, Det: 1, Vulnerability value: 0
    [Rewards] Dense
    [Version] 14
    [Observations] fully observed
    [Environment] Random
    [Attacker Starting Position] Start node
    [Local View] Yes
    [Reconnaissance activities] disabled
    [Reconnaissance bool features] No
    """
    def __init__(self, idsgame_config: IdsGameConfig = None, save_dir: str = None, initial_state_path: str = None):
        """
        Initialization of the environment

        :param save_dir: directory to save outputs of the env
        :param initial_state_path: path to the initial state (if none, use default)
        :param idsgame_config: configuration of the environment (if not specified a default config is used)
        """
        if idsgame_config is None:
            game_config = GameConfig(num_layers=1, num_servers_per_layer=1, num_attack_types=4, max_value=4,
                                     min_random_a_val=0, min_random_d_val=3, min_random_det_val=1)
            game_config.set_initial_state(defense_val=4, attack_val=0, num_vulnerabilities_per_node=1, det_val=1,
                                          vulnerability_val=0, num_vulnerabilities_per_layer=1)
            game_config.dense_rewards = True
            game_config.network_config.fully_observed = True
            game_config.network_config.relative_neighbor_positions = [(-1, 0), (1, 0)]
            if initial_state_path is not None:
                game_config.set_load_initial_state(initial_state_path)
            defender_agent = DefendMinimalValueBotAgent(game_config)
            idsgame_config = IdsGameConfig(game_config=game_config, defender_agent=defender_agent)
            idsgame_config.render_config.caption = "idsgame-minimal_defense-v14"
            idsgame_config.randomize_env = True
            idsgame_config.local_view_observations = True
        super().__init__(idsgame_config=idsgame_config, save_dir=save_dir)


class IdsGameRandomAttackV14Env(DefenderEnv):
    """
    [DefenderEnv] 1 layer, 1 server per layer, 4 attack-defense-values, random attacker
    [Initial State] Defense: 2, Attack:0, Num vulnerabilities: 1, Det: 1, Vulnerability value: 0
    [Rewards] Dense
    [Version] 14
    [Observations] fully observed
    [Environment] Random
    [Attacker Starting Position] Start node
    [Local View] Yes
    [Reconnaissance activities] disabled
    [Reconnaissance bool features] No
    """
    def __init__(self, idsgame_config: IdsGameConfig = None, save_dir: str = None, initial_state_path: str = None):
        """
        Initialization of the environment

        :param save_dir: directory to save outputs of the env
        :param initial_state_path: path to the initial state (if none, use default)
        :param idsgame_config: configuration of the environment (if not specified a default config is used)
        """
        if idsgame_config is None:
            game_config = GameConfig(num_layers=1, num_servers_per_layer=1, num_attack_types=4, max_value=4,
                                     min_random_a_val=0, min_random_d_val=3, min_random_det_val=1)
            game_config.set_initial_state(defense_val=4, attack_val=0, num_vulnerabilities_per_node=1, det_val=1,
                                          vulnerability_val=0, num_vulnerabilities_per_layer=1)
            game_config.dense_rewards = True
            game_config.network_config.fully_observed = True
            game_config.network_config.relative_neighbor_positions = [(-1, 0), (1, 0)]
            if initial_state_path is not None:
                game_config.set_load_initial_state(initial_state_path)
            attacker_agent = RandomAttackBotAgent(game_config)
            idsgame_config = IdsGameConfig(game_config=game_config, attacker_agent=attacker_agent)
            idsgame_config.render_config.caption = "idsgame-random_attack-v14"
            idsgame_config.randomize_env = True
            idsgame_config.local_view_observations = True
        super().__init__(idsgame_config=idsgame_config, save_dir=save_dir)


class IdsGameMaximalAttackV14Env(DefenderEnv):
    """
    [DefenderEnv] 1 layer, 1 server per layer, 4 attack-defense-values, attacker following the "attack maximal strategy"
    [Initial State] Defense: 2, Attack:0, Num vulnerabilities: 1, Det: 1, Vulnerability value: 0
    [Rewards] Dense
    [Version] 14
    [Observations] fully observed
    [Environment] Random
    [Attacker Starting Position] Start node
    [Local View] Yes
    [Reconnaissance activities] disabled
    [Reconnaissance bool features] No
    """
    def __init__(self, idsgame_config: IdsGameConfig = None, save_dir: str = None, initial_state_path: str = None):
        """
        Initialization of the environment

        :param save_dir: directory to save outputs of the env
        :param initial_state_path: path to the initial state (if none, use default)
        :param idsgame_config: configuration of the environment (if not specified a default config is used)
        """
        if idsgame_config is None:
            game_config = GameConfig(num_layers=1, num_servers_per_layer=1, num_attack_types=4, max_value=4,
                                     min_random_a_val=0, min_random_d_val=3, min_random_det_val=1)
            game_config.set_initial_state(defense_val=4, attack_val=0, num_vulnerabilities_per_node=1, det_val=1,
                                          vulnerability_val=0, num_vulnerabilities_per_layer=1)
            game_config.dense_rewards = True
            game_config.network_config.fully_observed = True
            game_config.network_config.relative_neighbor_positions = [(-1, 0), (1, 0)]
            if initial_state_path is not None:
                game_config.set_load_initial_state(initial_state_path)
            attacker_agent = AttackMaximalValueBotAgent(game_config)
            idsgame_config = IdsGameConfig(game_config=game_config, attacker_agent=attacker_agent)
            idsgame_config.render_config.caption = "idsgame-maximal_attack-v14"
            idsgame_config.randomize_env = True
            idsgame_config.local_view_observations = True
        super().__init__(idsgame_config=idsgame_config, save_dir=save_dir)


class IdsGameV14Env(AttackDefenseEnv):
    """
    [AttackDefenseEnv] 1 layer, 1 server per layer, 4 attack-defense-values
    [Initial State] Defense: 2, Attack:0, Num vulnerabilities: 1, Det: 1, Vulnerability value: 0
    [Rewards] Dense
    [Version] 14
    [Observations] fully observed
    [Environment] Random
    [Attacker Starting Position] Start node
    [Local View] Yes
    [Reconnaissance activities] disabled
    [Reconnaissance bool features] No
    """
    def __init__(self, idsgame_config: IdsGameConfig = None, save_dir: str = None, initial_state_path: str = None):
        """
        Initialization of the environment

        :param save_dir: directory to save outputs of the env
        :param initial_state_path: path to the initial state (if none, use default)
        :param idsgame_config: configuration of the environment (if not specified a default config is used)
        """
        if idsgame_config is None:
            game_config = GameConfig(num_layers=1, num_servers_per_layer=1, num_attack_types=4, max_value=4,
                                     min_random_a_val=0, min_random_d_val=3, min_random_det_val=1)
            game_config.set_initial_state(defense_val=4, attack_val=0, num_vulnerabilities_per_node=1, det_val=1,
                                          vulnerability_val=0, num_vulnerabilities_per_layer=1)
            game_config.dense_rewards = True
            game_config.network_config.fully_observed = True
            game_config.network_config.relative_neighbor_positions = [(-1, 0), (1, 0)]
            if initial_state_path is not None:
                game_config.set_load_initial_state(initial_state_path)
            idsgame_config = IdsGameConfig(game_config=game_config)
            idsgame_config.render_config.caption = "idsgame-v14"
            idsgame_config.randomize_env = True
            idsgame_config.local_view_observations = True
        super().__init__(idsgame_config=idsgame_config, save_dir=save_dir)



# -------- Version 15 ------------

class IdsGameRandomDefenseV15Env(AttackerEnv):
    """
    [AttackerEnv] 1 layer, 2 server per layer, 10 attack-defense-values, random defender
    [Initial State] Defense: 2, Attack:0, Num vulnerabilities: 1, Det: 2, Vulnerability value: 0
    [Rewards] Dense
    [Version] 15
    [Observations] fully observed
    [Environment] Random
    [Attacker Starting Position] Start node
    [Local View] Yes
    [Reconnaissance activities] disabled
    [Reconnaissance bool features] No
    """
    def __init__(self, idsgame_config: IdsGameConfig = None, save_dir: str = None, initial_state_path: str = None):
        """
        Initialization of the environment

        :param save_dir: directory to save outputs of the env
        :param initial_state_path: path to the initial state (if none, use default)
        :param idsgame_config: configuration of the environment (if not specified a default config is used)
        """
        if idsgame_config is None:
            game_config = GameConfig(num_layers=1, num_servers_per_layer=2, num_attack_types=10, max_value=10,
                                     min_random_a_val=0, min_random_d_val=2, min_random_det_val=1)
            game_config.set_initial_state(defense_val=4, attack_val=0, num_vulnerabilities_per_node=1, det_val=2,
                                          vulnerability_val=0, num_vulnerabilities_per_layer=1)
            game_config.dense_rewards = True
            game_config.network_config.fully_observed = True
            game_config.network_config.relative_neighbor_positions = [(-1, 0), (1, 0)]
            if initial_state_path is not None:
                game_config.set_load_initial_state(initial_state_path)
            defender_agent = RandomDefenseBotAgent(game_config)
            idsgame_config = IdsGameConfig(game_config=game_config, defender_agent=defender_agent)
            idsgame_config.render_config.caption = "idsgame-random_defense-v15"
            idsgame_config.randomize_env = True
            idsgame_config.local_view_observations = True
        super().__init__(idsgame_config=idsgame_config, save_dir=save_dir)


class IdsGameMinimalDefenseV15Env(AttackerEnv):
    """
    [AttackerEnv] 1 layer, 2 server per layer, 10 attack-defense-values, random defender
    [Initial State] Defense: 2, Attack:0, Num vulnerabilities: 1, Det: 2, Vulnerability value: 0
    [Rewards] Dense
    [Version] 15
    [Observations] fully observed
    [Environment] Random
    [Attacker Starting Position] Start node
    [Local View] Yes
    [Reconnaissance activities] disabled
    [Reconnaissance bool features] No
    """
    def __init__(self, idsgame_config: IdsGameConfig = None, save_dir: str = None, initial_state_path: str = None):
        """
        Initialization of the environment

        :param save_dir: directory to save outputs of the env
        :param initial_state_path: path to the initial state (if none, use default)
        :param idsgame_config: configuration of the environment (if not specified a default config is used)
        """
        if idsgame_config is None:
            game_config = GameConfig(num_layers=1, num_servers_per_layer=2, num_attack_types=10, max_value=10,
                                     min_random_a_val=0, min_random_d_val=2, min_random_det_val=1)
            game_config.set_initial_state(defense_val=4, attack_val=0, num_vulnerabilities_per_node=1, det_val=2,
                                          vulnerability_val=0, num_vulnerabilities_per_layer=1)
            game_config.dense_rewards = True
            game_config.network_config.fully_observed = True
            game_config.network_config.relative_neighbor_positions = [(-1, 0), (1, 0)]
            if initial_state_path is not None:
                game_config.set_load_initial_state(initial_state_path)
            defender_agent = DefendMinimalValueBotAgent(game_config)
            idsgame_config = IdsGameConfig(game_config=game_config, defender_agent=defender_agent)
            idsgame_config.render_config.caption = "idsgame-minimal_defense-v15"
            idsgame_config.randomize_env = True
            idsgame_config.local_view_observations = True
        super().__init__(idsgame_config=idsgame_config, save_dir=save_dir)


class IdsGameRandomAttackV15Env(DefenderEnv):
    """
    [AttackerEnv] 1 layer, 2 server per layer, 10 attack-defense-values, random defender
    [Initial State] Defense: 2, Attack:0, Num vulnerabilities: 1, Det: 2, Vulnerability value: 0
    [Rewards] Dense
    [Version] 15
    [Observations] fully observed
    [Environment] Random
    [Attacker Starting Position] Start node
    [Local View] Yes
    [Reconnaissance activities] disabled
    [Reconnaissance bool features] No
    """
    def __init__(self, idsgame_config: IdsGameConfig = None, save_dir: str = None, initial_state_path: str = None):
        """
        Initialization of the environment

        :param save_dir: directory to save outputs of the env
        :param initial_state_path: path to the initial state (if none, use default)
        :param idsgame_config: configuration of the environment (if not specified a default config is used)
        """
        if idsgame_config is None:
            game_config = GameConfig(num_layers=1, num_servers_per_layer=2, num_attack_types=10, max_value=10,
                                     min_random_a_val=0, min_random_d_val=2, min_random_det_val=1)
            game_config.set_initial_state(defense_val=4, attack_val=0, num_vulnerabilities_per_node=1, det_val=2,
                                          vulnerability_val=0, num_vulnerabilities_per_layer=1)
            game_config.dense_rewards = True
            game_config.network_config.fully_observed = True
            game_config.network_config.relative_neighbor_positions = [(-1, 0), (1, 0)]
            if initial_state_path is not None:
                game_config.set_load_initial_state(initial_state_path)
            attacker_agent = RandomAttackBotAgent(game_config)
            idsgame_config = IdsGameConfig(game_config=game_config, attacker_agent=attacker_agent)
            idsgame_config.render_config.caption = "idsgame-random_attack-v15"
            idsgame_config.randomize_env = True
            idsgame_config.local_view_observations = True
        super().__init__(idsgame_config=idsgame_config, save_dir=save_dir)


class IdsGameMaximalAttackV15Env(DefenderEnv):
    """
    [AttackerEnv] 1 layer, 2 server per layer, 10 attack-defense-values, random defender
    [Initial State] Defense: 2, Attack:0, Num vulnerabilities: 1, Det: 2, Vulnerability value: 0
    [Rewards] Dense
    [Version] 15
    [Observations] fully observed
    [Environment] Random
    [Attacker Starting Position] Start node
    [Local View] Yes
    [Reconnaissance activities] disabled
    [Reconnaissance bool features] No
    """
    def __init__(self, idsgame_config: IdsGameConfig = None, save_dir: str = None, initial_state_path: str = None):
        """
        Initialization of the environment

        :param save_dir: directory to save outputs of the env
        :param initial_state_path: path to the initial state (if none, use default)
        :param idsgame_config: configuration of the environment (if not specified a default config is used)
        """
        if idsgame_config is None:
            game_config = GameConfig(num_layers=1, num_servers_per_layer=2, num_attack_types=10, max_value=10,
                                     min_random_a_val=0, min_random_d_val=2, min_random_det_val=1)
            game_config.set_initial_state(defense_val=4, attack_val=0, num_vulnerabilities_per_node=1, det_val=2,
                                          vulnerability_val=0, num_vulnerabilities_per_layer=1)
            game_config.dense_rewards = True
            game_config.network_config.fully_observed = True
            game_config.network_config.relative_neighbor_positions = [(-1, 0), (1, 0)]
            if initial_state_path is not None:
                game_config.set_load_initial_state(initial_state_path)
            attacker_agent = AttackMaximalValueBotAgent(game_config)
            idsgame_config = IdsGameConfig(game_config=game_config, attacker_agent=attacker_agent)
            idsgame_config.render_config.caption = "idsgame-maximal_attack-v15"
            idsgame_config.randomize_env = True
            idsgame_config.local_view_observations = True
        super().__init__(idsgame_config=idsgame_config, save_dir=save_dir)


class IdsGameV15Env(AttackDefenseEnv):
    """
    [AttackerEnv] 1 layer, 2 server per layer, 10 attack-defense-values, random defender
    [Initial State] Defense: 2, Attack:0, Num vulnerabilities: 1, Det: 2, Vulnerability value: 0
    [Rewards] Dense
    [Version] 15
    [Observations] fully observed
    [Environment] Random
    [Attacker Starting Position] Start node
    [Local View] Yes
    [Reconnaissance activities] disabled
    [Reconnaissance bool features] No
    """
    def __init__(self, idsgame_config: IdsGameConfig = None, save_dir: str = None, initial_state_path: str = None):
        """
        Initialization of the environment

        :param save_dir: directory to save outputs of the env
        :param initial_state_path: path to the initial state (if none, use default)
        :param idsgame_config: configuration of the environment (if not specified a default config is used)
        """
        if idsgame_config is None:
            game_config = GameConfig(num_layers=1, num_servers_per_layer=2, num_attack_types=10, max_value=10,
                                     min_random_a_val=0, min_random_d_val=2, min_random_det_val=1)
            game_config.set_initial_state(defense_val=4, attack_val=0, num_vulnerabilities_per_node=1, det_val=2,
                                          vulnerability_val=0, num_vulnerabilities_per_layer=1)
            game_config.dense_rewards = True
            game_config.network_config.fully_observed = True
            game_config.network_config.relative_neighbor_positions = [(-1, 0), (1, 0)]
            if initial_state_path is not None:
                game_config.set_load_initial_state(initial_state_path)
            idsgame_config = IdsGameConfig(game_config=game_config)
            idsgame_config.render_config.caption = "idsgame-v15"
            idsgame_config.randomize_env = True
            idsgame_config.local_view_observations = True
        super().__init__(idsgame_config=idsgame_config, save_dir=save_dir)


# -------- Version 16 ------------

class IdsGameRandomDefenseV16Env(AttackerEnv):
    """
    [AttackerEnv] 1 layer, 1 server per layer, 4 attack-defense-values, random defender
    [Initial State] Defense: 2, Attack:0, Num vulnerabilities: 1, Det: 1, Vulnerability value: 0
    [Rewards] Dense
    [Version] 16
    [Observations] fully observed
    [Environment] Random
    [Attacker Starting Position] Random
    [Local View] No
    [Reconnaissance activities] disabled
    [Reconnaissance bool features] No
    """
    def __init__(self, idsgame_config: IdsGameConfig = None, save_dir: str = None, initial_state_path: str = None):
        """
        Initialization of the environment

        :param save_dir: directory to save outputs of the env
        :param initial_state_path: path to the initial state (if none, use default)
        :param idsgame_config: configuration of the environment (if not specified a default config is used)
        """
        if idsgame_config is None:
            game_config = GameConfig(num_layers=1, num_servers_per_layer=1, num_attack_types=4, max_value=4,
                                     min_random_a_val=0, min_random_d_val=3, min_random_det_val=1)
            game_config.set_initial_state(defense_val=4, attack_val=0, num_vulnerabilities_per_node=1, det_val=1,
                                          vulnerability_val=0, num_vulnerabilities_per_layer=1)
            game_config.dense_rewards_v2 = True
            game_config.network_config.fully_observed = True
            game_config.network_config.relative_neighbor_positions = [(-1, 0), (1, 0)]
            if initial_state_path is not None:
                game_config.set_load_initial_state(initial_state_path)
            defender_agent = RandomDefenseBotAgent(game_config)
            idsgame_config = IdsGameConfig(game_config=game_config, defender_agent=defender_agent)
            idsgame_config.render_config.caption = "idsgame-random_defense-v16"
            idsgame_config.randomize_env = True
            idsgame_config.randomize_starting_position = True
            idsgame_config.local_view_observations = False
        super().__init__(idsgame_config=idsgame_config, save_dir=save_dir)


class IdsGameMinimalDefenseV16Env(AttackerEnv):
    """
    [AttackerEnv] 1 layer, 1 server per layer, 4 attack-defense-values, defender following the "defend minimal strategy"
    [Initial State] Defense: 2, Attack:0, Num vulnerabilities: 1, Det: 1, Vulnerability value: 0
    [Rewards] Dense
    [Version] 16
    [Observations] fully observed
    [Environment] Random
    [Attacker Starting Position] Random
    [Local View] No
    [Reconnaissance activities] disabled
    [Reconnaissance bool features] No
    """
    def __init__(self, idsgame_config: IdsGameConfig = None, save_dir: str = None, initial_state_path: str = None):
        """
        Initialization of the environment

        :param save_dir: directory to save outputs of the env
        :param initial_state_path: path to the initial state (if none, use default)
        :param idsgame_config: configuration of the environment (if not specified a default config is used)
        """
        if idsgame_config is None:
            game_config = GameConfig(num_layers=1, num_servers_per_layer=1, num_attack_types=4, max_value=4,
                                     min_random_a_val=0, min_random_d_val=3, min_random_det_val=1)
            game_config.set_initial_state(defense_val=4, attack_val=0, num_vulnerabilities_per_node=1, det_val=1,
                                          vulnerability_val=0, num_vulnerabilities_per_layer=1)
            game_config.dense_rewards_v2 = True
            game_config.network_config.fully_observed = True
            game_config.network_config.relative_neighbor_positions = [(-1, 0), (1, 0)]
            if initial_state_path is not None:
                game_config.set_load_initial_state(initial_state_path)
            defender_agent = DefendMinimalValueBotAgent(game_config)
            idsgame_config = IdsGameConfig(game_config=game_config, defender_agent=defender_agent)
            idsgame_config.render_config.caption = "idsgame-minimal_defense-v16"
            idsgame_config.randomize_env = True
            idsgame_config.randomize_starting_position = True
            idsgame_config.local_view_observations = False
        super().__init__(idsgame_config=idsgame_config, save_dir=save_dir)


class IdsGameRandomAttackV16Env(DefenderEnv):
    """
    [DefenderEnv] 1 layer, 1 server per layer, 4 attack-defense-values, random attacker
    [Initial State] Defense: 2, Attack:0, Num vulnerabilities: 1, Det: 1, Vulnerability value: 0
    [Rewards] Dense
    [Version] 16
    [Observations] fully observed
    [Environment] Random
    [Attacker Starting Position] Random
    [Local View] No
    [Reconnaissance activities] disabled
    [Reconnaissance bool features] No
    """
    def __init__(self, idsgame_config: IdsGameConfig = None, save_dir: str = None, initial_state_path: str = None):
        """
        Initialization of the environment

        :param save_dir: directory to save outputs of the env
        :param initial_state_path: path to the initial state (if none, use default)
        :param idsgame_config: configuration of the environment (if not specified a default config is used)
        """
        if idsgame_config is None:
            game_config = GameConfig(num_layers=1, num_servers_per_layer=1, num_attack_types=4, max_value=4,
                                     min_random_a_val=0, min_random_d_val=3, min_random_det_val=1)
            game_config.set_initial_state(defense_val=4, attack_val=0, num_vulnerabilities_per_node=1, det_val=1,
                                          vulnerability_val=0, num_vulnerabilities_per_layer=1)
            game_config.dense_rewards_v2 = True
            game_config.network_config.fully_observed = True
            game_config.network_config.relative_neighbor_positions = [(-1, 0), (1, 0)]
            if initial_state_path is not None:
                game_config.set_load_initial_state(initial_state_path)
            attacker_agent = RandomAttackBotAgent(game_config)
            idsgame_config = IdsGameConfig(game_config=game_config, attacker_agent=attacker_agent)
            idsgame_config.render_config.caption = "idsgame-random_attack-v16"
            idsgame_config.randomize_env = True
            idsgame_config.randomize_starting_position = True
            idsgame_config.local_view_observations = False
        super().__init__(idsgame_config=idsgame_config, save_dir=save_dir)


class IdsGameMaximalAttackV16Env(DefenderEnv):
    """
    [DefenderEnv] 1 layer, 1 server per layer, 4 attack-defense-values, attacker following the "attack maximal strategy"
    [Initial State] Defense: 2, Attack:0, Num vulnerabilities: 1, Det: 1, Vulnerability value: 0
    [Rewards] Dense
    [Version] 16
    [Observations] fully observed
    [Environment] Random
    [Attacker Starting Position] Random
    [Local View] No
    [Reconnaissance activities] disabled
    [Reconnaissance bool features] No
    """
    def __init__(self, idsgame_config: IdsGameConfig = None, save_dir: str = None, initial_state_path: str = None):
        """
        Initialization of the environment

        :param save_dir: directory to save outputs of the env
        :param initial_state_path: path to the initial state (if none, use default)
        :param idsgame_config: configuration of the environment (if not specified a default config is used)
        """
        if idsgame_config is None:
            game_config = GameConfig(num_layers=1, num_servers_per_layer=1, num_attack_types=4, max_value=4,
                                     min_random_a_val=0, min_random_d_val=1, min_random_det_val=1)
            game_config.set_initial_state(defense_val=4, attack_val=0, num_vulnerabilities_per_node=1, det_val=4,
                                          vulnerability_val=0, num_vulnerabilities_per_layer=1)
            game_config.dense_rewards_v2 = True
            game_config.network_config.fully_observed = True
            game_config.network_config.relative_neighbor_positions = [(-1, 0), (1, 0)]
            if initial_state_path is not None:
                game_config.set_load_initial_state(initial_state_path)
            attacker_agent = AttackMaximalValueBotAgent(game_config)
            idsgame_config = IdsGameConfig(game_config=game_config, attacker_agent=attacker_agent)
            idsgame_config.render_config.caption = "idsgame-maximal_attack-v16"
            idsgame_config.randomize_env = True
            idsgame_config.randomize_starting_position = True
            idsgame_config.local_view_observations = False
        super().__init__(idsgame_config=idsgame_config, save_dir=save_dir)


class IdsGameV16Env(AttackDefenseEnv):
    """
    [AttackDefenseEnv] 1 layer, 1 server per layer, 4 attack-defense-values
    [Initial State] Defense: 2, Attack:0, Num vulnerabilities: 1, Det: 1, Vulnerability value: 0
    [Rewards] Dense
    [Version] 16
    [Observations] fully observed
    [Environment] Random
    [Local View] No
    [Attacker Starting Position] Random
    [Reconnaissance activities] disabled
    [Reconnaissance bool features] No
    """
    def __init__(self, idsgame_config: IdsGameConfig = None, save_dir: str = None, initial_state_path: str = None):
        """
        Initialization of the environment

        :param save_dir: directory to save outputs of the env
        :param initial_state_path: path to the initial state (if none, use default)
        :param idsgame_config: configuration of the environment (if not specified a default config is used)
        """
        if idsgame_config is None:
            game_config = GameConfig(num_layers=1, num_servers_per_layer=1, num_attack_types=4, max_value=4,
                                     min_random_a_val=0, min_random_d_val=3, min_random_det_val=1)
            game_config.set_initial_state(defense_val=4, attack_val=0, num_vulnerabilities_per_node=1, det_val=1,
                                          vulnerability_val=0, num_vulnerabilities_per_layer=1)
            game_config.dense_rewards_v2 = True
            game_config.network_config.fully_observed = True
            game_config.network_config.relative_neighbor_positions = [(-1, 0), (1, 0)]
            if initial_state_path is not None:
                game_config.set_load_initial_state(initial_state_path)
            idsgame_config = IdsGameConfig(game_config=game_config)
            idsgame_config.render_config.caption = "idsgame-v16"
            idsgame_config.randomize_env = True
            idsgame_config.randomize_starting_position = True
            idsgame_config.local_view_observations = False
        super().__init__(idsgame_config=idsgame_config, save_dir=save_dir)


# -------- Version 17 ------------

class IdsGameRandomDefenseV17Env(AttackerEnv):
    """
    [AttackerEnv] 1 layer, 1 server per layer, 4 attack-defense-values, random defender
    [Initial State] Defense: 2, Attack:0, Num vulnerabilities: 1, Det: 1, Vulnerability value: 0
    [Rewards] Dense
    [Version] 17
    [Observations] partially observed
    [Environment] Random
    [Local View] No
    [Attacker Starting Position] Random
    [Reconnaissance activities] disabled
    [Reconnaissance bool features] No
    """
    def __init__(self, idsgame_config: IdsGameConfig = None, save_dir: str = None, initial_state_path: str = None):
        """
        Initialization of the environment

        :param save_dir: directory to save outputs of the env
        :param initial_state_path: path to the initial state (if none, use default)
        :param idsgame_config: configuration of the environment (if not specified a default config is used)
        """
        if idsgame_config is None:
            game_config = GameConfig(num_layers=1, num_servers_per_layer=1, num_attack_types=4, max_value=4,
                                     min_random_a_val=0, min_random_d_val=3, min_random_det_val=1)
            game_config.set_initial_state(defense_val=4, attack_val=0, num_vulnerabilities_per_node=1, det_val=1,
                                          vulnerability_val=0, num_vulnerabilities_per_layer=1)
            game_config.dense_rewards_v2 = True
            game_config.network_config.fully_observed = False
            if initial_state_path is not None:
                game_config.set_load_initial_state(initial_state_path)
            defender_agent = RandomDefenseBotAgent(game_config)
            idsgame_config = IdsGameConfig(game_config=game_config, defender_agent=defender_agent)
            idsgame_config.render_config.caption = "idsgame-random_defense-v17"
            idsgame_config.randomize_env = True
            idsgame_config.randomize_starting_position = True
            idsgame_config.local_view_observations = False
        super().__init__(idsgame_config=idsgame_config, save_dir=save_dir)


class IdsGameMinimalDefenseV17Env(AttackerEnv):
    """
    [AttackerEnv] 1 layer, 1 server per layer, 4 attack-defense-values, defender following the "defend minimal strategy"
    [Initial State] Defense: 2, Attack:0, Num vulnerabilities: 1, Det: 1, Vulnerability value: 0
    [Rewards] Dense
    [Version] 17
    [Observations] partially observed
    [Environment] Random
    [Local View] No
    [Attacker Starting Position] Random
    [Reconnaissance activities] disabled
    [Reconnaissance bool features] No
    """
    def __init__(self, idsgame_config: IdsGameConfig = None, save_dir: str = None, initial_state_path: str = None):
        """
        Initialization of the environment

        :param save_dir: directory to save outputs of the env
        :param initial_state_path: path to the initial state (if none, use default)
        :param idsgame_config: configuration of the environment (if not specified a default config is used)
        """
        if idsgame_config is None:
            game_config = GameConfig(num_layers=1, num_servers_per_layer=1, num_attack_types=4, max_value=4,
                                     min_random_a_val=0, min_random_d_val=3, min_random_det_val=1)
            game_config.set_initial_state(defense_val=4, attack_val=0, num_vulnerabilities_per_node=1, det_val=1,
                                          vulnerability_val=0, num_vulnerabilities_per_layer=1)
            game_config.dense_rewards_v2 = True
            game_config.network_config.fully_observed = False
            if initial_state_path is not None:
                game_config.set_load_initial_state(initial_state_path)
            defender_agent = DefendMinimalValueBotAgent(game_config)
            idsgame_config = IdsGameConfig(game_config=game_config, defender_agent=defender_agent)
            idsgame_config.render_config.caption = "idsgame-minimal_defense-v17"
            idsgame_config.randomize_env = True
            idsgame_config.randomize_starting_position = True
            idsgame_config.local_view_observations = False
        super().__init__(idsgame_config=idsgame_config, save_dir=save_dir)


class IdsGameRandomAttackV17Env(DefenderEnv):
    """
    [DefenderEnv] 1 layer, 1 server per layer, 4 attack-defense-values, random attacker
    [Initial State] Defense: 2, Attack:0, Num vulnerabilities: 1, Det: 1, Vulnerability value: 0
    [Rewards] Dense
    [Version] 17
    [Observations] partially observed
    [Environment] Random
    [Local View] No
    [Attacker Starting Position] Random
    [Reconnaissance activities] disabled
    [Reconnaissance bool features] No
    """
    def __init__(self, idsgame_config: IdsGameConfig = None, save_dir: str = None, initial_state_path: str = None):
        """
        Initialization of the environment

        :param save_dir: directory to save outputs of the env
        :param initial_state_path: path to the initial state (if none, use default)
        :param idsgame_config: configuration of the environment (if not specified a default config is used)
        """
        if idsgame_config is None:
            game_config = GameConfig(num_layers=1, num_servers_per_layer=1, num_attack_types=4, max_value=4,
                                     min_random_a_val=0, min_random_d_val=3, min_random_det_val=1)
            game_config.set_initial_state(defense_val=4, attack_val=0, num_vulnerabilities_per_node=1, det_val=1,
                                          vulnerability_val=0, num_vulnerabilities_per_layer=1)
            game_config.dense_rewards_v2 = True
            game_config.network_config.fully_observed = False
            if initial_state_path is not None:
                game_config.set_load_initial_state(initial_state_path)
            attacker_agent = RandomAttackBotAgent(game_config)
            idsgame_config = IdsGameConfig(game_config=game_config, attacker_agent=attacker_agent)
            idsgame_config.render_config.caption = "idsgame-random_attack-v17"
            idsgame_config.randomize_env = True
            idsgame_config.randomize_starting_position = True
            idsgame_config.local_view_observations = False
        super().__init__(idsgame_config=idsgame_config, save_dir=save_dir)


class IdsGameMaximalAttackV17Env(DefenderEnv):
    """
    [DefenderEnv] 1 layer, 1 server per layer, 4 attack-defense-values, attacker following the "attack maximal strategy"
    [Initial State] Defense: 2, Attack:0, Num vulnerabilities: 1, Det: 1, Vulnerability value: 0
    [Rewards] Dense
    [Version] 17
    [Observations] partially observed
    [Environment] Random
    [Local View] No
    [Attacker Starting Position] Random
    [Reconnaissance activities] disabled
    [Reconnaissance bool features] No
    """
    def __init__(self, idsgame_config: IdsGameConfig = None, save_dir: str = None, initial_state_path: str = None):
        """
        Initialization of the environment

        :param save_dir: directory to save outputs of the env
        :param initial_state_path: path to the initial state (if none, use default)
        :param idsgame_config: configuration of the environment (if not specified a default config is used)
        """
        if idsgame_config is None:
            game_config = GameConfig(num_layers=1, num_servers_per_layer=1, num_attack_types=4, max_value=4,
                                     min_random_a_val=0, min_random_d_val=1, min_random_det_val=1)
            game_config.set_initial_state(defense_val=4, attack_val=0, num_vulnerabilities_per_node=1, det_val=4,
                                          vulnerability_val=0, num_vulnerabilities_per_layer=1)
            game_config.dense_rewards_v2 = True
            game_config.network_config.fully_observed = False
            if initial_state_path is not None:
                game_config.set_load_initial_state(initial_state_path)
            attacker_agent = AttackMaximalValueBotAgent(game_config)
            idsgame_config = IdsGameConfig(game_config=game_config, attacker_agent=attacker_agent)
            idsgame_config.render_config.caption = "idsgame-maximal_attack-v17"
            idsgame_config.randomize_env = True
            idsgame_config.randomize_starting_position = True
            idsgame_config.local_view_observations = False
        super().__init__(idsgame_config=idsgame_config, save_dir=save_dir)


class IdsGameV17Env(AttackDefenseEnv):
    """
    [AttackDefenseEnv] 1 layer, 1 server per layer, 4 attack-defense-values
    [Initial State] Defense: 2, Attack:0, Num vulnerabilities: 1, Det: 1, Vulnerability value: 0
    [Rewards] Dense
    [Version] 17
    [Observations] partially observed
    [Environment] Random
    [Local View] No
    [Attacker Starting Position] Random
    [Reconnaissance activities] disabled
    [Reconnaissance bool features] No
    """
    def __init__(self, idsgame_config: IdsGameConfig = None, save_dir: str = None, initial_state_path: str = None):
        """
        Initialization of the environment

        :param save_dir: directory to save outputs of the env
        :param initial_state_path: path to the initial state (if none, use default)
        :param idsgame_config: configuration of the environment (if not specified a default config is used)
        """
        if idsgame_config is None:
            game_config = GameConfig(num_layers=1, num_servers_per_layer=1, num_attack_types=4, max_value=4,
                                     min_random_a_val=0, min_random_d_val=3, min_random_det_val=1)
            game_config.set_initial_state(defense_val=4, attack_val=0, num_vulnerabilities_per_node=1, det_val=1,
                                          vulnerability_val=0, num_vulnerabilities_per_layer=1)
            game_config.dense_rewards_v2 = True
            game_config.network_config.fully_observed = False
            if initial_state_path is not None:
                game_config.set_load_initial_state(initial_state_path)
            idsgame_config = IdsGameConfig(game_config=game_config)
            idsgame_config.render_config.caption = "idsgame-v17"
            idsgame_config.randomize_env = True
            idsgame_config.randomize_starting_position = True
            idsgame_config.local_view_observations = False
        super().__init__(idsgame_config=idsgame_config, save_dir=save_dir)

# -------- Version 18 ------------

class IdsGameRandomDefenseV18Env(AttackerEnv):
    """
    [AttackerEnv] 1 layer, 1 server per layer, 4 attack-defense-values, random defender
    [Initial State] Defense: 2, Attack:0, Num vulnerabilities: 1, Det: 1, Vulnerability value: 0
    [Rewards] Dense
    [Version] 18
    [Observations] partially observed
    [Environment] Random
    [Local View] No
    [Attacker Starting Position] Random
    [Reconnaissance activities] enabled
    [Reconnaissance bool features] Yes
    """
    def __init__(self, idsgame_config: IdsGameConfig = None, save_dir: str = None, initial_state_path: str = None):
        """
        Initialization of the environment

        :param save_dir: directory to save outputs of the env
        :param initial_state_path: path to the initial state (if none, use default)
        :param idsgame_config: configuration of the environment (if not specified a default config is used)
        """
        if idsgame_config is None:
            game_config = GameConfig(num_layers=1, num_servers_per_layer=1, num_attack_types=4, max_value=4,
                                     min_random_a_val=0, min_random_d_val=3, min_random_det_val=1)
            game_config.set_initial_state(defense_val=4, attack_val=0, num_vulnerabilities_per_node=1, det_val=1,
                                          vulnerability_val=0, num_vulnerabilities_per_layer=1)
            game_config.dense_rewards_v2 = True
            game_config.network_config.fully_observed = False
            game_config.reconnaissance_actions = True
            game_config.set_attack_actions()
            if initial_state_path is not None:
                game_config.set_load_initial_state(initial_state_path)
            defender_agent = RandomDefenseBotAgent(game_config)
            idsgame_config = IdsGameConfig(game_config=game_config, defender_agent=defender_agent)
            idsgame_config.render_config.caption = "idsgame-random_defense-v18"
            idsgame_config.randomize_env = True
            idsgame_config.randomize_starting_position = True
            idsgame_config.local_view_observations = False
            idsgame_config.reconnaissance_bool_features = True
            idsgame_config.reconnaissance_actions = True
        super().__init__(idsgame_config=idsgame_config, save_dir=save_dir)


class IdsGameMinimalDefenseV18Env(AttackerEnv):
    """
    [AttackerEnv] 1 layer, 1 server per layer, 4 attack-defense-values, defender following the "defend minimal strategy"
    [Initial State] Defense: 2, Attack:0, Num vulnerabilities: 1, Det: 1, Vulnerability value: 0
    [Rewards] Dense
    [Version] 18
    [Observations] partially observed
    [Environment] Random
    [Local View] No
    [Attacker Starting Position] Random
    [Reconnaissance activities] enabled
    [Reconnaissance bool features] Yes
    """
    def __init__(self, idsgame_config: IdsGameConfig = None, save_dir: str = None, initial_state_path: str = None):
        """
        Initialization of the environment

        :param save_dir: directory to save outputs of the env
        :param initial_state_path: path to the initial state (if none, use default)
        :param idsgame_config: configuration of the environment (if not specified a default config is used)
        """
        if idsgame_config is None:
            # game_config = GameConfig(num_layers=1, num_servers_per_layer=1, num_attack_types=8, max_value=8,
            #                          min_random_a_val=0, min_random_d_val=8, min_random_det_val=1)
            # game_config.set_initial_state(defense_val=8, attack_val=0, num_vulnerabilities_per_node=1, det_val=1,
            #                               vulnerability_val=0, num_vulnerabilities_per_layer=1)
            game_config = GameConfig(num_layers=1, num_servers_per_layer=1, num_attack_types=4, max_value=7,
                                     min_random_a_val=0, min_random_d_val=7, min_random_det_val=1)
            game_config.set_initial_state(defense_val=7, attack_val=0, num_vulnerabilities_per_node=1, det_val=1,
                                          vulnerability_val=1, num_vulnerabilities_per_layer=1)
            game_config.dense_rewards_v2 = True
            game_config.network_config.fully_observed = False
            game_config.reconnaissance_actions = True
            game_config.network_config.relative_neighbor_positions = [(-1, 0), (1, 0)]
            game_config.network_config.max_neighbors = len(game_config.network_config.relative_neighbor_positions)
            game_config.set_attack_actions(local_view=True)
            if initial_state_path is not None:
                game_config.set_load_initial_state(initial_state_path)
            defender_agent = DefendMinimalValueBotAgent(game_config)
            idsgame_config = IdsGameConfig(game_config=game_config, defender_agent=defender_agent)
            idsgame_config.render_config.caption = "idsgame-minimal_defense-v18"
            idsgame_config.randomize_env = True
            idsgame_config.randomize_starting_position = True
            idsgame_config.local_view_observations = True
            idsgame_config.reconnaissance_bool_features = True
            idsgame_config.reconnaissance_actions = True
        super().__init__(idsgame_config=idsgame_config, save_dir=save_dir)


class IdsGameRandomAttackV18Env(DefenderEnv):
    """
    [DefenderEnv] 1 layer, 1 server per layer, 4 attack-defense-values, random attacker
    [Initial State] Defense: 2, Attack:0, Num vulnerabilities: 1, Det: 1, Vulnerability value: 0
    [Rewards] Dense
    [Version] 18
    [Observations] partially observed
    [Environment] Random
    [Local View] No
    [Attacker Starting Position] Random
    [Reconnaissance activities] enabled
    [Reconnaissance bool features] Yes
    """
    def __init__(self, idsgame_config: IdsGameConfig = None, save_dir: str = None, initial_state_path: str = None):
        """
        Initialization of the environment

        :param save_dir: directory to save outputs of the env
        :param initial_state_path: path to the initial state (if none, use default)
        :param idsgame_config: configuration of the environment (if not specified a default config is used)
        """
        if idsgame_config is None:
            game_config = GameConfig(num_layers=1, num_servers_per_layer=1, num_attack_types=4, max_value=4,
                                     min_random_a_val=0, min_random_d_val=3, min_random_det_val=1)
            game_config.set_initial_state(defense_val=4, attack_val=0, num_vulnerabilities_per_node=1, det_val=1,
                                          vulnerability_val=0, num_vulnerabilities_per_layer=1)
            game_config.dense_rewards_v2 = True
            game_config.network_config.fully_observed = False
            game_config.reconnaissance_actions = True
            game_config.set_attack_actions()
            if initial_state_path is not None:
                game_config.set_load_initial_state(initial_state_path)
            attacker_agent = RandomAttackBotAgent(game_config)
            idsgame_config = IdsGameConfig(game_config=game_config, attacker_agent=attacker_agent)
            idsgame_config.render_config.caption = "idsgame-random_attack-v18"
            idsgame_config.randomize_env = True
            idsgame_config.randomize_starting_position = True
            idsgame_config.local_view_observations = False
            idsgame_config.reconnaissance_bool_features = True
            idsgame_config.reconnaissance_actions = True
        super().__init__(idsgame_config=idsgame_config, save_dir=save_dir)


class IdsGameMaximalAttackV18Env(DefenderEnv):
    """
    [DefenderEnv] 1 layer, 1 server per layer, 4 attack-defense-values, attacker following the "attack maximal strategy"
    [Initial State] Defense: 2, Attack:0, Num vulnerabilities: 1, Det: 1, Vulnerability value: 0
    [Rewards] Dense
    [Version] 18
    [Observations] partially observed
    [Environment] Random
    [Local View] No
    [Attacker Starting Position] Random
    [Reconnaissance activities] enabled
    [Reconnaissance bool features] Yes
    """
    def __init__(self, idsgame_config: IdsGameConfig = None, save_dir: str = None, initial_state_path: str = None):
        """
        Initialization of the environment

        :param save_dir: directory to save outputs of the env
        :param initial_state_path: path to the initial state (if none, use default)
        :param idsgame_config: configuration of the environment (if not specified a default config is used)
        """
        if idsgame_config is None:
            game_config = GameConfig(num_layers=1, num_servers_per_layer=1, num_attack_types=4, max_value=4,
                                     min_random_a_val=0, min_random_d_val=1, min_random_det_val=1)
            game_config.set_initial_state(defense_val=4, attack_val=0, num_vulnerabilities_per_node=1, det_val=4,
                                          vulnerability_val=0, num_vulnerabilities_per_layer=1)
            game_config.dense_rewards_v2 = True
            game_config.network_config.fully_observed = False
            game_config.reconnaissance_actions = True
            game_config.set_attack_actions()
            if initial_state_path is not None:
                game_config.set_load_initial_state(initial_state_path)
            attacker_agent = AttackMaximalValueBotAgent(game_config)
            idsgame_config = IdsGameConfig(game_config=game_config, attacker_agent=attacker_agent)
            idsgame_config.render_config.caption = "idsgame-maximal_attack-v18"
            idsgame_config.randomize_env = True
            idsgame_config.randomize_starting_position = True
            idsgame_config.reconnaissance_bool_features = True
            idsgame_config.local_view_observations = False
            idsgame_config.reconnaissance_actions = True
        super().__init__(idsgame_config=idsgame_config, save_dir=save_dir)


class IdsGameV18Env(AttackDefenseEnv):
    """
    [AttackDefenseEnv] 1 layer, 1 server per layer, 4 attack-defense-values
    [Initial State] Defense: 2, Attack:0, Num vulnerabilities: 1, Det: 1, Vulnerability value: 0
    [Rewards] Dense
    [Version] 18
    [Observations] partially observed
    [Environment] Random
    [Local View] No
    [Reconnaissance bool features] Yes
    [Attacker Starting Position] Random
    [Reconnaissance activities] enabled
    """
    def __init__(self, idsgame_config: IdsGameConfig = None, save_dir: str = None, initial_state_path: str = None):
        """
        Initialization of the environment

        :param save_dir: directory to save outputs of the env
        :param initial_state_path: path to the initial state (if none, use default)
        :param idsgame_config: configuration of the environment (if not specified a default config is used)
        """
        if idsgame_config is None:
            game_config = GameConfig(num_layers=1, num_servers_per_layer=1, num_attack_types=4, max_value=4,
                                     min_random_a_val=0, min_random_d_val=3, min_random_det_val=1)
            game_config.set_initial_state(defense_val=4, attack_val=0, num_vulnerabilities_per_node=1, det_val=1,
                                          vulnerability_val=0, num_vulnerabilities_per_layer=1)
            game_config.dense_rewards_v2 = True
            game_config.network_config.fully_observed = False
            game_config.reconnaissance_actions = True
            game_config.set_attack_actions()
            if initial_state_path is not None:
                game_config.set_load_initial_state(initial_state_path)
            idsgame_config = IdsGameConfig(game_config=game_config)
            idsgame_config.render_config.caption = "idsgame-v18"
            idsgame_config.randomize_env = True
            idsgame_config.randomize_starting_position = True
            idsgame_config.reconnaissance_bool_features = True
            idsgame_config.local_view_observations = False
            idsgame_config.reconnaissance_actions = True
        super().__init__(idsgame_config=idsgame_config, save_dir=save_dir)

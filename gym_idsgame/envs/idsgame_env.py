from typing import Union
import gym
import numpy as np
from gym_idsgame.envs.constants import constants
import os
from gym_idsgame.envs.dao.game_state import GameState
from gym_idsgame.envs.dao.attack_defense_event import AttackDefenseEvent
from gym_idsgame.envs.dao.node_type import NodeType
from gym_idsgame.envs.dao.idsgame_config import IdsGameConfig
import gym_idsgame.envs.util.idsgame_util as util
from gym_idsgame.envs.rendering.agents.defender import Defender

class IdsGameEnv(gym.Env):
    """
    TODO
    """

    def __init__(self, idsgame_config: IdsGameConfig):
        """
        TODO
        """
        util.validate_config(idsgame_config)
        self.idsgame_config: IdsGameConfig = idsgame_config
        self.state: GameState = self.idsgame_config.game_config.initial_state
        self.observation_space = idsgame_config.game_config.get_attacker_observation_space()
        self.defender = Defender(self.idsgame_config.defender_policy)
        self.action_space = gym.spaces.Discrete(self.num_actions)
        self.viewer = None
        self.steps_beyond_done = None
        self.metadata = {
         'render.modes': ['human', 'rgb_array'],
         'video.frames_per_second' : 50 # Video rendering speed
        }
        self.reward_range = (float(constants.GAME_CONFIG.NEGATIVE_REWARD), float(constants.GAME_CONFIG.POSITIVE_REWARD))

    # def initial_state(self):
    #     attack_states = np.zeros((self.num_nodes, self.num_attack_types+1)) # Plus 1 to indicate whether the agent is currently at this node or not
    #     defense_states = np.zeros((self.num_nodes, self.num_attack_types+1)) # Plus 1 to indicate detection value
    #     for i in range(self.num_nodes):
    #         attack_states[i] = np.array([0]*(self.num_attack_types+1))
    #     for i in range(self.num_nodes):
    #         defense_states[i] = np.array([2] * (self.num_attack_types + 1))
    #         defense_states[i][1] = 0 # innate vulnerability
    #         defense_states[i][-1] = 2 # detection value
    #     attack_states[0][-1] = 1 # The agent starts at the "START" node
    #     state = np.array([attack_states, defense_states])
    #     return state

    def __get_server_under_attack(self, action:int) -> Union[int, Union[int,int], int]:
        server_id = action // self.num_attack_types
        server_pos = self.idsgame_config.game_config.network_config.get_node_pos(server_id)
        attack_type = self.__get_attack_type(action)
        return server_id, server_pos, attack_type

    def __get_attack_type(self, action):
        attack_type = action % (self.num_attack_types)
        return attack_type

    def __get_grid_pos_of_node(self, node):
        if node == 0:
            return self.num_rows-1, self.num_servers_per_layer //2
        if node == self.num_nodes-1:
            return 0, self.num_servers_per_layer //2
        n2 = (node-1)
        row = (self.num_rows-2) - (n2//self.num_servers_per_layer)
        col = n2 % self.num_servers_per_layer
        return row, col

    def __get_node_from_grid_pos(self, row, col):
        if row == 0 and col == self.num_servers_per_layer //2:
            return self.num_nodes-1
        if row == self.num_layers-1 and col == self.num_servers_per_layer //2:
            return 0
        n1 = row*self.num_cols + col
        n2 = n1 -(self.num_servers_per_layer-1) # first and second layer only contain 1 node
        return n2

    def __increment_attack_value(self, target_id, type):
        if values[server][type] < (self.max_value-1):
            values[server][type] += 1

    def __get_attacker_node(self):
        attack_states = self.state[0]
        for i in range(attack_states.shape[0]):
            if attack_states[i][-1] == 1:
                return i
        raise AssertionError("Could not find the current node of the attacker in the game state")

    def __is_attack_legal(self, target_node, attacker_node):
        attacker_row, attacker_col = self.__get_grid_pos_of_node(attacker_node)
        target_row, target_col = self.__get_grid_pos_of_node(target_node)
        attacker_matrix_id = attacker_row * self.num_cols + attacker_col
        target_attacker_id = target_row * self.num_cols + target_col
        link = self.adjacency_matrix[attacker_matrix_id][target_attacker_id]
        if int(link) == 1:
            return True
        return False

    def __simulate_attack(self, server, attack_type):
        if self.state[0][server][attack_type] > self.state[1][server][attack_type]:
            return True
        return False

    def __simulate_detection(self, target_node):
        p_detection = self.state[1][target_node][-1] / 10
        if np.random.rand() < p_detection:
            return True
        else:
            return False

    def __move_attacker(self, current_node, target_node):
        self.state[0][current_node][-1] = 0
        self.state[0][target_node][-1] = 1

    def __is_data_node(self, node):
        return node == self.num_nodes-1

    def __add_attack_event(self, target_node, attack_type):
        target_row, target_col = self.__get_grid_pos_of_node(target_node)
        attack_event = AttackDefenseEvent(target_col, target_row, attack_type)
        self.attack_events.append(attack_event)

    def __add_defense_event(self, target_node, defense_type):
        target_row, target_col = self.__get_grid_pos_of_node(target_node)
        defense_event = AttackDefenseEvent(target_col, target_row, defense_type)
        self.defense_events.append(defense_event)

    def __simulate_attach(self):
        pass

    def step(self, action):
        reward = 0
        done = False
        detected = False
        info = {}
        self.state.attack_events = []
        self.state.defense_events = []
        target_node_id, target_pos, attack_type = self.__get_server_under_attack(action)
        defense_pos, defense_type, defense_node_id = self.__defense_action()
        return None, None, None, None
        # self.attack_events = []
        # self.defense_events = []
        # reward = 0
        # done = False
        # detected = False
        # info = {}
        # target_node = self.__get_server_under_attack(action)
        # attack_type = self.__get_attack_type(action)
        # attacker_node = self.__get_attacker_node()
        # defense_row, defense_col, defense_type = self.__defense_action()
        # defense_node = self.__get_node_from_grid_pos(defense_row, defense_col)
        # self.__increment_attack_defense_value(self.state[1], defense_node, defense_type)
        # self.__add_defense_event(defense_node, defense_type)
        # if self.__is_attack_legal(target_node, attacker_node):
        #     self.__add_attack_event(target_node, attack_type)
        #     self.__increment_attack_defense_value(self.state[0], target_node, attack_type)
        #     attack_successful = self.__simulate_attack(target_node, attack_type)
        #     if attack_successful:
        #         self.__move_attacker(attacker_node, target_node)
        #         if self.__is_data_node(target_node):
        #             reward = constants.RENDERING.POSITIVE_REWARD
        #             done = True
        #     else:
        #         detected = self.__simulate_detection(target_node)
        #         if detected:
        #             reward = constants.RENDERING.NEGATIVE_REWARD
        #             done = True
        # observation = self.state[0]
        # if done:
        #     if self.steps_beyond_done is None:
        #         self.steps_beyond_done = 0
        #     else:
        #         gym.logger.warn("You are calling 'step()' even though this environment has already returned done = True. "
        #                         "You should always call 'reset()' once you receive 'done = True' -- "
        #                         "any further steps are undefined behavior.")
        #         self.steps_beyond_done += 1
        # self.game_step +=1
        # self.attacker_total_reward += reward
        # self.defender_total_reward -= reward
        # if self.viewer is not None:
        #     self.viewer.gameframe.set_state(self.convert_state_to_render_state(done, detected))
        # return observation, reward, done, info

    def reset(self):
        """
        Resets the environment and returns the initial state

        :return: the initial state
        """
        self.steps_beyond_done = None
        self.state = self.initial_state()
        if self.viewer is not None:
            self.viewer.gameframe.reset()
        observation = self.state[0]
        self.game_step = 0
        self.num_games += 1
        return observation

    def convert_state_to_render_state(self, done, detected):
        attacker_node = self.__get_attacker_node()
        attacker_row, attacker_col = self.__get_grid_pos_of_node(attacker_node)
        render_attack_values = np.zeros((self.num_rows, self.num_cols, self.num_attack_types))
        render_defense_values = np.zeros((self.num_rows, self.num_cols, self.num_attack_types))
        render_defense_det = np.zeros((self.num_rows, self.num_cols))
        for node in range(self.num_nodes):
            row, col = self.__get_grid_pos_of_node(node)
            attack_state = self.state[0][node][:-1]
            render_attack_values[row][col] = attack_state
            defense_state = self.state[1][node][:-1]
            render_defense_values[row][col] = defense_state
            render_defense_det[row][col] = self.state[1][node][-1]
        render_state = GameState(
            attack_values=render_attack_values.astype(np.int32),
            defense_values = render_defense_values.astype(np.int32),
            defense_det = render_defense_det.astype(np.int32),
            attacker_pos=(attacker_row, attacker_col),
            game_step=self.game_step,
            attacker_cumulative_reward=self.attacker_total_reward,
            defender_cumulative_reward=self.defender_total_reward,
            num_games=self.num_games,
            attack_events=self.attack_events,
            defense_events = self.defense_events,
            done=done,
            detected=detected
        )
        return render_state

    def render(self, mode='human'):
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
        arr = self.viewer.render(return_rgb_array = mode=='rgb_array')
        self.attack_events = []
        self.defense_events = []
        return arr

    def __setup_viewer(self):
        """
        Setup for the viewer to use for rendering
        :return: None
        """
        from gym_idsgame.envs.rendering.viewer import Viewer
        script_dir = os.path.dirname(__file__)
        resource_path = os.path.join(script_dir, './rendering/', constants.RENDERING.RESOURCES_DIR)
        self.viewer = Viewer(num_layers=self.num_layers, num_servers_per_layer=self.num_servers_per_layer,
                             num_attack_types=self.num_attack_types, max_value=self.max_value,
                             adjacency_matrix=self.adjacency_matrix, graph_layout=self.graph_layout,
                             blink_interval=self.blink_interval, num_blinks=self.num_blinks)
        self.viewer.agent_start()

    def close(self):
        """
        Closes the viewer (cleanup)

        :return: None
        """
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def __defense_action(self) -> Union[Union[int, int], int, int]:
        defense_row, defense_col, defense_type = self.defender.policy.action(self.state)
        defense_pos = (defense_row, defense_col)
        defense_node_id = self.idsgame_config.game_config.network_config.get_node_id(defense_pos)
        return defense_pos, defense_type, defense_node_id

    def __initialize_graph_config(self):
        self.graph_layout = np.zeros((self.num_rows, self.num_cols))
        for i in range(self.num_rows):
            for j in range(self.num_cols):
                if i == self.num_rows - 1:
                    if j == self.num_cols // 2:
                        self.graph_layout[i][j] = NodeType.START.value
                    else:
                        self.graph_layout[i][j] = NodeType.EMPTY.value
                elif i == 0:
                    if j == self.num_cols // 2:
                        self.graph_layout[i][j] = NodeType.DATA.value
                    else:
                        self.graph_layout[i][j] = NodeType.EMPTY.value
                else:
                    self.graph_layout[i][j] = NodeType.SERVER.value

        self.adjacency_matrix = np.zeros((self.num_rows * self.num_cols, self.num_cols * self.num_rows))
        for i in range(self.num_rows * self.num_cols):
            row_1 = i // self.num_cols
            col_1 = i % self.num_cols
            for j in range(self.num_rows * self.num_cols):
                row_2 = j // self.num_cols
                col_2 = j % self.num_cols
                if row_1 == 0:
                    if row_2 == 1 and col_1 == self.num_cols // 2:
                        self.adjacency_matrix[i][j] = 1
                        self.adjacency_matrix[j][i] = 1
                elif row_1 == self.num_rows - 1:
                    if row_2 == self.num_rows - 2 and col_1 == self.num_cols // 2:
                        self.adjacency_matrix[i][j] = 1
                        self.adjacency_matrix[j][i] = 1
                elif (row_2 == row_1 + 1 and col_1 == col_2):
                    self.adjacency_matrix[i][j] = 1
                    self.adjacency_matrix[j][i] = 1
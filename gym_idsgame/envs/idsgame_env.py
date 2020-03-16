import gym
import numpy as np
from gym_idsgame.envs.rendering import constants
import os

class IdsGameEnv(gym.Env):
    """
    TODO
    """

    def __init__(self, width, height):
        """
        Initializes the environment

        Observation:
            Type: Box(2)
            Num	Observation                 Min         Max
            0	column                      0           width-1
            1	row                         0           height-1

        Actions:
            Type: Discrete(4)
            Num	Action
            0	Left
            1	Right
            2   Up
            3   Down

        Reward:
            Reward is -1 for every step except when the agent moves to the goal state, where the reward is +1

        Starting State:
            (0, 0)

        Episode Termination:
            When agent reaches goal state

        :param width: the width of the grid
        :param height: the height of the grid
        """
        self.width = width
        self.height = height
        self.num_states = self.width * self.height
        self.x = 0
        self.y = 0
        self.action_descriptors = ["Left", "Right", "Up", "Down"]
        self.num_actions = len(self.action_descriptors)
        self.goal_state = [width - 1, height - 1]
        # Observation is a true Markov state of the form: [x,y] where x in [0, width) and y in [0, height)
        self.observation_space = gym.spaces.Box(low=np.array([0.0,0.0]),
                                                high=np.array([float(width-1), float(height-1)]), dtype=np.float32)
        # Action space is an integer in [0,4)
        self.action_space = gym.spaces.Discrete(4)
        self.viewer = None
        self.steps_beyond_done = None
        self.rect_size = 50
        self.metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50 # Video rendering speed
        }
        self.reward_range = (-float('inf'), float(1))


    def step(self, action):
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
        if self.action_descriptors[action] == "Left":
            if self.x > 0:
                self.x = self.x-1
        elif self.action_descriptors[action] == "Right":
            if self.x < self.width-1:
                self.x = self.x + 1
        elif self.action_descriptors[action] == "Up":
            if self.y > 0:
                self.y = self.y - 1
        elif self.action_descriptors[action] == "Down":
            if self.y < self.height - 1:
                self.y = self.y + 1

        state = self.__get_state()
        reward = 1 if np.array_equal(state, self.goal_state) else -1 # Reward is -1 if not at goal state, otherwise it is 1
        done = False
        if reward == 1:
            done = True
            if self.steps_beyond_done is None:
                self.steps_beyond_done = 0
            else:
                gym.logger.warn("You are calling 'step()' even though this environment has already returned done = True. "
                                "You should always call 'reset()' once you receive 'done = True' -- "
                                "any further steps are undefined behavior.")
                self.steps_beyond_done +=1
        if self.viewer is not None:
            self.viewer.gridframe.set_state(self.__get_state())
        return state, reward, done, {}


    def reset(self):
        """
        Resets the environment and returns the initial state

        :return: the initial state
        """
        self.steps_beyond_done = None
        self.x = 0
        self.y = 0
        if self.viewer is not None:
            self.viewer.gridframe.reset()
        return self.__get_state()

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
        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def __setup_viewer(self):
        """
        Setup for the viewer to use for rendering
        :return: None
        """
        from gym_idsgame.envs.rendering.viewer import Viewer
        script_dir = os.path.dirname(__file__)
        resource_path = os.path.join(script_dir, './rendering/', constants.IDSGAME.RESOURCES_DIR)
        self.viewer = Viewer(width=self.width*self.rect_size,
                             height=(self.height*self.rect_size) + constants.IDSGAME.PANEL_HEIGHT,
                             rect_size=self.rect_size,
                             resources_dir=resource_path)
        self.viewer.agent_start()

    def close(self):
        """
        Closes the viewer (cleanup)

        :return: None
        """
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def __get_state(self):
        """
        Returns a feature representation of the state

        :return: [x,y] list
        """
        return np.array([self.x, self.y])

    def print_gridworld(self):
        """
        Utility function for printing the grid world and the agent's current position

        :return: None
        """
        for i in range(0, self.height):
            print("|", end='')
            for j in range(0, self.width):
                if i == self.y and j == self.x:
                    print("  X  ", end='')
                elif np.all(np.array([i,j]), self.goal_state):
                    print(" +1  ", end='')
                else:
                    print(" -1  ", end='')
                print("|", end='')
            print("")

    def get_state_index(self, state):
        """
        Utility function for getting the index of a state (useful for tabular algorithms)

        :param state: the feature representation of the state
        :return: the index of the state
        """
        return state[1]*self.width + state[0]
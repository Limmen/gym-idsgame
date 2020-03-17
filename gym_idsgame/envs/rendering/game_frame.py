import pyglet
from gym_idsgame.envs.rendering.resource_network import ResourceNetwork
from gym_idsgame.envs.rendering.hacker import Hacker
from gym_idsgame.envs.rendering import constants
from gym_idsgame.envs.rendering.render_util import batch_label, batch_rect_fill, batch_line
from gym_idsgame.envs.rendering.resource import Resource
from gym_idsgame.envs.rendering.data import Data
import os
import numpy as np

class GameFrame(pyglet.window.Window):
    """
    A class representing the OpenGL/Pyglet Game Frame
    By subclassing pyglet.window.Window, event handlers can be defined simply by overriding functions, e.g.
    event handler for on_draw is defined by overriding the on_draw function.
    """

    def __init__(self, manual = True, num_layers = 1, num_servers_per_layer = 2, num_attack_types = 10, max_value = 10,
                 defense_policy=constants.BASELINE_POLICIES.NAIVE_DETERMINISTIC,
                 resources_dir=constants.IDSGAME.RESOURCES_DIR):
        self.num_layers = num_layers
        self.num_servers_per_layer = num_servers_per_layer
        self.num_attack_types = num_attack_types
        self.num_rows = self.num_layers + 2
        self.num_cols = num_servers_per_layer
        self.max_value = max_value
        self.defense_policy = defense_policy
        self.resources_dir = resources_dir
        self.rect_size = constants.IDSGAME.RECT_SIZE
        self.bg_color = constants.IDSGAME.WHITE
        self.border_color = constants.IDSGAME.BLACK
        self.avatar_filename = constants.IDSGAME.HACKER_AVATAR_FILENAME
        self.server_filename = constants.IDSGAME.SERVER_AVATAR_FILENAME
        self.data_filename = constants.IDSGAME.DATA_AVATAR_FILENAME
        self.resources_dir = constants.IDSGAME.RESOURCES_DIR
        self.agent_scale = 0.3
        self.resource_scale = 0.2
        self.data_scale = 0.2
        self.manual = manual
        self.line_width = constants.IDSGAME.LINE_WIDTH
        height = constants.IDSGAME.PANEL_HEIGHT + int((self.rect_size / 1.5)) * self.num_rows
        width = self.rect_size * self.num_cols
        caption = constants.IDSGAME.CAPTION
        super(GameFrame, self).__init__(height=height, width=width, caption=caption) # call constructor of parent class
        self.num_rows = (self.height - constants.IDSGAME.PANEL_HEIGHT) // int((self.rect_size/1.5))
        self.num_cols = self.width//self.rect_size
        self.num_cells = self.num_rows*self.num_cols
        self.resource_network = ResourceNetwork(self.rect_size, self.num_rows, self.num_cols)
        self.setup_resources_path()
        self.game_step = 0
        self.attack_type = 1
        self.create_batch()
        self.initialize_graph_config()
        self.done = False

    def initialize_graph_config(self):
        self.graph_layout = np.zeros((self.num_rows, self.num_cols))
        for i in range(self.num_rows):
            for j in range(self.num_cols):
                if i == self.num_rows-1:
                    if j == self.num_cols // 2:
                        self.graph_layout[i][j] = constants.NODE_TYPES.START
                    else:
                        self.graph_layout[i][j] = constants.NODE_TYPES.NONE
                elif i == 0:
                    if j == self.num_cols // 2:
                        self.graph_layout[i][j] = constants.NODE_TYPES.DATA
                    else:
                        self.graph_layout[i][j] = constants.NODE_TYPES.NONE
                else:
                    self.graph_layout[i][j] = constants.NODE_TYPES.RESOURCE

        self.adjacency_matrix = np.zeros((self.num_rows*self.num_cols,  self.num_cols*self.num_rows))
        for i in range(self.num_rows*self.num_cols):
            row_1 = i // self.num_cols
            col_1 = i % self.num_cols
            for j in range(self.num_rows*self.num_cols):
                row_2 = j // self.num_cols
                col_2 = j % self.num_cols
                if row_1 == 0:
                    if row_2 == 1 and col_1 == self.num_cols // 2:
                        self.adjacency_matrix[i][j] = 1
                        self.adjacency_matrix[j][i] = 1
                elif row_1 == self.num_rows-1:
                    if row_2 == self.num_rows-2 and col_1 == self.num_cols // 2:
                        self.adjacency_matrix[i][j] = 1
                        self.adjacency_matrix[j][i] = 1
                elif (row_2 == row_1 + 1 and col_1 == col_2):
                    self.adjacency_matrix[i][j] = 1
                    self.adjacency_matrix[j][i] = 1



    def create_batch(self):
        """
        Creates a batch of elements to render. By grouping elements in a batch we can utilize OpenGL batch rendering
        and reduce the cpu <–> gpu data transfers and the number of draw-calls.

        :return: None
        """
        self.batch = pyglet.graphics.Batch()
        self.background = pyglet.graphics.OrderedGroup(0)
        self.first_foreground = pyglet.graphics.OrderedGroup(1)
        self.second_foreground = pyglet.graphics.OrderedGroup(2)

        # ---- Background ----

        # Sets the background color
        batch_rect_fill(0, 0, self.width, self.height, self.bg_color, self.batch, self.background)

        # Resource Network
        for i in range(self.resource_network.num_rows):
            for j in range(self.resource_network.num_cols):
                # Data node
                if i == 0:
                    self.resource_network.grid[i][j].draw(i, j, self.border_color, self.batch, self.background,
                                                          self.second_foreground,
                                                          self.data_filename, self.data_scale,
                                                          data=(j == (self.num_cols // 2)))
                    if j == (self.num_cols // 2):
                        self.data_node = self.resource_network.grid[i][j].get_node()
                # Server node
                if i > 0 and i < self.resource_network.num_rows-1:
                    self.resource_network.grid[i][j].draw(i, j, self.border_color, self.batch, self.background,
                                                          self.second_foreground,
                                                          self.server_filename, self.resource_scale, server=True)
                # Start node
                if i == self.resource_network.num_rows-1:
                    self.resource_network.grid[i][j].draw(i, j, self.border_color, self.batch, self.background,
                                                          self.second_foreground,
                                                          self.avatar_filename, self.agent_scale,
                                                          start=(j == (self.num_cols//2)))
        # Hacker starts at the start node
        self.hacker = Hacker(self.avatar_filename, self.num_cols // 2,
                             self.resource_network.num_rows - 1, self.batch, self.first_foreground, self.second_foreground,
                             self.rect_size, scale=self.agent_scale)

        # Connect start node with server nodes
        root_edge = self.resource_network.root_edge(
            self.resource_network.grid[self.resource_network.num_rows-1][self.num_cols//2],
            self.resource_network.grid[self.resource_network.num_rows - 2][self.num_cols // 2],
            constants.IDSGAME.BLACK, self.batch, self.background, self.line_width
        )
        for i in range(self.num_cols):
            edges = self.resource_network.connect_start_and_server_nodes(
                self.resource_network.grid[self.resource_network.num_rows-1][self.num_cols//2],
                self.resource_network.grid[self.resource_network.num_rows - 2][i],
                constants.IDSGAME.BLACK, self.batch, self.background, self.line_width)
            edges.append(root_edge)
            self.resource_network.grid[self.resource_network.num_rows - 1][self.num_cols // 2].add_out_edge(edges)
            self.resource_network.grid[self.resource_network.num_rows - 2][i].add_in_edge(edges)

        # Connect server nodes with server nodes on next layer
        for i in range(1, self.resource_network.num_rows-2):
            for j in range(self.resource_network.num_cols):
                edges = self.resource_network.connect_server_and_server_nodes(
                    self.resource_network.grid[i][j],
                    self.resource_network.grid[i+1][j],
                    constants.IDSGAME.BLACK, self.batch, self.background, self.line_width
                )
                self.resource_network.grid[i+1][j].add_out_edge(edges)
                self.resource_network.grid[i][j].add_in_edge(edges)

        # Connect server nodes on final layer with data node
        for j in range(self.resource_network.num_cols):
            edges = self.resource_network.connect_server_and_data_nodes(
                self.resource_network.grid[1][j],
                self.resource_network.grid[0][self.num_cols//2],
                constants.IDSGAME.BLACK, self.batch, self.background, self.line_width
            )
            self.resource_network.grid[1][j].add_out_edge(edges)
            self.resource_network.grid[0][self.num_cols // 2].add_in_edge(edges)

        # Panel Labels
        batch_label("Attack Reward: ", constants.IDSGAME.PANEL_LEFT_MARGIN, self.height - constants.IDSGAME.PANEL_TOP_MARGIN,
                    constants.IDSGAME.PANEL_FONT_SIZE, constants.IDSGAME.BLACK_ALPHA, self.batch,
                    self.second_foreground)
        batch_label("Time-step: ", constants.IDSGAME.PANEL_LEFT_MARGIN,
                    self.height - constants.IDSGAME.PANEL_TOP_MARGIN * 2,
                    constants.IDSGAME.PANEL_FONT_SIZE, constants.IDSGAME.BLACK_ALPHA,
                    self.batch, self.second_foreground)

        batch_label("Attack Type: ", constants.IDSGAME.PANEL_LEFT_MARGIN * 4,
                    self.height - constants.IDSGAME.PANEL_TOP_MARGIN * 2,
                    constants.IDSGAME.PANEL_FONT_SIZE, constants.IDSGAME.BLACK_ALPHA,
                    self.batch, self.second_foreground)
        batch_label("Defense Reward: ", constants.IDSGAME.PANEL_LEFT_MARGIN * 4,
                    self.height - constants.IDSGAME.PANEL_TOP_MARGIN,
                    constants.IDSGAME.PANEL_FONT_SIZE, constants.IDSGAME.BLACK_ALPHA,
                    self.batch, self.second_foreground)
        self.attack_type_label = batch_label(str(self.attack_type), constants.IDSGAME.PANEL_LEFT_MARGIN * 5.2,
                                             self.height - constants.IDSGAME.PANEL_TOP_MARGIN * 2,
                                             constants.IDSGAME.PANEL_FONT_SIZE, constants.IDSGAME.BLACK_ALPHA,
                                             self.batch, self.second_foreground)
        self.a_reward_label = batch_label(str(self.hacker.cumulative_reward), constants.IDSGAME.PANEL_LEFT_MARGIN * 2.2,
                                          self.height - constants.IDSGAME.PANEL_TOP_MARGIN,
                                          constants.IDSGAME.PANEL_FONT_SIZE, constants.IDSGAME.BLACK_ALPHA, self.batch,
                                          self.second_foreground)
        self.d_reward_label = batch_label(str(self.data_node.cumulative_reward), constants.IDSGAME.PANEL_LEFT_MARGIN * 5.2,
                                          self.height - constants.IDSGAME.PANEL_TOP_MARGIN,
                                          constants.IDSGAME.PANEL_FONT_SIZE, constants.IDSGAME.BLACK_ALPHA, self.batch,
                                          self.second_foreground)
        self.step_label = batch_label(str(self.game_step), constants.IDSGAME.PANEL_LEFT_MARGIN * 2.2,
                                      self.height - constants.IDSGAME.PANEL_TOP_MARGIN * 2,
                                      constants.IDSGAME.PANEL_FONT_SIZE, constants.IDSGAME.BLACK_ALPHA, self.batch,
                                      self.second_foreground)

    def setup_resources_path(self):
        """
        Setup path to resources (e.g. images)

        :return: None
        """
        if os.path.exists(self.resources_dir):
            pyglet.resource.path = [self.resources_dir]
        else:
            script_dir = os.path.dirname(__file__)
            resource_path = os.path.join(script_dir, './', constants.IDSGAME.RESOURCES_DIR)
            pyglet.resource.path = [resource_path]
        pyglet.resource.reindex()

    def on_draw(self):
        """
        Event handler for on_draw event. OpenGL does not remember what was rendered on the previous frame so
        we redraw each frame every time. This method is typically called many times per second.

        Draws the GridWorld Frame

        :return: None
        """
        # Clear the window
        self.clear()
        # Draw batch with the frame contents
        self.batch.draw()
        # Make this window the current OpenGL rendering context
        self.switch_to()


    def on_mouse_press(self, x, y, button, modifiers):
        if self.done:
            return
        for i in range(self.resource_network.num_rows - 1):
            for j in range(self.resource_network.num_cols):
                node = self.resource_network.grid[i][j].get_node()
                if node is not None:
                    node.unschedule()
        for i in range(self.resource_network.num_rows-1):
            for j in range(self.resource_network.num_cols):
                node = self.resource_network.grid[i][j].get_node()
                if node is not None:
                    if x > node.x and x < (node.x + node.width):
                        if y > node.y and y < (node.y + node.height):
                            if self.adjacency_matrix[self.hacker.row*self.num_cols + self.hacker.col][node.row*self.num_cols + node.col] == 1:
                                defense_row, defense_col, defend_type = self.data_node.defense_action(self.graph_layout)
                                self.resource_network.grid[defense_row][defense_col].defend(defend_type)
                                if isinstance(node, Data):
                                    edges = self.resource_network.grid[self.hacker.row][self.hacker.col].resource.outgoing_edges
                                    attack_successful = node.simulate_attack(self.attack_type, edges)
                                else:
                                    attack_successful = node.simulate_attack(self.attack_type)
                                self.game_step += 1
                                if attack_successful:
                                    self.hacker.move_to(node.x, node.y, node.col, node.row)
                                    if isinstance(node, Data):
                                        self.done = True
                                        self.hacker.add_reward(constants.IDSGAME.POSITIVE_REWARD)
                                        node.add_reward(constants.IDSGAME.NEGATIVE_REWARD)
                                else:
                                    detected = node.simulate_detection()
                                    if detected:
                                        self.hacker.add_reward(constants.IDSGAME.NEGATIVE_REWARD)
                                        self.hacker.detected()
                                        self.data_node.add_reward(constants.IDSGAME.POSITIVE_REWARD)
                                        self.done = True

                            return



    def on_key_press(self, symbol, modifiers):
        """
        Event handler for on_key_press event.
        The user can move the agent with key presses.

        :param symbol: the symbol of the keypress
        :param modifiers: _
        :return: None
        """
        if self.manual:
            if symbol == pyglet.window.key._1:
                self.attack_type = 1
            elif symbol == pyglet.window.key._2:
                self.attack_type = 2
            elif symbol == pyglet.window.key._3:
                self.attack_type = 3
            elif symbol == pyglet.window.key._4:
                self.attack_type = 4
            elif symbol == pyglet.window.key._5:
                self.attack_type = 5
            elif symbol == pyglet.window.key._6:
                self.attack_type = 6
            elif symbol == pyglet.window.key._7:
                self.attack_type = 7
            elif symbol == pyglet.window.key._8:
                self.attack_type = 8
            elif symbol == pyglet.window.key._9:
                self.attack_type = 9
            elif symbol == pyglet.window.key._0:
                self.attack_type = 0
            elif symbol == pyglet.window.key.SPACE:
                self.reset()


    def update(self, dt):
        """
        Event handler for the update-event (timer-based typically), used to update the state of the grid.

        :param dt: the number of seconds since the function was last called
        :return: None
        """
        self.step_label.text = str(self.game_step)
        self.a_reward_label.text = str(self.hacker.cumulative_reward)
        self.d_reward_label.text = str(self.data_node.cumulative_reward)
        self.attack_type_label.text = str(self.attack_type)
        self.hacker.update()

    def set_state(self, state):
        """
        TODO

        :param state: the state
        :return: None
        """
        pass

    def reset(self):
        """
        Resets the agent state without closing the screen

        :return: None
        """
        self.done = False
        for i in range(self.resource_network.num_rows - 1):
            for j in range(self.resource_network.num_cols):
                node = self.resource_network.grid[i][j].get_node()
                if node is not None:
                    node.unschedule()
        self.hacker.reset()
        for i in range(self.resource_network.num_rows):
            for j in range(self.resource_network.num_cols):
                self.resource_network.grid[i][j].reset()
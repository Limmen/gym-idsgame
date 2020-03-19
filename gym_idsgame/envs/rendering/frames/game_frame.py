import pyglet
from gym_idsgame.envs.rendering.network.network import Network
from gym_idsgame.envs.constants import constants
from gym_idsgame.envs.rendering.util.render_util import batch_rect_fill
from gym_idsgame.envs.dao.render_state import RenderState
from gym_idsgame.envs.dao.attack_defense_event import AttackDefenseEvent
from gym_idsgame.envs.dao.render_config import RenderConfig
from gym_idsgame.envs.dao.game_state import GameState
from gym_idsgame.envs.rendering.agents.attacker import Attacker
from typing import List
import os
import numpy as np

class GameFrame(pyglet.window.Window):
    """
    A class representing the OpenGL/Pyglet Game Frame
    By subclassing pyglet.window.Window, event handlers can be defined simply by overriding functions, e.g.
    event handler for on_draw is defined by overriding the on_draw function.
    """

    def __init__(self, render_config: RenderConfig):
        self.render_config = render_config
        super(GameFrame, self).__init__(height=render_config.height, width=render_config.width, caption=render_config.caption) # call constructor of parent class
        self.resource_network = None
        self.attacker = None
        self.defender = None
        self.setup_resources_path()
        self.game_state = GameState()
        self.create_batch()
        self.set_state(self.render_config.game_config.initial_state)
        self.switch_to()

    def create_batch(self):
        """
        Creates a batch of elements to render. By grouping elements in a batch we can utilize OpenGL batch rendering
        and reduce the cpu <–> gpu data transfers and the number of draw-calls.

        :return: None
        """

        # Sets the background color
        batch_rect_fill(0, 0, self.render_config.width, self.render_config.height, self.render_config.bg_color,
                        self.render_config.batch, self.render_config.background)

        # Resource Network
        self.resource_network = Network(self.render_config)


        # Attacker
        #self.attacker = Attacker(self.render_config, 0,0)

        # # Hacker starts at the start node
        # self.hacker = Attacker(self.avatar_filename, self.num_cols // 2,
        #                        self.resource_network.num_rows - 1, self.batch, self.first_foreground, self.second_foreground,
        #                        self.rect_size, scale=self.agent_scale)
        #
        # # Connect start node with server nodes
        # root_edge = self.resource_network.root_edge(
        #     self.resource_network.grid[self.resource_network.num_rows-1][self.num_cols//2],
        #     self.resource_network.grid[self.resource_network.num_rows - 2][self.num_cols // 2],
        #     constants.GAMEFRAME.BLACK, self.batch, self.background, self.line_width
        # )
        # for i in range(self.num_cols):
        #     edges = self.resource_network.connect_start_and_server_nodes(
        #         self.resource_network.grid[self.resource_network.num_rows-1][self.num_cols//2],
        #         self.resource_network.grid[self.resource_network.num_rows - 2][i],
        #         constants.GAMEFRAME.BLACK, self.batch, self.background, self.line_width)
        #     edges.append(root_edge)
        #     self.resource_network.grid[self.resource_network.num_rows - 1][self.num_cols // 2].add_out_edge(edges)
        #     self.resource_network.grid[self.resource_network.num_rows - 2][i].add_in_edge(edges)
        #
        # # Connect server nodes with server nodes on next layer
        # for i in range(1, self.resource_network.num_rows-2):
        #     for j in range(self.resource_network.num_cols):
        #         edges = self.resource_network.connect_server_and_server_nodes(
        #             self.resource_network.grid[i][j],
        #             self.resource_network.grid[i+1][j],
        #             constants.GAMEFRAME.BLACK, self.batch, self.background, self.line_width
        #         )
        #         self.resource_network.grid[i+1][j].add_out_edge(edges)
        #         self.resource_network.grid[i][j].add_in_edge(edges)
        #
        # # Connect server nodes on final layer with data node
        # for j in range(self.resource_network.num_cols):
        #     edges = self.resource_network.connect_server_and_data_nodes(
        #         self.resource_network.grid[1][j],
        #         self.resource_network.grid[0][self.num_cols//2],
        #         constants.GAMEFRAME.BLACK, self.batch, self.background, self.line_width
        #     )
        #     self.resource_network.grid[1][j].add_out_edge(edges)
        #     self.resource_network.grid[0][self.num_cols // 2].add_in_edge(edges)
        #
        # # Panel Labels
        # batch_label("Attack Reward: ", constants.GAMEFRAME.PANEL_LEFT_MARGIN, self.height - constants.GAMEFRAME.PANEL_TOP_MARGIN,
        #             constants.GAMEFRAME.PANEL_FONT_SIZE, constants.GAMEFRAME.BLACK_ALPHA, self.batch,
        #             self.second_foreground)
        # batch_label("Time-step: ", constants.GAMEFRAME.PANEL_LEFT_MARGIN,
        #             self.height - constants.GAMEFRAME.PANEL_TOP_MARGIN * 2,
        #             constants.GAMEFRAME.PANEL_FONT_SIZE, constants.GAMEFRAME.BLACK_ALPHA,
        #             self.batch, self.second_foreground)
        #
        # batch_label("Attack Type: ", constants.GAMEFRAME.PANEL_LEFT_MARGIN * 4,
        #             self.height - constants.GAMEFRAME.PANEL_TOP_MARGIN * 2,
        #             constants.GAMEFRAME.PANEL_FONT_SIZE, constants.GAMEFRAME.BLACK_ALPHA,
        #             self.batch, self.second_foreground)
        # batch_label("Defense Reward: ", constants.GAMEFRAME.PANEL_LEFT_MARGIN * 4,
        #             self.height - constants.GAMEFRAME.PANEL_TOP_MARGIN,
        #             constants.GAMEFRAME.PANEL_FONT_SIZE, constants.GAMEFRAME.BLACK_ALPHA,
        #             self.batch, self.second_foreground)
        # batch_label("Num Games: ", constants.GAMEFRAME.PANEL_LEFT_MARGIN * 6.5,
        #             self.height - constants.GAMEFRAME.PANEL_TOP_MARGIN,
        #             constants.GAMEFRAME.PANEL_FONT_SIZE, constants.GAMEFRAME.BLACK_ALPHA,
        #             self.batch, self.second_foreground)
        # self.attack_type_label = batch_label(str(self.attack_type), constants.GAMEFRAME.PANEL_LEFT_MARGIN * 5.2,
        #                                      self.height - constants.GAMEFRAME.PANEL_TOP_MARGIN * 2,
        #                                      constants.GAMEFRAME.PANEL_FONT_SIZE, constants.GAMEFRAME.BLACK_ALPHA,
        #                                      self.batch, self.second_foreground)
        # self.a_reward_label = batch_label(str(self.hacker.cumulative_reward), constants.GAMEFRAME.PANEL_LEFT_MARGIN * 2.2,
        #                                   self.height - constants.GAMEFRAME.PANEL_TOP_MARGIN,
        #                                   constants.GAMEFRAME.PANEL_FONT_SIZE, constants.GAMEFRAME.BLACK_ALPHA, self.batch,
        #                                   self.second_foreground)
        # self.d_reward_label = batch_label(str(self.data_node.cumulative_reward), constants.GAMEFRAME.PANEL_LEFT_MARGIN * 5.2,
        #                                   self.height - constants.GAMEFRAME.PANEL_TOP_MARGIN,
        #                                   constants.GAMEFRAME.PANEL_FONT_SIZE, constants.GAMEFRAME.BLACK_ALPHA, self.batch,
        #                                   self.second_foreground)
        # self.step_label = batch_label(str(self.game_step), constants.GAMEFRAME.PANEL_LEFT_MARGIN * 2.2,
        #                               self.height - constants.GAMEFRAME.PANEL_TOP_MARGIN * 2,
        #                               constants.GAMEFRAME.PANEL_FONT_SIZE, constants.GAMEFRAME.BLACK_ALPHA, self.batch,
        #                               self.second_foreground)
        # self.num_games_label = batch_label(str(self.num_games),
        #                                    constants.GAMEFRAME.PANEL_LEFT_MARGIN * 7.5,
        #                                    self.height - constants.GAMEFRAME.PANEL_TOP_MARGIN,
        #                                    constants.GAMEFRAME.PANEL_FONT_SIZE, constants.GAMEFRAME.BLACK_ALPHA, self.batch,
        #                                    self.second_foreground)

    def setup_resources_path(self):
        """
        Setup path to resources (e.g. images)

        :return: None
        """
        if os.path.exists(self.render_config.resources_dir):
            pyglet.resource.path = [self.render_config.resources_dir]
        else:
            script_dir = os.path.dirname(__file__)
            resource_path = os.path.join(script_dir, './', constants.RENDERING.RESOURCES_DIR)
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
        self.render_config.batch.draw()
        # Make this window the current OpenGL rendering context
        self.switch_to()


    def on_mouse_press(self, x, y, button, modifiers):
        if self.done:
            return
        self.unschedule_events()
        for i in range(self.resource_network.num_rows-1):
            for j in range(self.resource_network.num_cols):
                node = self.resource_network.grid[i][j].get_node()
                if node is not None:
                    if x > node.x and x < (node.x + node.width):
                        if y > node.y and y < (node.y + node.height):
                            if self.adjacency_matrix[self.hacker.row*self.num_cols + self.hacker.col][node.row*self.num_cols + node.col] == 1:
                                defense_row, defense_col, defend_type = self.data_node.defense_action(self.graph_layout)
                                self.resource_network.grid[defense_row][defense_col].defend(defend_type)
                                attack_successful = True
                                # if isinstance(node, Data):
                                #     edges = self.resource_network.grid[self.hacker.row][self.hacker.col].resource.outgoing_edges
                                #     attack_successful = node.simulate_attack(self.attack_type, edges)
                                # else:
                                #     attack_successful = node.simulate_attack(self.attack_type)
                                self.game_step += 1
                                if attack_successful:
                                    self.hacker.move_to(node.x, node.y, node.col, node.row)
                                    # if isinstance(node, Data):
                                    #     self.done = True
                                    #     self.hacker.add_reward(constants.GAMEFRAME.POSITIVE_REWARD)
                                    #     node.add_reward(constants.GAMEFRAME.NEGATIVE_REWARD)
                                else:
                                    detected = node.simulate_detection()
                                    if detected:
                                        self.hacker.add_reward(constants.RENDERING.NEGATIVE_REWARD)
                                        self.hacker.detected()
                                        self.data_node.add_reward(constants.RENDERING.POSITIVE_REWARD)
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
        pass
        # self.step_label.text = str(self.game_step)
        # self.a_reward_label.text = str(self.hacker.cumulative_reward)
        # self.d_reward_label.text = str(self.data_node.cumulative_reward)
        # self.attack_type_label.text = str(self.game_state.attack_type)
        # self.num_games_label.text = str(self.game_state.num_games)
        # self.hacker.update()

    def set_node_states(self, attack_values, defense_values, det_values):
        pass
        # for i in range(self.resource_network.num_rows):
        #     for j in range(self.resource_network.num_cols):
        #         self.resource_network.grid[i][j].set_state(attack_values[i][j], defense_values[i][j], det_values[i][j])

    def set_state(self, render_state:RenderState):
        """
        TODO

        :param state: the state
        :return: None
        """
        pass
        # attack_values = np.copy(render_state.attack_values)
        # defense_values = np.copy(render_state.defense_values)
        # det_values = np.copy(render_state.defense_det)
        # self.set_node_states(attack_values, defense_values, det_values)
        # attacker_row, attacker_col = render_state.attacker_pos
        # self.attack_events = render_state.attack_events
        # self.defense_events = render_state.defense_events
        # hacker_node =  self.resource_network.grid[attacker_row][attacker_col].get_node()
        # if hacker_node is not None:
        #     self.hacker.move_to(hacker_node.x, hacker_node.y, hacker_node.col, hacker_node.row)
        # self.hacker.set_reward(render_state.attacker_cumulative_reward)
        # self.data_node.set_reward(render_state.defender_cumulative_reward)
        # self.game_step = render_state.game_step
        # self.step_label.text = str(self.game_step)
        # self.num_games = render_state.num_games
        # self.num_games_label.text = str(self.num_games)
        # self.a_reward_label.text = str(self.hacker.cumulative_reward)
        # self.d_reward_label.text = str(self.data_node.cumulative_reward)
        # self.attack_type_label.text = str(self.attack_type)
        # self.hacker.update()
        # self.done = render_state.done
        # if render_state.detected:
        #     self.hacker.detected()

    def simulate_events(self, i):
        self.simulate_defense_events(self.defense_events, i)
        self.simulate_attack_events(self.attack_events, i)

    def reset_events(self):
        self.attack_events = []
        self.defense_events = []

    def simulate_attack_events(self, attack_events: List[AttackDefenseEvent], i):
        for attack in attack_events:
            self.attack_type = attack.attack_defense_type
            target_node = self.resource_network.grid[attack.target_row][attack.target_col].get_node()
            # if isinstance(target_node, Data):
            #     edges = []
            #     if isinstance(self.resource_network.grid[self.hacker.row][self.hacker.col], Server):
            #         edges = self.resource_network.grid[self.hacker.row][self.hacker.col].resource.outgoing_edges
            #     self.resource_network.grid[attack.target_row][attack.target_col].manual_blink_attack(i, edges)
            # else:
            #     self.resource_network.grid[attack.target_row][attack.target_col].manual_blink_attack(i)

    def test(self):
        if self.defense_event is not None:
            defense = self.defense_event
            pyglet.clock.schedule(self.resource_network.grid[defense.target_row][defense.target_col].data.defense_black)
            pyglet.clock.tick(poll=True)

    def simulate_defense_events(self, defense_events: List[AttackDefenseEvent], i):
        for defense in defense_events:
            self.resource_network.grid[defense.target_row][defense.target_col].manual_blink_defense(i)

    def unschedule_events(self):
        for i in range(self.resource_network.num_rows - 1):
            for j in range(self.resource_network.num_cols):
                node = self.resource_network.grid[i][j].get_node()
                if node is not None:
                    node.unschedule()

    def reset(self):
        """
        Resets the agent state without closing the screen

        :return: None
        """
        self.done = False
        self.unschedule_events()
        self.hacker.reset()
        attack_values = np.copy(self.init_state.attack_values)
        defense_values = np.copy(self.init_state.defense_values)
        det_values = np.copy(self.init_state.defense_det)
        self.set_node_states(attack_values, defense_values, det_values)
        self.num_games += 1
        self.game_step = 0
        self.switch_to()
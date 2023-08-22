"""
The main gameframe for the gym-idsgame environment
"""
from typing import List, Union
import os
import pyglet
import gym_idsgame.envs.util.idsgame_util as util
from gym_idsgame.envs.rendering.network.network import Network
from gym_idsgame.envs.constants import constants
from gym_idsgame.envs.rendering.util.render_util import batch_rect_fill
from gym_idsgame.envs.dao.game_state import GameState
from gym_idsgame.envs.dao.attack_defense_event import AttackDefenseEvent
from gym_idsgame.envs.dao.idsgame_config import IdsGameConfig
from gym_idsgame.envs.rendering.agents.attacker import Attacker
from gym_idsgame.envs.rendering.frames.panels.game_panel import GamePanel
from gym_idsgame.envs.dao.node_type import NodeType
from gym_idsgame.envs.rendering.network.nodes.node import Node

class GameFrame(pyglet.window.Window):
    """
    A class representing the OpenGL/Pyglet Game Frame

    By subclassing pyglet.window.Window, event handlers can be defined simply by overriding functions, e.g.
    event handler for on_draw is defined by overriding the on_draw function.
    """

    def __init__(self, idsgame_config: IdsGameConfig):
        """
        Constructor, initializes the frame

        :param idsgame_config: Config for the IdsGameEnv
        """
        self.idsgame_config = idsgame_config
        # call constructor of parent class
        super(GameFrame, self).__init__(height=idsgame_config.render_config.height,
                                        width=idsgame_config.render_config.width,
                                        caption=idsgame_config.render_config.caption)
        self.resource_network = None
        self.attacker_sprite = None
        self.defender_agent = None
        self.attacker_agent = None
        self.game_state: GameState = None
        self.setup_resources_path()
        self.create_batch()
        self.set_state(self.idsgame_config.game_config.initial_state)
        self.switch_to()
        self.attacker_view = self.idsgame_config.render_config.attacker_view
        self.defender_view = self.idsgame_config.render_config.defender_view
        if self.attacker_view and not self.defender_view:
            self.toggle_attacker_view()
        elif not self.attacker_view and self.defender_view:
            self.toggle_defender_view()
        else:
            self.toggle_full_view()

    def create_batch(self) -> None:
        """
        Creates a batch of elements to render. By grouping elements in a batch we can utilize OpenGL batch rendering
        and reduce the cpu <–> gpu data transfers and the number of draw-calls.

        :return: None
        """

        # Sets the background color
        batch_rect_fill(0, 0, self.idsgame_config.render_config.width, self.idsgame_config.render_config.height,
                        self.idsgame_config.render_config.bg_color,
                        self.idsgame_config.render_config.batch, self.idsgame_config.render_config.background)

        # Resource Network
        self.resource_network = Network(self.idsgame_config)

        # Resource Network Links
        self.resource_network.create_links()

        # Attacker Sprite
        attacker_row, attacker_col = self.idsgame_config.game_config.network_config.start_pos
        self.attacker_sprite = Attacker(self.idsgame_config, attacker_col, attacker_row)

        # Defender Agent
        self.defender_agent = self.idsgame_config.defender_agent

        # Attacker Agent
        self.attacker_agent = self.idsgame_config.attacker_agent

        # Game Panel
        self.game_panel = GamePanel(self.idsgame_config)

    def setup_resources_path(self) -> None:
        """
        Setup path to resources (e.g. images)

        :return: None
        """
        if os.path.exists(self.idsgame_config.render_config.resources_dir):
            pyglet.resource.path = [self.idsgame_config.render_config.resources_dir]
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
        self.idsgame_config.render_config.batch.draw()
        # Make this window the current OpenGL rendering context
        self.switch_to()

    def on_mouse_press(self, x:int, y:int, button, modifiers) -> None:
        # Dont do anything if agent is playing
        if not self.idsgame_config.game_config.manual_attacker and not self.idsgame_config.game_config.manual_defender:
            return
        # Dont do anything if game is over
        if self.game_state.done:
            return

        # Unschedule events from previous press, if any
        self.unschedule_events()
        # 1. Find the node in the network that was pressed
        for i in range(self.idsgame_config.game_config.num_rows):
            for j in range(self.idsgame_config.game_config.num_cols):
                node = self.resource_network.grid[i][j]
                if node.node_type != NodeType.EMPTY:

                    if node.node_type == NodeType.START:
                        if node.x-node.radius < x < (node.x + node.radius) and node.y-node.radius < y < (node.y + node.radius):
                            # 1.5 Special case: if it is the start node, let the attacker move there without making
                            # any attack or risk to be detected
                            if node.node_type == NodeType.START and util.is_attack_legal(
                                    node.pos, self.attacker_sprite.pos, self.idsgame_config.game_config.network_config):
                                attack_row, attack_col = self.game_state.attacker_pos
                                self.resource_network.grid[attack_row][attack_col].visualize_attack(
                                    self.game_state.attack_defense_type, node.pos, [])
                                self.game_state.attacker_pos = node.pos
                                self.game_state.game_step += 1
                                return

                    if (node.node_type != NodeType.START and
                            node.x < x < (node.x + node.width) and node.y < y < (node.y + node.height)):

                        # Manual Defender
                        if self.idsgame_config.game_config.manual_defender:

                            detect = modifiers & pyglet.window.key.MOD_SHIFT

                            # 2. Update defense state
                            self.game_state.defend(node.id, self.game_state.attack_defense_type,
                                                   self.idsgame_config.game_config.max_value,
                                                   self.idsgame_config.game_config.network_config, detect=detect)

                            # 3. Update attack state
                            attack_id = self.attacker_agent.action(self.game_state)
                            attack_node_id, attack_node_pos, attack_type, reconnaissance = util.interpret_attack_action(
                                attack_id, self.idsgame_config.game_config)
                            attack_row, attack_col = attack_node_pos
                            legal = util.is_attack_legal(attack_node_pos, self.attacker_sprite.pos,
                                                 self.idsgame_config.game_config.network_config)

                            if not reconnaissance:
                                self.game_state.attack(attack_node_id, attack_type,
                                                       self.idsgame_config.game_config.max_value,
                                                       self.idsgame_config.game_config.network_config,
                                                       reconnaissance_enabled=self.idsgame_config.reconnaissance_actions)
                            else:
                                self.game_state.reconnaissance(attack_node_id, attack_type, reconnaissance_reward=self.idsgame_config.reconnaissance_reward)

                            # 4. Visualize defense
                            self.resource_network.grid[node.row][node.col].visualize_defense(detect=detect)

                            attack_successful = False

                            if not reconnaissance:
                                # 6. Visualize attack
                                edges = []
                                if self.resource_network.grid[attack_row][attack_col].node_type == NodeType.DATA:
                                    edges = self.resource_network.get(self.attacker_sprite.pos).outgoing_edges

                                self.resource_network.grid[attack_row][attack_col].visualize_attack(
                                    attack_type, self.game_state.attacker_pos, edges)

                                # 7. Simulate attack outcome
                                attack_successful = self.game_state.simulate_attack(
                                    attack_node_id, attack_type, self.idsgame_config.game_config.network_config)
                            else:
                                # 6. Visualize reconnaissance
                                self.resource_network.grid[attack_row][attack_col].visualize_reconnaissance()

                            # 8. Update game state based on the outcome of the attack
                            self.game_state.game_step += 1
                            if attack_successful:
                                self.game_state.attacker_pos = (attack_row, attack_col)
                                if self.resource_network.grid[attack_row][attack_col].node_type == NodeType.DATA:
                                    self.game_state.done = True
                                    self.game_state.hacked = True
                            else:
                                detected = self.game_state.simulate_detection(
                                    self.resource_network.grid[attack_row][attack_col].id, reconnaissance=reconnaissance,
                                reconnaissance_detection_factor=self.idsgame_config.reconnaissance_detection_factor)
                                if detected:
                                    self.game_state.done = True
                                    self.game_state.detected = True

                        # Manual Attacker
                        elif self.idsgame_config.game_config.manual_attacker:
                            # 2. Check that the selected node can be attacked (there is a link to it from the current
                            # position of the attacker)
                            if util.is_attack_legal(node.pos, self.attacker_sprite.pos,
                                                    self.idsgame_config.game_config.network_config):
                                # 3. Update defense state
                                defend_id = self.defender_agent.action(self.game_state)
                                defend_node_id, defend_node_pos, defend_type = util.interpret_defense_action(
                                    defend_id, self.idsgame_config.game_config)
                                defense_row, defense_col = defend_node_pos
                                detect = defend_type == self.idsgame_config.game_config.max_value+1

                                defend_node_id = self.idsgame_config.game_config.network_config.get_node_id(
                                    (defense_row, defense_col))
                                self.game_state.defend(defend_node_id, defend_type,
                                                       self.idsgame_config.game_config.max_value,
                                                       self.idsgame_config.game_config.network_config, detect=detect)

                                # 4. Update attack state
                                reconnaissance = self.game_state.attack_defense_type >= \
                                                 self.idsgame_config.game_config.num_attack_types
                                if not reconnaissance:
                                    self.game_state.attack(node.id, self.game_state.attack_defense_type,
                                                           self.idsgame_config.game_config.max_value,
                                                           self.idsgame_config.game_config.network_config,
                                                           reconnaissance_enabled=self.idsgame_config.reconnaissance_actions)
                                else:
                                    attack_type = self.game_state.attack_defense_type - \
                                                  self.idsgame_config.game_config.num_attack_types
                                    self.game_state.reconnaissance(node.id, attack_type, reconnaissance_reward=self.idsgame_config.reconnaissance_reward)

                                # 5. Visualize defense
                                self.resource_network.grid[defense_row][defense_col].visualize_defense(detect)

                                attack_successful = False

                                if not reconnaissance:
                                    # 6. Visualize attack
                                    edges = []
                                    if node.node_type == NodeType.DATA:
                                        edges = self.resource_network.get(self.attacker_sprite.pos).outgoing_edges
                                    node.visualize_attack(self.game_state.attack_defense_type, self.game_state.attacker_pos,
                                                          edges)

                                    # 7. Simulate attack outcome
                                    attack_successful = self.game_state.simulate_attack(
                                        node.id, self.game_state.attack_defense_type,
                                        self.idsgame_config.game_config.network_config)
                                else:
                                    # 6. Visualize reconnaissance activity
                                    node.visualize_reconnaissance()

                                # 8. Update game state based on the outcome of the attack
                                self.game_state.game_step += 1
                                if attack_successful:
                                    self.game_state.attacker_pos = node.pos
                                    if node.node_type == NodeType.DATA:
                                        self.game_state.done = True
                                        self.game_state.hacked = True
                                else:
                                    detected = self.game_state.simulate_detection(node.id, reconnaissance=reconnaissance,
                                                                                  reconnaissance_detection_factor=self.idsgame_config.reconnaissance_detection_factor)
                                    if detected:
                                        self.game_state.done = True
                                        self.game_state.detected = True

    def on_key_press(self, symbol, modifiers) -> None:
        """
        Event handler for on_key_press event.
        The user can move the agent with key presses.

        :param symbol: the symbol of the keypress
        :param modifiers: _
        :return: None
        """
        if self.idsgame_config.game_config.manual_attacker or self.idsgame_config.game_config.manual_defender:
            if symbol == pyglet.window.key._1 or int(symbol) == 49:
                if self.idsgame_config.game_config.manual_attacker:
                    cond = self.idsgame_config.game_config.num_attack_types > 1
                    if self.idsgame_config.reconnaissance_actions:
                        cond = (self.idsgame_config.game_config.num_attack_types + 1) > 1
                    if cond:
                        self.game_state.attack_defense_type = 1
                elif  self.idsgame_config.game_config.manual_defender:
                    if self.idsgame_config.game_config.num_attack_types > 1:
                        self.game_state.attack_defense_type = 1
            elif symbol == pyglet.window.key._2 or int(symbol) == 50:
                if self.idsgame_config.game_config.manual_attacker:
                    cond = self.idsgame_config.game_config.num_attack_types  > 2
                    if self.idsgame_config.reconnaissance_actions:
                        cond = (self.idsgame_config.game_config.num_attack_types + 1) > 2
                    if cond:
                        self.game_state.attack_defense_type = 2
                elif self.idsgame_config.game_config.manual_defender:
                    if self.idsgame_config.game_config.num_attack_types > 2:
                        self.game_state.attack_defense_type = 2
            elif symbol == pyglet.window.key._3 or int(symbol) == 51:
                if self.idsgame_config.game_config.manual_attacker:
                    cond = self.idsgame_config.game_config.num_attack_types  > 3
                    if self.idsgame_config.reconnaissance_actions:
                        cond = (self.idsgame_config.game_config.num_attack_types + 1) > 3
                    if cond:
                        self.game_state.attack_defense_type = 3
                elif self.idsgame_config.game_config.manual_defender or int(symbol) == 53:
                    if self.idsgame_config.game_config.num_attack_types > 3:
                        self.game_state.attack_defense_type = 3
            elif symbol == pyglet.window.key._4 or int(symbol) == 52:
                if self.idsgame_config.game_config.manual_attacker:
                    cond = self.idsgame_config.game_config.num_attack_types//self.idsgame_config.game_config.num_nodes > 4
                    if self.idsgame_config.reconnaissance_actions:
                        cond = (self.idsgame_config.game_config.num_attack_types+1) > 4
                    if cond:
                        self.game_state.attack_defense_type = 4
                elif self.idsgame_config.game_config.manual_defender:
                    if self.idsgame_config.game_config.num_attack_types > 4:
                        self.game_state.attack_defense_type = 4
            elif symbol == pyglet.window.key._5 or int(symbol) == 53:
                if self.idsgame_config.game_config.manual_attacker:
                    cond = self.idsgame_config.game_config.num_attack_types  > 5
                    if self.idsgame_config.reconnaissance_actions:
                        cond = (self.idsgame_config.game_config.num_attack_types + 1) > 5
                    if cond:
                        self.game_state.attack_defense_type = 5
                elif self.idsgame_config.game_config.manual_defender:
                    if self.idsgame_config.game_config.num_attack_types > 5:
                        self.game_state.attack_defense_type = 5
            elif symbol == pyglet.window.key._6 or int(symbol) == 54:
                if self.idsgame_config.game_config.manual_attacker:
                    cond = self.idsgame_config.game_config.num_attack_types  > 6
                    if self.idsgame_config.reconnaissance_actions:
                        cond = (self.idsgame_config.game_config.num_attack_types + 1) > 6
                    if cond:
                        self.game_state.attack_defense_type = 6
                elif self.idsgame_config.game_config.manual_defender:
                    if self.idsgame_config.game_config.num_attack_types > 6:
                        self.game_state.attack_defense_type = 6
            elif symbol == pyglet.window.key._7 or int(symbol) == 55:
                if self.idsgame_config.game_config.manual_attacker:
                    cond = self.idsgame_config.game_config.num_attack_types  > 7
                    if self.idsgame_config.reconnaissance_actions:
                        cond = (self.idsgame_config.game_config.num_attack_types + 1) > 7
                    if cond:
                        self.game_state.attack_defense_type = 7
                elif self.idsgame_config.game_config.manual_defender:
                    if self.idsgame_config.game_config.num_attack_types > 7:
                        self.game_state.attack_defense_type = 7
            elif symbol == pyglet.window.key._8 or int(symbol) == 56:
                if self.idsgame_config.game_config.manual_attacker:
                    cond = self.idsgame_config.game_config.num_attack_types  > 8
                    if self.idsgame_config.reconnaissance_actions:
                        cond = (self.idsgame_config.game_config.num_attack_types + 1) > 8
                    if cond:
                        self.game_state.attack_defense_type = 8
                elif self.idsgame_config.game_config.manual_defender:
                    if self.idsgame_config.game_config.num_attack_types > 8:
                        self.game_state.attack_defense_type = 8
            elif symbol == pyglet.window.key._9 or int(symbol) == 57:
                if self.idsgame_config.game_config.manual_attacker:
                    cond = self.idsgame_config.game_config.num_attack_types  > 9
                    if self.idsgame_config.reconnaissance_actions:
                        cond = (self.idsgame_config.game_config.num_attack_types + 1) > 9
                    if cond:
                        self.game_state.attack_defense_type = 9
                elif self.idsgame_config.game_config.manual_defender:
                    if self.idsgame_config.game_config.num_attack_types > 9:
                        self.game_state.attack_defense_type = 9
            elif symbol == pyglet.window.key._0 or symbol == 48:
                if self.idsgame_config.game_config.manual_attacker:
                    cond = self.idsgame_config.game_config.num_attack_types  > 0
                    if self.idsgame_config.reconnaissance_actions:
                        cond = (self.idsgame_config.game_config.num_attack_types + 1) > 0
                    if cond:
                        self.game_state.attack_defense_type = 0
                elif self.idsgame_config.game_config.manual_defender:
                    if self.idsgame_config.game_config.num_attack_types > 0:
                        self.game_state.attack_defense_type = 0
            elif symbol == pyglet.window.key.SPACE:
                self.reset(update_stats=True)
            elif symbol == pyglet.window.key.A or symbol == 97:
                self.toggle_attacker_view()
            elif symbol == pyglet.window.key.D or symbol == 100:
                self.toggle_defender_view()
            elif symbol == pyglet.window.key.F or symbol == 102:
                self.toggle_full_view()

    def toggle_attacker_view(self):
        """
        Toggle attacker's view, hide defender details (partially observed)

        :return: None
        """
        self.attacker_view = True
        self.defender_view = False
        for i in range(self.idsgame_config.game_config.num_rows - 1):
            for j in range(self.idsgame_config.game_config.num_cols):
                node = self.resource_network.grid[i][j]
                if node is not None and node.node_type != NodeType.EMPTY:
                    node.toggle_attacker_view()
        self.attacker_sprite.show()

    def toggle_defender_view(self):
        """
        Toggle defender's view, hide attacker details (partially observed)

        :return: None
        """
        self.attacker_view = False
        self.defender_view = True
        for i in range(self.idsgame_config.game_config.num_rows - 1):
            for j in range(self.idsgame_config.game_config.num_cols):
                node = self.resource_network.grid[i][j]
                if node is not None and node.node_type != NodeType.EMPTY:
                    node.toggle_defender_view()
        self.attacker_sprite.hide()

    def toggle_full_view(self):
        """
        Toggle fully observed view

        :return: None
        """
        self.attacker_view = True
        self.defender_view = True
        for i in range(self.idsgame_config.game_config.num_rows - 1):
            for j in range(self.idsgame_config.game_config.num_cols):
                node = self.resource_network.grid[i][j]
                if node is not None and node.node_type != NodeType.EMPTY:
                    node.toggle_full_view()
        self.attacker_sprite.show()


    def update(self, dt) -> None:
        """
        Event handler for the update-event (timer-based typically), used to update the state of the grid.

        :param dt: the number of seconds since the function was last called
        :return: None
        """
        self.set_state(self.game_state)


    def set_state(self, game_state:GameState) -> None:
        """
        Updates the current state

        :param state: the new state
        :return: None
        """
        self.game_state = game_state.copy()
        self.game_panel.update_state_text(self.game_state)
        self.attacker_sprite.move_to_pos(self.game_state.attacker_pos)
        if game_state.detected:
            self.attacker_sprite.detected()
        else:
            self.attacker_sprite.undetect()
        self.resource_network.set_node_states(self.game_state)

    def simulate_events(self, i:int) -> None:
        """
        Simulates attack/defense events manually. Method used when rendering in agent-mode

        :param i: the index of the event visualization
        :return:  None
        """
        self.simulate_defense_events(self.game_state.defense_events, i)
        self.simulate_attack_events(self.game_state.attack_events, i)

    def reset_events(self) -> None:
        """
        Resets the events for agent-mode rendering

        :return: None
        """
        self.game_state.attack_events = []
        self.game_state.defense_events = []

    def simulate_attack_events(self, attack_events: List[AttackDefenseEvent], i: int) -> None:
        """
        Simulate attack events manually for rendering in agent-mode

        :param attack_events: the list of attack events to simulate
        :param i: the index of the event visualization
        :return: None
        """
        for attack in attack_events:
            self.attack_type = attack.attack_defense_type
            target_node: Node = self.resource_network.grid[attack.target_row][attack.target_col]
            attack_row, attack_col = attack.attacker_pos
            edges = []
            if not attack.reconnaissance:
                if target_node.node_type == NodeType.DATA:
                    edges = self.resource_network.get(attack.attacker_pos).outgoing_edges
                if target_node.node_type == NodeType.START:
                    self.resource_network.grid[attack_row][attack_col].manual_blink_attack(
                        i, target_node.pos, edges)
                    return
                self.resource_network.grid[attack.target_row][attack.target_col].manual_blink_attack(
                    i, attack.attacker_pos, edges)
            else:
                self.resource_network.grid[attack.target_row][attack.target_col].manual_reconnaissance(i)

    def simulate_defense_events(self, defense_events: List[AttackDefenseEvent], i:int) -> None:
        """
        Simulate defense events manually for rendering in agent-mode

        :param defense_events: the list of defense events to simulate
        :param i: the index of the event visualization
        :return: None
        """
        for defense in defense_events:
            detect = defense.attack_defense_type == self.idsgame_config.game_config.num_attack_types
            self.resource_network.grid[defense.target_row][defense.target_col].manual_blink_defense(i, detect=detect)

    def unschedule_events(self) -> None:
        """
        Utility method for unscheduling events. When the user triggers an action before the last action completed,
        this method will be called to avoid getting spam-visualizations in the UI.

        :return: None
        """
        for i in range(self.idsgame_config.game_config.num_rows - 1):
            for j in range(self.idsgame_config.game_config.num_cols):
                node = self.resource_network.grid[i][j]
                if node is not None:
                    node.unschedule()

    def reset(self, update_stats=False) -> None:
        """
        Resets the agent state without closing the screen

        :param update_stats: boolean flag whether to update the game statistics
        :return: None
        """
        self.game_state.new_game(self.idsgame_config.game_config.initial_state,
                                 a_reward=constants.GAME_CONFIG.POSITIVE_REWARD,
                                 d_reward=constants.GAME_CONFIG.POSITIVE_REWARD,
                                 update_stats=update_stats,
                                 randomize_state=self.idsgame_config.randomize_env,
                                 network_config=self.idsgame_config.game_config.network_config,
                                 num_attack_types=self.idsgame_config.game_config.num_attack_types,
                                 defense_val=self.idsgame_config.game_config.defense_val,
                                 attack_val=self.idsgame_config.game_config.attack_val,
                                 det_val=self.idsgame_config.game_config.det_val,
                                 vulnerability_val=self.idsgame_config.game_config.vulnerabilitiy_val,
                                 num_vulnerabilities_per_layer=
                                 self.idsgame_config.game_config.num_vulnerabilities_per_layer,
                                 num_vulnerabilities_per_node=self.idsgame_config.game_config.num_vulnerabilities_per_node,
                                 randomize_visibility=self.idsgame_config.randomize_visibility,
                                 visibility_p=self.idsgame_config.visibility_p
                                 )
        if self.idsgame_config.randomize_starting_position:
            self.game_state.randomize_attacker_position(self.idsgame_config.game_config.network_config)
        self.set_state(self.game_state)
        self.reset_events()
        self.unschedule_events()
        self.switch_to()
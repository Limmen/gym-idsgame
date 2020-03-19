import pyglet
from gym_idsgame.envs.rendering.network.network import Network
from gym_idsgame.envs.constants import constants
from gym_idsgame.envs.rendering.util.render_util import batch_rect_fill
from gym_idsgame.envs.dao.render_state import RenderState
from gym_idsgame.envs.dao.attack_defense_event import AttackDefenseEvent
from gym_idsgame.envs.dao.render_config import RenderConfig
from gym_idsgame.envs.rendering.agents.attacker import Attacker
from gym_idsgame.envs.rendering.agents.defender import Defender
from gym_idsgame.envs.rendering.frames.panels.game_panel import GamePanel
from gym_idsgame.envs.dao.node_type import NodeType
from typing import List
import os

class GameFrame(pyglet.window.Window):
    """
    A class representing the OpenGL/Pyglet Game Frame

    By subclassing pyglet.window.Window, event handlers can be defined simply by overriding functions, e.g.
    event handler for on_draw is defined by overriding the on_draw function.
    """

    def __init__(self, render_config: RenderConfig):
        """
        Constructor, initializes the frame

        :param render_config: the render config, e.g the font size, avatars, line width, colors, etc.
        """
        self.render_config = render_config
        super(GameFrame, self).__init__(height=render_config.height, width=render_config.width,
                                        caption=render_config.caption) # call constructor of parent class
        self.resource_network = None
        self.attacker = None
        self.defender = None
        self.render_state = None
        self.setup_resources_path()
        self.create_batch()
        self.set_state(self.render_config.game_config.initial_state)
        self.switch_to()

    def create_batch(self) -> None:
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

        # Resource Network Links
        self.resource_network.create_links()

        # Attacker
        attacker_row, attacker_col = self.render_config.game_config.network_config.start_pos
        self.attacker = Attacker(self.render_config, attacker_col, attacker_row)

        # Defender
        self.defender = Defender(self.render_config.defender_policy)

        # Game Panel
        self.game_panel = GamePanel(self.render_config)

    def setup_resources_path(self) -> None:
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


    def on_mouse_press(self, x:int, y:int, button, modifiers) -> None:
        # Dont do anything if agent is playing
        if not self.render_config.game_config.manual:
            return
        # Dont do anything if game is over
        if self.render_state.done:
            return

        # Unschedule events from previous press, if any
        self.unschedule_events()
        # 1. Find the node in the network that was pressed
        for i in range(self.render_config.game_config.num_rows-1):
            for j in range(self.render_config.game_config.num_cols):
                node = self.resource_network.grid[i][j]
                if node.node_type != NodeType.EMPTY:
                    if node.x < x < (node.x + node.width) and node.y < y < (node.y + node.height):

                        # 2. Check that the selected node can be attacked (there is a link to it from the current
                        # position of the attacker)
                        if self.resource_network.is_attack_legal(self.attacker.pos, node.pos):

                            # 3. Simulate defense
                            defense_row, defense_col, defend_type = self.defender.policy.action(self.render_state)
                            self.resource_network.grid[defense_row][defense_col].defend(defend_type)
                            edges = []
                            if node.node_type == NodeType.DATA:
                                edges = self.resource_network.get(self.attacker.pos).outgoing_edges

                            # 4. Simulate attack
                            attack_successful = node.simulate_attack(self.render_state.attack_type, edges)

                            # 5. Update state
                            self.render_state.game_step += 1
                            if attack_successful:
                                self.render_state.attacker_pos = node.pos
                                if node.node_type == NodeType.DATA:
                                    self.render_state.done = True
                                    self.render_state.attacker_cumulative_reward += constants.GAME_CONFIG.POSITIVE_REWARD
                                    self.render_state.defender_cumulative_reward -= constants.GAME_CONFIG.POSITIVE_REWARD
                            else:
                                detected = node.simulate_detection()
                                if detected:
                                    self.render_state.done = True
                                    self.render_state.detected = True
                                    self.render_state.attacker_cumulative_reward -= constants.GAME_CONFIG.POSITIVE_REWARD
                                    self.render_state.defender_cumulative_reward += constants.GAME_CONFIG.POSITIVE_REWARD

    def on_key_press(self, symbol, modifiers) -> None:
        """
        Event handler for on_key_press event.
        The user can move the agent with key presses.

        :param symbol: the symbol of the keypress
        :param modifiers: _
        :return: None
        """
        if self.render_config.game_config.manual:
            if symbol == pyglet.window.key._1:
                self.render_state.attack_type = 1
            elif symbol == pyglet.window.key._2:
                self.render_state.attack_type = 2
            elif symbol == pyglet.window.key._3:
                self.render_state.attack_type = 3
            elif symbol == pyglet.window.key._4:
                self.render_state.attack_type = 4
            elif symbol == pyglet.window.key._5:
                self.render_state.attack_type = 5
            elif symbol == pyglet.window.key._6:
                self.render_state.attack_type = 6
            elif symbol == pyglet.window.key._7:
                self.render_state.attack_type = 7
            elif symbol == pyglet.window.key._8:
                self.render_state.attack_type = 8
            elif symbol == pyglet.window.key._9:
                self.render_state.attack_type = 9
            elif symbol == pyglet.window.key._0:
                self.render_state.attack_type = 0
            elif symbol == pyglet.window.key.SPACE:
                self.reset()


    def update(self, dt) -> None:
        """
        Event handler for the update-event (timer-based typically), used to update the state of the grid.

        :param dt: the number of seconds since the function was last called
        :return: None
        """
        self.set_state(self.render_state)

    def set_state(self, render_state:RenderState) -> None:
        """
        Updates the current state

        :param state: the new state
        :return: None
        """
        self.render_state = render_state.copy()
        self.game_panel.update_state_text(self.render_state)
        self.attacker.move_to_pos(self.render_state.attacker_pos)
        if render_state.detected:
            self.attacker.detected()
        else:
            self.attacker.undetect()
        self.resource_network.set_node_states(self.render_state)

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
        for i in range(self.render_config.game_config.num_rows - 1):
            for j in range(self.render_config.game_config.num_cols):
                node = self.resource_network.grid[i][j]
                if node is not None:
                    node.unschedule()

    def reset(self):
        """
        Resets the agent state without closing the screen

        :return: None
        """
        self.render_state.new_game(self.render_config.game_config.initial_state)
        self.set_state(self.render_state)
        self.unschedule_events()
        self.switch_to()
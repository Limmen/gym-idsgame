from gym_idsgame.envs.rendering.util.render_util import batch_label
from gym_idsgame.envs.constants import constants
from gym_idsgame.envs.dao.render_config import RenderConfig
from gym_idsgame.envs.dao.render_state import RenderState

class GamePanel:
    """
    DTO for the top-panel in the game that visualizes the current game-step, cumulative rewards, etc.
    """

    def __init__(self, render_config: RenderConfig):
        """
        Constructor, initializes the panel

        :param render_config: the rendering config, e.g line width, font size etc.
        """
        self.render_config = render_config
        self.set_labels()

    def set_labels(self) -> None:
        """
        Creates the labels of the panel (should only be called once)

        :return: Noneg
        """
        batch_label("Attack Reward: ", constants.RENDERING.PANEL_LEFT_MARGIN,
                    self.render_config.height - constants.RENDERING.PANEL_TOP_MARGIN,
                    constants.RENDERING.PANEL_FONT_SIZE, constants.RENDERING.BLACK_ALPHA, self.render_config.batch,
                    self.render_config.second_foreground)
        batch_label("Time-step: ", constants.RENDERING.PANEL_LEFT_MARGIN,
                    self.render_config.height - constants.RENDERING.PANEL_TOP_MARGIN * 2,
                    constants.RENDERING.PANEL_FONT_SIZE, constants.RENDERING.BLACK_ALPHA,
                    self.render_config.batch, self.render_config.second_foreground)

        batch_label("Attack Type: ", constants.RENDERING.PANEL_LEFT_MARGIN * 4,
                    self.render_config.height - constants.RENDERING.PANEL_TOP_MARGIN * 2,
                    constants.RENDERING.PANEL_FONT_SIZE, constants.RENDERING.BLACK_ALPHA,
                    self.render_config.batch, self.render_config.second_foreground)
        batch_label("Defense Reward: ", constants.RENDERING.PANEL_LEFT_MARGIN * 4,
                    self.render_config.height - constants.RENDERING.PANEL_TOP_MARGIN,
                    constants.RENDERING.PANEL_FONT_SIZE, constants.RENDERING.BLACK_ALPHA,
                    self.render_config.batch, self.render_config.second_foreground)
        batch_label("Num Games: ", constants.RENDERING.PANEL_LEFT_MARGIN * 6.5,
                    self.render_config.height - constants.RENDERING.PANEL_TOP_MARGIN,
                    constants.RENDERING.PANEL_FONT_SIZE, constants.RENDERING.BLACK_ALPHA,
                    self.render_config.batch, self.render_config.second_foreground)
        self.attack_type_label = batch_label("0", constants.RENDERING.PANEL_LEFT_MARGIN * 5.2,
                                             self.render_config.height - constants.RENDERING.PANEL_TOP_MARGIN * 2,
                                             constants.RENDERING.PANEL_FONT_SIZE, constants.RENDERING.BLACK_ALPHA,
                                             self.render_config.batch, self.render_config.second_foreground)
        self.a_reward_label = batch_label("0",
                                          constants.RENDERING.PANEL_LEFT_MARGIN * 2.2,
                                          self.render_config.height - constants.RENDERING.PANEL_TOP_MARGIN,
                                          constants.RENDERING.PANEL_FONT_SIZE, constants.RENDERING.BLACK_ALPHA,
                                          self.render_config.batch, self.render_config.second_foreground)
        self.d_reward_label = batch_label("0",
                                          constants.RENDERING.PANEL_LEFT_MARGIN * 5.2,
                                          self.render_config.height - constants.RENDERING.PANEL_TOP_MARGIN,
                                          constants.RENDERING.PANEL_FONT_SIZE, constants.RENDERING.BLACK_ALPHA,
                                          self.render_config.batch, self.render_config.second_foreground)
        self.step_label = batch_label("0", constants.RENDERING.PANEL_LEFT_MARGIN * 2.2,
                                      self.render_config.height - constants.RENDERING.PANEL_TOP_MARGIN * 2,
                                      constants.RENDERING.PANEL_FONT_SIZE, constants.RENDERING.BLACK_ALPHA,
                                      self.render_config.batch, self.render_config.second_foreground)
        self.num_games_label = batch_label("0",
                                           constants.RENDERING.PANEL_LEFT_MARGIN * 7.5,
                                           self.render_config.height - constants.RENDERING.PANEL_TOP_MARGIN,
                                           constants.RENDERING.PANEL_FONT_SIZE, constants.RENDERING.BLACK_ALPHA,
                                           self.render_config.batch, self.render_config.second_foreground)

    def update_state_text(self, render_state: RenderState) -> None:
        """
        Updates the text of the labels on the panel

        :param render_state: the state to reflect in the labels
        :return: Noneg
        """
        self.attack_type_label.text = str(render_state.attack_type)
        self.a_reward_label.text = str(render_state.attacker_cumulative_reward)
        self.d_reward_label.text = str(render_state.defender_cumulative_reward)
        self.step_label.text = str(render_state.game_step)
        self.num_games_label.text = str(render_state.num_games)

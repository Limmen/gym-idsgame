from gym_idsgame.envs.rendering.util.render_util import batch_label
from gym_idsgame.envs.constants import constants
from gym_idsgame.envs.dao.idsgame_config import IdsGameConfig
from gym_idsgame.envs.dao.render_state import RenderState

class GamePanel:
    """
    DTO for the top-panel in the game that visualizes the current game-step, cumulative rewards, etc.
    """

    def __init__(self, idsgame_config: IdsGameConfig):
        """
        Constructor, initializes the panel

        :param idsgame_config: IdsGameEnv config
        """
        self.idsgame_config = idsgame_config
        self.set_labels()

    def set_labels(self) -> None:
        """
        Creates the labels of the panel (should only be called once)

        :return: None
        """
        batch_label("Attack Reward: ", constants.RENDERING.PANEL_LEFT_MARGIN,
                    self.idsgame_config.render_config.height - constants.RENDERING.PANEL_TOP_MARGIN,
                    constants.RENDERING.PANEL_FONT_SIZE, constants.RENDERING.BLACK_ALPHA,
                    self.idsgame_config.render_config.batch,
                    self.idsgame_config.render_config.second_foreground)
        batch_label("Time-step: ", constants.RENDERING.PANEL_LEFT_MARGIN,
                    self.idsgame_config.render_config.height - constants.RENDERING.PANEL_TOP_MARGIN * 2,
                    constants.RENDERING.PANEL_FONT_SIZE, constants.RENDERING.BLACK_ALPHA,
                    self.idsgame_config.render_config.batch, self.idsgame_config.render_config.second_foreground)

        batch_label("Attack Type: ", constants.RENDERING.PANEL_LEFT_MARGIN * 4,
                    self.idsgame_config.render_config.height - constants.RENDERING.PANEL_TOP_MARGIN * 2,
                    constants.RENDERING.PANEL_FONT_SIZE, constants.RENDERING.BLACK_ALPHA,
                    self.idsgame_config.render_config.batch, self.idsgame_config.render_config.second_foreground)
        batch_label("Defense Reward: ", constants.RENDERING.PANEL_LEFT_MARGIN * 4,
                    self.idsgame_config.render_config.height - constants.RENDERING.PANEL_TOP_MARGIN,
                    constants.RENDERING.PANEL_FONT_SIZE, constants.RENDERING.BLACK_ALPHA,
                    self.idsgame_config.render_config.batch, self.idsgame_config.render_config.second_foreground)
        batch_label("Num Games: ", constants.RENDERING.PANEL_LEFT_MARGIN * 6.5,
                    self.idsgame_config.render_config.height - constants.RENDERING.PANEL_TOP_MARGIN,
                    constants.RENDERING.PANEL_FONT_SIZE, constants.RENDERING.BLACK_ALPHA,
                    self.idsgame_config.render_config.batch, self.idsgame_config.render_config.second_foreground)
        batch_label("P(breached): ", constants.RENDERING.PANEL_LEFT_MARGIN * 6.5,
                    self.idsgame_config.render_config.height - constants.RENDERING.PANEL_TOP_MARGIN*2,
                    constants.RENDERING.PANEL_FONT_SIZE, constants.RENDERING.BLACK_ALPHA,
                    self.idsgame_config.render_config.batch, self.idsgame_config.render_config.second_foreground)
        self.attack_type_label = batch_label("0", constants.RENDERING.PANEL_LEFT_MARGIN * 5.2,
                                             self.idsgame_config.render_config.height
                                             - constants.RENDERING.PANEL_TOP_MARGIN * 2,
                                             constants.RENDERING.PANEL_FONT_SIZE, constants.RENDERING.BLACK_ALPHA,
                                             self.idsgame_config.render_config.batch,
                                             self.idsgame_config.render_config.second_foreground)
        self.a_reward_label = batch_label("0",
                                          constants.RENDERING.PANEL_LEFT_MARGIN * 2.2,
                                          self.idsgame_config.render_config.height -
                                          constants.RENDERING.PANEL_TOP_MARGIN,
                                          constants.RENDERING.PANEL_FONT_SIZE, constants.RENDERING.BLACK_ALPHA,
                                          self.idsgame_config.render_config.batch,
                                          self.idsgame_config.render_config.second_foreground)
        self.d_reward_label = batch_label("0",
                                          constants.RENDERING.PANEL_LEFT_MARGIN * 5.2,
                                          self.idsgame_config.render_config.height -
                                          constants.RENDERING.PANEL_TOP_MARGIN,
                                          constants.RENDERING.PANEL_FONT_SIZE, constants.RENDERING.BLACK_ALPHA,
                                          self.idsgame_config.render_config.batch,
                                          self.idsgame_config.render_config.second_foreground)
        self.step_label = batch_label("0", constants.RENDERING.PANEL_LEFT_MARGIN * 2.2,
                                      self.idsgame_config.render_config.height -
                                      constants.RENDERING.PANEL_TOP_MARGIN * 2,
                                      constants.RENDERING.PANEL_FONT_SIZE, constants.RENDERING.BLACK_ALPHA,
                                      self.idsgame_config.render_config.batch,
                                      self.idsgame_config.render_config.second_foreground)
        self.num_games_label = batch_label("0",
                                           constants.RENDERING.PANEL_LEFT_MARGIN * 7.5,
                                           self.idsgame_config.render_config.height -
                                           constants.RENDERING.PANEL_TOP_MARGIN,
                                           constants.RENDERING.PANEL_FONT_SIZE, constants.RENDERING.BLACK_ALPHA,
                                           self.idsgame_config.render_config.batch,
                                           self.idsgame_config.render_config.second_foreground)
        self.hack_probability = batch_label("0.0",
                                           constants.RENDERING.PANEL_LEFT_MARGIN * 7.5,
                                           self.idsgame_config.render_config.height -
                                            constants.RENDERING.PANEL_TOP_MARGIN*2,
                                           constants.RENDERING.PANEL_FONT_SIZE, constants.RENDERING.BLACK_ALPHA,
                                           self.idsgame_config.render_config.batch,
                                            self.idsgame_config.render_config.second_foreground)

    def update_state_text(self, render_state: RenderState) -> None:
        """
        Updates the text of the labels on the panel

        :param render_state: the state to reflect in the labels
        :return: None
        """
        self.attack_type_label.text = str(render_state.attack_type)
        self.a_reward_label.text = str(render_state.attacker_cumulative_reward)
        self.d_reward_label.text = str(render_state.defender_cumulative_reward)
        self.step_label.text = str(render_state.game_step)
        self.num_games_label.text = str(render_state.num_games)
        hack_probability = 0.0
        if render_state.num_hacks > 0:
            hack_probability = float(render_state.num_hacks)/float(render_state.num_games)
        self.hack_probability.text = "{0:.2f}".format(hack_probability)

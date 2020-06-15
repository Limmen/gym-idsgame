"""
Abstract bot-agent for the gym-idsgame environment
"""
from abc import abstractmethod
from gym_idsgame.envs.dao.game_state import GameState
from gym_idsgame.envs.dao.game_config import GameConfig
from gym_idsgame.agents.agent import Agent

class BotAgent(Agent):
    """
    Abstract class representing a fixed agent-policy
    """

    def __init__(self, game_config: GameConfig):
        super(BotAgent, self).__init__(game_config)

    @abstractmethod
    def action(self, game_state: GameState) -> int:
        """
        Abstract method to be implemented by sub-classes.
        A method that takes in the current state and outputs an action

        :param game_state: the current state
        :return: action_id
        """
        pass

"""
Tests for game_config.py
"""

import pytest
import logging
from gym_idsgame.envs.dao.game_config import GameConfig

class TestConfigSuite():
    pytest.logger = logging.getLogger("gameconfig_tests")

    def test_initialization(self):
        game_config = GameConfig()
        assert game_config.initial_state is not None
        assert game_config.network_config is not None
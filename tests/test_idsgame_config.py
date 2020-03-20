"""
Tests for idsgame_config.py
"""

import pytest
import logging
from gym_idsgame.envs.dao.idsgame_config import IdsGameConfig

class TestIdsGameConfigSuite():
    pytest.logger = logging.getLogger("idsgame_config")

    def test_initialization(self):
        idsgame_config = IdsGameConfig()
        assert idsgame_config.render_config is not None
        assert idsgame_config.game_config is not None
        assert idsgame_config.defender_policy is not None
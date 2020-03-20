"""
Tests for render_config.py
"""

import pytest
import logging
from gym_idsgame.envs.dao.render_config import RenderConfig

class TestIdsGameConfigSuite():
    pytest.logger = logging.getLogger("render_config")

    def test_initialization(self):
        render_config = RenderConfig()
        assert render_config.rect_size is not None
        assert render_config.bg_color is not None
        assert render_config.attacker_filename is not None
        assert render_config.server_filename is not None
        assert render_config.data_filename is not None
        assert render_config.cage_filename is not None
        assert render_config.attacker_scale is not None
        assert render_config.server_scale is not None
        assert render_config.data_scale is not None
        assert render_config.cage_scale is not None
        assert render_config.line_width is not None
        assert render_config.caption is not None
        assert render_config.resources_dir is not None
        assert render_config.blink_interval is not None
        assert render_config.num_blinks is not None
        assert render_config.batch is not None
        assert render_config.background is not None
        assert render_config.first_foreground is not None
        assert render_config.second_foreground is not None
        assert render_config.height is not None
        assert render_config.width is not None
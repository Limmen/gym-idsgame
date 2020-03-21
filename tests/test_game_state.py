"""
Tests for game_state.py
"""

import pytest
import logging
import numpy as np
from gym_idsgame.envs.dao.game_state import GameState
from gym_idsgame.envs.dao.network_config import NetworkConfig

class TestGameStateSuite():
    pytest.logger = logging.getLogger("gamestate_tests")

    def test_initialization(self):
        GameState()

    def test_default_state(self):
        state = GameState()
        num_nodes = 8
        num_attack_types = 10
        state.default_state(list(range(num_nodes)), (3,1), num_attack_types)
        assert state.attack_values.shape == (8, 10)
        assert state.defense_values.shape == (8, 10)
        assert state.defense_det.shape == (8,)
        assert state.attacker_pos == (3,1)
        assert state.done == False
        assert state.hacked == False
        assert state.num_hacks == 0
        assert state.detected == False
        assert state.num_games == 0

    def test_copy(self):
        state = GameState()
        num_nodes = 8
        num_attack_types = 10
        state.default_state(list(range(num_nodes)), (3, 1), num_attack_types)
        copy = state.copy()
        assert copy.num_hacks == state.num_hacks
        assert np.array_equal(copy.attack_values, state.attack_values)
        assert np.array_equal(copy.defense_det, state.defense_det)
        assert np.array_equal(copy.defense_values, state.defense_values)

    def test_new_game(self):
        state = GameState()
        num_nodes = 8
        num_attack_types = 10
        state.default_state(list(range(num_nodes)), (3, 1), num_attack_types)
        init_state = state.copy()
        old_game_count = state.num_games
        state.new_game(init_state)
        assert state.num_games == old_game_count+1
        assert state.done == False
        assert state.detected == False
        state.default_state(list(range(num_nodes)), (3, 1), num_attack_types)
        init_state = state.copy()
        state.hacked = True
        old_hacked_count = 0
        state.new_game(init_state)
        assert state.num_hacks == old_hacked_count + 1

    def test_attack(self):
        state = GameState()
        network_config = NetworkConfig(3,3)
        num_nodes = 8
        num_attack_types = 10
        state.default_state(list(range(num_nodes)), (3, 1), num_attack_types)
        attack_node_id = 3
        attack_type =4
        max_value = 10
        old_count = state.attack_values[attack_node_id][attack_type]
        state.attack(attack_node_id, attack_type, max_value, network_config)
        assert state.attack_values[attack_node_id][attack_type] < max_value
        assert state.attack_values[attack_node_id][attack_type] == old_count+1
        state.attack_values[attack_node_id][attack_type] = 10
        state.attack(attack_node_id, attack_type, max_value, network_config)
        assert state.attack_values[attack_node_id][attack_type] == max_value

    def test_defend(self):
        state = GameState()
        network_config = NetworkConfig(3,3)
        num_nodes = 8
        num_attack_types = 10
        state.default_state(list(range(num_nodes)), (3, 1), num_attack_types)
        defend_node_id = 3
        defense_type = 4
        max_value = 10
        old_count = state.defense_values[defend_node_id][defense_type]
        state.defend(defend_node_id, defense_type, max_value, network_config)
        assert state.defense_values[defend_node_id][defense_type] < max_value
        assert state.defense_values[defend_node_id][defense_type] == old_count + 1
        state.defense_values[defend_node_id][defense_type] = 10
        state.defend(defend_node_id, defense_type, max_value, network_config)
        assert state.defense_values[defend_node_id][defense_type] == max_value

    def test_simulate_attack(self):
        state = GameState()
        network_config = NetworkConfig(3, 3)
        num_nodes = 8
        num_attack_types = 10
        state.default_state(list(range(num_nodes)), (3, 1), num_attack_types)
        attack_node_id = 3
        attack_type = 4
        state.defense_values[attack_node_id][attack_type] = 5
        state.attack_values[attack_node_id][attack_type] = 5
        assert not state.simulate_attack(attack_node_id, attack_type, network_config)
        state.defense_values[attack_node_id][attack_type] = 5
        state.attack_values[attack_node_id][attack_type] = 6
        assert state.simulate_attack(attack_node_id, attack_type, network_config)

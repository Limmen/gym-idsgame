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
        rows = 4
        cols = 4
        network_config = NetworkConfig(rows, cols)
        num_nodes = len(network_config.node_list)
        num_attack_types = 10
        state.default_state(list(range(num_nodes)), (3,1), num_attack_types, network_config)
        assert state.attack_values.shape == (num_nodes, 10)
        assert state.defense_values.shape == (num_nodes, 10)
        assert state.defense_det.shape == (num_nodes,)
        assert state.attacker_pos == (3,1)
        assert state.done == False
        assert state.hacked == False
        assert state.num_hacks == 0
        assert state.detected == False
        assert state.num_games == 0

    def test_copy(self):
        state = GameState()
        num_attack_types = 10
        rows = 4
        cols = 4
        network_config = NetworkConfig(rows, cols)
        num_nodes = len(network_config.node_list)
        state.default_state(list(range(num_nodes)), (3, 1), num_attack_types, network_config)
        copy = state.copy()
        assert copy.num_hacks == state.num_hacks
        assert np.array_equal(copy.attack_values, state.attack_values)
        assert np.array_equal(copy.defense_det, state.defense_det)
        assert np.array_equal(copy.defense_values, state.defense_values)

    def test_new_game(self):
        rows = 4
        cols = 4
        network_config = NetworkConfig(rows, cols)
        state = GameState()
        num_nodes = len(network_config.node_list)
        num_attack_types = 10
        state.default_state(list(range(num_nodes)), (3, 1), num_attack_types, network_config)
        init_state = state.copy()
        old_game_count = state.num_games
        state.new_game(init_state)
        assert state.num_games == old_game_count+1
        assert state.done == False
        assert state.detected == False
        state.default_state(list(range(num_nodes)), (3, 1), num_attack_types, network_config)
        init_state = state.copy()
        state.hacked = True
        old_hacked_count = 0
        state.new_game(init_state)
        assert state.num_hacks == old_hacked_count + 1

    def test_attack(self):
        state = GameState()
        rows = 4
        cols = 4
        network_config = NetworkConfig(rows, cols)
        num_nodes = len(network_config.node_list)
        num_attack_types = 10
        state.default_state(list(range(num_nodes)), (3, 1), num_attack_types, network_config)
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
        rows = 4
        cols = 4
        network_config = NetworkConfig(rows, cols)
        num_nodes = len(network_config.node_list)
        num_attack_types = 10
        state.default_state(list(range(num_nodes)), (3, 1), num_attack_types, network_config)
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
        rows = 4
        cols = 4
        network_config = NetworkConfig(rows, cols)
        num_nodes = len(network_config.node_list)
        state = GameState()
        num_attack_types = 10
        state.default_state(list(range(num_nodes)), (3, 1), num_attack_types, network_config)
        attack_node_id = 3
        attack_type = 4
        state.defense_values[attack_node_id][attack_type] = 5
        state.attack_values[attack_node_id][attack_type] = 5
        assert not state.simulate_attack(attack_node_id, attack_type, network_config)
        state.defense_values[attack_node_id][attack_type] = 5
        state.attack_values[attack_node_id][attack_type] = 6
        assert state.simulate_attack(attack_node_id, attack_type, network_config)

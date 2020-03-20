"""
Tests for network_config.py
"""

import pytest
import logging
from gym_idsgame.envs.dao.network_config import NetworkConfig

class TestIdsGameConfigSuite():
    pytest.logger = logging.getLogger("network_config")

    def test_initialization(self):
        num_rows = 2
        num_cols = 3
        network_config = NetworkConfig(num_rows, num_cols)
        assert network_config.graph_layout.shape == (num_rows, num_cols)
        for i in range(network_config.num_rows):
            for j in range(network_config.num_cols):
                assert network_config.graph_layout[i][j] is not None
        assert network_config.adjacency_matrix.shape == (num_rows*num_cols, num_rows*num_cols)
        for i in range(network_config.num_rows*network_config.num_cols):
            for j in range(network_config.num_cols*network_config.num_rows):
                assert network_config.adjacency_matrix[i][j] == 1 or network_config.adjacency_matrix[i][j] == 0

    def test_get_coords_of_adjacency_matrix_id(self):
        num_rows = 2
        num_cols = 3
        network_config = NetworkConfig(num_rows, num_cols)
        assert network_config.get_coords_of_adjacency_matrix_id(0) == (0, 0)
        assert network_config.get_coords_of_adjacency_matrix_id(1) == (0, 1)
        assert network_config.get_coords_of_adjacency_matrix_id(2) == (0, 2)
        assert network_config.get_coords_of_adjacency_matrix_id(3) == (1, 0)
        assert network_config.get_coords_of_adjacency_matrix_id(4) == (1, 1)
        assert network_config.get_coords_of_adjacency_matrix_id(5) == (1, 2)


    def test_start_pos(self):
        num_rows = 2
        num_cols = 3
        network_config = NetworkConfig(num_rows, num_cols)
        assert network_config.start_pos == (1, 1)

    def test_start_row(self):
        num_rows = 2
        num_cols = 3
        network_config = NetworkConfig(num_rows, num_cols)
        assert network_config.start_row == 1

    def test_start_col(self):
        num_rows = 2
        num_cols = 3
        network_config = NetworkConfig(num_rows, num_cols)
        assert network_config.start_col == 1

    def test_data_pos(self):
        num_rows = 2
        num_cols = 3
        network_config = NetworkConfig(num_rows, num_cols)
        assert network_config.data_pos == (0,1)

    def test_data_row(self):
        num_rows = 2
        num_cols = 3
        network_config = NetworkConfig(num_rows, num_cols)
        assert network_config.data_row == 0

    def test_data_col(self):
        num_rows = 2
        num_cols = 3
        network_config = NetworkConfig(num_rows, num_cols)
        assert network_config.data_col == 1

    def test_node_list(self):
        num_rows = 3
        num_cols = 3
        network_config = NetworkConfig(num_rows, num_cols)
        node_list = network_config.node_list
        assert len(node_list) == num_rows*num_cols - 4
        for i in range(len(node_list)):
            assert node_list[i] is not None

    def test_get_node_pos(self):
        num_rows = 3
        num_cols = 3
        network_config = NetworkConfig(num_rows, num_cols)
        assert network_config.get_node_pos(0) == (0, 1)
        assert network_config.get_node_pos(1) == (1, 0)
        assert network_config.get_node_pos(2) == (1, 1)
        assert network_config.get_node_pos(3) == (1, 2)
        assert network_config.get_node_pos(4) == (2, 1)


    def test_get_node_id(self):
        num_rows = 3
        num_cols = 3
        network_config = NetworkConfig(num_rows, num_cols)
        assert network_config.get_node_id((0,1)) == 0
        assert network_config.get_node_id((1, 0)) == 1
        assert network_config.get_node_id((1, 1)) == 2
        assert network_config.get_node_id((1, 2)) == 3
        assert network_config.get_node_id((2, 1)) == 4
        assert network_config.get_node_id((2, 2)) == -1
        assert network_config.get_node_id((2, 0)) == -1
        assert network_config.get_node_id((0, 0)) == -1
        assert network_config.get_node_id((0, 2)) == -1

import numpy as np
import pytest
from qaco_elite.grid import GridWorld, make_random_world, dijkstra_shortest_path
from qaco_elite.aco import path_length, turn_cost


def test_no_diagonal_corner_cutting():
    w = GridWorld(np.array([[0, 1], [0, 0]], dtype=bool))
    assert 3 not in [x for x, _, _ in w.neighbors[0]]
    assert dijkstra_shortest_path(w)[0] == pytest.approx(2.0)


def test_open_grid_diagonal_shortest_path():
    w = GridWorld(np.zeros((5, 5), dtype=bool))
    dist, path = dijkstra_shortest_path(w)
    assert dist == pytest.approx(4 * np.sqrt(2))
    assert path_length(w, path) == pytest.approx(dist)
    assert turn_cost(w, path) == 0.0


def test_four_direction_turn_cost():
    w = GridWorld(np.zeros((3, 3), dtype=bool))
    assert turn_cost(w, [0, 1, 4, 7, 8]) == 4.0  # 45,45,90 degrees


def test_seeded_map_reproducibility_and_connectivity():
    a, count = make_random_world(size=12, density=0.3, seed=94)
    b, count2 = make_random_world(size=12, density=0.3, seed=94)
    assert count == count2
    assert np.array_equal(a.obstacles, b.obstacles)
    assert np.isfinite(dijkstra_shortest_path(a)[0])

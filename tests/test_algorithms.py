import numpy as np
import pytest
from qaco_elite import GridWorld, QLearningConfig, ACOConfig, learn_q_values, q_action_prior, run_aco
from qaco_elite.aco import path_length


def test_training_deterministic_and_prior_valid():
    world = GridWorld(np.zeros((4, 4), dtype=bool))
    cfg = QLearningConfig(episodes=60, max_steps=60)
    a, ac = learn_q_values(world, cfg, seed=21)
    b, bc = learn_q_values(world, cfg, seed=21)
    assert np.array_equal(a, b)
    assert np.array_equal(ac, bc)
    prior = q_action_prior(world, a)
    assert np.all(prior[world.transitions >= 0] >= 0.05 - 1e-12)
    assert np.all(prior[world.transitions < 0] == 0)


@pytest.mark.parametrize("method", ["Plain-ACO", "Q-ACO", "Q-ACO-Elite"])
def test_algorithm_returns_legal_path(method):
    world = GridWorld(np.zeros((6, 6), dtype=bool))
    q, _ = learn_q_values(world, QLearningConfig(episodes=150, max_steps=120), seed=13)
    prior = q_action_prior(world, q)
    config = ACOConfig(ants=10, iterations=6, max_walk_steps=300)
    result = run_aco(world, config, method, prior if method != "Plain-ACO" else None, seed=14)
    assert result.best_path[0] == world.start_id
    assert result.best_path[-1] == world.goal_id
    for s, next_s in zip(result.best_path[:-1], result.best_path[1:]):
        assert next_s in [j for j, _, _ in world.neighbors[s]]
    assert result.best_length == pytest.approx(path_length(world, result.best_path))
    assert len(result.best_so_far) == config.iterations
    assert np.all(np.diff(result.best_so_far) <= 1e-9)
    assert result.convergence_iteration is not None


def test_missing_q_and_bad_method_raise():
    world = GridWorld(np.zeros((3, 3), dtype=bool))
    with pytest.raises(ValueError):
        run_aco(world, ACOConfig(ants=1, iterations=1), "Q-ACO")
    with pytest.raises(ValueError):
        run_aco(world, ACOConfig(ants=1, iterations=1), "No-ACO")

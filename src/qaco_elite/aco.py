"""Plain ACO, Q-guided ACO, and Q-guided ACO with elite reinforcement."""
from dataclasses import dataclass
from math import atan2, pi
import numpy as np
from .grid import GridWorld


@dataclass(frozen=True)
class ACOConfig:
    ants: int = 50
    iterations: int = 100
    pheromone_alpha: float = 1.0
    distance_beta: float = 5.0
    q_exponent: float = 2.0
    evaporation: float = 0.2
    elite_weight: float = 5.0
    initial_pheromone: float = 1.0  # Paper unspecified.
    deposit_strength: float = 1.0  # Paper I, unspecified numerical value.
    max_walk_steps: int = 2300  # Backtrack cap: paper does not specify deadlock handling.


@dataclass
class ACOResult:
    algorithm: str
    best_path: list[int]
    best_length: float
    turn_cost: float
    convergence_iteration: int | None
    best_so_far: list[float]
    iteration_best: list[float]
    successful_ants: list[int]


def path_length(world: GridWorld, path: list[int]):
    if not path:
        return float("inf")
    return sum(float(np.hypot(*np.subtract(world.to_xy(b), world.to_xy(a))))
               for a, b in zip(path[:-1], path[1:]))


def turn_cost(world: GridWorld, path: list[int]):
    """45 degree turns cost 1, 90 degree turns 2, etc. (angle / 45 degrees)."""
    if len(path) < 3:
        return 0.0
    angles = []
    for a, b in zip(path[:-1], path[1:]):
        r1, c1 = world.to_xy(a)
        r2, c2 = world.to_xy(b)
        angles.append(atan2(r2 - r1, c2 - c1))
    return float(sum(round(abs((b - a + pi) % (2 * pi) - pi) / (pi / 4))
                     for a, b in zip(angles[:-1], angles[1:])))


def _construct_ant(world, edge_weights, rng, step_cap):
    """Randomized tabu walk with explicit stack backtracking on dead ends.

    The paper does not specify its ant deadlock handling; rejected edges are
    remembered only for the current ant's current DFS branch.
    """
    path = [world.start_id]
    visited = {world.start_id}
    tried = {}  # At each node, choices exhausted in the current DFS tree.
    transitions = 0
    while path and transitions < step_cap:
        s = path[-1]
        if s == world.goal_id:
            return list(path)
        cand = [(nxt, act) for nxt, act, _ in world.neighbors[s]
                if nxt not in visited and nxt not in tried.get(s, set())]
        if not cand:
            dead = path.pop()
            visited.remove(dead)
            if path:
                tried.setdefault(path[-1], set()).add(dead)
            continue
        weights = np.array([edge_weights[s, a] for _, a in cand], dtype=float)
        if not np.isfinite(weights).all() or weights.sum() <= 0:
            weights = np.ones(len(cand), dtype=float)
        idx = int(rng.choice(len(cand), p=weights / weights.sum()))
        nxt = cand[idx][0]
        visited.add(nxt)
        path.append(nxt)
        transitions += 1
    return []


def run_aco(world: GridWorld, config: ACOConfig, algorithm: str,
            prior: np.ndarray | None = None, seed: int = 0):
    """Run one of three paper methods with identical world and ACO settings."""
    if algorithm not in ("Plain-ACO", "Q-ACO", "Q-ACO-Elite"):
        raise ValueError("invalid algorithm")
    use_q = algorithm != "Plain-ACO"
    if use_q and (prior is None or prior.shape != world.transitions.shape):
        raise ValueError("Q-ACO requires positive action prior")
    rng = np.random.default_rng(seed)
    tau = np.full((world.n_states, 8), config.initial_pheromone, dtype=float)
    # The paper's Eq (4) is eta_ij=1/d_ij (one-step, not goal-distance).
    eta = np.zeros_like(tau)
    for s, edges in enumerate(world.neighbors):
        for _, a, step in edges:
            eta[s, a] = step ** (-config.distance_beta)
    q_factor = np.power(prior, config.q_exponent) if use_q else 1.0
    best_path = []
    best_length = float("inf")
    history, per_iteration, successes = [], [], []
    first_found_at = None
    for iteration in range(1, config.iterations + 1):
        edge_weights = (tau ** config.pheromone_alpha) * eta * q_factor
        paths = []
        lengths = []
        for _ in range(config.ants):
            path = _construct_ant(world, edge_weights, rng, config.max_walk_steps)
            if path:
                length = path_length(world, path)
                paths.append(path)
                lengths.append(length)
                if length < best_length - 1e-10:
                    best_path, best_length = path, length
                    first_found_at = iteration
        successes.append(len(paths))
        per_iteration.append(float(min(lengths)) if lengths else float("nan"))
        history.append(best_length)
        tau *= 1.0 - config.evaporation
        tau = np.maximum(tau, 1e-12)
        for path, length in zip(paths, lengths):
            delta = config.deposit_strength / length
            for s, nxt in zip(path[:-1], path[1:]):
                a = next(a for j, a, _ in world.neighbors[s] if j == nxt)
                tau[s, a] += delta
        if algorithm == "Q-ACO-Elite" and best_path:
            delta = config.elite_weight * config.deposit_strength / best_length
            for s, nxt in zip(best_path[:-1], best_path[1:]):
                a = next(a for j, a, _ in world.neighbors[s] if j == nxt)
                tau[s, a] += delta
    return ACOResult(algorithm, best_path, best_length, turn_cost(world, best_path),
                     first_found_at, history, per_iteration, successes)

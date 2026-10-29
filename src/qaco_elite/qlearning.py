"""Heuristically enhanced Q learning with annealed Boltzmann exploration."""
from dataclasses import dataclass
import numpy as np
from .grid import GridWorld


@dataclass(frozen=True)
class QLearningConfig:
    episodes: int = 3500  # Unspecified in paper: documented experimental assumption.
    max_steps: int = 450  # Unspecified in paper.
    learning_rate: float = 0.2
    gamma: float = 0.9
    heuristic_weight: float = 0.5
    temperature_initial: float = 5.0
    temperature_min: float = 0.01
    temperature_decay: float = 0.01


def learn_q_values(world: GridWorld, config: QLearningConfig, seed: int = 0):
    """Train tabular Q(s,a). Invalid actions remain selectable, with -10 penalty.

    Uses Eq (9)-(14); h is evaluated at the *successor* state, in accordance
    with the prose description for Eq (12). Terminal bootstrap is zero.
    """
    rng = np.random.default_rng(seed)
    q = np.zeros((world.n_states, 8), dtype=float)
    distances = world.euclidean_to_goal()
    hmax, hmin = float(np.max(distances)), float(np.min(distances))
    heuristic = (hmax - distances) / max(1e-12, hmax - hmin)
    alpha, gamma, lam = config.learning_rate, config.gamma, config.heuristic_weight
    completions = np.zeros(config.episodes, dtype=bool)
    for ep in range(config.episodes):
        s = world.start_id
        temp = max(config.temperature_min, config.temperature_initial * np.exp(-config.temperature_decay * ep))
        for _ in range(config.max_steps):
            logits = q[s] / temp
            weights = np.exp(logits - np.max(logits))
            weights /= weights.sum()
            a = int(rng.choice(8, p=weights))
            nxt = int(world.transitions[s, a])
            invalid = nxt < 0
            if invalid:
                nxt = s
            done = nxt == world.goal_id
            reward = -10.0 if invalid else (10.0 if done else -1.0)
            future = 0.0 if done else float(np.max(q[nxt]))
            # q(s,a) <- q + alpha * [r + gamma(max Q(s',.) + lambda*h_norm(s')) - q]
            target = reward + gamma * (future + lam * heuristic[nxt])
            q[s, a] += alpha * (target - q[s, a])
            s = nxt
            if done:
                completions[ep] = True
                break
    return q, completions


def q_action_prior(world: GridWorld, q: np.ndarray, *, floor: float = 0.05):
    """Positive, per-state min-max conversion of Q for Eq (15).

    Paper raises Q_ij to a positive exponent but does not explain how negative
    tabular Q values are made admissible. This documented conversion prevents
    squaring a large negative value into falsely strong pheromone guidance.
    """
    if not 0 < floor <= 1:
        raise ValueError("floor must be in (0,1]")
    prior = np.zeros_like(q, dtype=float)
    for s in range(world.n_states):
        allowed = world.transitions[s] >= 0
        if not np.any(allowed):
            continue
        vals = q[s, allowed]
        spread = float(vals.max() - vals.min())
        if spread < 1e-12:
            prior[s, allowed] = 1.0
        else:
            prior[s, allowed] = floor + (1 - floor) * (vals - vals.min()) / spread
    return prior

"""Eight-neighbour occupancy grids with no diagonal corner cutting."""
from dataclasses import dataclass, field
import heapq
from math import sqrt
import numpy as np

# Eight discrete actions; the Q table has exactly this order.
DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1),
              (-1, -1), (-1, 1), (1, -1), (1, 1))


@dataclass
class GridWorld:
    obstacles: np.ndarray
    start: tuple[int, int] = (0, 0)
    goal: tuple[int, int] | None = None
    neighbors: tuple = field(init=False, repr=False)
    transitions: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        self.obstacles = np.asarray(self.obstacles, dtype=bool)
        if self.obstacles.ndim != 2:
            raise ValueError("occupancy map must be a 2D array")
        self.h, self.w = self.obstacles.shape
        if self.goal is None:
            self.goal = (self.h - 1, self.w - 1)
        for point in (self.start, self.goal):
            if not (0 <= point[0] < self.h and 0 <= point[1] < self.w):
                raise ValueError("start/goal outside map")
            if self.obstacles[point]:
                raise ValueError("start/goal must be traversable")
        self.transitions = np.full((self.h * self.w, 8), -1, dtype=np.int32)
        adjacency = []
        for r in range(self.h):
            for c in range(self.w):
                edges = []
                if not self.obstacles[r, c]:
                    for a, (dr, dc) in enumerate(DIRECTIONS):
                        nr, nc = r + dr, c + dc
                        if not (0 <= nr < self.h and 0 <= nc < self.w):
                            continue
                        if self.obstacles[nr, nc]:
                            continue
                        if dr and dc and (self.obstacles[r + dr, c] or self.obstacles[r, c + dc]):
                            continue
                        nxt = nr * self.w + nc
                        self.transitions[r * self.w + c, a] = nxt
                        edges.append((nxt, a, sqrt(dr * dr + dc * dc)))
                adjacency.append(tuple(edges))
        self.neighbors = tuple(adjacency)

    @property
    def n_states(self):
        return self.h * self.w

    @property
    def start_id(self):
        return self.start[0] * self.w + self.start[1]

    @property
    def goal_id(self):
        return self.goal[0] * self.w + self.goal[1]

    def to_xy(self, state: int):
        return divmod(int(state), self.w)

    def euclidean_to_goal(self):
        rr, cc = np.indices((self.h, self.w))
        return np.hypot(rr - self.goal[0], cc - self.goal[1]).ravel()


def dijkstra_shortest_path(world: GridWorld):
    """Exact shortest path under Euclidean one-step lengths."""
    q = [(0.0, world.start_id)]
    d = {world.start_id: 0.0}
    prev = {}
    while q:
        cost, s = heapq.heappop(q)
        if cost > d[s] + 1e-12:
            continue
        if s == world.goal_id:
            path = [s]
            while path[-1] != world.start_id:
                path.append(prev[path[-1]])
            return cost, path[::-1]
        for nxt, _, step in world.neighbors[s]:
            new = cost + step
            if new < d.get(nxt, float("inf")) - 1e-12:
                d[nxt], prev[nxt] = new, s
                heapq.heappush(q, (new, nxt))
    return float("inf"), []


def make_random_world(size=20, density=0.2, seed=0, max_attempts=5000):
    """Sample independent Bernoulli obstacles until a valid connected map appears.

    Original paper does not publish its maps/seeds; this generator supplies
    explicitly reproducible *new* maps, not a reconstruction of its figures.
    """
    if not 0 <= density < 1:
        raise ValueError("density must be in [0,1)")
    rng = np.random.default_rng(seed)
    for attempt in range(1, max_attempts + 1):
        grid = rng.random((size, size)) < density
        grid[0, 0] = grid[-1, -1] = False
        world = GridWorld(grid)
        if np.isfinite(dijkstra_shortest_path(world)[0]):
            return world, attempt
    raise RuntimeError(f"no connected map after {max_attempts} samples")

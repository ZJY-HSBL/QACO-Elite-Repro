# v1.0.0 — Algorithm reproduction baseline

- Paper-faithful components: 8-neighbour grid/no corner cutting, Boltzmann heuristic Q-learning, Q-guided ACO, global-best elite pheromone deposit.
- Comparison: Plain-ACO vs Q-ACO vs Q-ACO-Elite, 20×20 grid, densities 0.2/0.3/0.4, 50 ants × 100 ACO iterations.
- Deterministic seeded simulation, independent Dijkstra optimum, three path-quality metrics, CSV/JSON evidence, matplotlib plots and SVG README figures.
- Documentation of all assumptions not specified in the publication; English/Chinese README, automated unit tests.
- Single-seed result: improved Q-guided search over Plain-ACO on all three new maps; no final path-length improvement due to elite deposit in this trial. **No claim of exact numerical reproduction or statistical superiority.**

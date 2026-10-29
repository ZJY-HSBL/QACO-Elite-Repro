"""Reproducible three-density benchmark with CSV/JSON and plotted evidence."""
import argparse
import csv
import json
import platform
import sys
from dataclasses import asdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from .grid import make_random_world, dijkstra_shortest_path
from .qlearning import QLearningConfig, learn_q_values, q_action_prior
from .aco import ACOConfig, run_aco

PAPER = {
    "0.2": {"Plain-ACO": (36.83, 72, 37), "Q-ACO": (30.97, 14, 15), "Q-ACO-Elite": (30.97, 11, 15)},
    "0.3": {"Plain-ACO": (37.41, 57, 38), "Q-ACO": (32.73, 12, 14), "Q-ACO-Elite": (32.73, 8, 14)},
    "0.4": {"Plain-ACO": (58.83, 100, 62), "Q-ACO": (44.83, 37, 32), "Q-ACO-Elite": (44.24, 19, 28)},
}
METHODS = ("Plain-ACO", "Q-ACO", "Q-ACO-Elite")
COLORS = {"Plain-ACO": "#e8793a", "Q-ACO": "#3178a5", "Q-ACO-Elite": "#27926d"}


def plot_paths(world, results, optimal, density, out):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.15), constrained_layout=True)
    values = [("Dijkstra optimal", optimal[1], "#7a59a3")] + [(m, results[m].best_path, COLORS[m]) for m in METHODS]
    for ax, (name, path, color) in zip(axes, values):
        ax.imshow(world.obstacles.astype(int), cmap="Greys", vmin=0, vmax=1, interpolation="nearest", origin="upper")
        ax.set_xticks(np.arange(-.5, world.w, 1), minor=True)
        ax.set_yticks(np.arange(-.5, world.h, 1), minor=True)
        ax.grid(which="minor", color="#b9b9b9", linewidth=.35)
        if path:
            pts = np.array([world.to_xy(s) for s in path])
            ax.plot(pts[:, 1], pts[:, 0], color=color, linewidth=2.1, zorder=3)
        ax.scatter([world.start[1]], [world.start[0]], c="#20b55c", s=72, marker="o", edgecolors="white", zorder=5)
        ax.scatter([world.goal[1]], [world.goal[0]], c="#d63c43", s=80, marker="*", edgecolors="white", zorder=5)
        ax.set_title(name, fontsize=11, weight="bold")
        ax.set_xticks([0, 5, 10, 15, 19]); ax.set_yticks([0, 5, 10, 15, 19])
        ax.set_aspect("equal")
    fig.suptitle(f"AGV paths | 20x20 | target obstacle density = {density:.1f}", fontsize=13, weight="bold")
    fig.savefig(out, dpi=175, bbox_inches="tight")
    plt.close(fig)


def plot_convergence(results, density, out):
    fig, ax = plt.subplots(figsize=(8.5, 4.5), constrained_layout=True)
    for name in METHODS:
        r = results[name]
        y = np.array(r.best_so_far, dtype=float)
        y[~np.isfinite(y)] = np.nan
        ax.plot(range(1, len(y) + 1), y, label=name, color=COLORS[name], linewidth=2)
    ax.set(xlabel="ACO iteration", ylabel="Best-so-far path length (grid units)",
           title=f"Convergence | obstacle density {density:.1f}")
    ax.grid(alpha=.25)
    ax.legend()
    fig.savefig(out, dpi=175, bbox_inches="tight")
    plt.close(fig)


def plot_comparison(rows, out):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), constrained_layout=True)
    metrics = [("best_length", "Path length (lower is better)"),
               ("convergence_iteration", "First iteration attaining final best"),
               ("turn_cost", "Total turning cost (lower is better)")]
    for ax, (field, label) in zip(axes, metrics):
        x = np.arange(3)
        for j, method in enumerate(METHODS):
            values = [next(r[field] for r in rows if r['density']==d and r['algorithm']==method)
                      for d in [0.2, 0.3, 0.4]]
            ax.bar(x + (j - 1) * .24, values, width=.22, color=COLORS[method], label=method)
        ax.set_xticks(x, ['0.2', '0.3', '0.4']); ax.set_xlabel('Obstacle probability')
        ax.set_title(label, fontsize=10, weight='bold')
        ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Independent reproduction experiment (NOT original paper data)", fontsize=12, weight='bold')
    fig.savefig(out, dpi=175, bbox_inches="tight")
    plt.close(fig)


def benchmark(output: Path, seed: int, qconfig: QLearningConfig, aconfig: ACOConfig):
    output.mkdir(parents=True, exist_ok=True)
    rows, artifacts = [], {}
    config = {"seed": seed, "size": 20, "q_learning": asdict(qconfig), "aco": asdict(aconfig),
              "software": {"python": platform.python_version(), "numpy": np.__version__, "matplotlib": matplotlib.__version__},
              "maps": {}, "note": "New seeded maps, not the original unpublished maps; all comparisons within each map are paired."}
    for index, density in enumerate([.2, .3, .4]):
        print(f"[density={density:.1f}] generating connected 20x20 map...", flush=True)
        world, attempts = make_random_world(size=20, density=density, seed=seed + 100 * index)
        optimal = dijkstra_shortest_path(world)
        key = f"{density:.1f}"
        config['maps'][key] = {"seed": seed + 100 * index, "samples_until_connected": attempts,
                               "actual_obstacle_fraction": float(world.obstacles.mean()),
                               "dijkstra_optimum": optimal[0]}
        np.save(output / f"map_{key}.npy", world.obstacles.astype(np.uint8))
        print(f"[density={density:.1f}] Dijkstra = {optimal[0]:.3f}, Q-learning training...", flush=True)
        q, completed = learn_q_values(world, qconfig, seed=seed + 100 * index + 1)
        np.save(output / f"q_values_{key}.npy", q)
        config['maps'][key]['q_episodes_completed'] = int(completed.sum())
        prior = q_action_prior(world, q)
        outputs = {}
        for method in METHODS:
            print(f"[density={density:.1f}] {method}...", flush=True)
            r = run_aco(world, aconfig, method, prior if method != 'Plain-ACO' else None,
                        seed=seed + 100 * index + 2)
            outputs[method] = r
            rec = {'density': density, 'algorithm': method,
                   'best_length': round(r.best_length, 8),
                   'convergence_iteration': r.convergence_iteration,
                   'turn_cost': r.turn_cost, 'success_count': int(sum(r.successful_ants)),
                   'ants_total': aconfig.ants * aconfig.iterations,
                   'optimality_gap_pct': round(100 * (r.best_length / optimal[0] - 1), 4),
                   'paper_length': PAPER[key][method][0],
                   'paper_iteration': PAPER[key][method][1],
                   'paper_turn_cost': PAPER[key][method][2]}
            rows.append(rec)
            print(f"  length={r.best_length:.2f}; first-best={r.convergence_iteration}; turn={r.turn_cost:g}; gap={rec['optimality_gap_pct']:.2f}%", flush=True)
        plot_paths(world, outputs, optimal, density, output / f"paths_{key}.png")
        plot_convergence(outputs, density, output / f"convergence_{key}.png")
        artifacts[key] = {m: {'path': r.best_path, 'best_so_far': [v if np.isfinite(v) else None for v in r.best_so_far],
                               'iteration_best': [v if np.isfinite(v) else None for v in r.iteration_best],
                               'successful_ants_per_iteration': r.successful_ants} for m, r in outputs.items()}
    plot_comparison(rows, output / 'benchmark_comparison.png')
    with (output / 'results.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    (output / 'run_config.json').write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding='utf-8')
    (output / 'trajectories.json').write_text(json.dumps(artifacts, ensure_ascii=False), encoding='utf-8')
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description="Q-ACO-Elite paper-style benchmark")
    parser.add_argument('--output', type=Path, default=Path('results'))
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--q-episodes', type=int, default=1200,
                        help="Unspecified by article; selected as disclosed reproduction parameter")
    parser.add_argument('--ants', type=int, default=50)
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--walk-cap', type=int, default=2300)
    args = parser.parse_args(argv)
    rows = benchmark(args.output, args.seed, QLearningConfig(episodes=args.q_episodes),
                     ACOConfig(ants=args.ants, iterations=args.iterations, max_walk_steps=args.walk_cap))
    print(f"\nWrote {len(rows)} records, plots, maps and Q tables to {args.output}", flush=True)


if __name__ == '__main__':
    main()

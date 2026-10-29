"""Independent, paired multi-map replication and component ablations.

This is an additional experiment: the article did not release maps, seeds,
training budgets, positive-Q transformation or deadlock policy. Do not report
these independently sampled results as the original article's measurements.
"""
import argparse
import csv
import hashlib
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .aco import ACOConfig, run_aco
from .experiment import METHODS, PAPER, plot_paths
from .grid import dijkstra_shortest_path, make_random_world
from .qlearning import QLearningConfig, learn_q_values, q_action_prior

ABLATIONS = ('Elite-NoHeuristic', 'Elite-NoQ')
ALL_METHODS = (*METHODS, *ABLATIONS)
PAIRS = (
    ('Plain-ACO', 'Q-ACO'),
    ('Q-ACO', 'Q-ACO-Elite'),
    ('Elite-NoHeuristic', 'Q-ACO-Elite'),
    ('Elite-NoQ', 'Q-ACO-Elite'),
)


def paired_statistics(before, after, *, seed=0, resamples=5000):
    """Paired improvement before-after; positive means the latter is smaller.

    Percentile bootstrap CI for the mean, plus two-sided random sign-flip
    p-value. The latter is exploratory under the exchangeable paired null.
    """
    before, after = np.asarray(before, dtype=float), np.asarray(after, dtype=float)
    if before.shape != after.shape or before.ndim != 1 or not len(before):
        raise ValueError('nonempty same-length 1D paired observations required')
    if not np.isfinite(before).all() or not np.isfinite(after).all():
        raise ValueError('paired observations must be finite')
    if resamples < 100:
        raise ValueError('at least 100 resamples required')
    diff = before - after
    rng = np.random.default_rng(seed)
    sample_index = rng.integers(len(diff), size=(resamples, len(diff)))
    boots = diff[sample_index].mean(axis=1)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(resamples, len(diff)))
    null = (signs * diff).mean(axis=1)
    p = (1 + np.count_nonzero(np.abs(null) >= abs(diff.mean()) - 1e-12)) / (resamples + 1)
    return dict(n=len(diff), improvement_mean=float(diff.mean()),
                improvement_sd=float(diff.std(ddof=1)) if len(diff) > 1 else 0.0,
                ci_low=float(np.quantile(boots, .025)),
                ci_high=float(np.quantile(boots, .975)), p_raw=float(p))


def holm_adjust(pvalues):
    """Holm family-wise adjusted p values, in original comparison order."""
    values = np.asarray(pvalues, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=float)
    lower_bound = 0.0
    for rank, ix in enumerate(order):
        lower_bound = max(lower_bound, (len(values) - rank) * values[ix])
        adjusted[ix] = min(1.0, lower_bound)
    return adjusted.tolist()


def _save_csv(path, rows):
    if not rows:
        raise ValueError('cannot write empty records')
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _render(trajectories, rows, effects, output):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), constrained_layout=True)
    for ax, density in zip(axes, (.2, .3, .4)):
        for name in METHODS:
            curves = np.asarray([x['history'] for x in trajectories if x['density'] == density
                                 and x['method'] == name], dtype=float)
            curves[~np.isfinite(curves)] = np.nan
            ax.plot(np.arange(1, curves.shape[1] + 1), np.nanmean(curves, axis=0), label=name)
        ax.set(title=f'Obstacle probability {density:.1f}', xlabel='ACO iteration',
               ylabel='Mean best-so-far path length')
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle('Independent paired random-map validation | NOT original paper curves')
    fig.savefig(output / 'mean_convergence.png', dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), constrained_layout=True)
    for ax, density in zip(axes, (.2, .3, .4)):
        group = [x for x in effects if x['density'] == density and x['metric'] == 'best_length']
        y = np.arange(len(group))
        mean = np.asarray([x['improvement_mean'] for x in group])
        lo = np.asarray([x['ci_low'] for x in group])
        hi = np.asarray([x['ci_high'] for x in group])
        ax.errorbar(mean, y, xerr=np.vstack((mean - lo, hi - mean)), fmt='o', capsize=3)
        ax.axvline(0, ls='--', lw=1, c='gray')
        ax.set_yticks(y, [f"{x['before']} -> {x['after']}" for x in group], fontsize=8)
        ax.set(xlabel='Mean paired path length reduction (95% bootstrap CI)',
               title=f'Density {density:.1f}')
        ax.grid(axis='x', alpha=.2)
    fig.suptitle('Paired differences on the SAME maps and random seeds')
    fig.savefig(output / 'paired_effects.png', dpi=170)
    plt.close(fig)


def validate(output, *, seeds=6, base_seed=20261030, episodes=600,
             ants=50, iterations=100, walk_cap=1800, resamples=5000):
    if seeds < 2:
        raise ValueError('at least two paired seeds needed for uncertainty estimation')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    qcfg = QLearningConfig(episodes=episodes)
    acfg = ACOConfig(ants=ants, iterations=iterations, max_walk_steps=walk_cap)
    rows, trajectories, records = [], [], []
    for di, density in enumerate((.2, .3, .4)):
        for run in range(seeds):
            map_seed = base_seed + di * 100000 + run
            q_seed, aco_seed = map_seed + 30000, map_seed + 60000
            world, attempts = make_random_world(size=20, density=density, seed=map_seed)
            optimum, optimal_path = dijkstra_shortest_path(world)
            start = time.perf_counter()
            q, completions = learn_q_values(world, qcfg, seed=q_seed)
            qtime = time.perf_counter() - start
            start = time.perf_counter()
            q_noheur, noheur_completions = learn_q_values(
                world, replace(qcfg, heuristic_weight=0.0), seed=q_seed)
            noheur_time = time.perf_counter() - start
            prior, prior_noheur = q_action_prior(world, q), q_action_prior(world, q_noheur)
            configurations = (
                ('Plain-ACO', 'Plain-ACO', None, acfg, 0.0),
                ('Q-ACO', 'Q-ACO', prior, acfg, qtime),
                ('Q-ACO-Elite', 'Q-ACO-Elite', prior, acfg, qtime),
                ('Elite-NoHeuristic', 'Q-ACO-Elite', prior_noheur, acfg, noheur_time),
                ('Elite-NoQ', 'Q-ACO-Elite', np.ones_like(prior), acfg, 0.0),
            )
            results = {}
            for label, algorithm, information, settings, train_seconds in configurations:
                start = time.perf_counter()
                result = run_aco(world, settings, algorithm, prior=information, seed=aco_seed)
                elapsed = time.perf_counter() - start
                if not result.best_path or result.best_length + 1e-8 < optimum:
                    raise AssertionError(f'invalid path or exceeds lower bound: {label} {map_seed}')
                results[label] = result
                rows.append(dict(density=density, run=run, map_seed=map_seed,
                                 q_seed=q_seed, aco_seed=aco_seed, method=label,
                                 obstacle_fraction=float(world.obstacles.mean()),
                                 rejected_maps=attempts - 1,
                                 map_sha256=hashlib.sha256(world.obstacles.tobytes()).hexdigest(),
                                 exact_shortest_length=round(optimum, 9),
                                 best_length=round(result.best_length, 9),
                                 optimality_gap_pct=round(100 * (result.best_length / optimum - 1), 7),
                                 turn_cost=result.turn_cost,
                                 convergence_iteration=result.convergence_iteration,
                                 successful_ants=sum(result.successful_ants),
                                 attempted_ants=ants * iterations,
                                 q_training_seconds=round(train_seconds, 5),
                                 aco_seconds=round(elapsed, 5),
                                 end_to_end_seconds=round(train_seconds + elapsed, 5)))
                trajectories.append(dict(density=density, run=run, method=label,
                                         history=result.best_so_far))
            if run == 0:
                plot_paths(world, {m: results[m] for m in METHODS},
                           (optimum, optimal_path), density, output / f'example_paths_{density:.1f}.png')
            records.append(dict(density=density, run=run, map_seed=map_seed,
                                q_completed=int(completions.sum()),
                                noheur_q_completed=int(noheur_completions.sum()),
                                q_training_seconds=qtime,
                                q_noheur_training_seconds=noheur_time))
            print(f"density={density:.1f} seed={run+1}/{seeds}: " +
                  ' / '.join(f'{k}={results[k].best_length:.2f}' for k in METHODS), flush=True)
    summary, effects = [], []
    for density in (.2, .3, .4):
        group = [r for r in rows if r['density'] == density]
        for method in ALL_METHODS:
            part = [r for r in group if r['method'] == method]
            summary.append(dict(density=density, method=method, n=len(part),
                                length_mean=float(np.mean([r['best_length'] for r in part])),
                                length_sd=float(np.std([r['best_length'] for r in part], ddof=1)),
                                gap_mean_pct=float(np.mean([r['optimality_gap_pct'] for r in part])),
                                turn_mean=float(np.mean([r['turn_cost'] for r in part])),
                                convergence_mean=float(np.mean([r['convergence_iteration'] for r in part])),
                                success_rate=float(np.mean([r['successful_ants']/r['attempted_ants'] for r in part])),
                                end_to_end_seconds_mean=float(np.mean([r['end_to_end_seconds'] for r in part]))))
        length_family = []
        for before, after in PAIRS:
            for metric in ('best_length', 'turn_cost', 'convergence_iteration'):
                p1 = sorted((r for r in group if r['method'] == before), key=lambda r: r['run'])
                p2 = sorted((r for r in group if r['method'] == after), key=lambda r: r['run'])
                if [r['map_seed'] for r in p1] != [r['map_seed'] for r in p2]:
                    raise AssertionError('unpaired experiment')
                stat = paired_statistics([r[metric] for r in p1], [r[metric] for r in p2],
                                         seed=base_seed + int(density * 100) * 100 + len(effects),
                                         resamples=resamples)
                effects.append(dict(density=density, before=before, after=after,
                                    metric=metric, **stat, p_holm_length_family=''))
                if metric == 'best_length':
                    length_family.append(len(effects) - 1)
        adjusted = holm_adjust([effects[idx]['p_raw'] for idx in length_family])
        for idx, val in zip(length_family, adjusted):
            effects[idx]['p_holm_length_family'] = val
    _save_csv(output / 'individual_runs.csv', rows)
    _save_csv(output / 'summary.csv', summary)
    _save_csv(output / 'paired_effects.csv', effects)
    (output / 'trajectories.json').write_text(json.dumps(trajectories), encoding='utf-8')
    (output / 'q_training.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
    metadata = dict(protocol='independent paired random maps, not the authors original maps',
                    base_seed=base_seed, map_count=seeds * 3, seeds_per_density=seeds,
                    densities=[.2, .3, .4], q_learning=asdict(qcfg), aco=asdict(acfg),
                    bootstraps_and_signflips=resamples, paper_reference=PAPER,
                    implementations={'Elite-NoHeuristic':'Q-ACO-Elite with lambda=0',
                                     'Elite-NoQ':'Elite ACO with uniform transition prior'},
                    observations='Path length is primary. Paired 95% bootstrap intervals; Holm corrected sign-flip tests over four path-length comparisons per density. Exploratory, conditional on map generator and chosen hyperparameters.')
    (output / 'manifest.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    _render(trajectories, rows, effects, output)
    return rows, summary, effects


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('results/validation'))
    parser.add_argument('--seeds', type=int, default=6)
    parser.add_argument('--base-seed', type=int, default=20261030)
    parser.add_argument('--q-episodes', type=int, default=600)
    parser.add_argument('--ants', type=int, default=50)
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--walk-cap', type=int, default=1800)
    parser.add_argument('--resamples', type=int, default=5000)
    args = parser.parse_args(argv)
    validate(args.output, seeds=args.seeds, base_seed=args.base_seed,
             episodes=args.q_episodes, ants=args.ants, iterations=args.iterations,
             walk_cap=args.walk_cap, resamples=args.resamples)
    print(f'Validation artifacts saved to {args.output}', flush=True)


if __name__ == '__main__':
    main()

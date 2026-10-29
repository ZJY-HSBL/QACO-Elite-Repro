"""Statistical and artifact validation for reproducibility study."""
import csv
import json
import numpy as np
import pytest
from qaco_elite.validation import paired_statistics, holm_adjust, validate


def test_paired_statistics_repeatable_and_direction():
    a = [9, 8, 7, 10, 11]
    b = [6, 6, 6, 7, 7]
    x = paired_statistics(a, b, seed=17, resamples=1000)
    y = paired_statistics(a, b, seed=17, resamples=1000)
    assert x == y
    assert x['improvement_mean'] == pytest.approx(2.6)
    assert x['ci_low'] > 0
    assert 0 < x['p_raw'] <= 1


def test_holm_order_and_bounds():
    assert holm_adjust([.04, .001, .02]) == pytest.approx([.04, .003, .04])
    assert holm_adjust([.7, .9, 1.]) == pytest.approx([1., 1., 1.])


def test_bad_pair_input():
    with pytest.raises(ValueError):
        paired_statistics([1, 2], [1])
    with pytest.raises(ValueError):
        paired_statistics([1, np.inf], [1, 2])
    with pytest.raises(ValueError):
        paired_statistics([1, 2], [1, 2], resamples=5)


def test_smoke_validation_artifacts(tmp_path):
    rows, summary, effects = validate(tmp_path, seeds=2, episodes=12,
                                      ants=2, iterations=2, walk_cap=900, resamples=100)
    assert len(rows) == 3 * 2 * 5
    assert len(summary) == 3 * 5
    assert len(effects) == 3 * 4 * 3
    for name in ('individual_runs.csv', 'summary.csv', 'paired_effects.csv',
                 'mean_convergence.png', 'paired_effects.png', 'manifest.json'):
        assert (tmp_path / name).stat().st_size > 0
    data = json.loads((tmp_path / 'manifest.json').read_text())
    assert data['map_count'] == 6
    with (tmp_path / 'individual_runs.csv').open() as f:
        records = list(csv.DictReader(f))
    assert len({(r['density'], r['map_seed']) for r in records}) == 6
    assert all(float(r['best_length']) >= float(r['exact_shortest_length']) - 1e-7
               for r in records)

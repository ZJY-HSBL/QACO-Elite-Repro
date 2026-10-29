# v1.1.0 — Paired validation and component ablations

基于 2026 年刘炀、吴艺茹、张恒《融合改进 Q 学习和蚁群算法的 AGV 路径规划》开展进一步独立验证。

- Preserve the equation-level Plain-ACO / Q-ACO / Q-ACO-Elite implementations and published 20×20, three-density comparison.
- Add independent **paired random-map** evaluation with all seeds, effective obstacle fractions, optimality gaps, turning costs, convergence iterations, ant success rates, offline learning and online optimization times.
- Add two ablations: **Elite-NoHeuristic** (Q-learning λ=0) and **Elite-NoQ** (elite-only with uniform prior).
- Add reproducible paired bootstrap 95% confidence intervals, exploratory sign-flip tests and Holm correction for four path-length comparisons per density.
- Add source-level automated checks plus raw CSV/JSON and PNG evidence; GitHub Actions runs all tests and regenerates the published evidence archive.
- Paper's original random maps, seeds and several method details are **not available**. Results are independent computational replications, not bit-for-bit reproduction or the original author's datasets.

Run: `python -m qaco_elite.validation --output results/validation --seeds 6 --q-episodes 450 --ants 20 --iterations 40 --walk-cap 750 --resamples 10000`

See `docs/reproduction_notes.md` and `docs/validation_protocol.md` before interpreting performance claims.

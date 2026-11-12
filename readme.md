# forex-research

Systematic test of FX and multi-asset trading ideas with pre-registered rules, mirror benchmarks and block bootstraps.

Start with summary.md: goals, decisions, every result, the survey, and the open options.

- src/, scripts/, tests/: Python 3, pandas, numpy, pytest.
- results/: outputs of every task (two large CSVs omitted; regenerate with the scripts).
- spec.md: the original backtest spec.
- data/readme.md: how to rebuild the data folders (raw data is not in the repo).

Run tests: python -m pytest -q

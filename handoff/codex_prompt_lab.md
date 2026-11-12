# Codex prompt: build the idea lab

You are working in the forex-research repo root. Python: ~/fxenv/bin/python.

Read lab_spec.md completely before writing code. It defines the data split, the building
blocks, the fills, the scorecard, the file format and the guarantees. If this prompt and the
specification disagree, the specification wins. If anything is ambiguous, STOP and ask; do not
choose.

## Hard rules

- Do not modify existing files in src/, scripts/, tests/ or results/. Add new files only.
  Allowed edit: .gitignore (add cache/ if missing and results/lab/*/trades.csv).
- FX data only through src.loader.load_continuous_pair. Never pass final=True. Never parse a
  date on or after 2025-11-01.
- The lab must not compute anything from bars closing on or after 2020-01-01. Drop them right
  after loading, before any feature is built. Do not run the confirm command on real data in
  this task; test it on synthetic data only.
- The existing test suite runs unmodified, including its own reads of download/ and fresh/.
- ASCII only. Lowercase file names. Simple, readable code: numpy, pandas, tomllib, pytest.
- No tuning and no extra features. Build exactly the blocks in the specification.

## Code (new files)

- src/lab_data.py: load the 15-minute bars for a period, build the 1h, 4h and 1d bars, the
  volatility s and e(H), with an on-disk cache under cache/ keyed by the source files' size and
  modification time.
- src/lab_blocks.py: levels, entries, biases, lights, stops, targets and manage rules, each a
  small function registered by its type name. Strict validation of the TOML file.
- src/lab_sim.py: the trade simulator (reusing src.costs and src.instruments), the mirror, the
  random baseline.
- src/lab_score.py: the scorecard and the ledger (reuse the weekly-block inference of
  src.signal_eval where it fits).
- scripts/lab.py: the run, list and confirm commands.
- ideas/: the idea files supplied with this task (copy them unchanged).

Speed: one idea should run in a few minutes on all 10 pairs. Compute levels and candidate
entries with arrays; a plain loop per trade path is fine. Print elapsed time.

## Tests (tests/test_lab.py, synthetic data only, fast)

1. No lookahead for every block type (truncation test), including the confirmation delay of
   swings and the visibility of higher time frame bars.
2. Period guards: the lab loader returns nothing from 2020-01-01 on; the confirm loader nothing
   before it or after 2025-10-31; confirm refuses without a pre-registration that contains the
   idea's sha256 and refuses a second run of the same sha256.
3. Each level, entry, bias, light, stop, target and manage rule on a small hand example.
4. Simulator against src.signal_eval.simulate_trade: with manage = breakeven at 1.0 the fills,
   exit reason and R are identical on synthetic paths (long and short, stop, target, time exit).
5. Stops never loosen; a moved stop acts from the next bar; the stop wins a bar that touches both.
6. Mirror: reversed side, same distances and rules. d and coin as defined.
7. Random baseline: same seed gives identical trades; counts match per pair and year.
8. File validation: unknown keys, unknown types and missing required keys raise an error.
9. Ledger: one appended line per run; an identical rerun gives an identical scorecard.

Run the whole suite: python -m pytest -q. All existing tests must still pass, unmodified.

## First runs (after the tests pass)

Run every file in ideas/ once, in name order, with the run command. Do not change any idea
file. Do not interpret the results.

## Final reply (in this order)

1. Files created or changed.
2. The pytest summary line.
3. The output of "python scripts/lab.py list".
4. For each idea, the full report.txt.
5. Anything in the specification you had to interpret, or "none".

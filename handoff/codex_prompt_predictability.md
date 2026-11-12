# Codex prompt, task 15: is FX direction predictable by a fitted model?

You are working in the forex-research repo root. Python: /opt/miniconda3/envs/pix2pixpolar-dev/bin/python.

Read prereg_predictability.md completely before writing code. It is the frozen specification:
every input, model, boundary, measure and verdict is defined there. This prompt only says how to
build and report it. If this prompt and the pre-registration disagree, the pre-registration
wins. If anything is ambiguous, STOP and ask; do not choose.

## Hard rules

- Do not change any rule, parameter or threshold. Do not add, drop or tune configurations.
- Do not modify existing files in src/, scripts/, tests/ or results/. Add new files only.
  Allowed edits to existing files: .gitignore (add cache/) and data/readme.md (append the
  download command below). Do not edit summary.md, spec.md or any prereg file.
- Do not modify scripts/sim_predict_power.py (already in the repo; not part of this task).
- FX data only through src.loader.load_continuous_pair. Never pass final=True. Do not parse any
  date on or after 2025-11-01 in any file.
- The existing test suite runs unmodified, including its own reads of download/ and fresh/.
- ASCII only in code, comments, output and file contents. Lowercase file names.
- Simple, readable code. numpy, pandas, scikit-learn, pytest; matplotlib only for the chart.
- Run the evaluation once. If you rerun after seeing results, give the reason under Deviations.
- Report what the files say. No interpretation beyond the pre-registered verdicts.

## Step A: environment and data

1. Check that scikit-learn imports in the Python above and print its version. If it is missing,
   STOP and report. Do not install anything.
2. Download three FRED series (lowercase file names):

       for s in VIXCLS DCOILBRENTEU DGS2; do curl -sL -o "data/fred/$(echo $s | tr A-Z a-z).csv" "https://fred.stlouisfed.org/graph/fredgraph.csv?id=$s"; done

   If a download fails, print the command and STOP. Append the command to data/readme.md under
   a heading "FRED daily series, task 15". Blank and "." values are missing observations.

## Step B: code (new files)

src/predict_data.py
- Readers for the eight ETF files (reuse src.multi_asset_data._read_etf without modifying it)
  and the three FRED files. Rows dated 2025-11-01 or later are skipped before any numeric
  parsing. Print one audit line per file (rows, first, last, missing).
- FX: load_continuous_pair(pair, "1h") for the ten pairs in src.loader.fresh_pairs.
- external_inputs(...): the 22 G3 inputs by observation date, and the mapping to decision times
  (latest observation dated strictly before the UTC date of t).

src/predict.py
- inputs_g1, inputs_g2: the own-history and dollar-factor inputs.
- targets(...): z for h = 1, 4, 24, the eligibility mask and the decision grids.
- fit_ridge / forecast_ridge (closed form with numpy), fit_trees / forecast_trees
  (HistGradientBoostingRegressor, pair as categorical input).
- select_once(...): alpha per pair and horizon, capacity per horizon, on the initial window.
- walk_forward(...): the twelve yearly refits with the purged boundary t + h <= Y-01-01.
- score(...): correlations, trades under rules ALL and SEL, gross, cost and net in units of the
  expected move, weekly sums.
- inference(...): weekly t, 5000 week resamples (seed 0), halves.
- verdict(...): the table of section 8, as a pure function of numbers.
- placebo_signs(...): one random sign per calendar week, as in gate L2.

scripts/run_predict.py
- Order: environment check; data audit; build inputs and targets once (they do not depend on
  the model); selection; primary walk-forward (6 cells); leak gate L2; report-only R1 to R4;
  save; print. Print a progress line with elapsed time after every refit.
- Build the input table once and keep it in memory (float32 is fine). An optional cache under
  cache/ is allowed; it must be rebuilt if any source file changes.
- Expected run time is 10 to 90 minutes. Do not shrink the specification to save time. If it
  cannot finish, STOP and report where the time goes.
- Print the full report to stdout and write the same text to results/predict_report.txt.

## Step C: tests (tests/test_predict.py, synthetic data only, fast)

1. No lookahead: truncating every source after time t leaves all inputs at or before t unchanged
   (G1, G2 and G3). An external observation dated X is not used on UTC date X.
2. Targets: exist only when a bar closes exactly h hours later; both t and t+h must be within
   00:00 to 20:00 UTC (20:00 allowed, 21:00 not).
3. Decision grids: 1h every eligible hour, 4h at 00/04/08/12/16, 24h at 08:00.
4. Walk-forward: no training row has t + h after the boundary; scored rows belong to year Y;
   selection touches the initial window only; forecasts at or before t do not change when later
   data change.
5. Ridge: equals a hand-solved example; constant inputs dropped; centring and scaling use the
   training window only; the intercept is not penalized.
6. Trees: same seed gives identical forecasts; the pair column is categorical.
7. Costs: a hand example in expected-move units, without a rollover, with one rollover, and
   with the Wednesday triple.
8. Rules: ALL trades every grid decision with f != 0; SEL trades only if |f| >= cost.
9. Inference: weekly sums include zero-trade weeks; t on a hand example; same seed, same draws.
10. Verdict: truth table for the three verdicts, including each single failing condition.
11. Placebo: the sign is constant within a week, shared by all pairs, and reproducible.
12. Locks: no reader parses 2025-11-03; the runner never calls a loader with final=True.
13. Planted signal: on synthetic data M1 recovers a planted linear relation (correlation at
    least half of the planted value) and a no-signal run gives |t| below 3 (fixed seed).

Run the whole suite: python -m pytest -q. All existing tests must still pass, unmodified.

## Step D: outputs

- results/predict_report.txt
- results/predict_summary.csv: long format, one row per (variant, model, horizon, rule, sample,
  statistic) with value, ci_low, ci_high where they exist. Variants: primary, r1, r2, placebo.
  Samples: all, each half, each pair, each year.
- results/predict_weekly.csv: week, variant, model, horizon, rule, pair, n, gross_sum, cost_sum,
  net_sum. float_format %.10g.
- results/predict_forecasts_24h.csv: every 24h grid decision of the primary variant, both
  models: time, pair, model, forecast, target, close, s, cost, gross, net, traded_sel.
  float_format %.10g.
- results/predict_cumulative.png: three stacked panels (1h, 4h, 24h), cumulative gross and net
  under rule ALL for M1 and M2 against time. Legends outside the plot area (right side). If
  matplotlib is missing, skip the chart and say so.

## Step E: report layout (plain text, fixed-width tables)

1. Environment (python, numpy, pandas, scikit-learn versions) and data: one line per pair (rows,
   first, last) and per external file; eligible rows, scored rows and grid trades per horizon.
2. Selection: alpha per pair and horizon, capacity per horizon, with the holdout correlations.
3. Leak gates: L1 (names of the passing tests) and the L2 placebo table with PASS or FAIL.
4. Primary table, one line per model and horizon: scored rows, Pearson and Spearman correlation,
   mean cost, break-even correlation; rule ALL: trades, d [95% CI], weekly t of gross, mean net
   [95% CI], weekly t of net, mean net in each half; rule SEL: trades, mean gross, mean net
   [95% CI], weekly t of net, mean net in each half; verdict.
5. "Overall verdict: <TRADABLE|PREDICTABLE, BELOW COST|NO PREDICTABILITY>" with the best cell.
6. Report-only R1 to R4, each labelled "report-only".
7. Deviations: every deviation from the pre-registration, or "none".
8. Configuration count line: 6 decision-bearing cells, 9 report-only fits, 2 placebo fits.

## Final reply (in this order)

1. Files created or changed.
2. The pytest summary line.
3. The full text of results/predict_report.txt.
4. Deviations or open questions, or "none".

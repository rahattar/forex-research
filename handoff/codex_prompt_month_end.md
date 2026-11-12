# Codex prompt, task 16: month-end hedging flow into the London 4 pm fix

You are working in the forex-research repo root. Python: ~/fxenv/bin/python.

Read prereg_month_end.md completely before writing code. It is the frozen specification. This
prompt only says how to build and report it. If this prompt and the pre-registration disagree,
the pre-registration wins. If anything is ambiguous, STOP and ask; do not choose.

## Hard rules

- Do not change any rule, parameter or threshold. Do not add, drop or tune configurations.
- Do not modify existing files in src/, scripts/, tests/ or results/. Add new files only.
  Do not edit summary.md, spec.md or any prereg file.
- Do not modify scripts/sim_month_end_power.py (already in the repo; not part of this task).
- FX data only through src.loader.load_continuous_pair. Never pass final=True. Do not parse any
  date on or after 2025-11-01 in any file. No downloads are needed.
- The existing test suite runs unmodified, including its own reads of download/ and fresh/.
- ASCII only in code, comments, output and file contents. Lowercase file names.
- Simple, readable code. numpy, pandas, pytest; matplotlib only for the chart.
- Run the evaluation once. If you rerun after seeing results, give the reason under Deviations.
- Report what the files say. No interpretation beyond the pre-registered verdict.

## Code (new files)

src/month_end.py
- month_end_days(...): D_m with the fallback rule; London times converted with zoneinfo.
- signal(...): S_m from spy Adj Close (reuse src.multi_asset_data._read_etf without modifying it).
- pair_trades(...): one row per month, pair and window with close_in, close_out, s, h, the log
  move, gross, cost and net.
- month_values(...): G_m, C_m, N_m and the number of pairs.
- inference(...): t, the month bootstrap (5000, seed 0), halves.
- verdict(...): the table of section 6, as a pure function of numbers.

scripts/run_month_end.py
- Order: data audit; month-end days and signals; primary test; verdict; R1 to R5; save; print.
- Print the full report to stdout and write the same text to results/month_end_report.txt.

## Tests (tests/test_month_end.py, synthetic data only, fast)

1. Month-end day: last weekday; the fallback when fewer than 4 pairs have both bars; a month
   that cannot be placed is skipped and counted.
2. London time: 2015-01-30 gives 12:00 and 16:00 UTC; 2015-07-31 gives 11:00 and 15:00 UTC.
3. Signal: uses only spy dates strictly before D_m; changing spy on or after D_m changes nothing;
   the base is the last spy date of the previous calendar month.
4. Direction: S_m > 0 sells USD (long eurusd, short usdjpy); S_m < 0 the reverse.
5. Gross, cost and net on a hand example, for a USD-quote and a USD-base pair.
6. Month value: mean over available pairs; fewer than 4 pairs gives no value.
7. Inference: t on a hand example; same seed, same intervals; halves split by year of D_m.
8. Verdict: truth table for ALIVE, EXISTED DECAYED and NOT FOUND, including each failing condition.
9. Placebo day: seven calendar days earlier, no fallback, its own signal.
10. Locks: no reader parses 2025-11-03; the runner never calls a loader with final=True.

Run the whole suite: python -m pytest -q. All existing tests must still pass, unmodified.

## Outputs

- results/month_end_report.txt
- results/month_end_trades.csv: one row per (variant, month, pair): variant (primary, r1, r2 with
  the window, r3, r5), d_m, t_in_utc, t_out_utc, signal, strong, q, pair, usd_sign, close_in,
  close_out, s, h, gross, cost, net. float_format %.10g.
- results/month_end_months.csv: one row per (variant, month): d_m, signal, strong, q, pairs, g,
  c, n. float_format %.10g.
- results/month_end_cumulative.png: cumulative N_m against time for the primary test (strong
  months) and for R1 (all months), with the 2018 split marked. Legend outside the plot area
  (right side). If matplotlib is missing, skip the chart and say so.

## Report layout (plain text, fixed-width tables)

1. Data audit: one line per pair (hourly rows, first, last) and spy; months placed, months
   moved by the fallback (list them), months skipped; strong months in each half.
2. Primary table: n months, mean G [95% CI], t of G, mean C, mean N [95% CI], t of N, for all
   strong months, the early half and the late half. In pips for eurusd as a guide: mean gross
   and mean cost of the eurusd leg.
3. "Verdict: <ALIVE|EXISTED, DECAYED|NOT FOUND>" with the three t values used.
4. Report-only R1 to R5, each labelled "report-only", same columns where they apply.
5. Deviations: every deviation from the pre-registration, or "none".
6. Configuration count line: 1 decision-bearing, 6 report-only.

## Final reply (in this order)

1. Files created or changed.
2. The pytest summary line.
3. The full text of results/month_end_report.txt.
4. Deviations or open questions, or "none".

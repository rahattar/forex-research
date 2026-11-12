# Forex research: full summary and state of the project

Owner: Alex (control scientist, MPC and model-based control background; trades a small personal account as a hobby).
Period of work: 2026-11-09 to 2026-11-11. Code written by Codex from prompts drafted with Claude; Claude reviewed every stage and ran its own checks.

This document is the handoff. Read it fully before proposing any new test.

---

## 1. Goal and constraints as stated by Alex

- Learn to trade FX combining fundamental (macro) and technical analysis, without watching screens all day (full-time job).
- Live budget: 200 USD of fun money, already written off; no top-ups. Risk 5% per trade by default, 10% only on top setups. Daily stop at two losses or 15%, weekly stop at 25%. Stops and targets always attached at entry. Never widen a stop. Minimum reward to risk 1.5.
- Wants "some earning" by year end: a minimum of 30% on the account.
- Alex's diagnosis of his own past failure: lack of discipline on stops, targets, and stopping after losses.
- Later decision (important): strategy tests must NOT be capped at 200 USD. Test strategies on a large notional (100,000 USD or more) or in account-free units (R per trade, Sharpe), and scale back to real money only at the end.
- Later decision: remove the 48-hour time stop as a default (it did not matter; see section 5).
- Pre-2010 data: Alex does not want it. Not downloaded.
- Debt: Alex is paying down credit cards and a HELOC first; the trading money is separate fun money.
- Alex does not want to stop the research. He wants a better technique.

Workflow used: Claude drafts prompts, Alex runs them in Codex (to save Claude usage), Claude reviews outputs and the files on disk.

---

## 2. Data on disk (not in this repo; see data/readme.md to rebuild)

- download/: Dukascopy m15 bid CSVs for eurusd, gbpusd, usdjpy, 2019-01-01 to 2026-10-30 (plus an eurusd h1 file that has bad open/close values in ~0.3% of bars and must not be used).
- fresh/download/: Dukascopy m15 bid, eurusd/gbpusd/usdjpy 2010-2018; audusd, usdcad, usdchf, nzdusd, eurjpy, gbpjpy, eurgbp 2010-01-01 to 2025-10-31.
- data/fred/: dff (US fed funds), ecbdfr (ECB deposit), iudsoia (UK SONIA), boj_policy_rate.csv (hand-built BoJ step table; Mar-2024, Jul-2024, Jan-2025 rows from memory, not verified), ir3tib01*m156n (3-month interbank, eight currencies, monthly), dtb3 (US T-bill).
- data/etf/: yfinance daily for spy, iwm, efa, eem, ief, tlt, lqd, hyg, gld, dbc, vnq, fxe, fxy, uup, 2000 to 2025-10-31.
- Locked final test period: 2025-11-01 to 2026-10-31. Never opened. The loader refuses it unless final=True.
- Known data facts: Dukascopy pads weekends with flat bars (dropped by the loader). Data is BID only, so spreads are modeled; around the daily rollover the bid dips, which can create fake time-of-day patterns.

---

## 3. Code (this repo)

- src/loader.py: cleaning, closing-time index, 1h by resampling m15 for all pairs, daily bars from completed days, train/val/test split with a hard lock, fresh-data loaders.
- src/instruments.py, costs.py, sizing.py: pip sizes, spreads (eurusd 1.0, gbpusd 1.2, usdjpy 1.0, audusd 1.3, usdcad 1.7, usdchf 1.6, nzdusd 1.8, eurjpy 1.8, gbpjpy 2.8, eurgbp 1.5 pips), 1 pip stop slippage, gap fills at the worse of stop and open, stop-first rule when one bar touches both, swap -0.2 pips per night (triple Wednesday), lot sizing rounded down with 0.01 minimum.
- src/engine.py, strategies.py: account-level engine (balance, halts, limits) and strategies A, B (trend_swing), C (trend_swing_15m).
- src/signal_eval.py: account-free evaluation; every signal traded at 1R; mirror trade (same time, opposite side, same distances); direction skill d = (R_strategy - R_mirror) / 2; calendar-week block bootstrap.
- src/rates.py, daily_carry.py, multi_asset.py, multi_asset_data.py: rates, daily momentum and carry, multi-asset ETF tests.
- scripts/: one runner per task (test1_residual, test2_trend, test3_volatility, run_engine, run_variants, test4_rates, run_signal_eval, run_fresh_confirm, daily momentum, run_daily_carry, run_multi_asset).
- tests/: 402 pytest tests at last run (lookahead, lock, cost arithmetic, mirror construction, reproducibility).
- results/: CSVs and printed reports for every task (two large files omitted: fresh_signal_trades.csv and multi_asset_returns.csv; regenerate with the scripts).

Codex ran tests with /opt/miniconda3/envs/pix2pixpolar-dev/bin/python on Alex's Mac.

---

## 4. Methodology that matured over the project (keep it)

1. Pre-register the rule and pass criterion before looking at the data it is tested on.
2. Measure strategies without account effects first: R per trade or Sharpe. The early 200 USD engine skipped most signals (minimum lot, halts) and distorted everything.
3. Benchmark direction with the mirror trade, not a random seed run with different trade sets.
4. Block bootstrap by calendar week (trades) or month (portfolios).
5. Confirm on data nobody has looked at. Fixed rules make every unseen year a fair test; small validation windows have low power (10 months, ~100 trades, under 50% power for a 0.1R effect).
6. Count configurations tested; a new signal should clear roughly t = 3 (Harvey, Liu, Zhu).
7. Costs: random entries at 20 to 40 pip stops lose about 0.06 to 0.10R per trade to spread and slippage. Any direction skill must exceed that.

---

## 5. Results, in order

| # | Test | Result |
|---|---|---|
| 1 | Stretch of 1h price from slow EMA (fade or follow), 1 to 24h, 3 pairs | No edge after spread; 1h horizons lose exactly the spread |
| 2 | Daily trend state (close vs SMA50/200) and pullback, 1 to 10 days | Nothing held from train to validation; market drift dominated |
| 3 | Volatility forecast: EWMA (10 day) of 24h range | WORKS: about 20% lower MAE than naive on all pairs and both splits; stable calibration (range above 2x forecast only 3 to 5% of the time) |
| 4 | Engine: daily trend + 1h pullback + vol stop + 2R, 200 USD account | Indistinguishable from random entries; 2R target hit only 7% of the time |
| 5 | Seven variants: A vol stop, B swing stop (1.5/2/3R), C swing stop + 15m trigger | No variant passed; C looked good on train (97 to 100 percentile) and failed on validation |
| 6 | Rates (policy rate level and 60/120 day change), 3 pairs | No pass; carry paid but price moves ate it in 2025 |
| 7 | Independent review | The 200 USD account and the random benchmark were distorting results; validation was underpowered |
| 8 | Signal-level re-measure (task 9) | Swing-stop family showed d about +0.06R on 2019-2025, interval just including zero |
| 9 | Pre-registered fresh confirmation (task 10), B2, 10 pairs, 2010-2025, 8,945 trades | FAIL: d = -0.047R [-0.078, -0.016]; negative on 9 of 10 pairs, 12 of 16 years |
| 10 | Exits (Claude check): no time stop, no breakeven, all combinations | Irrelevant: net about -0.12R in all versions; with pure stop/2R exits the target hit 29.6% vs 33% for a coin flip |
| 11 | Daily scale: time-series momentum, cross-sectional momentum, 55-day breakout (task 11) | FAIL: net Sharpe -0.22, -0.47 (significant), breakout d -0.07R |
| 12 | Carry, 8 currencies, 2010-2025 (task 12) | FAIL: Sharpe 0.01; 1.4%/yr carry lost to price moves and markup |
| 13 | Multi-asset ETFs 2005-2025 (task 13) | Static diversified S1: Sharpe 0.49 [0.06, 0.93] (real premium). Trend long/flat T1: Sharpe 0.51, max drawdown halved (5.9% vs 12.1%), verdict REDUCES RISK ONLY. Long/short trend weaker (0.25). 60/40: 8.5%/yr, Sharpe 0.64, drawdown 31%. SPY: 11%/yr, drawdown 55% |
| 14 | Adaptive side switching (Claude check) on the 8,945 B2 trade pairs | Did not beat the best fixed side (always mirror -0.029R); even hindsight best side per year is -0.020R |
| 15 | Hour-of-day pattern (iterative-learning idea; Claude check), learn 2010-2017, test 2018-2025 | Real pattern before costs (2.0 pips/day/pair, t 3.8, 9 of 10 pairs) but negative after spread (-1.1 pips/day). Likely partly a bid-only rollover artifact |

Bottom line so far: across roughly 500 configurations, no rule predicts FX direction from public price or rate data after retail costs, at any horizon from 1 hour to 12 months. The only reliable findings are (a) volatility is forecastable, (b) holding a diversified multi-asset portfolio earns a premium, (c) a trend filter on that portfolio cuts drawdowns roughly in half without adding return.

On the 30% goal: nothing supports expecting it. With no edge, a +30% year is roughly a 1 in 7 to 1 in 9 shot at 5% risk, with negative expected value.

---

## 6. Survey conclusions (professional and individual traders)

Professionals:
- Edge comes from breadth (many independent bets) and low cost, more than from better forecasts (Grinold fundamental law: performance ~ skill per bet x sqrt(number of independent bets)).
- Main approaches: diversified risk premia and trend across 50 to 100+ markets; statistical arbitrage (tiny edge, many bets, low cost, shorting); market making; information advantage (alternative data, ML at scale); volatility trading.
- Research discipline: t above 3 for new factors; walk-forward; cost modeling.

Individuals (unbiased full-population studies):
- Taiwan day traders: under 1% predictably profitable after fees (Barber, Lee, Liu, Odean), but that small group persists, so skill exists.
- Brazil futures day traders: 97% of those persisting 300+ days lost money (Chague et al.).
- Retail FX/CFD accounts: 74 to 89% lose (ESMA); 89% in a four-year AMF study.
- Winners are mostly: short-horizon quasi market makers with low costs and many screen hours; disciplined trend followers (Turtles, which is the breakout family that failed on 2010-2025 FX); niche players in small markets; and above all strict risk control. An individual's real advantages: small size, patience, no benchmark.

Control theory already in finance:
- Stochastic optimal control: Merton, market making (Avellaneda-Stoikov), execution (Almgren-Chriss).
- LQ tracking with costs: Garleanu and Pedersen (trade partway toward an aim portfolio).
- MPC: Boyd et al., Multi-Period Trading via Convex Optimization (cvxportfolio).
- Kalman filters: hedge ratios, trend extraction, nowcasting. Trend rules are all linear filters of past returns (Levine and Pedersen), so a better filter shape does not add information.
- Regime models (HMM), volatility targeting and CPPI (proportional drawdown control), robust control (Hansen and Sargent), model-free feedback trading (Barmish and Primbs, guarantees assume zero costs).
- Online portfolio selection / universal portfolios (Cover; Li and Hoi survey) as the adaptive-learning analogue.

Key insight (separation principle): the forecaster (estimator) carries about zero directional information in our data; a better controller cannot create information. Control adds value on decision and risk regulation: multi-period cost-aware rebalancing, volatility and drawdown regulation, monitoring (CUSUM fault detection for strategy decay), robustness margins (fractional Kelly as gain margin), learning repeatable execution-cost patterns.

Adaptive control limit found: detecting a 0.05R effect needs about 1,600 trades (about 3 years on 10 pairs); regimes change at least that fast, so an adaptive law cannot identify direction in time.

---

## 7. Open options at handoff (Alex had not chosen yet)

1. Download ask prices (Dukascopy offers them; dukascopy-node price type ask) and re-test the hour-of-day pattern on mid prices. Decides whether anything real is there or it was the bid-only rollover artifact.
2. Build a controlled premium portfolio: T1 (14 ETFs, trend long/flat) as the plant; run-to-run or MPC rebalancing with a volatility target, drawdown constraint, cost and tax awareness; judge against the simple monthly rule. Expected gain: lower cost and steadier risk, not higher return.
3. Fully invested trend portfolio at 8 to 10% volatility (T1 ran at 3.6% with 41% average cash).
4. Volatility targeting of 60/40 and T1 using the forecast from test 3 (published evidence mixed on returns, clearer on risk).
5. Weekend macro agent (news reading) as a forward paper test, scored with the mirror method; cannot be backtested honestly.
6. Live 200 USD: if traded, use daily-chart trades with wide stops, 2 to 3 per month, journal every call with its mirror; the tight-stop plan loses about 25% a year to costs alone.

Pending from earlier and never done: rewritten "prompt 7" (realistic time-varying spreads, 4h structure stops, trailing exit, efficiency-ratio filter). Superseded in importance by the fresh-data failure; only worth running if the hour-of-day or execution-cost work continues.

---

## 8. Practical notes for continuing

- Alex prefers maximum conciseness, ASCII-only text, technically rigorous answers, minimal hand-holding, lowercase file names, and chart legends outside the plot area.
- Codex outputs should be reviewed against files on disk, not taken at face value (one Codex reply re-ran the wrong task).
- macOS volume is case-insensitive: rename lowercase via a temporary name.
- Keep the locked year locked until a single final run of a fully specified strategy.

---

## 9. Update of 2026-11-12 (work done after the handoff above)

Read sections 1 to 8 first. This section replaces section 7 as the current state.

### 9.1 What changed

- The stock and portfolio work (options 2 to 4 of section 7) moved to its own repo,
  github.com/rahattar/stock-research. Its summary.md is the entry point. Task 13 files stay here
  unchanged. Do not continue stock work in this repo.
- FX work continued here in three steps: task 15 (fitted forecast model, run), task 16
  (month-end fix flow, drafted, not run) and the lab (built, seven runs).
- The owner's position: forecasting is not finished for him. He wants to keep exploring,
  including unusual ideas, and wants to talk an idea through before any code or prompt.

### 9.2 Task 15: is direction predictable by a fitted model? (handoff/prereg_predictability.md)

Status: results as reported by Codex. Not verified against files (section 9.5).

- Two models forecast the next 1h, 4h and 24h move of each pair: ridge regression per pair (M1)
  and gradient-boosted trees pooled over pairs (M2). 42 inputs: the pair's own history, the
  dollar factor, and the previous day's moves in stocks, bonds, credit, gold, commodities, oil,
  VIX and the US 2-year yield. Walk-forward, scored years 2014 to 2025, after spread and swap.
- Units: one expected move (hourly volatility times sqrt of the horizon).

| Model, horizon | Correlation | Direction skill d (t) | Mean cost | Net per trade | Verdict |
|---|---|---|---|---|---|
| M1, 1h | 0.007 | 0.0043 (2.7) | 0.150 | -0.146 | NO PREDICTABILITY |
| M1, 4h | 0.010 | 0.0048 (1.5) | 0.075 | -0.070 | NO PREDICTABILITY |
| M1, 24h | 0.006 | 0.0075 (0.9) | 0.037 | -0.030 | NO PREDICTABILITY |
| M2, 1h | 0.012 | 0.0092 (4.7) | 0.150 | -0.141 | PREDICTABLE, BELOW COST |
| M2, 4h | 0.007 | 0.0082 (2.1) | 0.075 | -0.067 | NO PREDICTABILITY |
| M2, 24h | -0.006 | -0.0032 (-0.3) | 0.037 | -0.040 | NO PREDICTABILITY |

- Break-even correlation: 0.19 (1h), 0.09 (4h), 0.05 (24h). Achieved: 0.012 at best.
- In rough terms the best forecast is right 50.6% of the time; paying the spread needs about
  60% at 1h, 55% at 4h and 52.5% at 24h.
- The 1h effect (about 0.1 pip per trade against 1.6 pips of cost) is positive in 9 of 10 pairs
  and 10 of 12 years and may be the bid-only spread effect flagged in advance.
- Trading only the strongest forecasts was negative after costs in every cell. Other markets
  added nothing provable. Clock inputs roughly doubled the linear model's gross skill, still
  far below cost. The placebo leak check passed (t 0.2 and 1.2). 466 tests passed.
- Conclusion: "forecast the trend from prices and other markets, then trade it" does not pay at
  1h to 24h. The ask-price rerun named in the pre-registration was not done: it would not change
  a trading decision.

### 9.3 Why no time frame escapes (approximate)

| Holding time | Hit rate needed to pay costs | Found |
|---|---|---|
| 15 min | about 70% | not tested; worse than 1h by construction |
| 1h | 60% | 50.6% |
| 4h | 55% | 50.5% |
| 24h | 52.5% | 50.5% |
| 1 to 2 weeks | about 51.5% (the lowest) | hand rules failed; 16 years can only prove 56% or more |
| 1 to 12 months | about 52% (swap grows) | momentum and carry negative |

### 9.4 The lab (handoff/lab_spec.md)

- A test bench built by Codex: scripts/lab.py run ideas/x.toml runs one idea on all 10 pairs for
  2010 to 2019 and prints a fixed scorecard (result per trade in R, the mirror, a random-entry
  baseline with the same exits, shape, by pair and year). Every run is appended to
  results/lab/ledger.csv.
- 2020-01 to 2025-10 is sealed: one confirm run per idea, only with a pre-registration that
  contains the idea file's sha256. Lab results are not claims.
- Building blocks: levels (swings, previous day or week, round numbers, rolling extremes),
  entries (break, bounce, 15-minute trigger, random), bias and lights (trend, hours, cost,
  volatility state), stops (level, volatility, pips), targets (R multiple, level, none), stop
  moves (break-even, trail by level, trail by volatility). Trendlines are left out on purpose.
- Workflow: the owner describes an idea in words; Claude writes the TOML file.

First seven runs (as reported; d is half the gap between the trade and its mirror):

| Run | Idea | Mean R | d (t) | Win rate |
|---|---|---|---|---|
| 01 | Break of a 1h swing low or high, 1:1 at one expected 4h move | -0.086 | +0.002 (0.4) | 46% |
| 02 | Break of the previous day's range | -0.084 | +0.004 (0.6) | 46% |
| 03 | Break of a 100-pip round number | -0.065 | +0.016 (2.5) | 47% |
| 04 | 15m trigger with the 1h trend, stop beyond the 1h swing, target 2.5R | -0.105 | -0.017 (-2.1) | 36% |
| 05 | Run 04 with the stop moved to break-even after 1R | -0.111 | -0.015 (-2.2) | 30% |
| 06 | Run 04 with the stop trailed behind 1h swings, no target | -0.156 | -0.006 (-0.6) | 22% |
| 07 | Run 04 with two lights: 4h trend agrees, cost at most 0.10R | -0.052 | -0.013 (-1.1) | 40% |

Reading
- No "support that price does not recover from" at swings or the previous day's range.
- Round numbers show a small continuation (9 of 10 pairs, 8 of 10 years), consistent with the
  published order-clustering effect, at about a fifth of the cost. One t of 2.5 in seven runs is
  weak evidence.
- Mechanical stop moves changed the shape, not the mean (trailing made it worse).
- The cost light halved the loss by halving the cost per trade; the trend light added nothing.
- Open check: run 06 pays 0.16R of cost against 0.10R in run 04; probably re-entries with closer
  stops after quick stop-outs, not verified.

### 9.5 State of the files

- At the time of this commit the code and results of task 15 and the lab exist only on the
  owner's Mac (src/predict*.py, scripts/run_predict.py, tests/test_predict.py,
  results/predict_*, src/lab_*.py, scripts/lab.py, tests/test_lab.py, results/lab/, ideas/, and
  the root copies of the specifications). When they are pushed, the root copies are canonical.
- handoff/ holds what Claude wrote and what Codex reported: the pre-registrations, the lab
  specification, the three Codex prompts, the two simulation scripts, the seven idea files, and
  reports_as_pasted/ (task 15 report and the first lab runs, as pasted into the chat).
- The task 14 files (vol_target*) in the owner's forex folder belong to stock-research.
- Verification still to do once the files are pushed: recompute task 15 from
  results/predict_weekly.csv and predict_forecasts_24h.csv; read src/predict.py and the lab
  code for lookahead; recompute the lab scorecards from the trades files.

### 9.6 Next steps, in the order last agreed or proposed

1. Lab, round numbers (proposed, not yet approved): levels every 50 pips; the same break with
   stop and target one expected day away; the same break with the cost light; the opposite
   trade (bounce off a round number). The skill would have to roughly triple to pay.
2. Task 16, month-end hedging flow into the London 4 pm fix (handoff/prereg_month_end.md and
   its prompt): ready to run, needs no downloads. Published effect; practitioners report it
   faded. Predicted: NOT FOUND 50%, EXISTED, DECAYED 35%, ALIVE 15%.
3. Drift after central-bank rate decisions (dates are in data/rates).
4. Any idea the owner describes: write it as a lab file first.

Closed: direction from price rules (tests 1 to 12), from a fitted model on prices and other
markets (task 15), stops and targets as a source of edge (tests 5 and 10, lab runs 04 to 06).

### 9.7 Practical notes added in this session

- Python for new tasks: ~/fxenv/bin/python, a venv that inherits the pix2pixpolar-dev packages
  and adds scikit-learn (numpy 2.4.6, pandas 3.0.4, scikit-learn 1.9.1). 534 tests pass.
- The Codex sandbox has no network. The owner runs downloads himself (yfinance in a throwaway
  venv, FRED by curl), then tells Codex the files are present.
- Codex stops and asks when a specification is ambiguous. Answers are appended to the
  specification as a dated "Clarifications" section before it continues.
- Thresholds are set by simulation on synthetic data before a run, and predictions are written
  into the pre-registration.
- A failed data gate was amended once (task 14, stock repo) with a derivation and a one-shot
  rule; treat any amendment after a failure with suspicion and record it as a deviation.

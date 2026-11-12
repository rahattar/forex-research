# Pre-registration, task 15: is FX direction predictable by a fitted model?

Written 2026-11-11, before any model has been fitted in this project.
Frozen once committed. Any deviation must be listed in the report under "Deviations".

## 1. Purpose

Tests 1 to 12 tried hand-set rules (moving averages, breakouts, momentum, rates). No rule was
fitted to data and no other market was used as an input. This task does the identification step
that was skipped: fit models that forecast the next 1h, 4h and 24h move of each pair, and
measure on unseen years whether the forecast carries enough information to pay the spread.

Trading rules with 15-minute entries come afterwards, and only if this task finds something.

## 2. What has already been seen

- FX prices 2010-01 to 2025-10 for all 10 pairs have been used in tests 1 to 12 and in three
  checks. Nothing here is fresh data. The protection is the walk-forward design: every forecast
  that is scored was made by a model fitted only on earlier data, and the specification below is
  fixed before any model is fitted.
- Known from test 15: an hour-of-day pattern exists before costs and is at least partly a
  bid-only artifact. Clock features are therefore kept out of the primary models (section 5).
- A pass here would still need confirmation on the locked year. Locked period: 2025-11-01 to
  2026-10-31. Not read.

## 3. Data

- FX: src.loader.load_continuous_pair(pair, "1h") for the 10 pairs, 2010-01 to 2025-10-31,
  hourly bid bars labelled by closing time (UTC). x_t = ln(close) of the bar closing at t.
- Other markets, daily:
  - data/etf/: spy, efa, eem (stocks), ief, tlt (bonds), hyg (credit), gld (gold), dbc
    (commodities). Adj Close.
  - data/fred/: vixcls.csv (VIX), dcoilbrenteu.csv (Brent oil), dgs2.csv (US 2-year yield).
    New downloads. Rows dated 2025-11-01 or later are never parsed.
- Costs: src.instruments.Config (spread per pair, swap -0.2 pips per night).

## 4. Samples, targets and the bid-only guard

- r_t = x_t - x_(t-1 bar). v_t = ewm(span=240, adjust=False, min_periods=240) of r_t^2.
  s_t = sqrt(v_t) is the hourly volatility known at t.
- Target for horizon h in {1, 4, 24} hours: z_(t,h) = (x_(t+h) - x_t) / (s_t * sqrt(h)).
  It exists only if a bar closes exactly h hours after t.
- Guard: bid prices dip around the daily rollover (21:00 or 22:00 UTC). A row is eligible only
  if both t and t+h have a UTC clock time from 00:00 to 20:00 inclusive. Training and scoring
  use eligible rows only.
- Decision grid for scoring trades (non-overlapping positions per pair):
  - 1h: every eligible hour (00:00 to 19:00 UTC).
  - 4h: 00:00, 04:00, 08:00, 12:00, 16:00 UTC.
  - 24h: 08:00 UTC.

## 5. Inputs (all known at t)

Lookbacks are counted in observed hourly bars. ewm_S means ewm(span=S, adjust=False).mean().
A row is usable only if every input exists; rows before a pair's 481st bar are dropped.

G1, the pair's own history (11 inputs)
- Past moves: (x_t - x_(t-L)) / (s_t * sqrt(L)) for L = 1, 4, 24, 120, 480.
- Distance from averages: (x_t - EMA_S(x)_t) / (s_t * sqrt(S)) for S = 24, 120
  (ewm(span=S, adjust=False)).
- Channel position: (close_t - min) / (max - min) - 0.5 over the high and low of the last L bars
  including t, for L = 24, 120, 480 (0 if max = min).
- Volatility state: 0.5 * ln(ewm_24(r^2) / ewm_480(r^2)).

G2, the dollar factor (9 inputs)
- u_t = mean over the USD pairs with a bar at t of sign_p * r_(p,t) / s_(p,t), with sign +1 for
  usdjpy, usdcad, usdchf and -1 for eurusd, gbpusd, audusd, nzdusd.
- u is defined on the union of the pairs' bar times. U_L = (sum of u over the last L union
  times up to and including t) / sqrt(L) for L = 1, 4, 24, 120.
- The pair's own sign (+1, -1, or 0 for eurjpy, gbpjpy, eurgbp), and sign * U_L for the four L.

G3, other markets (22 inputs)
- For each of spy, efa, eem, ief, tlt, hyg, gld, dbc (log Adj Close), Brent (log price), VIX
  (log level) and the US 2-year yield (level): the 1-observation change and the 5-observation
  change, each divided by (sd of the last 60 1-observation changes, the current one included,
  * sqrt(k)), k = 1 or 5. Observations are the series' own valid dates (missing values skipped).
- Timing: at decision time t use the latest observation dated strictly before the UTC date of t.

Primary models use G1 + G2 + G3 (42 inputs). Hour of day and day of week are excluded from the
primary models and appear only in a report-only variant (section 9).

## 6. Models

Both forecast z. Training targets are clipped to [-5, 5]. Nothing is tuned on scored data.

M1, linear: ridge regression, one model per pair and horizon.
- Inputs and target centred, inputs scaled to unit sd, using the training window only. Inputs
  that are constant in the training window are dropped.
- Minimizes the sum of squared errors + alpha * (sum of squared coefficients), intercept not
  penalized. Alpha from {1e2, 1e3, 1e4, 1e5, 1e6}.

M2, flexible: sklearn HistGradientBoostingRegressor, one pooled model per horizon over all pairs,
with the pair as a categorical input.
- loss squared_error, learning_rate 0.05, min_samples_leaf 2000, early_stopping False,
  random_state 0.
- Capacity from {(max_depth 3, max_iter 100), (max_depth 6, max_iter 300)}.

Selection of alpha and capacity: once per model and horizon, on the initial training window
only. The cut is the time below which 80% of that window's eligible rows fall (all pairs
pooled; the same cut for every pair). Fit on the rows with t + h at or before the cut, score
the rows with t after the cut, keep the choice with the highest Pearson correlation between
forecast and clipped target (M1: per pair; M2: pooled; ties: the stronger penalty, the smaller
capacity). The choice then stays fixed for all refits.

## 7. Walk-forward

- Scored years: 2014 to 2025 (2025 ends 2025-10-31). Twelve refits.
- For scored year Y the model is fitted on all eligible rows with t + h at or before
  Y-01-01 00:00 UTC, and forecasts all eligible rows with t in year Y.
- Initial training window: rows with t + h at or before 2014-01-01.

## 8. Measures and verdicts

For each model and horizon, on scored rows only:

- Forecast quality: correlation of forecast f and target z over all eligible rows (Pearson with
  z clipped to [-5, 5], and Spearman).
- Trades on the decision grid, in units of one expected move. The expected move in pips is
  e_t = s_t * sqrt(h) * close_t / pip_size.
  - gross = sign(f) * z   (z not clipped)
  - cost = (spread_pips[pair] - swap_pips(t, t+h)) / e_t   (swap_pips is negative or zero)
  - net = gross - cost
  - Rule ALL: trade every grid decision with f != 0.
  - Rule SEL: trade only if |f| >= cost (the model expects to earn more than the cost).
- Direction skill d = mean gross. With the mirror trade as benchmark this is the same number:
  (net of the trade - net of its mirror) / 2 = gross.
- Break-even correlation: mean cost / 0.8 (for a normal forecast, d = 0.8 * correlation).
- Inference: calendar weeks (Monday start, UTC, by decision time), all pairs together, weeks
  without trades count as zero. t = mean of weekly sums / (sd of weekly sums / sqrt(weeks)).
  95% intervals for the mean per trade from 5000 week resamples, seed 0 (ratio of sums).
- Halves: 2014 to 2019 and 2020 to 2025.

Verdict per model and horizon:

| Verdict | Condition |
|---|---|
| TRADABLE | for rule ALL or rule SEL: mean net > 0, weekly t of net >= 3.0, at least 500 trades, and mean net > 0 in both halves |
| PREDICTABLE, BELOW COST | not tradable, and for rule ALL: d > 0, weekly t of gross >= 3.0, and d > 0 in both halves |
| NO PREDICTABILITY | otherwise |

Overall verdict: the best of the six cells; all six are reported. Twelve tests at t >= 3 give a
family-wise false-positive rate of about 1.6%.

Caveat fixed in advance: at 1h, and partly at 4h, bid-only data can show a reversal that comes
from the spread moving, not from the price. A gross-only finding at those horizons is not
evidence about mid prices; it would need ask data. It cannot produce a false TRADABLE verdict
by itself because net results pay the full spread, but the static spread may understate costs
in thin hours.

Leak gates (the evaluation is void if one fails)
- L1, in the test suite: truncating all inputs after time t leaves every input and every
  forecast at or before t unchanged; no training row has t + h after the fit boundary.
- L2, placebo run at the 24h horizon for both models: targets multiplied by a random sign per
  calendar week of t (numpy default_rng(0), one draw per week in time order, shared by all
  pairs), then selection and walk-forward rerun on the changed targets. The weekly t of gross
  under rule ALL must be below 3.0 in absolute value for both models. (Net is not tested: with
  no skill it equals minus the cost.)

Power (rough, the run reports exact figures)
- Rule ALL has about 600,000 trades at 1h, 150,000 at 4h and 24,000 at 24h. With pairs moving
  together, the smallest direction skill visible at t = 3 is about 0.007, 0.013 and 0.034.
- Mean cost is about 0.15, 0.075 and 0.035. A TRADABLE verdict therefore needs a correlation of
  about 0.20 at 1h, 0.11 at 4h and 0.09 at 24h. At 24h a real correlation between 0.04 and 0.09
  would show as positive but unproven.
- scripts/sim_predict_power.py: on synthetic data both models recover a planted linear signal
  of correlation 0.03 (t about 12) and return nothing when there is no signal; only M2 recovers
  a planted interaction (about 60% of it at 0.06).

## 9. Report-only items (no verdict, no selection)

- R1: both models with hour of day (0 to 23) and day of week (0 to 4) added (one-hot for M1,
  integers for M2), all horizons. Exposed to the bid-only clock artifact.
- R2: M1 with G1 only, all horizons (what the pair's own history gives without other markets).
- R3: per pair, per year and per half tables for the six primary cells.
- R4: for M1, the ten largest standardized coefficients per horizon in the last refit, pooled
  over pairs by mean absolute value.

Configuration count: 6 decision-bearing cells (2 models, 3 horizons, 2 rules each), 9
report-only fits (R1: 6, R2: 3) and 2 placebo fits. The FX project total rises from about 500
to about 517.

## 10. Predictions, written before the run

- Leak gates pass.
- Correlations: within 0.01 of zero at 1h and 4h apart from a small reversal at 1h; within 0.02
  of zero at 24h.
- Rule SEL trades rarely or never: the models seldom expect more than the cost.
- Verdicts: no TRADABLE cell (95%). NO PREDICTABILITY overall about 55%; PREDICTABLE, BELOW COST
  about 40%, most likely at 1h and from the bid-only reversal.
- R1 shows a clock effect before costs, as in test 15. R2 is no worse than the full M1.

## 11. What follows

- TRADABLE: design the trade with 15-minute entries for that cell, pre-register it, confirm on
  the locked year.
- PREDICTABLE, BELOW COST at 4h or 24h: the problem is cost, not direction. Next step would be
  cheaper execution (limit orders, ask data), not entry rules.
- PREDICTABLE, BELOW COST at 1h only: download ask prices and repeat on mid prices before
  believing it.
- NO PREDICTABILITY: the family "forecast the trend from past prices and other markets, then
  trade it" is closed for 1h to 24h. The month-end fix test is next.

## 12. Clarifications

Added 2026-11-11, before any model was fitted. None changes a verdict rule.

- C1, ridge scaling: sample standard deviation (ddof = 1), from the training window.
- C2, selection cut: no interpolation. Per horizon, sort the eligible rows of the initial window
  (all pairs pooled) by decision time; the cut is the decision time of the row at position
  ceil(0.8 * N), counting from 1. Fit on rows with t + h at or before the cut, score rows with
  t after the cut.
- C3, Spearman correlation: against z not clipped. (Pearson uses z clipped, as in section 8.)
- C4, R1 and R2: each variant selects its own alpha or capacity on the initial window with the
  same procedure, because its inputs differ. The placebo also selects its own (section 8, L2).
- C5, undefined selection scores: a holdout correlation that cannot be computed (constant
  forecast or fewer than 2 rows) ranks below every defined one. If none is defined, take the
  strongest penalty or the smaller capacity.
- C6, inference edge cases: a weekly t with fewer than 2 weeks or zero spread is undefined and
  does not pass. A bootstrap resample with no trades is dropped. "Mean net in a half" is the
  mean per trade in that half.
- C7, numeric details: ridge is solved in float64. The pair category for M2 is the position of
  the pair in src.loader.fresh_pairs (0 to 9).

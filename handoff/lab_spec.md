# Lab specification: a fast test bench for trading ideas

Written 2026-11-12. This is a tool, not a test. Results from the lab are not claims.

## 1. Purpose and rules of use

- Try any entry, stop, target, stop-moving rule or filter in minutes, on 2010-01-01 to
  2019-12-31 only, for all 10 pairs.
- Every run prints the same scorecard and is appended to a ledger, so the number of ideas tried
  is always known.
- 2020-01-01 to 2025-10-31 is sealed. The lab cannot read it. An idea gets one run there, through
  the confirm command, and only with a pre-registration file that names it.
- 2025-11-01 to 2026-10-31 stays locked as before.
- Both periods were used by earlier tests with other rules. The sealed period is an honest
  holdout for new rules as long as it is never looked at while exploring.

## 2. Data and clock

- Bars: src.loader.load_continuous_pair(pair, "15m"), bid, labelled by closing time (UTC).
  The lab keeps only bars closing before 2020-01-01 00:00 UTC; confirm keeps only bars closing
  from 2020-01-01 00:00 UTC to 2025-10-31.
- Higher time frames are built from the 15-minute bars: "1h", "4h" (bars closing at 00, 04, 08,
  12, 16, 20 UTC) and "1d" (bars closing at 17:00 America/New_York). A higher bar is visible at
  time t only if its closing time is at or before t.
- Decisions are made at each 15-minute close t and use only bars closing at or before t.
  The entry is at that close.
- Volatility: s = sqrt of ewm(span=240, adjust=False, min_periods=240) of squared 1h log returns,
  read at the last 1h bar closed at or before t. Expected move over H hours, in price units:
  e(H) = close_t * s * sqrt(H).

## 3. Fills and costs (same conventions as src.signal_eval.simulate_trade)

- Reuse src.costs: entry_fill_price, exit_fill_price, bar_exit, swap_pips; src.instruments
  Config for spread, slippage and pip size. Long entries pay the spread at entry; short stops and
  targets are ask levels; the stop wins when a bar touches both; the entry bar cannot exit.
- risk_pips = stop distance from the entry bid + spread + slippage. R = net pips / risk_pips.
- One open trade per pair. No entries on Friday from 20:00 UTC. Trades still open at the end of
  the data are closed at the last close.
- max_hold_hours (default 48): the trade is closed at the first close at or after that time.

## 4. Building blocks

An idea is one TOML file (section 6). Unknown keys or values are an error, never ignored.

Levels. Each gives a support (below) and a resistance (above), known at t:
- swing: timeframe, k. A bar is a swing low if its low is strictly below the lows of the k bars
  before and the k bars after it; it is confirmed at the close of the k-th bar after it. The
  level is the most recent confirmed swing low (support) or swing high (resistance).
- rolling: timeframe, n. Lowest low and highest high of the last n visible bars.
- prev_day, prev_week: low and high of the last visible "1d" bar, or of the last completed
  Monday-to-Friday week of "1d" bars.
- round: step_pips. The largest multiple of the step strictly below the close and the smallest
  strictly above it.

Entry (one per idea):
- break: level, direction. Support is broken at t if close_t < L and close_(t-1) >= L, where L is
  the support as known at t-1. Resistance likewise. direction "with" sells a broken support and
  buys a broken resistance; "against" does the opposite.
- bounce: level. If low_t <= L and close_t > L (L = support known at t-1), buy. If
  high_t >= L and close_t < L (L = resistance known at t-1), sell.
- trigger: buy candidate if close_t > high_(t-1); sell candidate if close_t < low_(t-1).
  Requires a bias; only the bias side is taken.
- random: probability per bar, seed. Side random, or the bias side if a bias is given.
- sides: "both" (default), "long" or "short".

Bias (optional, sets the allowed side):
- trend: timeframe, fast, slow. Sign of EMA_fast - EMA_slow on visible closes
  (ewm(span, adjust=False)); needs slow bars of history.
- sma: timeframe, n. Sign of close - SMA_n on visible closes.

Lights (optional list; all must agree):
- hours: start, end (UTC hours, start inclusive, end exclusive).
- cost: max_cost_r. (spread + slippage) / risk_pips must be at most this.
- vol_ratio: min, max. sqrt(ewm_24(r^2) / ewm_480(r^2)) on 1h returns must lie in [min, max].
- trend_agree: timeframe, fast, slow. The trend sign must equal the trade side.

Stop (one per idea):
- level: a level block, buffer. Long: support - buffer * e(1). Short: resistance + buffer * e(1).
- vol: multiple, hours. Distance = multiple * e(hours).
- pips: value.
- A signal is skipped (and counted) if the stop is on the wrong side of the entry or closer
  than 2 spreads.

Target (one per idea):
- r: multiple. Distance = multiple * risk in price + spread (as src.strategies.swing_signal).
- level: a level block. Long: resistance. Short: support. Skipped if not beyond the entry.
- none: no target; the trade ends by stop, moved stop or time.

Manage (optional list, applied at each 15-minute close after entry; a stop never loosens; a
changed stop is active from the next bar):
- breakeven: at_r. When the best price since entry reaches at_r * risk, the stop moves to the
  entry fill.
- trail_level: a level block, buffer. Long: stop = max(stop, support - buffer * e(1)).
- trail_vol: multiple, hours. Long: stop = max(stop, highest close since entry -
  multiple * e(hours)), with e read at the current bar. Shorts mirrored.

## 5. Scorecard (identical for every run)

- Counts: signals, skipped (by reason), trades, trades per year, by pair.
- Strategy: mean R [95% CI], weekly t, win rate, mean win, mean loss, payoff ratio, skewness,
  median hold in hours, shares of exit reasons, longest losing streak, mean cost in R.
- Mirror: the same signals with the side reversed and the same stop and target distances and
  manage rules. d = (R - R_mirror) / 2 with [95% CI] and weekly t. coin = (R + R_mirror) / 2.
- Random baseline: 5 seeds. For each seed, the same number of trades per pair and calendar
  year, entry bars drawn uniformly from that pair-year, random side, stop, target and manage
  built by the idea's own rules at that bar (redraw if skipped, at most 20 times). Mean R and
  win rate averaged over seeds.
- Breakdown: mean R and d by pair and by year.
- Inference as in src.signal_eval (calendar-week blocks, 5000 resamples, seed 0).
- One summary line: "direction skill d = x (t = y); exits and costs alone give z".

## 6. Idea file format (TOML, read with tomllib)

    name = "break_1h_swing"
    description = "free text"
    max_hold_hours = 48
    [entry]
    type = "break"
    direction = "with"
    sides = "both"
    level = { type = "swing", timeframe = "1h", k = 3 }
    [bias]
    type = "trend"
    timeframe = "1h"
    fast = 20
    slow = 50
    [[lights]]
    type = "cost"
    max_cost_r = 0.15
    [stop]
    type = "vol"
    multiple = 1.0
    hours = 4
    [target]
    type = "r"
    multiple = 1.0
    [[manage]]
    type = "breakeven"
    at_r = 1.0

## 7. Commands and outputs

- python scripts/lab.py run ideas/x.toml
  Prints the scorecard; writes results/lab/<name>/report.txt and trades.csv; appends one line
  to results/lab/ledger.csv (time, name, sha256 of the idea file, trades, mean R, d, t of d,
  coin, random-baseline R).
- python scripts/lab.py list
  Prints the ledger.
- python scripts/lab.py confirm ideas/x.toml --prereg prereg_x.md
  Runs on the sealed period only if the pre-registration file exists and contains the sha256 of
  the idea file. Writes results/confirm/<name>/ and results/confirm/ledger.csv. A second confirm
  of the same sha256 is refused.

## 8. Guarantees the tests must enforce

- No lookahead: for every level, bias, light, stop, target and manage block, truncating the
  data after t leaves every value and decision at or before t unchanged.
- The lab loader never returns a bar closing on or after 2020-01-01; confirm never returns one
  before it or after 2025-10-31.
- A run with manage = breakeven at 1.0 reproduces src.signal_eval.simulate_trade on the same
  signal (same fills, exit reason and R).
- Mirror trades have the same distances and rules; random baselines are reproducible by seed.
- Same idea file, same data: identical scorecard.

## 9. Clarifications

Added 2026-11-12, before any code was written.

- C1, two candidates on one bar: if a bounce or a break gives both a buy and a sell candidate on
  the same bar, both are skipped and counted under the reason "ambiguous".
- C2, random baseline: it measures what the exits and costs give with no entry information.
  - The side is always random. The bias and any trend_agree light are ignored.
  - The hours, cost and vol_ratio lights apply, and so does the Friday rule. A draw that fails
    them counts as a skipped draw and is redrawn.
  - Baseline trades are simulated independently of each other and of the strategy's trades: the
    one-open-trade-per-pair rule does not apply to them.
- C3, failed redraws: if 20 draws fail, that baseline trade is dropped. The run continues and
  the scorecard prints the requested and the achieved baseline count per seed.
- C4, seeds: 0, 1, 2, 3, 4, each with numpy.random.default_rng(seed). Draws go pair by pair in
  the order of src.loader.fresh_pairs, then year by year, then trade by trade.
- C5, strategy signals: the one-open-trade-per-pair rule is applied in time order. A signal
  that arrives while the pair has an open trade is skipped and counted under "pair busy".

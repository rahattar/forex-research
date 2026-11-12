# Forex backtest spec

Owner: Alex. Personal project, 200 USD fun-money account. Python 3, pandas, numpy. Keep code simple and readable.

## Data (folder: download/)
- eurusd-h1, eurusd-m15, gbpusd-m15, usdjpy-m15 CSVs, 2019-01-01 to 2026-10-30.
- Columns: timestamp (UTC ms), open, high, low, close. BID prices only. No volume.
- Weekend padding exists as flat bars (open=high=low=close, Fri evening to Sun). Drop all flat bars.
- Build 1h for EURUSD, GBPUSD and USDJPY by resampling m15; never read the native h1 CSV. Never use bars from the future.

## Data split (by time, never random)
- Train: 2019-01 to 2024-12. Fit and tune only here.
- Validation: 2025-01 to 2025-10. Choose between candidates here.
- Final test: 2025-11 to 2026-10. LOCKED. Code must not load it unless run with an explicit --final flag, and the final run happens once.

## Trading rules to encode
- Decision cycle: hourly. 15m bars only to time the entry. Stop and target come from 1h levels.
- Direction filter: daily trend (price vs 50 and 200 day averages, plus swing structure). Macro layer is NOT in test 1 (daily trend stands in for it).
- Setups later: pullback, breakout retest, channel bounce, range reversal, failed breakout. Each trade is tagged with its setup type.
- Reward to risk >= 1.5 (gate: skip the trade if it fails, never adjust the target to pass).
- Stop never widened. Move to breakeven at +1R. Time limit on a trade (parameter, default 48h).

## Money and risk
- Start balance 200 USD, compounding. Size from the stop: lot = risk_usd / (stop_pips * pip_value_per_lot).
- Risk per trade 5% default (parameter). Min lot 0.01, lot step 0.01. Skip a trade if the min lot risks more than the cap.
- Daily stop: 2 losses or 15% of balance. Weekly stop: 25%. Trading halts until the next period.
- Several positions allowed. Parameters (undecided, make configurable): min gap between entries (hours), combined open risk cap (10 to 15%), max open positions, block highly correlated pairs.
- Log balance after every closed trade and report max drawdown.

## Costs (configurable, pessimistic)
- Spread per trade, pips: EURUSD 1.0, GBPUSD 1.2, USDJPY 1.0 (bid data, so ask = bid + spread for buys). Wider off-hours.
- Slippage: 1 pip on every stop-out.
- Swap: model overnight cost as a parameter (default small negative per night).
- If one bar touches both stop and target, assume the stop was hit first.
- No entries within 30 min of high-impact news or on Friday evening (news calendar not available yet: use Friday rule only in test 1).

## First test (before any strategy)
1. Slow mean on 1h close (try EMA spans equivalent to 1, 3, 5 days). Residual = price minus slow mean.
2. On train only: autocorrelation of residual changes, and out-of-sample sign predictability of the next 1h to 24h move from residual stretch, net of spread.
3. Repeat on validation. Report the numbers in a short table. No tuning on validation beyond choosing among the few candidates.

## Deliverables
- loader.py (clean, resample, split), costs and sizing helpers, backtest engine, test1 script, a short results.md.
- Unit tests for: no lookahead, flat-bar removal, split boundaries, stop-first rule, position sizing and min lot.

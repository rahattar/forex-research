# Pre-registration, task 16: month-end hedging flow into the London 4 pm fix

Written 2026-11-12, before any month-end return has been computed in this project.
Frozen once committed. Any deviation must be listed in the report under "Deviations".

## 1. Purpose

Tests 1 to 15 looked for direction in prices, rates and other markets and found none that pays
the spread. This task tests a different kind of claim: a forced flow. Funds that hedge foreign
shares adjust their hedges once a month at the London 4 pm fix. When US shares rose during the
month, foreign holders must sell more dollars; when they fell, they must buy dollars back. The
published finding (Melvin and Prins 2015) is that the month's equity return predicts the
currency move in the hours before the month-end fix. Practitioners report that the effect
faded in recent years. The test asks two things: did it exist in our data, and is it alive now.

## 2. What has already been seen

- FX prices 2010-01 to 2025-10 have been used by earlier tests, but never conditioned on the
  calendar day of the month or on the month's equity return.
- The specification is fixed before any month-end return is computed. Thresholds in section 6
  were set on synthetic data (scripts/sim_month_end_power.py).
- Locked period: 2025-11-01 to 2026-10-31. Not read.

## 3. Data

- FX: src.loader.load_continuous_pair for the seven USD pairs: eurusd, gbpusd, audusd, nzdusd
  (USD is the quote currency, usd_sign = -1) and usdjpy, usdcad, usdchf (usd_sign = +1).
  Hourly bars for the primary test; 15-minute bars only where a time is not on the hour.
  x = ln(bid close) of the bar closing at the stated time.
- US shares: data/etf/spy.csv, Adj Close (src.multi_asset_data._read_etf).
- Costs: src.instruments.Config spread per pair. No swap (positions close the same day).

## 4. Definitions

Month-end day D_m, for each month m from 2010-01 to 2025-10 (190 months)
- The last weekday (Monday to Friday) of the calendar month.
- If fewer than 4 of the 7 pairs have bars closing at both T_in and T_out on that day, use the
  previous weekday (at most three steps back; otherwise the month is skipped and reported).

Times
- T_in = 12:00 and T_out = 16:00 Europe/London local time on D_m, converted to UTC with the
  zoneinfo database (12:00 and 16:00 UTC in winter, 11:00 and 15:00 UTC in summer).

Signal (known the evening before D_m)
- S_m = (SPY Adj Close on the last SPY date strictly before D_m)
        / (SPY Adj Close on the last SPY date of the previous calendar month) - 1.
- Strong month: |S_m| >= 0.02.
- Position on the dollar: q_m = -1 (sell USD) if S_m > 0, +1 (buy USD) if S_m < 0.

Per pair p and month m, with h = hours from T_in to T_out (4 for the primary window)
- s = hourly volatility at T_in: sqrt of ewm(span=240, adjust=False, min_periods=240) of the
  squared hourly log returns of the pair, read at the hourly bar closing at T_in.
- gross = q_m * usd_sign_p * (x(T_out) - x(T_in)) / (s * sqrt(h))
- cost  = spread_pips[p] * pip_size / (close(T_in) * s * sqrt(h))
- net   = gross - cost
- A pair-month is used only if the bars at T_in and T_out and the value s all exist.

Per month: G_m, C_m, N_m = the mean over the available pairs of gross, cost and net (at least 4
pairs). One number per month; months are the unit of inference.

## 5. Inference

- t = mean / (sd / sqrt(n)) over months, sd with ddof 1.
- 95% intervals from 5000 resamples of months with replacement, numpy default_rng(0),
  percentile method.
- Halves by the year of D_m: early 2010 to 2017, late 2018 to 2025.

## 6. Decision rule (strong months, primary window, seven-pair basket)

| Verdict | Condition |
|---|---|
| ALIVE | t of N over all strong months >= 2.0, and t of N over the late half >= 1.0 |
| EXISTED, DECAYED | not ALIVE, and t of G over the early half >= 2.0 |
| NOT FOUND | otherwise |

The bar is t = 2, not 3, because this is a published effect with a stated mechanism, tested
once with one configuration (the project standard: about 3 for new signals, about 2 for
established ones).

Operating characteristics (20,000 synthetic histories per row; 56 early and 54 late strong
months, cost 0.075, monthly sd 1 in units of the expected 4h move; 0.30 is about 6 pips on
EURUSD):

| True gross effect, early / late | ALIVE | EXISTED, DECAYED | NOT FOUND |
|---|---|---|---|
| 0 / 0 | 0.00 | 0.02 | 0.97 |
| 0.15 / 0.15 | 0.10 | 0.14 | 0.76 |
| 0.30 / 0.30 | 0.58 | 0.16 | 0.25 |
| 0.45 / 0.45 | 0.94 | 0.04 | 0.02 |
| 0.45 / 0 | 0.05 | 0.85 | 0.09 |
| 0.45 / 0.15 | 0.31 | 0.61 | 0.09 |

Read: the test sees an effect of about 6 to 9 pips per strong month-end and cannot see 3 pips.
ALIVE means "worth a forward test", not "proven": about 12 trades a year cannot prove more.

## 7. Report-only items (no verdict, no selection)

- R1: all months, not only strong ones (same rule and window).
- R2: other windows on strong months, London time: 08:00 to 16:00, 14:00 to 16:00, and 12:00 to
  15:45 (15-minute bar).
- R3: after the fix, strong months: 16:00 to 20:00 London, same position q_m. The literature
  expects a partial reversal (negative gross).
- R4: breakdowns of the primary test: per pair, per year, by half, by the sign of S_m, by
  thirds of |S_m| over all months; and the slope of the dollar basket return (USD up positive,
  mean over pairs of usd_sign_p * (x(T_out) - x(T_in)) / (s * sqrt(h))) on S_m over all months.
- R5: placebo day. The same trade seven calendar days before D_m, with the signal measured to
  the last SPY date strictly before that day and the same 2% rule. Should show nothing.

Details for the report-only items: h is 8, 2, 3.75 and 4 hours for the R2 and R3 windows. R5 has
no fallback day: a placebo day with fewer than 4 pairs is skipped. Thirds of |S_m| use cut
points over all 190 months. When the fallback moves D_m back, the signal still ends at the last
SPY date strictly before the final D_m.

Configuration count: 1 decision-bearing, 6 report-only (R1, three windows, R3, R5). The FX
project total rises from about 517 to about 524.

## 8. Limits stated in advance

- Bid-only prices. Spreads are at their tightest at these hours, so the bid-only bias is small
  here, but the static spread is still an assumption.
- The signal uses US shares only. The published version uses each country's own share market.
  A weaker result here does not disprove the published one.
- A year has 12 month-ends and about 7 strong ones. Nothing about this can be proven quickly.

## 9. Predictions, written before the run

- About 110 strong months of 190.
- Gross positive in the early half, near zero in the late half.
- Verdict: NOT FOUND 50%, EXISTED, DECAYED 35%, ALIVE 15%.
- R5 (placebo day) shows nothing; R3 is negative if the effect exists.

## 10. What follows

- ALIVE: forward paper test at each month-end, the locked year as a check (12 month-ends), and
  a version with each country's own share market.
- EXISTED, DECAYED: recorded as the first effect this project detected, which shows the method
  can find a real one. Not tradable now.
- NOT FOUND: closed. Next structural tests on the same data: order clustering at round numbers,
  then drift after central-bank rate decisions.

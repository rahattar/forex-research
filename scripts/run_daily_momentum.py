"""Pre-registered M1, M2 and D1; no hist/, rates, test data or parameter search."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.daily_breakout import evaluate_breakout, trade_bootstrap, trade_statistics
from src.daily_momentum import (daily_panel, evaluate_m1, evaluate_m2, half, implementability, long_usd_returns,
                                m1_verdict, portfolio_summary, restricted_reads, return_statistics)
from src.instruments import Config
from src.loader import fresh_pairs, load_continuous_pair


results_dir = Path(__file__).resolve().parents[1] / "results"
bootstrap_resamples = 5000
bootstrap_seed = 0


def component_frame(frame, name):
    result = frame.copy()
    result["gross"] = frame[f"gross_{name}"]
    result["net"] = frame[f"net_{name}"]
    result["gross_leverage"] = frame[f"weight_{name}"].shift(1).fillna(0).abs()
    changes = frame[f"weight_{name}"].diff()
    changes.iloc[0] = frame[f"weight_{name}"].iloc[0]
    result["turnover"] = changes.abs()
    result["cost"] = frame[f"cost_{name}"]
    result["financing"] = frame[f"financing_{name}"]
    return result


def summarize_daily(portfolios, trades, closes, resamples=5000, seed=0):
    rows = []
    usd = long_usd_returns(closes)
    for name, frame in portfolios.items():
        for period, mask in (("all", np.ones(len(frame), dtype=bool)),
                             ("2010_2017", frame.index < half), ("2018_2025", frame.index >= half)):
            sample = frame.loc[mask]
            row = {"strategy": name, "group": "portfolio", "period": period, "value": "all",
                   **portfolio_summary(sample, resamples, seed)}
            if name == "M1":
                row["long_usd_gross_correlation"] = sample.gross.corr(usd.reindex(sample.index))
                row["long_usd_net_correlation"] = sample.net.corr(usd.reindex(sample.index))
            rows.append(row)
        for year in range(2010, 2026):
            sample = frame.loc[frame.index.year == year]
            rows.append({"strategy": name, "group": "year", "period": "all", "value": str(year),
                         "n": len(sample), "net_total_return": return_statistics(sample.net)["total_return"]})
        components = [column.removeprefix("weight_") for column in frame if column.startswith("weight_")]
        for component in components:
            rows.append({"strategy": name, "group": "contribution", "period": "all", "value": component,
                         "n": len(frame), "gross_annual_contribution": frame[f"gross_{component}"].mean() * 252,
                         "net_annual_contribution": frame[f"net_{component}"].mean() * 252})
            if name == "baseline":
                rows.append({"strategy": name, "group": "pair", "period": "all", "value": component,
                             **portfolio_summary(component_frame(frame, component), resamples, seed)})
    rows.append({"strategy": "D1", "group": "trades", "period": "all", "value": "all",
                 **trade_statistics(trades), **trade_bootstrap(trades, resamples, seed)})
    for group, column, categories in (("pair", "pair", fresh_pairs), ("side", "side", ("long", "short"))):
        for value in categories:
            sample = trades.loc[trades[column] == value]
            rows.append({"strategy": "D1", "group": group, "period": "all", "value": value,
                         **trade_statistics(sample)})
    for year in range(2010, 2026):
        sample = trades.loc[trades.entry_time.dt.year == year]
        rows.append({"strategy": "D1", "group": "year", "period": "all", "value": str(year), **trade_statistics(sample)})
    result = pd.DataFrame(rows)
    m1 = result.loc[(result.strategy == "M1") & (result['group'] == "portfolio") & (result.period == "all")].iloc[0]
    positions = implementability(closes, m1.net_sharpe)
    positions = positions.assign(strategy="M1", group="implementability", period="all", value=positions.pair).drop(columns="pair")
    return pd.concat([result, positions], ignore_index=True)


def print_tables(summary):
    portfolios = summary.loc[summary['group'] == "portfolio"]
    print("Portfolio metrics: returns are geometric annualized; volatility/Sharpe use 252 days and zero cash interest.")
    print("95% intervals resample calendar months, 5000 draws, seed 0. M2 and D1 are secondary/report-only.")
    print(f"{'rule':<8} {'period':<10} {'mode':<5} {'ann_return [CI]':>26} {'ann_vol [CI]':>23} "
          f"{'Sharpe [CI]':>24} {'MDD':>7} {'gross_lev':>9} {'turn/yr':>8} {'drag/yr':>8}")
    for row in portfolios.itertuples(index=False):
        for mode in ("gross", "net"):
            cells = []
            for metric in ("annual_return", "annual_vol", "sharpe"):
                value, low, high = (getattr(row, f"{mode}_{metric}{suffix}") for suffix in ("", "_ci_low", "_ci_high"))
                cells.append(f"{value:.2%} [{low:.2%},{high:.2%}]" if metric != "sharpe" else f"{value:.3f} [{low:.3f},{high:.3f}]")
            print(f"{row.strategy:<8} {row.period:<10} {mode:<5} {cells[0]:>26} {cells[1]:>23} {cells[2]:>24} "
                  f"{getattr(row, mode + '_max_drawdown'):>6.1%} {row.average_gross_leverage:>9.3f} "
                  f"{row.annual_turnover:>8.3f} {row.annual_cost_drag:>7.2%}")
    print("Evaluation dates:")
    for row in portfolios.loc[portfolios.period == "all"].itertuples(index=False):
        print(f"{row.strategy}: {row.first} to {row.last}; {int(row.n)} completed daily observations.")
    print("M1 correlation with equal-weight long-USD across seven USD pairs (gross/net):")
    for row in portfolios.loc[portfolios.strategy == "M1"].itertuples(index=False):
        print(f"{row.period}: {row.long_usd_gross_correlation:.3f}/{row.long_usd_net_correlation:.3f}")
    print("Year-by-year net portfolio return; D1 cells are n/mean strategy R/mean d, by entry year.")
    print(f"{'year':<5} {'M1':>10} {'M2':>10} {'baseline':>10} {'D1':>24}")
    for year in range(2010, 2026):
        sample = summary.loc[(summary['group'] == "year") & (summary.value == str(year))].set_index("strategy")
        d1 = sample.loc["D1"]
        cells = [f"{sample.loc[name, 'net_total_return']:.2%}" for name in ("M1", "M2", "baseline")]
        print(f"{year:<5} " + " ".join(f"{cell:>10}" for cell in cells) + f" {int(d1.n):>5}/{d1.mean_r_strategy:.3f}/{d1.mean_d:.3f}")
    print("Annual arithmetic contributions (gross/net), percentage points; components sum to portfolio arithmetic means.")
    for name in ("M1", "M2"):
        sample = summary.loc[(summary.strategy == name) & (summary['group'] == "contribution")]
        print(name + ": " + "; ".join(f"{row.value} {100 * row.gross_annual_contribution:.3f}/{100 * row.net_annual_contribution:.3f}"
                                     for row in sample.itertuples(index=False)))
    print("Always-long per-pair baseline, same weekly M1 volatility scaling: gross/net annual return, net vol/Sharpe/MDD.")
    for row in summary.loc[(summary.strategy == "baseline") & (summary['group'] == "pair")].itertuples(index=False):
        print(f"{row.value:<7} {row.gross_annual_return:>7.2%}/{row.net_annual_return:.2%} "
              f"{row.net_annual_vol:>7.2%}/{row.net_sharpe:.3f}/{row.net_max_drawdown:.2%}")
    d1 = summary.loc[(summary.strategy == "D1") & (summary['group'] == "trades")].iloc[0]
    print(f"D1: n={int(d1.n)}, strategy R={d1.mean_r_strategy:.3f} [{d1.r_ci_low:.3f},{d1.r_ci_high:.3f}], "
          f"mirror R={d1.mean_r_mirror:.3f}, d={d1.mean_d:.3f} [{d1.d_ci_low:.3f},{d1.d_ci_high:.3f}], "
          f"cost R={d1.mean_cost_r:.3f}, average hold={d1.average_hold_days:.2f} days.")
    print("D1 exits, strategy/mirror: count(mean R).")
    for reason in ("stop", "channel", "end_of_data"):
        print(f"{reason:<12} {int(d1[reason + '_n'])}({d1[reason + '_mean_r']:.3f}) / "
              f"{int(d1['mirror_' + reason + '_n'])}({d1['mirror_' + reason + '_mean_r']:.3f})")
    print("D1 detail: n, strategy R, mirror R, d, cost R, hold days. Exit breakdowns are also in daily_summary.csv.")
    for row in summary.loc[(summary.strategy == "D1") & summary['group'].isin(("pair", "side"))].itertuples(index=False):
        print(f"{row.value:<7} {int(row.n):>4} {row.mean_r_strategy:>8.3f} {row.mean_r_mirror:>8.3f} "
              f"{row.mean_d:>8.3f} {row.mean_cost_r:>7.3f} {row.average_hold_days:>8.2f}")
    positions = summary.loc[summary['group'] == "implementability"]
    print("Implementability: median absolute base-currency units / share of pair-weeks >=1000 units.")
    print(f"{'pair':<7} {'$200/10%':>19} {'$200/20%':>19} {'$2000/10%':>19} {'$2000/20%':>19}")
    settings = ((200, .1), (200, .2), (2000, .1), (2000, .2))
    for pair in fresh_pairs:
        cells = []
        for balance, target in settings:
            row = positions.loc[(positions.value == pair) & (positions.balance == balance) & (positions.target_vol == target)].iloc[0]
            cells.append(f"{row.median_units:.1f}/{row.share_ge_1000:.1%}")
        print(f"{pair:<7} " + " ".join(f"{cell:>19}" for cell in cells))
    print("Sharpe-implied annual USD mean = balance * stated target volatility * full-sample M1 net Sharpe; not a dollar forecast.")
    for balance, target in settings:
        row = positions.loc[(positions.balance == balance) & (positions.target_vol == target)].iloc[0]
        print(f"${balance}/{target:.0%}: ${row.expected_annual_usd:.2f}")
    print(f"M1 verdict: {m1_verdict(summary)}")


def main():
    with restricted_reads():
        hourly, d1_frames = {}, []
        cfg = Config()
        print("Continuous data, closing timestamps, test boundary exclusive at 2025-11-01 UTC:", flush=True)
        for pair in fresh_pairs:
            hourly[pair] = load_continuous_pair(pair, "1h")
            frame = hourly[pair]
            join = frame.attrs
            detail = (f"; join overlap={join['join_overlap']}, gap={join['join_gap_hours']:.2f}h "
                      f"({join['join_before']} -> {join['join_after']})") if "join_overlap" in join else ""
            print(f"{pair}: {len(frame)} hourly bars, {frame.index[0]} -> {frame.index[-1]}{detail}", flush=True)
        closes = daily_panel(hourly)
        print(f"Joint completed-day calendar: {len(closes)} rows, {closes.index[0]} -> {closes.index[-1]}; no price interpolation.", flush=True)
        portfolios = {"M1": evaluate_m1(closes, cfg), "M2": evaluate_m2(closes, cfg),
                      "baseline": evaluate_m1(closes, cfg, always_long=True)}
        for pair in fresh_pairs:
            trades = evaluate_breakout(hourly[pair], pair, cfg)
            d1_frames.append(trades)
            print(f"D1 {pair}: {len(trades)} strategy/mirror trade pairs", file=sys.stderr, flush=True)
        trades = pd.concat(d1_frames, ignore_index=True).sort_values(["entry_time", "pair"]).reset_index(drop=True)
        summary = summarize_daily(portfolios, trades, closes, bootstrap_resamples, bootstrap_seed)
        results_dir.mkdir(parents=True, exist_ok=True)
        for name, filename in (("M1", "daily_m1_returns.csv"), ("M2", "daily_m2_returns.csv")):
            portfolios[name].to_csv(results_dir / filename, index_label="timestamp", float_format="%.12g")
        trades.to_csv(results_dir / "daily_d1_trades.csv", index=False, float_format="%.12g")
        summary.to_csv(results_dir / "daily_summary.csv", index=False, float_format="%.12g")
        print_tables(summary)
        print("Saved daily_m1_returns.csv, daily_m2_returns.csv, daily_d1_trades.csv and daily_summary.csv. No tuning or other selection.")


if __name__ == "__main__":
    main()

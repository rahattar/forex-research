"""Pre-registered ETF trend versus static risk holding, with paired inference."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.multi_asset import evaluate, price_panel, strategies, summarize, trend_verdict
from src.multi_asset_data import cash_returns, load_inputs, restricted_asset_reads, tickers


results_dir = Path(__file__).resolve().parents[1] / "results"
bootstrap_resamples = 5000
bootstrap_seed = 0


def number(value):
    return f"{value:.3f}" if np.isfinite(value) else "-"


def print_tables(summary):
    main = summary.loc[summary['group'] == "portfolio"]
    print("Annual returns are geometric; volatility and excess-return Sharpe use 252 days. All intervals use 5000 paired calendar-month resamples, seed 0.")
    print(f"{'period':<12} {'book':<4} {'total':>8} {'excess':>8} {'vol':>7} {'excess Sharpe [95% CI]':>27} "
          f"{'MDD':>7} {'cash':>7} {'turn/yr':>8} {'cost/yr':>8}")
    for row in main.itertuples(index=False):
        ci = f"{number(row.excess_sharpe)} [{number(row.sharpe_ci_low)},{number(row.sharpe_ci_high)}]"
        print(f"{row.period:<12} {row.strategy:<4} {row.annual_total_return:>7.2%} {row.annual_excess_return:>7.2%} "
              f"{row.total_vol:>6.2%} {ci:>27} {row.max_drawdown:>6.2%} {row.average_cash_weight:>6.1%} "
              f"{row.annual_turnover:>8.3f} {row.annual_cost:>7.3%}")
    print("Total-return drawdown dates: peak/trough/recovery; worst calendar year (including partial endpoint years).")
    for row in main.itertuples(index=False):
        dates = [str(getattr(row, "drawdown_" + key)).split("T")[0] for key in ("start", "trough", "recovery")]
        print(f"{row.period:<12} {row.strategy:<4} {' / '.join(dates)}; worst {int(row.worst_year)} {row.worst_year_return:.2%}")
    print("T1 minus S1: paired Sharpe difference and total-return drawdown difference; negative DD difference means smaller T1 drawdown.")
    for row in summary.loc[summary['group'] == "comparison"].itertuples(index=False):
        print(f"{row.period:<12} Sharpe {row.sharpe_difference:.3f} [{row.sharpe_difference_ci_low:.3f},{row.sharpe_difference_ci_high:.3f}]; "
              f"DD T1/S1 {row.t1_drawdown:.2%}/{row.s1_drawdown:.2%}; difference {row.drawdown_difference:.2%} "
              f"[{row.drawdown_difference_ci_low:.2%},{row.drawdown_difference_ci_high:.2%}]; relative reduction {row.relative_drawdown_reduction:.1%}")
    print("Year-by-year total returns:")
    years = sorted(summary.loc[summary['group'] == "year", "value"].unique())
    print("year " + " ".join(f"{name:>9}" for name in strategies))
    for year in years:
        sample = summary.loc[(summary['group'] == "year") & (summary.value == year)].set_index("strategy")
        print(f"{year} " + " ".join(f"{sample.loc[name, 'total_return']:>8.2%}" for name in strategies))
    print("Asset contributions: annual arithmetic total/excess percentage points; T1 holding share counts executed rebalance months, including pre-eligibility zeros.")
    print(f"{'asset':<5} {'S1 total/excess':>18} {'T1 total/excess':>18} {'early excess S1/T1':>20} "
          f"{'late excess S1/T1':>20} {'T1 held all/early/late':>26}")
    for asset in (*tickers, "cash"):
        sample = summary.loc[(summary['group'] == "asset") & (summary.value == asset)].set_index(["strategy", "period"])
        s1, t1 = sample.loc[("S1", "all")], sample.loc[("T1", "all")]
        cells = [f"{s1.annual_total_contribution * 100:.3f}/{s1.annual_excess_contribution * 100:.3f}",
                 f"{t1.annual_total_contribution * 100:.3f}/{t1.annual_excess_contribution * 100:.3f}"]
        for period in ("before_2017", "from_2017"):
            cells.append(f"{sample.loc[('S1', period), 'annual_excess_contribution'] * 100:.3f}/"
                         f"{sample.loc[('T1', period), 'annual_excess_contribution'] * 100:.3f}")
        shares = "/".join(f"{sample.loc[('T1', period), 'held_share_months']:.1%}"
                          for period in ("all", "before_2017", "from_2017")) if asset != "cash" else "-"
        print(f"{asset:<5} {cells[0]:>18} {cells[1]:>18} {cells[2]:>20} {cells[3]:>20} {shares:>26}")
    print("Median target position USD at executed rebalances (fixed account balances, fractional positions):")
    print(f"{'asset':<5} {'S1 $2000':>12} {'T1 $2000':>12} {'S1 $100000':>13} {'T1 $100000':>13}")
    for asset in tickers:
        cells = []
        for balance, name in ((2000, "S1"), (2000, "T1"), (100000, "S1"), (100000, "T1")):
            row = summary.loc[(summary['group'] == "position") & (summary.strategy == name) &
                              (summary.balance == balance) & (summary.value == asset)].iloc[0]
            cells.append(f"{row.median_position_usd:.2f}")
        print(f"{asset:<5} {cells[0]:>12} {cells[1]:>12} {cells[2]:>13} {cells[3]:>13}")
    print("Annual USD implied by full-period geometric annual excess return:")
    for name in strategies:
        sample = summary.loc[(summary['group'] == "annual_usd") & (summary.strategy == name)].set_index("balance")
        print(f"{name:<4} $2000: {sample.loc[2000, 'annual_usd']:.2f}; $100000: {sample.loc[100000, 'annual_usd']:.2f}")
    s1 = main.loc[(main.strategy == "S1") & (main.period == "all")].iloc[0]
    print(f"S1 premium: excess-return Sharpe {s1.excess_sharpe:.6f}, 95% CI [{s1.sharpe_ci_low:.6f}, {s1.sharpe_ci_high:.6f}]")
    print(f"T1 against S1: {trend_verdict(summary)}")


def main():
    with restricted_asset_reads():
        inputs, observed_cash, _ = load_inputs()
        prices = price_panel(inputs)
        cash = cash_returns(observed_cash, prices.index)
        frames, weights, executions, _, signal_date = evaluate(prices, cash)
        calendar = frames["S1"].index
        print(f"First qualifying signal {signal_date.date()}; first execution {calendar[0].date()}; "
              f"last return {calendar[-1].date()}; {len(calendar)} common daily returns.", flush=True)
        summary = summarize(frames, weights, executions, bootstrap_resamples, bootstrap_seed)
        returns = pd.concat([frames[name].assign(strategy=name) for name in strategies]).rename_axis("date").reset_index()
        results_dir.mkdir(parents=True, exist_ok=True)
        returns.to_csv(results_dir / "multi_asset_returns.csv", index=False, float_format="%.12g")
        summary.to_csv(results_dir / "multi_asset_summary.csv", index=False, float_format="%.12g")
        print_tables(summary)
        print("Saved multi_asset_returns.csv and multi_asset_summary.csv. All six books use the same dates; no tuning or other selection.")


if __name__ == "__main__":
    main()

"""Fixed C1/C2 carry confirmation and carry-aware momentum re-reports."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.daily_carry import (add_carry, blend_returns, c1_verdict, c1_weights, c2_weights, carry_statistics,
                             currency_contributions, evaluate_currency_book, evaluate_pair_book, implementation,
                             long_usd_weights, monthly_holdings, restricted_carry_reads)
from src.daily_momentum import (currency_pairs, daily_panel, evaluate_m1, evaluate_m2, half, inverse_pairs,
                                return_statistics)
from src.instruments import Config
from src.loader import fresh_pairs, load_continuous_pair
from src.rates import load_monthly_rates, monthly_sources


results_dir = Path(__file__).resolve().parents[1] / "results"
bootstrap_resamples = 5000
bootstrap_seed = 0


def component_frame(frame, pair):
    result = frame.copy()
    for name in ("spot", "carry", "markup", "cost", "gross", "net"):
        result[name] = frame[f"{name}_{pair}"]
    weights = frame[f"weight_{pair}"]
    result["gross_leverage"] = weights.shift(1).fillna(0).abs()
    result["turnover"] = weights.diff().fillna(weights.iloc[0]).abs()
    return result


def summarize_carry(portfolios, c1_allocations, c2_targets, closes, resamples=5000, seed=0):
    rows = []
    periods = (("all", None), ("2010_2017", False), ("2018_2025", True))
    for strategy, frame in portfolios.items():
        for period, after in periods:
            sample = frame if after is None else frame.loc[(frame.index >= half) == after]
            row = {"strategy": strategy, "group": "portfolio", "period": period, "value": "all",
                   **carry_statistics(sample, resamples, seed)}
            if strategy == "C1":
                for name in ("M1_carry", "M2_carry"):
                    row[f"correlation_{name}"] = sample.net.corr(portfolios[name].net)
            rows.append(row)
            if strategy in ("C1", "C2"):
                contributions = currency_contributions(sample, currency_book=strategy == "C1")
                for currency, contribution in contributions.iterrows():
                    rows.append({"strategy": strategy, "group": "currency", "period": period, "value": currency,
                                 **{f"annual_{key}_contribution": value for key, value in contribution.items()}})
        for year in range(2010, 2026):
            sample = frame.loc[frame.index.year == year]
            rows.append({"strategy": strategy, "group": "year", "period": "all", "value": str(year), "n": len(sample),
                         "net_total_return": return_statistics(sample.net)["total_return"]})
        if strategy == "long_all":
            for pair in closes:
                rows.append({"strategy": strategy, "group": "pair", "period": "all", "value": pair,
                             **carry_statistics(component_frame(frame, pair), resamples, seed)})
    for strategy, weights in (("C1", c1_allocations), ("C2", c2_targets)):
        months = monthly_holdings(weights, currency_book=strategy == "C1")
        for period, after in periods:
            sample = months if after is None else months.loc[(months.index >= "2018-01") == after]
            for currency in monthly_sources:
                rows.append({"strategy": strategy, "group": "holdings", "period": period, "value": currency,
                             "n_months": len(sample), "long_share": sample[currency].gt(0).mean(),
                             "short_share": sample[currency].lt(0).mean(), "flat_share": sample[currency].eq(0).mean()})
        for month, row in months.iterrows():
            for currency in monthly_sources:
                rows.append({"strategy": strategy, "group": "holding_month", "period": "all", "value": currency,
                             "month": month, "exposure": row[currency]})
        if strategy == "C1":
            positions = pd.DataFrame(index=weights.index)
            for currency, pair in currency_pairs.items():
                positions[pair] = weights[currency] * (-1 if pair in inverse_pairs else 1)
        else:
            positions = weights
        main = next(row for row in rows if row["strategy"] == strategy and row["group"] == "portfolio" and row["period"] == "all")
        units = implementation(closes, positions, main["net_sharpe"], main["net_annual_vol"],
                               "month" if strategy == "C1" else "week")
        rows.extend(units.assign(strategy=strategy, group="implementability", period="all").to_dict("records"))
    return pd.DataFrame(rows)


def print_tables(summary):
    main = summary.loc[summary['group'] == "portfolio"]
    print("Gross = spot + actual carry; net = gross - one broker markup - trading cost. Annual component columns are arithmetic percentage points.")
    print("CAGR is geometric. Volatility/Sharpe use 252 days; 95% calendar-month bootstrap intervals use 5000 resamples, seed 0.")
    print(f"{'rule':<10} {'period':<10} {'spot':>7} {'carry':>7} {'markup':>7} {'trade':>7} {'CAGR G/N':>15} "
          f"{'vol G/N':>13} {'Sharpe G [CI]':>25} {'Sharpe N [CI]':>25} {'DD G/N':>13} {'turn/yr':>8}")
    for row in main.itertuples(index=False):
        components = [getattr(row, f"annual_{name}_contribution") * 100 for name in ("spot", "carry", "markup", "cost")]
        gross = f"{row.gross_sharpe:.3f}[{row.gross_sharpe_ci_low:.3f},{row.gross_sharpe_ci_high:.3f}]"
        net = f"{row.net_sharpe:.3f}[{row.net_sharpe_ci_low:.3f},{row.net_sharpe_ci_high:.3f}]"
        annual = f"{row.gross_annual_return:.2%}/{row.net_annual_return:.2%}"
        vol = f"{row.gross_annual_vol:.2%}/{row.net_annual_vol:.2%}"
        drawdown = f"{row.gross_max_drawdown:.1%}/{row.net_max_drawdown:.1%}"
        print(f"{row.strategy:<10} {row.period:<10} " + " ".join(f"{value:>7.3f}" for value in components) +
              f" {annual:>15} {vol:>13} {gross:>25} {net:>25} {drawdown:>13} {row.annual_turnover:>8.3f}")
    print("Net drawdown dates (peak / trough / recovery), worst net month, and evaluation window:")
    for row in main.itertuples(index=False):
        dates = [str(getattr(row, "net_drawdown_" + suffix)).split("T")[0] for suffix in ("start", "trough", "recovery")]
        print(f"{row.strategy:<10} {row.period:<10} {' / '.join(dates)}; worst {row.worst_month} {row.worst_month_return:.2%}; "
              f"{row.first[:10]}..{row.last[:10]} n={int(row.n)}")
    print("Year-by-year net returns; 2010 and 2025 may be partial. Secondary results apply no selection.")
    strategies = list(main.strategy.drop_duplicates())
    print("year " + " ".join(f"{name:>11}" for name in strategies))
    for year in range(2010, 2026):
        sample = summary.loc[(summary['group'] == "year") & (summary.value == str(year))].set_index("strategy")
        print(f"{year} " + " ".join(f"{sample.loc[name, 'net_total_return']:>10.2%}" for name in strategies))
    print("Currency annual contributions, percentage points (spot/carry/markup/trade/net).")
    for row in summary.loc[(summary['group'] == "currency") & (summary.period == "all")].itertuples(index=False):
        values = [getattr(row, f"annual_{name}_contribution") * 100 for name in ("spot", "carry", "markup", "cost", "net")]
        print(f"{row.strategy:<3} {row.value:<3} " + " / ".join(f"{value:.3f}" for value in values))
    print("Currency shares of months long/short; C1 counts ranked choices including USD cash; C2 uses month-average net currency exposure.")
    for currency in monthly_sources:
        cells = []
        for strategy in ("C1", "C2"):
            for period in ("all", "2010_2017", "2018_2025"):
                row = summary.loc[(summary.strategy == strategy) & (summary['group'] == "holdings") &
                                  (summary.period == period) & (summary.value == currency)].iloc[0]
                cells.append(f"{row.long_share:.1%}/{row.short_share:.1%}")
        print(f"{currency}: C1 all/early/late " + " ".join(cells[:3]) + "; C2 all/early/late " + " ".join(cells[3:]))
    print("C1 daily-net correlation with carry-aware M1 / M2:")
    for row in main.loc[main.strategy == "C1"].itertuples(index=False):
        print(f"{row.period}: {row.correlation_M1_carry:.3f}/{row.correlation_M2_carry:.3f}")
    print("Always-long pair baselines: net CAGR, net volatility, net Sharpe [95% CI], net max DD; full metrics saved.")
    for row in summary.loc[(summary.strategy == "long_all") & (summary['group'] == "pair")].itertuples(index=False):
        print(f"{row.value:<7} {row.net_annual_return:>7.2%} {row.net_annual_vol:>7.2%} "
              f"{row.net_sharpe:>6.3f}[{row.net_sharpe_ci_low:.3f},{row.net_sharpe_ci_high:.3f}] {row.net_max_drawdown:>6.2%}")
    positions = summary.loc[summary['group'] == "implementability"]
    print("Registered-weight implementability, no volatility rescaling: median units (active-only median) / share of rebalance periods >=1000 units.")
    for strategy in ("C1", "C2"):
        for pair in positions.loc[positions.strategy == strategy, "value"].drop_duplicates():
            cells = []
            for balance in (200, 2000):
                row = positions.loc[(positions.strategy == strategy) & (positions.value == pair) & (positions.balance == balance)].iloc[0]
                cells.append(f"${balance}: {row.median_units:.1f}({row.median_active_units:.1f})/{row.share_ge_1000:.1%}")
            print(f"{strategy} {pair}: " + "; ".join(cells))
        for balance in (200, 2000):
            row = positions.loc[(positions.strategy == strategy) & (positions.balance == balance)].iloc[0]
            print(f"{strategy} ${balance}: annual arithmetic USD mean {row.expected_annual_usd:.2f} (=balance * realized vol * net Sharpe).")
    c1 = main.loc[(main.strategy == "C1") & (main.period == "all")].iloc[0]
    print(f"C1 primary net Sharpe={c1.net_sharpe:.6f}, 95% CI [{c1.net_sharpe_ci_low:.6f}, {c1.net_sharpe_ci_high:.6f}].")
    print(f"C1 verdict: {c1_verdict(summary)}")


def main():
    with restricted_carry_reads():
        # The previous task must exist, and its code/output files are never written.
        for name in ("daily_m1_returns.csv", "daily_m2_returns.csv", "daily_d1_trades.csv", "daily_summary.csv"):
            if not (results_dir / name).is_file():
                raise ValueError("Task 11 must be complete before Task 12")
        cfg = Config()
        rates = load_monthly_rates()
        print(f"Eight monthly interbank rates, following-month availability; calendar {rates.index[0]}..{rates.index[-1]}.", flush=True)
        hourly = {pair: load_continuous_pair(pair, "1h") for pair in fresh_pairs}
        closes = daily_panel(hourly)
        allocations = c1_weights(closes, rates)
        pair_weights = c2_weights(closes, rates)
        portfolios = {"C1": evaluate_currency_book(closes, allocations, rates, cfg),
                      "C2": evaluate_pair_book(closes, pair_weights, rates, cfg),
                      "M1_carry": add_carry(evaluate_m1(closes, cfg), rates),
                      "M2_carry": add_carry(evaluate_m2(closes, cfg), rates, currency_book=True)}
        portfolios["blend"] = blend_returns(portfolios["C1"], portfolios["M1_carry"])
        portfolios["long_usd"] = evaluate_pair_book(closes, long_usd_weights(closes), rates, cfg)
        portfolios["long_all"] = evaluate_pair_book(closes, c2_weights(closes, rates, always_long=True), rates, cfg)
        summary = summarize_carry(portfolios, allocations, pair_weights, closes, bootstrap_resamples, bootstrap_seed)
        for strategy, filename in (("C1", "carry_c1_returns.csv"), ("C2", "carry_c2_returns.csv")):
            portfolios[strategy].to_csv(results_dir / filename, index_label="timestamp", float_format="%.12g")
        summary.to_csv(results_dir / "carry_summary.csv", index=False, float_format="%.12g")
        print_tables(summary)
        print("Saved carry_c1_returns.csv, carry_c2_returns.csv and carry_summary.csv. Task 11 unchanged; no tuning or other selection.")


if __name__ == "__main__":
    main()

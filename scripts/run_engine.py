"""Fixed-parameter train/val runs and 200 seeded random controls per split."""

from functools import partial
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.engine import EngineConfig, PreparedMarket, run_engine
from src.loader import load_pair, pairs
from src.strategies import build_features, random_entry, signal_probabilities, trend_pullback


results_dir = Path(__file__).resolve().parents[1] / "results"
benchmark_seeds = tuple(range(200))


def percentile_rank(values, score):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values) or not np.isfinite(score):
        return np.nan
    return 100 * ((values < score).sum() + 0.5 * (values == score).sum()) / len(values)


def print_summaries(summary):
    trend = summary.loc[summary["strategy"] == "trend_pullback"]
    print("Trend pullback: fixed parameters, fresh $200 per split; drawdown is realized balance.")
    print(f"{'split':<5} {'trades':>6} {'win':>7} {'avg_r':>8} {'expect_r':>8} {'pf':>7} {'balance':>9} {'dd_usd':>8} {'dd_pct':>8} {'day_halt':>8} {'week_halt':>9} {'min_lot':>8} {'hold_h':>8}")
    for row in trend.itertuples(index=False):
        print(f"{row.split:<5} {row.trades:>6} {row.win_rate:>6.1%} {row.average_r:>8.3f} {row.expectancy_r:>8.3f} "
              f"{row.profit_factor:>7.3f} {row.ending_balance:>9.2f} {row.max_drawdown_usd:>8.2f} "
              f"{row.max_drawdown_pct:>7.1%} {row.daily_halts:>8} {row.weekly_halts:>9} "
              f"{row.skipped_min_lot:>8} {row.average_hold_hours:>8.2f}")
    print(f"Random entry: {len(benchmark_seeds)} seeds per split, p5 / median / p95; percentile ranks use midranks.")
    print(f"{'split':<5} {'expectancy_r p5/median/p95':>29} {'ending_balance p5/median/p95':>32} {'trend_exp_pct':>13} {'trend_bal_pct':>13}")
    for which in ("train", "val"):
        sample = summary.loc[(summary["split"] == which) & (summary["strategy"] == "random_entry")]
        actual = trend.loc[trend["split"] == which].iloc[0]
        finite_expectancy = sample.loc[np.isfinite(sample["expectancy_r"]), "expectancy_r"]
        exp = np.percentile(finite_expectancy, [5, 50, 95]) if len(finite_expectancy) else np.full(3, np.nan)
        balances = np.nanpercentile(sample["ending_balance"], [5, 50, 95])
        exp_text = " / ".join(f"{value:.3f}" for value in exp)
        bal_text = " / ".join(f"{value:.2f}" for value in balances)
        print(f"{which:<5} {exp_text:>29} {bal_text:>32} "
              f"{percentile_rank(sample['expectancy_r'], actual.expectancy_r):>12.1f}% "
              f"{percentile_rank(sample['ending_balance'], actual.ending_balance):>12.1f}%")


def main():
    cfg = EngineConfig()
    hourly = {pair: load_pair(pair, "1h") for pair in pairs}
    quarters = {pair: load_pair(pair, "15m") for pair in pairs}
    features = {pair: build_features(hourly[pair], pair) for pair in pairs}
    results_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for which in ("train", "val"):
        market = PreparedMarket(hourly, quarters, which, features)
        result = run_engine(market, partial(trend_pullback, cfg=cfg), cfg)
        result.trades.to_csv(results_dir / f"engine_trades_{which}.csv", index=False, float_format="%.12g")
        summaries.append({"strategy": "trend_pullback", "seed": -1, **result.summary})
        rates, counts, opportunities = signal_probabilities(market, cfg)
        print(f"{which} raw signal rates: " + ", ".join(
            f"{pair}={counts[pair]}/{opportunities[pair]} ({rates[pair]:.4%})" for pair in pairs), flush=True)
        for seed in benchmark_seeds:
            benchmark = run_engine(market, random_entry(rates, seed, cfg), cfg)
            summaries.append({"strategy": "random_entry", "seed": seed, **benchmark.summary})
            if (seed + 1) % 25 == 0:
                print(f"{which}: completed {seed + 1}/{len(benchmark_seeds)} random seeds", flush=True)
    summary = pd.DataFrame(summaries)
    summary.to_csv(results_dir / "engine_summary.csv", index=False, float_format="%.12g")
    print_summaries(summary)
    print("Saved engine_trades_{train,val}.csv and engine_summary.csv. Test remains locked.")


if __name__ == "__main__":
    main()

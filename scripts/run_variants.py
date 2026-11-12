"""Seven pre-registered variants, each with 200 entry-matched random controls."""

from functools import partial
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_engine import percentile_rank
from src.engine import EngineConfig, PreparedMarket, run_engine
from src.loader import load_pair, pairs
from src.strategies import MatchedRandom, build_features, matched_entries, trend_pullback, trend_swing, trend_swing_15m


variants = (
    ("A", "baseline", 2.0),
    ("B1.5", "swing", 1.5), ("B2", "swing", 2.0), ("B3", "swing", 3.0),
    ("C1.5", "swing_15m", 1.5), ("C2", "swing_15m", 2.0), ("C3", "swing_15m", 3.0),
)
benchmark_seeds = tuple(range(200))
results_dir = Path(__file__).resolve().parents[1] / "results"
pass_metric = "expectancy_rank"


def variant_strategy(kind, rr, cfg):
    if kind == "baseline":
        return partial(trend_pullback, cfg=cfg)
    if kind == "swing":
        return partial(trend_swing, rr=rr, cfg=cfg)
    if kind == "swing_15m":
        return trend_swing_15m(rr, cfg)
    raise ValueError(f"Unknown registered variant: {kind}")


def quantiles(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return np.percentile(values, [5, 50, 95]) if len(values) else np.full(3, np.nan)


def summarize_variant(name, kind, rr, result, benchmarks):
    row = {"variant": name, "strategy": kind, "rr": rr, **result.summary}
    trades = result.trades
    row["target_hit_pct"] = (trades.exit_reason == "target").mean() if len(trades) else np.nan
    for reason in ("stop", "target", "time", "end_of_split"):
        sample = trades.loc[trades.exit_reason == reason, "r_multiple"]
        row[f"{reason}_n"] = len(sample)
        row[f"{reason}_mean_r"] = sample.mean()
    for metric, label in (("expectancy_r", "expectancy"), ("ending_balance", "balance"), ("trades", "executed_n")):
        for percentile, value in zip(("p5", "median", "p95"), quantiles(benchmarks[metric])):
            row[f"random_{label}_{percentile}"] = value
    row["expectancy_rank"] = percentile_rank(benchmarks.expectancy_r, result.summary["expectancy_r"])
    row["balance_rank"] = percentile_rank(benchmarks.ending_balance, result.summary["ending_balance"])
    row["matched_attempts"] = len(trades)
    row["random_seeds"] = len(benchmarks)
    row["random_finite_expectancies"] = int(np.isfinite(benchmarks.expectancy_r).sum())
    return row


def apply_pass_rule(summary):
    """Expectancy rank >=90 on both named splits; no other criterion."""
    passed = {}
    for name, rows in summary.groupby("variant", sort=False):
        passed[name] = (set(rows["split"]) == {"train", "val"} and len(rows) == 2 and
                        bool((rows[pass_metric] >= 90).all()))
    result = summary.copy()
    result["pass_both_splits"] = result.variant.map(passed)
    return result


def print_table(summary):
    print("A=unchanged trend_pullback; B=trend_swing; C=trend_swing_15m. S/T/H/E=stop/target/time/split-end.")
    print("Random columns are p5/median/p95; ranks are expectancy-R/balance midranks; exits are count(mean R).")
    print(f"{'v':<4} {'split':<5} {'n':>4} {'win':>6} {'tgt':>6} {'expR':>7} {'PF':>6} {'bal':>8} {'DD$/%':>13} "
          f"{'random_R':>21} {'random_$':>25} {'rankR/$':>11} {'S':>13} {'T':>13} {'H':>13} {'E':>11}")
    for row in summary.itertuples(index=False):
        random_r = "/".join(f"{value:.3f}" for value in (row.random_expectancy_p5, row.random_expectancy_median, row.random_expectancy_p95))
        random_balance = "/".join(f"{value:.2f}" for value in (row.random_balance_p5, row.random_balance_median, row.random_balance_p95))
        drawdown = f"{row.max_drawdown_usd:.2f}/{row.max_drawdown_pct:.1%}"
        ranks = f"{row.expectancy_rank:.1f}/{row.balance_rank:.1f}"
        exits = [f"{getattr(row, reason + '_n')}({getattr(row, reason + '_mean_r'):.3f})"
                 for reason in ("stop", "target", "time", "end_of_split")]
        print(f"{row.variant:<4} {row.split:<5} {row.trades:>4} {row.win_rate:>5.1%} {row.target_hit_pct:>5.1%} "
              f"{row.expectancy_r:>7.3f} {row.profit_factor:>6.3f} {row.ending_balance:>8.2f} {drawdown:>13} "
              f"{random_r:>21} {random_balance:>25} {ranks:>11} "
              f"{exits[0]:>13} {exits[1]:>13} {exits[2]:>13} {exits[3]:>11}")
    passing = summary.loc[summary.pass_both_splits, "variant"].drop_duplicates().tolist()
    print("Pre-registered pass rule: expectancy-R rank >=90 on BOTH train and val. Passing variants: " +
          (", ".join(passing) if passing else "none") + ".")
    print("Matched entries are scheduled attempts; all normal account gates may reject attempts without rescheduling.")


def main():
    cfg = EngineConfig()
    hourly = {pair: load_pair(pair, "1h") for pair in pairs}
    quarters = {pair: load_pair(pair, "15m") for pair in pairs}
    features = {pair: build_features(hourly[pair], pair) for pair in pairs}
    summaries, all_random, all_trades = [], [], []
    results_dir.mkdir(parents=True, exist_ok=True)
    for which in ("train", "val"):
        market = PreparedMarket(hourly, quarters, which, features)
        for name, kind, rr in variants:
            result = run_engine(market, variant_strategy(kind, rr, cfg), cfg)
            all_trades.append(result.trades.assign(variant=name, split=which, rr=rr))
            template = matched_entries(result.trades, cfg)
            benchmarks = []
            for seed in benchmark_seeds:
                replay = run_engine(market, MatchedRandom(template, seed), cfg)
                benchmarks.append({"variant": name, "seed": seed, "matched_attempts": len(template), **replay.summary})
                if (seed + 1) % 50 == 0:
                    print(f"{which} {name}: {len(result.trades)} real trades; {seed + 1}/{len(benchmark_seeds)} matched seeds complete", file=sys.stderr, flush=True)
            benchmark_frame = pd.DataFrame(benchmarks)
            all_random.append(benchmark_frame)
            summaries.append(summarize_variant(name, kind, rr, result, benchmark_frame))
    summary = apply_pass_rule(pd.DataFrame(summaries))
    order = {name: i for i, (name, _, _) in enumerate(variants)}
    summary = summary.assign(_order=summary.variant.map(order)).sort_values(["_order", "split"]).drop(columns="_order")
    summary.to_csv(results_dir / "variants_summary.csv", index=False, float_format="%.12g")
    pd.concat(all_trades, ignore_index=True).to_csv(results_dir / "variants_trades.csv", index=False, float_format="%.12g")
    pd.concat(all_random, ignore_index=True).to_csv(results_dir / "variants_random.csv", index=False, float_format="%.12g")
    print_table(summary)
    print(f"Exactly {len(variants)} variants, train/val only, {len(benchmark_seeds)} seeds per variant/split.")
    print("Saved variants_summary.csv, variants_trades.csv and variants_random.csv. Test remains locked.")


if __name__ == "__main__":
    main()

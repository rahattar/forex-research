"""Frozen seven-variant confirmation on isolated fresh data; primary B2 only."""

from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_variants import variants, variant_strategy
from src.engine import EngineConfig
from src.loader import fresh_data_report, fresh_hourly, fresh_pairs, load_fresh_pair, pairs
from src.signal_eval import FreshMarket, basic_statistics, block_bootstrap, evaluate_signals
from src.strategies import build_features


results_dir = Path(__file__).resolve().parents[1] / "results"
bootstrap_resamples = 5000
bootstrap_seed = 0
start = pd.Timestamp("2010-01-01", tz="UTC")
boundary = pd.Timestamp("2019-01-01", tz="UTC")
end = pd.Timestamp("2025-11-01", tz="UTC")


def summarize_fresh(trades, resamples=5000, seed=0):
    rows = []
    for name, _, _ in variants:
        sample = trades.loc[trades.variant == name]
        rows.append({"variant": name, "group": "all", "value": "pooled", **basic_statistics(sample),
                     **block_bootstrap(sample, start, end, resamples, seed)})
        if name in ("B2", "C2"):
            old = sample.pair.isin(pairs)
            before = sample.entry_time < boundary
            blocks = (("original_2010_2018", old, start, boundary),
                      ("new_2010_2018", ~old & before, start, boundary),
                      ("new_2019_2025", ~old & ~before, boundary, end))
            for label, mask, first, last in blocks:
                subset = sample.loc[mask]
                rows.append({"variant": name, "group": "block", "value": label, **basic_statistics(subset),
                             **block_bootstrap(subset, first, last, resamples, seed)})
        for group, column, categories in (("pair", "pair", fresh_pairs), ("side", "side", ("long", "short"))):
            for value in categories:
                subset = sample.loc[sample[column] == value]
                rows.append({"variant": name, "group": group, "value": value, **basic_statistics(subset)})
        for year in range(2010, 2026):
            subset = sample.loc[sample.entry_time.dt.year == year]
            rows.append({"variant": name, "group": "year", "value": str(year), **basic_statistics(subset)})
    return pd.DataFrame(rows)


def primary_result(summary):
    """The sole pre-registered decision is pooled B2's strict lower-CI bound."""
    primary = summary.loc[(summary.variant == "B2") & (summary['group'] == "all")]
    if len(primary) != 1:
        raise ValueError("Expected exactly one pooled B2 row")
    return "PASS" if primary.iloc[0].d_ci_low > 0 else "FAIL"


def print_main(rows, label):
    print(label)
    print(f"{'v':<4} {'sample':<19} {'n':>6} {'R_strategy [95% CI]':>27} {'R_mirror':>9} "
          f"{'d [95% CI]':>27} {'week_t':>7} {'cost_R':>7} {'MDD_R':>8}")
    for row in rows.itertuples(index=False):
        r = f"{row.mean_r_strategy:.3f} [{row.r_ci_low:.3f},{row.r_ci_high:.3f}]"
        d = f"{row.mean_d:.3f} [{row.d_ci_low:.3f},{row.d_ci_high:.3f}]"
        print(f"{row.variant:<4} {row.value:<19} {row.n:>6} {r:>27} {row.mean_r_mirror:>9.3f} {d:>27} "
              f"{row.weekly_d_t_stat:>7.2f} {row.mean_cost_r:>7.3f} {row.max_drawdown_r:>8.2f}")


def print_tables(summary):
    main = summary.loc[summary['group'] == "all"]
    print_main(main, "Pooled fresh confirmation. A=baseline; B=swing; C=swing_15m. All secondary results are report-only.")
    print("Strategy exits: count(mean R). S/T/H/E = stop/target/time/fold-end; mirror exit details are also saved.")
    print(f"{'v':<4} {'S':>17} {'T':>17} {'H':>17} {'E':>17}")
    for row in main.itertuples(index=False):
        cells = [f"{getattr(row, reason + '_n')}({getattr(row, reason + '_mean_r'):.3f})"
                 for reason in ("stop", "target", "time", "end_of_split")]
        print(f"{row.variant:<4} " + " ".join(f"{cell:>17}" for cell in cells))
    print_main(summary.loc[summary['group'] == "block"], "B2/C2 reporting blocks, by entry date; new-pair paths remain continuous across 2019.")
    for group, categories in (("pair", fresh_pairs), ("side", ("long", "short")), ("year", range(2010, 2026))):
        print(f"Per {group}: cells are n/mean R strategy/mean d. 2025 ends before November; original pairs end in 2018.")
        print(f"{group:<7} " + " ".join(f"{name:>23}" for name, _, _ in variants))
        for value in categories:
            cells = []
            for name, _, _ in variants:
                row = summary.loc[(summary.variant == name) & (summary['group'] == group) & (summary.value == str(value))].iloc[0]
                cells.append(f"{row.n}/{row.mean_r_strategy:.3f}/{row.mean_d:.3f}")
            print(f"{str(value):<7} " + " ".join(f"{cell:>23}" for cell in cells))
    b2 = main.loc[main.variant == "B2"].iloc[0]
    print(f"Primary hypothesis B2: {primary_result(summary)} — pooled mean d={b2.mean_d:.6f}, "
          f"95% calendar-week bootstrap CI [{b2.d_ci_low:.6f}, {b2.d_ci_high:.6f}]; PASS iff lower bound > 0.")
    print(f"Exactly seven frozen variants; {bootstrap_resamples} week-block resamples, seed {bootstrap_seed}; "
          "all pairs kept together within each week. No other selection or tuning.")


def main():
    # All files must validate before the first strategy sees any data.
    fresh_data_report()
    cfg = EngineConfig()
    frames = []
    for pair in fresh_pairs:
        quarters = load_fresh_pair(pair, "15m")
        hourly = fresh_hourly(quarters)
        features = build_features(hourly, pair)
        market = FreshMarket({pair: hourly}, {pair: quarters}, {pair: features})
        for name, kind, rr in variants:
            result = evaluate_signals(market, variant_strategy(kind, rr, cfg), cfg)
            frames.append(result.trades.assign(variant=name, split="fresh", rr=rr))
            print(f"{pair} {name}: {len(result.trades)} strategy/mirror pairs; {result.counts}", file=sys.stderr, flush=True)
        # Pair occupancy is independent; processing pairs separately bounds memory.
        del market, quarters, hourly, features
    trades = pd.concat(frames, ignore_index=True)
    trades = trades.sort_values(["variant", "entry_time", "pair"]).reset_index(drop=True)
    summary = summarize_fresh(trades, bootstrap_resamples, bootstrap_seed)
    results_dir.mkdir(parents=True, exist_ok=True)
    trades.to_csv(results_dir / "fresh_signal_trades.csv", index=False, float_format="%.12g")
    summary.to_csv(results_dir / "fresh_signal_summary.csv", index=False, float_format="%.12g")
    print_tables(summary)
    print("Saved fresh_signal_trades.csv and fresh_signal_summary.csv. Existing train/val roots and the locked test were not loaded.")


if __name__ == "__main__":
    main()

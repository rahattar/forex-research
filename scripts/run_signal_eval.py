"""Seven unchanged variants, no account gates, exact mirrors, weekly bootstrap."""

from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_variants import variants, variant_strategy
from src.engine import EngineConfig, PreparedMarket
from src.loader import load_pair, pairs
from src.signal_eval import engine_consistency, evaluate_signals, summarize
from src.strategies import build_features


results_dir = Path(__file__).resolve().parents[1] / "results"
bootstrap_resamples = 5000
bootstrap_seed = 0


def print_tables(summary):
    main = summary.loc[summary['group'] == "all"]
    print("Account-free signal evaluation; confidence intervals are 95% calendar-week block bootstrap intervals.")
    print(f"{'v':<4} {'period':<6} {'n':>5} {'R_strategy [CI]':>27} {'R_mirror':>9} {'d [CI]':>27} {'cost_R':>7} {'weekly_t':>8} {'MDD_R':>8}")
    for row in main.itertuples(index=False):
        r = f"{row.mean_r_strategy:.3f} [{row.r_ci_low:.3f},{row.r_ci_high:.3f}]"
        d = f"{row.mean_d:.3f} [{row.d_ci_low:.3f},{row.d_ci_high:.3f}]"
        print(f"{row.variant:<4} {row.period:<6} {row.n:>5} {r:>27} {row.mean_r_mirror:>9.3f} {d:>27} "
              f"{row.mean_cost_r:>7.3f} {row.weekly_d_t_stat:>8.2f} {row.max_drawdown_r:>8.2f}")
    print("Strategy exits: count(mean R); S/T/H/E = stop/target/time/split-end. Mirror breakdown is also in the CSV.")
    print(f"{'v':<4} {'period':<6} {'S':>15} {'T':>15} {'H':>15} {'E':>13}")
    for row in main.itertuples(index=False):
        cells = [f"{getattr(row, reason + '_n')}({getattr(row, reason + '_mean_r'):.3f})"
                 for reason in ("stop", "target", "time", "end_of_split")]
        print(f"{row.variant:<4} {row.period:<6} " + " ".join(f"{cell:>15}" for cell in cells))
    print("Year-by-year, by entry year: cells are n/mean R strategy/mean d; 2025 is partial validation.")
    print(f"{'v':<4} " + " ".join(f"{year:>20}" for year in range(2019, 2026)))
    for variant in main.variant.drop_duplicates():
        sample = summary.loc[(summary.variant == variant) & (summary['group'] == "year")].set_index("value")
        cells = [f"{int(sample.loc[str(year), 'n'])}/{sample.loc[str(year), 'mean_r_strategy']:.3f}/{sample.loc[str(year), 'mean_d']:.3f}"
                 for year in range(2019, 2026)]
        print(f"{variant:<4} " + " ".join(f"{cell:>20}" for cell in cells))
    print("Pooled pair/side detail: cells are n/mean R strategy/mean d. Joint pair-side and split details are in the CSV.")
    print(f"{'v':<4} " + " ".join(f"{value:>21}" for value in (*pairs, "long", "short")))
    for variant in main.variant.drop_duplicates():
        sample = summary.loc[(summary.variant == variant) & (summary.period == "pooled")]
        cells = []
        for group, value in [("pair", pair) for pair in pairs] + [("side", "long"), ("side", "short")]:
            row = sample.loc[(sample['group'] == group) & (sample.value == value)].iloc[0]
            cells.append(f"{row.n}/{row.mean_r_strategy:.3f}/{row.mean_d:.3f}")
        print(f"{variant:<4} " + " ".join(f"{cell:>21}" for cell in cells))
    print("Engine consistency: common entries, maximum absolute raw R difference, then FX-normalized difference.")
    print(f"{'v':<4} {'period':<6} {'matches/engine':>14} {'raw_max':>12} {'fx_normalized':>14} {'path_mismatch':>13}")
    for row in main.itertuples(index=False):
        count = f"{int(row.engine_matches)}/{int(row.engine_trades)}"
        print(f"{row.variant:<4} {row.period:<6} {count:>14} {row.engine_max_abs_r_diff:>12.3g} "
              f"{row.engine_max_abs_r_diff_fx_normalized:>14.3g} {int(row.engine_exit_path_mismatches):>13}")
    positive = main.loc[main.pooled_d_ci_above_zero, "variant"].tolist()
    print("Report-only variants with pooled 95% mean-d interval entirely above zero: " + (", ".join(positive) if positive else "none") + ".")


def main():
    cfg = EngineConfig()
    hourly = {pair: load_pair(pair, "1h") for pair in pairs}
    quarters = {pair: load_pair(pair, "15m") for pair in pairs}
    features = {pair: build_features(hourly[pair], pair) for pair in pairs}
    frames = []
    for which in ("train", "val"):
        market = PreparedMarket(hourly, quarters, which, features)
        for name, kind, rr in variants:
            result = evaluate_signals(market, variant_strategy(kind, rr, cfg), cfg)
            frames.append(result.trades.assign(variant=name, split=which, rr=rr))
            print(f"{which} {name}: {len(result.trades)} strategy/mirror pairs; {result.counts}", file=sys.stderr, flush=True)
    trades = pd.concat(frames, ignore_index=True)
    variant_names = [name for name, _, _ in variants]
    summary = summarize(trades, bootstrap_resamples, bootstrap_seed, variant_names)
    reference = pd.read_csv(results_dir / "variants_trades.csv", parse_dates=["entry_time", "exit_time"])
    consistency = engine_consistency(trades, reference, cfg, variant_names)
    summary = summary.merge(consistency, on=["variant", "period"], how="left", validate="many_to_one")
    summary.loc[summary['group'] != "all", consistency.columns.difference(["variant", "period"])] = float("nan")
    results_dir.mkdir(parents=True, exist_ok=True)
    trades.to_csv(results_dir / "signal_trades.csv", index=False, float_format="%.12g")
    summary.to_csv(results_dir / "signal_summary.csv", index=False, float_format="%.12g")
    print_tables(summary)
    print(f"Exactly seven existing variants; {bootstrap_resamples} bootstrap resamples, seed {bootstrap_seed}.")
    print("No account state or account gates; only Friday entry exclusion and one open strategy trade per pair.")
    print("Saved signal_trades.csv and signal_summary.csv. Test remains locked; fresh/ and data/ are never read.")


if __name__ == "__main__":
    main()

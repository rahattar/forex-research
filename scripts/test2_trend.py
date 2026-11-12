"""Completed-daily trend and hourly pullback diagnostics; train/val only."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.test1_residual import forward_move, friday_cutoff_hour, residual_features, trade_statistics
from src.instruments import Config
from src.loader import daily_bars, load_pair, pairs, split


horizons = (24, 48, 120, 240)
spans = (24, 72)
thresholds = (0.5, 1.0, 1.5)
rules = ("trend", "pullback")
result_path = Path(__file__).resolve().parents[1] / "results" / "test2_grid.csv"


def daily_trend(hourly):
    """State is available at daily completion; NaN until 200 completed days."""
    daily = daily_bars(hourly)
    close = daily["close"]
    sma50 = close.rolling(50, min_periods=50).mean()
    sma200 = close.rolling(200, min_periods=200).mean()
    state = pd.Series(0.0, index=daily.index, name="state")
    state.loc[(close > sma50) & (sma50 > sma200)] = 1
    state.loc[(close < sma50) & (sma50 < sma200)] = -1
    return state.where(sma200.notna())


def entry_allowed(index):
    return ~((index.dayofweek == 4) & (index.hour >= friday_cutoff_hour))


def signal_returns(hourly, signals, h, pair, which, cfg):
    """Execute strictly after the signal, with signal/entry/exit in one split."""
    close = split(hourly, which)["close"]
    signals = split(signals.to_frame("side"), which)["side"].dropna()
    positions = close.index.searchsorted(signals.index, side="right")
    keep = positions < len(close)
    entries = close.index.take(positions[keep])
    move = forward_move(close, h, pair).reindex(entries)
    gross = pd.Series(move.to_numpy() * signals.to_numpy()[keep], index=entries)
    gross = gross.loc[gross.notna() & entry_allowed(gross.index)]
    return pd.DataFrame({"gross": gross, "net": gross - cfg.spread_pips[pair]})


def pooled_statistics(samples, h):
    """Pool individual returns; thin each pair separately before pooling t data."""
    gross = pd.concat([sample["gross"] for sample in samples], ignore_index=True)
    net = pd.concat([sample["net"] for sample in samples], ignore_index=True)
    independent = pd.concat([sample["net"].iloc[::h] for sample in samples], ignore_index=True)
    stats = trade_statistics(gross, gross - net, h=1)
    thinned = trade_statistics(independent, spread=0, h=1)
    stats["n_nonoverlap"] = thinned["n_trades"]
    stats["mean_net_nonoverlap_pips"] = thinned["mean_net_pips"]
    stats["t_stat"] = thinned["t_stat"]
    return stats


def build_grid(cfg=None):
    cfg = Config() if cfg is None else cfg
    rows, pool = [], {}
    configs = [("trend", 0, 0.0)] + [
        ("pullback", span, k) for span in spans for k in thresholds
    ]
    for pair in pairs:
        hourly = load_pair(pair, "1h")
        hourly = pd.concat([split(hourly, "train"), split(hourly, "val")])
        states = daily_trend(hourly)
        hourly_states = states.reindex(hourly.index, method="ffill")
        signals = {("trend", 0, 0.0): states.loc[states.notna() & states.ne(0)]}
        for span in spans:
            z = residual_features(hourly["close"], span)["z"]
            for k in thresholds:
                signals[("pullback", span, k)] = hourly_states.loc[hourly_states * z < -k]
        for rule, span, k in configs:
            for h in horizons:
                for which in ("train", "val"):
                    sample = signal_returns(hourly, signals[(rule, span, k)], h, pair, which, cfg)
                    key = (rule, span, k, h, which)
                    pool.setdefault(key, []).append(sample)
                    rows.append({
                        "pair": pair, "rule": rule, "span": span, "h": h, "k": k,
                        "split": which, "spread_pips": cfg.spread_pips[pair],
                        **trade_statistics(sample["gross"], cfg.spread_pips[pair], h),
                    })
        # Drift baselines share the post-200-day history and entry-hour filter.
        for which in ("train", "val"):
            part = split(hourly, which)
            ready = hourly_states.reindex(part.index).notna() & entry_allowed(part.index)
            for h in horizons:
                move = forward_move(part["close"], h, pair).loc[ready].dropna()
                for rule, direction in (("always_long", 1), ("always_short", -1)):
                    rows.append({
                        "pair": pair, "rule": rule, "span": 0, "h": h, "k": 0.0,
                        "split": which, "spread_pips": cfg.spread_pips[pair],
                        **trade_statistics(direction * move, cfg.spread_pips[pair], h),
                    })
    for (rule, span, k, h, which), samples in pool.items():
        rows.append({
            "pair": "pooled", "rule": rule, "span": span, "h": h, "k": k,
            "split": which, "spread_pips": np.nan, **pooled_statistics(samples, h),
        })
    return pd.DataFrame(rows)


def compact_summary(grid, pooled=False, count=5):
    keys = ["pair", "rule", "span", "h", "k"]
    metrics = ["n_trades", "mean_gross_pips", "mean_net_pips", "hit_rate", "t_stat", "n_nonoverlap"]
    eligible = grid.loc[grid["rule"].isin(rules) & (grid["pair"].eq("pooled") == pooled)]
    train = eligible.loc[eligible["split"] == "train", keys + metrics]
    train = train.sort_values(
        ["rule", "mean_net_pips", "pair", "span", "h", "k"],
        ascending=[True, False, True, True, True, True], na_position="last",
    ).groupby("rule", sort=False).head(count)
    val = eligible.loc[eligible["split"] == "val", keys + metrics]
    return train.merge(val, on=keys, suffixes=("_train", "_val"), validate="one_to_one")


def print_summary(summary):
    print(
        f"{'pair':<7} {'rule':<8} {'span':>4} {'h':>3} {'k':>3} "
        f"{'train_n':>7} {'val_n':>6} {'gross_tr':>8} {'gross_va':>8} "
        f"{'net_tr':>8} {'net_va':>8} {'hit_tr':>7} {'hit_va':>7} {'t_tr(n)':>12} {'t_va(n)':>12}"
    )
    for row in summary.itertuples(index=False):
        print(
            f"{row.pair:<7} {row.rule:<8} {row.span:>4} {row.h:>3} {row.k:>3.1f} "
            f"{row.n_trades_train:>7} {row.n_trades_val:>6} "
            f"{row.mean_gross_pips_train:>8.2f} {row.mean_gross_pips_val:>8.2f} "
            f"{row.mean_net_pips_train:>8.2f} {row.mean_net_pips_val:>8.2f} "
            f"{row.hit_rate_train:>6.1%} {row.hit_rate_val:>6.1%} "
            f"{f'{row.t_stat_train:.2f}({row.n_nonoverlap_train})':>12} "
            f"{f'{row.t_stat_val:.2f}({row.n_nonoverlap_val})':>12}"
        )


def print_baseline(grid):
    print("Unconditional hourly baseline after warm-up: gross/net pips, long (L) and short (S).")
    print(f"{'pair':<7} {'h':>3} {'train_n':>7} {'val_n':>6} {'train_L':>15} {'train_S':>15} {'val_L':>15} {'val_S':>15}")
    for pair in pairs:
        for h in horizons:
            subset = grid.loc[(grid["pair"] == pair) & (grid["h"] == h)]
            cells, counts = [], []
            for which in ("train", "val"):
                for side in ("long", "short"):
                    row = subset.loc[(subset["split"] == which) & (subset["rule"] == f"always_{side}")].iloc[0]
                    cells.append(f"{row.mean_gross_pips:.2f}/{row.mean_net_pips:.2f}")
                    if side == "long":
                        counts.append(row.n_trades)
            print(f"{pair:<7} {h:>3} {counts[0]:>7} {counts[1]:>6} " + " ".join(f"{cell:>15}" for cell in cells))


def main():
    grid = build_grid()
    result_path.parent.mkdir(parents=True, exist_ok=True)
    grid.to_csv(result_path, index=False, float_format="%.10g")
    print("Top 5 TRAIN rows per rule across pairs, ranked by mean net pips:")
    print_summary(compact_summary(grid))
    print("Pooled-over-pairs: best TRAIN configuration per rule, weighted by trade count:")
    print_summary(compact_summary(grid, pooled=True, count=1))
    print_baseline(grid)
    trials = grid.loc[grid["pair"].isin(pairs) & grid["rule"].isin(rules)]
    count = len(trials[["pair", "rule", "span", "h", "k"]].drop_duplicates())
    print(f"Trend combinations tested: {count}; pooled rows and 24 baseline combinations are descriptive.")
    print("T-stat uses every h-th qualifying observation within each pair; NaN means insufficient/constant data.")
    print("Horizons are elapsed UTC hours; entries from Friday 20:00 UTC are excluded. No multiple-test correction.")
    print("Saved results/test2_grid.csv; final test remains locked.")


if __name__ == "__main__":
    main()

"""Causal residual diagnostics on train/validation only; no trading engine."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.instruments import Config, pip_size
from src.loader import load_pair, pairs, split


spans = (24, 72, 120)
horizons = (1, 4, 12, 24)
thresholds = (1.0, 1.5, 2.0)
rules = ("fade", "momentum")
friday_cutoff_hour = 20
result_path = Path(__file__).resolve().parents[1] / "results" / "test1_grid.csv"


def residual_features(close, span, window=240):
    """Use observed completed bars; the scale excludes the current change.

    Both residual and scale are in price units. Sample standard deviation uses
    a full window of past changes; a zero scale leaves z undefined.
    """
    ema = close.ewm(span=span, adjust=False).mean()
    residual = close - ema
    scale = close.diff().rolling(window, min_periods=window).std(ddof=1).shift(1)
    z = residual / scale.where(scale > 0)
    return pd.DataFrame({"close": close, "residual": residual, "scale": scale, "z": z})


def forward_move(close, h, pair):
    """Pip move to the exact UTC timestamp h hours later, within this split."""
    future = close.reindex(close.index + pd.Timedelta(hours=h))
    future = pd.Series(future.to_numpy(), index=close.index)
    return (future - close) / pip_size(pair)


def spearman(left, right):
    """Average ranks followed by Pearson correlation, without a scipy dependency."""
    values = pd.concat([left, right], axis=1).dropna()
    if len(values) < 2 or (values.nunique() < 2).any():
        return np.nan
    return float(values.iloc[:, 0].rank().corr(values.iloc[:, 1].rank()))


def trade_statistics(gross, spread, h):
    """All qualifying trades for averages; every h-th trade for the net t-stat.

    Qualified entries are at least one hour apart, so this subsample cannot have
    overlapping h-hour forward intervals. No p-value or multiple-test correction
    is inferred from this diagnostic statistic.
    """
    net = gross - spread
    independent = net.iloc[::h]
    count = len(independent)
    standard_error = independent.std(ddof=1) / np.sqrt(count) if count >= 2 else np.nan
    t_stat = independent.mean() / standard_error if standard_error > 0 else np.nan
    return {
        "n_trades": len(gross),
        "mean_gross_pips": gross.mean(),
        "mean_net_pips": net.mean(),
        "hit_rate": (net > 0).mean() if len(net) else np.nan,
        "n_nonoverlap": count,
        "mean_net_nonoverlap_pips": independent.mean(),
        "t_stat": t_stat,
    }


def pair_grid(pair, hourly, cfg):
    """Keep labels within splits and warm validation features on train.

    The residual-change autocorrelation uses h observed-bar lags.
    """
    allowed = pd.concat([split(hourly, "train"), split(hourly, "val")])
    rows = []
    for span in spans:
        features = residual_features(allowed["close"], span)
        for which in ("train", "val"):
            part = split(features, which)
            residual_changes = part["residual"].diff()
            for h in horizons:
                forward = forward_move(part["close"], h, pair)
                valid = part["z"].notna() & forward.notna()
                z, move = part.loc[valid, "z"], forward.loc[valid]
                rho = spearman(z, move)
                acf = residual_changes.autocorr(lag=h) if len(residual_changes) > h + 1 else np.nan
                entry_allowed = ~((z.index.dayofweek == 4) & (z.index.hour >= friday_cutoff_hour))
                for k in thresholds:
                    selected = (z.abs() > k) & entry_allowed
                    momentum = np.sign(z.loc[selected]) * move.loc[selected]
                    for rule in rules:
                        gross = -momentum if rule == "fade" else momentum
                        rows.append({
                            "pair": pair, "span": span, "h": h, "k": k,
                            "rule": rule, "split": which, "n_observations": len(z),
                            "spearman": rho, "residual_change_acf_h_bars": acf,
                            "spread_pips": cfg.spread_pips[pair],
                            **trade_statistics(gross, cfg.spread_pips[pair], h),
                        })
    return pd.DataFrame(rows)


def build_grid(cfg=None):
    cfg = Config() if cfg is None else cfg
    return pd.concat([pair_grid(pair, load_pair(pair, "1h"), cfg) for pair in pairs], ignore_index=True)


def compact_summary(grid):
    """Select five candidates per pair/rule by training mean net pips only."""
    keys = ["pair", "rule", "span", "h", "k"]
    metrics = ["n_trades", "mean_net_pips", "hit_rate", "t_stat"]
    train = grid.loc[grid["split"] == "train", keys + metrics]
    train = train.sort_values(
        ["pair", "rule", "mean_net_pips", "span", "h", "k"],
        ascending=[True, True, False, True, True, True], na_position="last",
    ).groupby(["pair", "rule"], sort=False).head(5)
    val = grid.loc[grid["split"] == "val", keys + metrics]
    return train.merge(val, on=keys, suffixes=("_train", "_val"), validate="one_to_one")


def print_summary(summary):
    print("Best 5 per pair/rule by TRAIN mean net pips; validation does not select rows.")
    print(
        f"{'pair':<7} {'rule':<8} {'span':>4} {'h':>2} {'k':>3} "
        f"{'train_n':>7} {'val_n':>6} {'train_net':>9} {'val_net':>8} "
        f"{'train_hit':>9} {'val_hit':>8} {'train_t':>8} {'val_t':>7}"
    )
    for row in summary.itertuples(index=False):
        print(
            f"{row.pair:<7} {row.rule:<8} {row.span:>4} {row.h:>2} {row.k:>3.1f} "
            f"{row.n_trades_train:>7} {row.n_trades_val:>6} "
            f"{row.mean_net_pips_train:>9.3f} {row.mean_net_pips_val:>8.3f} "
            f"{row.hit_rate_train:>8.1%} {row.hit_rate_val:>8.1%} "
            f"{row.t_stat_train:>8.2f} {row.t_stat_val:>7.2f}"
        )


def main():
    grid = build_grid()
    result_path.parent.mkdir(parents=True, exist_ok=True)
    grid.to_csv(result_path, index=False, float_format="%.10g")
    print_summary(compact_summary(grid))
    count = len(grid[["pair", "span", "h", "k", "rule"]].drop_duplicates())
    print(f"Total combinations tested: {count} ({len(grid)} rows across train and val).")
    print("Many tests can produce chance false positives; no multiple-testing correction applied.")
    print("Net/hit rates subtract one spread only; t-stat uses every h-th qualifying net return.")
    print("Horizons are elapsed UTC hours; Friday entries from 20:00 UTC are excluded.")
    print("Saved results/test1_grid.csv; the final test split remains locked.")


if __name__ == "__main__":
    main()

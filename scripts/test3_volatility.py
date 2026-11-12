"""Causal next-24-observed-bar range forecasts, with train-only seasonality."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.instruments import pip_size
from src.loader import load_pair, pairs, split


days = (5, 10, 20)
window = 24
result_path = Path(__file__).resolve().parents[1] / "results" / "test3_vol.csv"


def past_range(hourly, pair):
    """Range of the 24 completed bars ending at the signal time, in pips."""
    return (hourly["high"].rolling(window).max() - hourly["low"].rolling(window).min()) / pip_size(pair)


def future_range(hourly, pair):
    """Target uses bars t+1 through t+24 and excludes the completed signal bar."""
    return past_range(hourly, pair).shift(-window)


def seasonal_factor(hourly):
    """Prior-train mean relative hourly range, by UTC hour, normalized to mean 1.

    Relative range divides high-low by its trailing 24-bar mean. At each train
    origin, only earlier observations fit the profile. Training ends freeze the
    estimates for validation. Unseen hours initially receive factor one.
    """
    ranges = hourly["high"] - hourly["low"]
    scale = ranges.rolling(window).mean()
    relative = ranges / scale.where(scale > 0)
    train_index = split(hourly, "train").index
    values = relative.where(hourly.index.isin(train_index))
    valid = values.notna().astype(int)
    filled = values.fillna(0)
    hours = hourly.index.hour
    hour_sum = filled.groupby(hours).cumsum() - filled
    hour_count = valid.groupby(hours).cumsum() - valid
    total_sum = filled.cumsum() - filled
    total_count = valid.cumsum() - valid
    overall = total_sum / total_count.replace(0, np.nan)
    factor = (hour_sum / hour_count.replace(0, np.nan)) / overall.where(overall > 0)
    return factor.fillna(1.0)


def volatility_forecasts(hourly, pair):
    """Fixed EMA spans update on completed ranges; no validation fitting."""
    naive = past_range(hourly, pair)
    forecasts = {"naive": naive}
    profile = seasonal_factor(hourly)
    for span in days:
        ewma = naive.ewm(span=span * 24, adjust=False).mean()
        forecasts[f"ewma_{span}d"] = ewma
        forecasts[f"seasonal_{span}d"] = ewma * profile
    return pd.DataFrame(forecasts)


def forecast_metrics(actual, predicted, naive_mae):
    errors = actual - predicted
    mae = errors.abs().mean()
    total_variance = ((actual - actual.mean()) ** 2).sum()
    return {
        "n": len(actual),
        "mae_pips": mae,
        "r_squared": 1 - (errors ** 2).sum() / total_variance if total_variance > 0 else np.nan,
        "mae_ratio": mae / naive_mae if naive_mae > 0 else np.nan,
        "exceed_1x": (actual > predicted).mean() if len(actual) else np.nan,
        "exceed_1_5x": (actual > 1.5 * predicted).mean() if len(actual) else np.nan,
        "exceed_2x": (actual > 2 * predicted).mean() if len(actual) else np.nan,
    }


def pair_report(pair, hourly):
    hourly = pd.concat([split(hourly, "train"), split(hourly, "val")])
    forecasts = volatility_forecasts(hourly, pair)
    rows = []
    for which in ("train", "val"):
        part = split(hourly, which)
        actual = future_range(part, pair)
        predicted = forecasts.reindex(part.index)
        valid = actual.notna() & predicted.notna().all(axis=1)
        actual, predicted = actual.loc[valid], predicted.loc[valid]
        naive_mae = (actual - predicted["naive"]).abs().mean()
        for model in predicted:
            rows.append({"pair": pair, "model": model, "split": which,
                         **forecast_metrics(actual, predicted[model], naive_mae)})
    return pd.DataFrame(rows)


def compact_table(report):
    keys = ["pair", "model"]
    train = report.loc[report["split"] == "train"].drop(columns="split")
    val = report.loc[report["split"] == "val"].drop(columns="split")
    return train.merge(val, on=keys, suffixes=("_train", "_val"), validate="one_to_one")


def print_table(table):
    print("Next-24-bar range forecasts: MAE in pips; exceedance fractions are validation only.")
    print(f"{'pair':<7} {'model':<13} {'mae_tr':>7} {'mae_va':>7} {'r2_tr':>7} {'r2_va':>7} {'ratio_tr':>8} {'ratio_va':>8} {'>1x':>6} {'>1.5x':>6} {'>2x':>6}")
    for row in table.itertuples(index=False):
        print(f"{row.pair:<7} {row.model:<13} {row.mae_pips_train:>7.2f} {row.mae_pips_val:>7.2f} "
              f"{row.r_squared_train:>7.3f} {row.r_squared_val:>7.3f} {row.mae_ratio_train:>8.3f} {row.mae_ratio_val:>8.3f} "
              f"{row.exceed_1x_val:>5.1%} {row.exceed_1_5x_val:>5.1%} {row.exceed_2x_val:>5.1%}")


def main():
    report = pd.concat([pair_report(pair, load_pair(pair, "1h")) for pair in pairs], ignore_index=True)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(result_path, index=False, float_format="%.10g")
    print_table(compact_table(report))
    print(f"Volatility configurations evaluated: {len(report[['pair', 'model']].drop_duplicates())}.")
    print("24 bars may span weekends; EWMA spans are days * 24 observed bars.")
    print("Seasonality uses prior training hourly ranges only, frozen on validation; EWMA state updates causally.")
    print("Saved results/test3_vol.csv; final test remains locked.")


if __name__ == "__main__":
    main()

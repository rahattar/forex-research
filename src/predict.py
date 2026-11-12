"""Frozen Task 15 features, fitted forecasts, scoring, and weekly inference."""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .instruments import Config, pip_size
from .loader import fresh_pairs
from .predict_data import external_inputs


usd_signs = dict(eurusd=-1, gbpusd=-1, usdjpy=1, audusd=-1, usdcad=1, usdchf=1, nzdusd=-1,
                 eurjpy=0, gbpjpy=0, eurgbp=0)
horizons = (1, 4, 24)
alphas = (1e2, 1e3, 1e4, 1e5, 1e6)
capacities = ((3, 100), (6, 300))


def hourly_scale(bars):
    return np.sqrt(np.log(bars.close).diff().pow(2).ewm(span=240, adjust=False, min_periods=240).mean())


def inputs_g1(bars):
    x = np.log(bars.close)
    r2 = x.diff().pow(2)
    s = hourly_scale(bars)
    features = {"move_" + str(k): x.diff(k) / (s * np.sqrt(k)) for k in (1, 4, 24, 120, 480)}
    features.update({"ema_" + str(k): (x - x.ewm(span=k, adjust=False).mean()) / (s * np.sqrt(k)) for k in (24, 120)})
    for k in (24, 120, 480):
        low, high = bars.low.rolling(k).min(), bars.high.rolling(k).max()
        features["channel_" + str(k)] = ((bars.close - low) / (high - low) - .5).mask(high == low, 0)
    features["vol_state"] = .5 * np.log(r2.ewm(span=24, adjust=False).mean() / r2.ewm(span=480, adjust=False).mean())
    frame = pd.DataFrame(features).replace([np.inf, -np.inf], np.nan)
    frame.iloc[:480] = np.nan
    return frame


def inputs_g2(bars):
    components = [usd_signs[pair] * np.log(frame.close).diff() / hourly_scale(frame)
                  for pair, frame in bars.items() if usd_signs[pair]]
    u = pd.concat(components, axis=1).mean(axis=1).sort_index()
    factor = pd.DataFrame({"usd_" + str(k): u.rolling(k, min_periods=k).sum() / np.sqrt(k) for k in (1, 4, 24, 120)})
    results = {}
    for pair, frame in bars.items():
        own = factor.reindex(frame.index).copy()
        own["usd_sign"] = usd_signs[pair]
        for k in (1, 4, 24, 120):
            own["signed_usd_" + str(k)] = own["usd_" + str(k)] * usd_signs[pair]
        results[pair] = own
    return results


def targets(bars, h):
    if h not in horizons:
        raise ValueError("Unregistered horizon")
    times = bars.index
    endpoint = times + pd.Timedelta(hours=h)
    future = bars.close.reindex(endpoint).to_numpy()
    s = hourly_scale(bars)
    z = (np.log(future) - np.log(bars.close.to_numpy())) / (s.to_numpy() * np.sqrt(h))
    eligible = np.isfinite(z) & (s.to_numpy() > 0) & (times.hour <= 20) & (endpoint.hour <= 20)
    hours = times.hour
    grid = hours <= 19 if h == 1 else np.isin(hours, (0, 4, 8, 12, 16) if h == 4 else (8,))
    return pd.DataFrame({"target": z, "endpoint": endpoint, "eligible": eligible, "grid": grid & eligible}, index=times)


def expected_cost(times, h, close, s, pair, cfg=None):
    """Vectorized version of costs.swap_pips for the three fixed horizons."""
    cfg = Config() if cfg is None else cfg
    end = times + pd.Timedelta(hours=h)
    nights = np.zeros(len(times))
    for day in range(h // 24 + 1):
        rollover = times.normalize() + pd.Timedelta(days=day, hours=22)
        touched = (rollover > times) & (rollover <= end) & (rollover.dayofweek < 5)
        nights += touched * np.where(rollover.dayofweek == 2, 3, 1)
    expected = np.asarray(s) * np.sqrt(h) * np.asarray(close) / pip_size(pair)
    return (cfg.spread_pips[pair] - cfg.swap_pips_per_night * nights) / expected


def build_table(bars, external):
    """Build all inputs once; horizons and model variants share these columns."""
    factors = inputs_g2(bars)
    union = pd.DatetimeIndex(sorted(set().union(*(frame.index for frame in bars.values()))))
    _, other = external_inputs(external, union)
    pieces = []
    for pair, frame in bars.items():
        own = inputs_g1(frame)
        part = pd.concat([own, factors[pair], other.reindex(frame.index)], axis=1).astype("float32")
        part["pair"] = fresh_pairs.index(pair)
        part["close"] = frame.close
        part["s"] = hourly_scale(frame)
        for h in horizons:
            response = targets(frame, h)
            part["z" + str(h)] = response.target
            part["eligible" + str(h)] = response.eligible
            part["grid" + str(h)] = response.grid
            part["cost" + str(h)] = expected_cost(frame.index, h, frame.close, part.s, pair)
        pieces.append(part)
    result = pd.concat(pieces).sort_index(kind="stable")
    primary = list(own.columns) + list(factors[pair].columns) + list(other.columns)
    for hour in range(24):
        result["hour_" + str(hour)] = (result.index.hour == hour).astype("float32")
    for day in range(5):
        result["day_" + str(day)] = (result.index.dayofweek == day).astype("float32")
    result["clock_hour"] = result.index.hour.astype("float32")
    result["clock_day"] = result.index.dayofweek.astype("float32")
    return result, {"primary": primary, "g1": list(own.columns),
                    "r1_m1": primary + ["hour_" + str(h) for h in range(24)] + ["day_" + str(d) for d in range(5)],
                    "r1_m2": primary + ["clock_hour", "clock_day"]}


@dataclass
class Ridge:
    mean: np.ndarray
    scale: np.ndarray
    keep: np.ndarray
    beta: np.ndarray
    intercept: float


def fit_ridge(x, y, alpha):
    x, y = np.asarray(x, dtype=np.float64), np.clip(np.asarray(y, dtype=np.float64), -5, 5)
    if len(y) < 2:
        raise ValueError("Ridge requires at least two training rows")
    mean, scale = x.mean(axis=0), x.std(axis=0, ddof=1)
    keep = (np.ptp(x, axis=0) > 0) & (scale > 0) & np.isfinite(scale)
    standardized = (x[:, keep] - mean[keep]) / scale[keep]
    centered = y - y.mean()
    gram = standardized.T @ standardized
    beta = np.linalg.solve(gram + alpha * np.eye(keep.sum()), standardized.T @ centered)
    return Ridge(mean, scale, keep, beta, float(y.mean()))


def forecast_ridge(model, x):
    x = np.asarray(x, dtype=np.float64)
    return model.intercept + ((x[:, model.keep] - model.mean[model.keep]) / model.scale[model.keep]) @ model.beta


def fit_trees(x, y, pair, capacity):
    x = np.asarray(x)
    model = HistGradientBoostingRegressor(loss="squared_error", learning_rate=.05, min_samples_leaf=2000,
                                         early_stopping=False, random_state=0, max_depth=capacity[0], max_iter=capacity[1],
                                         categorical_features=[False] * x.shape[1] + [True])
    return model.fit(np.column_stack([x, pair]), np.clip(y, -5, 5))


def forecast_trees(model, x, pair):
    return model.predict(np.column_stack([x, pair]))


def correlation(x, y):
    if len(x) < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def usable(table, columns, h):
    return table["eligible" + str(h)].to_numpy() & np.isfinite(table[columns].to_numpy()).all(axis=1)


def select_once(table, columns, h, model_name, progress=None):
    """Select only within the initial window, purging endpoints at its cut."""
    x = table[columns].to_numpy()
    y, pairs = table["z" + str(h)].to_numpy(), table.pair.to_numpy()
    ends = table.index + pd.Timedelta(hours=h)
    initial = usable(table, columns, h) & (ends <= pd.Timestamp("2014-01-01", tz="UTC"))
    initial_times = table.index[initial].sort_values()
    if len(initial_times) == 0:
        raise ValueError("No initial training rows")
    cut = initial_times[int(np.ceil(.8 * len(initial_times))) - 1]
    train = initial & (ends <= cut)
    hold = initial & (table.index > cut)
    choices, records = {}, []
    groups = sorted(np.unique(pairs[initial])) if model_name == "m1" else [None]
    for group in groups:
        a, b = train.copy(), hold.copy()
        if group is not None:
            a &= pairs == group
            b &= pairs == group
        candidates = tuple(reversed(alphas)) if model_name == "m1" else capacities
        best, best_score = candidates[0], -np.inf
        for choice in candidates:
            fitted = fit_ridge(x[a], y[a], choice) if model_name == "m1" else fit_trees(x[a], y[a], pairs[a], choice)
            forecasts = (forecast_ridge(fitted, x[b]) if model_name == "m1" else forecast_trees(fitted, x[b], pairs[b])) if b.any() else np.empty(0)
            corr = correlation(forecasts, np.clip(y[b], -5, 5))
            ranking = corr if np.isfinite(corr) else -np.inf
            if ranking > best_score:
                best, best_score = choice, ranking
            if progress is not None:
                progress("selection %s h=%d pair=%s choice=%s" % (model_name, h, "pooled" if group is None else fresh_pairs[group], choice))
        choices[group] = best
        records.append({"pair": "pooled" if group is None else fresh_pairs[group], "choice": best,
                        "correlation": best_score if np.isfinite(best_score) else np.nan, "cut": cut,
                        "train_rows": int(a.sum()), "holdout_rows": int(b.sum()),
                        "last_training_endpoint": ends[a].max()})
    return {"choices": choices, "records": records, "cut": cut}


def walk_forward(table, columns, h, model_name, selection, progress=None, years=range(2014, 2026)):
    x = table[columns].to_numpy()
    y, pairs = table["z" + str(h)].to_numpy(), table.pair.to_numpy()
    eligible = usable(table, columns, h)
    ends = table.index + pd.Timedelta(hours=h)
    predictions = np.full(len(table), np.nan)
    last, audit = {}, []
    for year in years:
        boundary = pd.Timestamp("%d-01-01" % year, tz="UTC")
        train = eligible & (ends <= boundary)
        scored = eligible & (table.index.year == year)
        groups = sorted(np.unique(pairs[train])) if model_name == "m1" else [None]
        for group in groups:
            a, b = train.copy(), scored.copy()
            if group is not None:
                a &= pairs == group
                b &= pairs == group
            choice = selection["choices"][group]
            fitted = fit_ridge(x[a], y[a], choice) if model_name == "m1" else fit_trees(x[a], y[a], pairs[a], choice)
            if b.any():
                predictions[b] = forecast_ridge(fitted, x[b]) if model_name == "m1" else forecast_trees(fitted, x[b], pairs[b])
            last[group] = fitted
            audit.append({"year": year, "pair": group, "boundary": boundary, "last_training_endpoint": ends[a].max(),
                          "first_scored_time": table.index[b].min(), "last_scored_time": table.index[b].max()})
            if progress is not None:
                progress("refit %s h=%d year=%d pair=%s train=%d scored=%d" %
                         (model_name, h, year, "pooled" if group is None else fresh_pairs[group], a.sum(), b.sum()))
    return predictions, last, audit


def week_start(times):
    return times.normalize() - pd.to_timedelta(times.dayofweek, unit="D")


def placebo_signs(times, seed=0):
    weeks = week_start(times)
    unique = weeks.unique().sort_values()
    signs = np.random.default_rng(seed).choice([-1, 1], size=len(unique))
    return signs[unique.get_indexer(weeks)]


def score(table, forecasts, h):
    eligible = np.isfinite(forecasts)
    y = table["z" + str(h)].to_numpy()[eligible]
    f = np.asarray(forecasts)[eligible]
    pearson = correlation(f, np.clip(y, -5, 5))
    spearman = correlation(pd.Series(f).rank().to_numpy(), pd.Series(y).rank().to_numpy())
    grid = eligible & table["grid" + str(h)].to_numpy()
    result = table.loc[grid, ["pair", "close", "s", "z" + str(h), "cost" + str(h)]].copy()
    result = result.rename(columns={"z" + str(h): "target", "cost" + str(h): "cost"})
    result["forecast"] = np.asarray(forecasts)[grid]
    result["gross"] = np.sign(result.forecast) * result.target
    result["net"] = result.gross - result.cost
    result["traded_all"] = result.forecast != 0
    result["traded_sel"] = result.forecast.abs() >= result.cost
    return result, {"scored_rows": int(eligible.sum()), "pearson": pearson, "spearman": spearman,
                    "mean_cost": float(result.cost.mean()), "break_even_correlation": float(result.cost.mean() / .8)}


def weekly_table(trades, rule, weeks, pairs=None):
    pairs = tuple(range(len(fresh_pairs))) if pairs is None else tuple(pairs)
    active = trades.loc[trades["traded_" + rule.lower()]].copy()
    active["week"] = week_start(active.index)
    active["n"] = 1
    grouped = active.groupby(["week", "pair"])[["n", "gross", "cost", "net"]].sum()
    index = pd.MultiIndex.from_product([weeks, pairs], names=["week", "pair"])
    return grouped.reindex(index, fill_value=0).rename(columns={"gross": "gross_sum", "cost": "cost_sum", "net": "net_sum"})


def weekly_t(values):
    values = np.asarray(values, dtype=float)
    if len(values) < 2 or values.std(ddof=1) == 0:
        return np.nan
    return float(values.mean() / (values.std(ddof=1) / np.sqrt(len(values))))


@lru_cache(maxsize=16)
def bootstrap_indices(n, seed, resamples):
    return np.random.default_rng(seed).integers(0, n, size=(resamples, n), dtype=np.int32)


def inference(weekly, seed=0, resamples=5000):
    pooled = weekly.groupby(level="week").sum()
    values = pooled[["n", "gross_sum", "net_sum", "cost_sum"]].to_numpy(dtype=float)
    n = values[:, 0].sum()
    result = {"n": int(n), "gross": values[:, 1].sum() / n if n else np.nan,
              "net": values[:, 2].sum() / n if n else np.nan, "cost": values[:, 3].sum() / n if n else np.nan,
              "t_gross": weekly_t(values[:, 1]), "t_net": weekly_t(values[:, 2])}
    if not len(values) or not n:
        result.update(gross_low=np.nan, gross_high=np.nan, net_low=np.nan, net_high=np.nan)
        return result
    indices = bootstrap_indices(len(values), seed, resamples)
    draws = np.empty((resamples, 2))
    for start in range(0, resamples, 250):
        sums = values[indices[start:start + 250], :3].sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            draws[start:start + len(sums)] = sums[:, 1:] / sums[:, :1]
    for i, name in enumerate(("gross", "net")):
        valid = draws[:, i][np.isfinite(draws[:, i])]
        low, high = np.quantile(valid, [.025, .975]) if len(valid) else (np.nan, np.nan)
        result[name + "_low"], result[name + "_high"] = float(low), float(high)
    return result


def verdict(all_rule, sel_rule):
    for rule in (all_rule, sel_rule):
        if (rule["net"] > 0 and rule["t_net"] >= 3 and rule["n"] >= 500 and
                rule["net_first"] > 0 and rule["net_second"] > 0):
            return "TRADABLE"
    if (all_rule["gross"] > 0 and all_rule["t_gross"] >= 3 and
            all_rule["gross_first"] > 0 and all_rule["gross_second"] > 0):
        return "PREDICTABLE, BELOW COST"
    return "NO PREDICTABILITY"

"""Fixed daily momentum portfolios, monthly inference, and implementability."""

import builtins
from contextlib import contextmanager
import io
from pathlib import Path

import numpy as np
import pandas as pd

from .instruments import Config, pip_size
from .loader import daily_bars, fresh_pairs


start = pd.Timestamp("2010-01-01", tz="UTC")
half = pd.Timestamp("2018-01-01", tz="UTC")
end = pd.Timestamp("2025-11-01", tz="UTC")
currency_pairs = {"eur": "eurusd", "gbp": "gbpusd", "jpy": "usdjpy", "aud": "audusd",
                  "cad": "usdcad", "chf": "usdchf", "nzd": "nzdusd"}
inverse_pairs = ("usdjpy", "usdcad", "usdchf")


@contextmanager
def restricted_reads():
    """Reject any application access under hist/ or data/, including symlinks."""
    project = Path(__file__).resolve().parents[1]
    blocked = [project / "hist", project / "data"]
    originals = builtins.open, io.open

    def guard(original):
        def checked(filename, *args, **kwargs):
            if isinstance(filename, (str, Path)) and any(Path(filename).resolve().is_relative_to(root) for root in blocked):
                raise ValueError("Task 11 cannot access hist/ or data/")
            return original(filename, *args, **kwargs)
        return checked

    builtins.open, io.open = guard(originals[0]), guard(originals[1])
    try:
        yield
    finally:
        builtins.open, io.open = originals


def validate_times(frame):
    if not isinstance(frame.index, pd.DatetimeIndex) or str(frame.index.tz) != "UTC":
        raise ValueError("Daily research requires UTC closing timestamps")
    if not frame.index.is_unique or not frame.index.is_monotonic_increasing:
        raise ValueError("Daily research requires unique sorted timestamps")
    if len(frame) and (frame.index[0] < start or frame.index[-1] >= end):
        raise ValueError("Daily research cannot receive the locked test period or pre-2010 data")


def daily_panel(hourly):
    if set(hourly) != set(fresh_pairs):
        raise ValueError("Daily portfolios require all ten registered pairs")
    days = {}
    for pair, frame in hourly.items():
        validate_times(frame)
        days[pair] = daily_bars(frame)
    closes = pd.concat({pair: days[pair].close for pair in fresh_pairs}, axis=1).dropna()
    validate_times(closes)
    return closes


def first_in_period(index, frequency):
    if frequency == "week":
        keys = index.normalize() - pd.to_timedelta(index.dayofweek, unit="D")
    elif frequency == "month":
        keys = index.year * 12 + index.month
    else:
        raise ValueError("Rebalances must be weekly or monthly")
    return ~pd.Index(keys).duplicated()


def m1_features(closes):
    validate_times(closes)
    signs = [np.sign(closes / closes.shift(lag) - 1) for lag in (21, 63, 252)]
    signal = sum(signs) / 3
    volatility = closes.pct_change(fill_method=None).ewm(span=60, adjust=False, min_periods=60).std() * np.sqrt(252)
    return signal, volatility


def m1_weights(closes, *, target=0.10, always_long=False):
    signal, volatility = m1_features(closes)
    valid = signal.notna().all(axis=1) & volatility.gt(0).all(axis=1)
    if not valid.any():
        raise ValueError("Insufficient completed days for all ten M1 signals")
    if always_long:
        signal = signal * 0 + 1
    targets = (signal * target / volatility / 10).clip(-0.30, 0.30)
    schedule = pd.Series(first_in_period(closes.index, "week"), index=closes.index)
    targets = targets.where(valid & schedule, axis=0)
    held = targets.ffill().fillna(0)
    first = valid.index[valid][0]
    return held.loc[first:], volatility, signal, schedule


def currency_prices(closes):
    result = pd.DataFrame({"usd": 1.0}, index=closes.index)
    for currency, pair in currency_pairs.items():
        result[currency] = 1 / closes[pair] if pair in inverse_pairs else closes[pair]
    return result


def m2_weights(closes):
    validate_times(closes)
    prices = currency_prices(closes)
    momentum = prices / prices.shift(63) - 1
    momentum["usd"] = 0.0
    ready = momentum.notna().all(axis=1)
    if not ready.any():
        raise ValueError("Insufficient completed days for M2")
    schedule = first_in_period(prices.index, "month")
    targets = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)
    for position in np.flatnonzero(schedule & ready):
        # Alphabetical order makes exact momentum ties deterministic.
        ranked = momentum.iloc[position].sort_index().sort_values(kind="stable")
        targets.iloc[position] = 0.0
        targets.loc[prices.index[position], ranked.index[:2]] = -0.25
        targets.loc[prices.index[position], ranked.index[-2:]] = 0.25
    return targets.ffill().fillna(0).loc[ready.index[ready][0]:], prices


def portfolio_returns(prices, weights, cost_prices, unit_costs, *, financing=0.0075):
    """A weight set at close t earns the return t->t+1; turnover is paid at t.

    Financing charges elapsed calendar days on the prior held gross weight.
    unit_costs are price units (pip size already applied), zero for USD cash.
    """
    validate_times(prices)
    validate_times(weights)
    returns = prices.pct_change(fill_method=None).reindex(weights.index)
    held = weights.shift(1).fillna(0)
    changes = weights.diff()
    changes.iloc[0] = weights.iloc[0]
    costs = changes.abs() * pd.Series(unit_costs) / cost_prices.reindex(weights.index)
    days = weights.index.to_series().diff().dt.total_seconds().div(86400).fillna(0)
    finance = held.abs().mul(days * financing / 365, axis=0)
    gross = held * returns.fillna(0)
    net = gross - costs - finance
    out = pd.DataFrame({"gross": gross.sum(axis=1), "net": net.sum(axis=1), "cost": costs.sum(axis=1),
                        "financing": finance.sum(axis=1), "gross_leverage": held.abs().sum(axis=1),
                        "turnover": changes.abs().sum(axis=1), "rebalance": changes.ne(0).any(axis=1)}, index=weights.index)
    for column in weights:
        out[f"weight_{column}"] = weights[column]
        out[f"gross_{column}"] = gross[column]
        out[f"net_{column}"] = net[column]
        out[f"cost_{column}"] = costs[column]
        out[f"financing_{column}"] = finance[column]
    return out


def evaluate_m1(closes, cfg=None, *, always_long=False):
    cfg = Config() if cfg is None else cfg
    weights, _, _, _ = m1_weights(closes, always_long=always_long)
    costs = {pair: (0.5 * cfg.spread_pips[pair] + 0.5) * pip_size(pair) for pair in closes}
    return portfolio_returns(closes, weights, closes, costs)


def evaluate_m2(closes, cfg=None):
    cfg = Config() if cfg is None else cfg
    weights, prices = m2_weights(closes)
    cost_prices = pd.DataFrame({"usd": 1.0}, index=closes.index)
    costs = {"usd": 0.0}
    for currency, pair in currency_pairs.items():
        cost_prices[currency] = closes[pair]
        costs[currency] = (0.5 * cfg.spread_pips[pair] + 0.5) * pip_size(pair)
    # The USD cash leg is recorded but has no FX notional, turnover or financing.
    weights["usd"] = 0.0
    return portfolio_returns(prices, weights, cost_prices, costs)


def return_statistics(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {name: np.nan for name in ("annual_return", "annual_vol", "sharpe", "max_drawdown", "total_return")}
    std = values.std(ddof=1) if len(values) > 1 else np.nan
    equity = np.r_[1.0, np.cumprod(1 + values)]
    return {"annual_return": np.expm1(np.log1p(values).sum() * 252 / len(values)),
            "annual_vol": std * np.sqrt(252), "sharpe": values.mean() / std * np.sqrt(252) if std > 0 else np.nan,
            "max_drawdown": (1 - equity / np.maximum.accumulate(equity)).max(), "total_return": equity[-1] - 1}


def month_keys(times):
    index = pd.DatetimeIndex(times)
    return index.year * 12 + index.month


def portfolio_bootstrap(frame, resamples=5000, seed=0):
    """Resample whole calendar months; moments give exact shuffled-day metrics."""
    if frame.empty or resamples < 1:
        raise ValueError("Monthly bootstrap needs observations and positive resamples")
    values = frame[["gross", "net"]].to_numpy(dtype=float)
    keys = month_keys(frame.index)
    _, labels = np.unique(keys, return_inverse=True)
    count = np.bincount(labels)
    sums = np.stack([np.bincount(labels, weights=values[:, i]) for i in range(2)], axis=1)
    squares = np.stack([np.bincount(labels, weights=values[:, i] ** 2) for i in range(2)], axis=1)
    logs = np.stack([np.bincount(labels, weights=np.log1p(values[:, i])) for i in range(2)], axis=1)
    rng = np.random.default_rng(seed)
    draws = []
    for offset in range(0, resamples, 250):
        indices = rng.integers(0, len(count), size=(min(250, resamples - offset), len(count)))
        n = count[indices].sum(axis=1)[:, None]
        totals = sums[indices].sum(axis=1)
        variance = np.maximum((squares[indices].sum(axis=1) - totals ** 2 / n) / (n - 1), 0)
        std = np.sqrt(variance)
        sharpe = np.divide(totals / n * np.sqrt(252), std, out=np.full_like(std, np.nan), where=std > 0)
        annual = np.expm1(logs[indices].sum(axis=1) * 252 / n)
        draws.append(np.stack([annual, std * np.sqrt(252), sharpe], axis=2))
    intervals = np.nanpercentile(np.concatenate(draws), [2.5, 97.5], axis=0)
    result = {"bootstrap_resamples": resamples, "bootstrap_seed": seed, "n_months": len(count)}
    for i, mode in enumerate(("gross", "net")):
        for j, metric in enumerate(("annual_return", "annual_vol", "sharpe")):
            result[f"{mode}_{metric}_ci_low"] = intervals[0, i, j]
            result[f"{mode}_{metric}_ci_high"] = intervals[1, i, j]
    return result


def portfolio_summary(frame, resamples=5000, seed=0):
    row = {"n": len(frame), "first": frame.index[0].isoformat(), "last": frame.index[-1].isoformat(),
           "average_gross_leverage": frame.gross_leverage.mean(), "annual_turnover": frame.turnover.mean() * 252,
           "annual_execution_drag": frame.cost.mean() * 252, "annual_financing_drag": frame.financing.mean() * 252,
           "annual_cost_drag": (frame.cost + frame.financing).mean() * 252}
    for mode in ("gross", "net"):
        row.update({f"{mode}_{key}": value for key, value in return_statistics(frame[mode]).items()})
    row.update(portfolio_bootstrap(frame, resamples, seed))
    return row


def m1_verdict(summary):
    main = summary.loc[(summary.strategy == "M1") & (summary['group'] == "portfolio") & (summary.period == "all")].iloc[0]
    halves = summary.loc[(summary.strategy == "M1") & (summary['group'] == "portfolio") & summary.period.isin(("2010_2017", "2018_2025"))]
    if main.net_sharpe_ci_low > 0:
        return "PASS"
    if main.net_sharpe >= 0.30 and len(halves) == 2 and (halves.net_total_return > 0).all():
        return "PROMISING"
    return "FAIL"


def long_usd_returns(closes):
    returns = closes[list(currency_pairs.values())].pct_change(fill_method=None)
    for pair in returns:
        if pair not in inverse_pairs:
            returns[pair] *= -1
    return returns.mean(axis=1)


def implementability(closes, net_sharpe):
    rows = []
    for target in (0.10, 0.20):
        weights, _, signal, schedule = m1_weights(closes, target=target)
        eligible = weights.index.intersection(schedule.index[schedule & signal.notna().all(axis=1)])
        base_prices = pd.DataFrame(index=closes.index)
        for pair in closes:
            base = pair[:3]
            base_prices[pair] = 1.0 if base == "usd" else closes[currency_pairs[base]]
        for balance in (200, 2000):
            units = weights.loc[eligible].abs() * balance / base_prices.loc[eligible]
            for pair in closes:
                rows.append({"balance": balance, "target_vol": target, "pair": pair,
                             "median_units": units[pair].median(), "share_ge_1000": units[pair].ge(1000).mean(),
                             "pair_weeks": len(units), "expected_annual_usd": balance * target * net_sharpe})
    return pd.DataFrame(rows)

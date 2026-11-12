"""Pre-registered monthly currency carry and weekly volatility-scaled pair carry."""

import builtins
from contextlib import contextmanager
import io
from pathlib import Path

import numpy as np
import pandas as pd

from .daily_momentum import (currency_pairs, currency_prices, end, first_in_period, inverse_pairs,
                             m1_features, portfolio_returns, portfolio_summary, validate_times)
from .instruments import Config, pip_size
from .rates import monthly_diff, monthly_sources


primary_start = pd.Timestamp("2010-02-01", tz="UTC")


@contextmanager
def restricted_carry_reads():
    """No hist/ access; the only real data/ files allowed are the eight inputs."""
    project = Path(__file__).resolve().parents[1]
    allowed = {(project / "data" / "fred" / name).resolve() for name in monthly_sources.values()}
    originals = builtins.open, io.open

    def guard(original):
        def checked(filename, *args, **kwargs):
            if isinstance(filename, (str, Path)):
                source = Path(filename).resolve()
                if source.is_relative_to(project / "hist"):
                    raise ValueError("Task 12 cannot access hist/")
                if source.is_relative_to(project / "data") and source not in allowed:
                    raise ValueError("Task 12 reads only the eight monthly interbank rate files")
            return original(filename, *args, **kwargs)
        return checked

    builtins.open, io.open = guard(originals[0]), guard(originals[1])
    try:
        yield
    finally:
        builtins.open, io.open = originals


def validate_rates(rates):
    validate_times(rates)
    if set(rates) != set(monthly_sources):
        raise ValueError("Carry needs all eight currency rates")
    if len(rates) and not rates.index.equals(pd.date_range(rates.index[0], rates.index[-1], freq="1D")):
        raise ValueError("Carry rates must use an uninterrupted daily calendar")


def rate_accrual(rates, index):
    """Integrated decimal interest in [previous close, current close).

    A newly usable month's rate accrues from that day onward, including inside
    a multi-day weekend interval. It never applies to earlier holding days.
    """
    validate_rates(rates)
    calendar = pd.DatetimeIndex(index)
    if len(calendar) and (calendar[0] < rates.index[0] or calendar[-1] >= end):
        raise ValueError("Accrual dates exceed available rates or the test lock")
    required = rates.loc[calendar[0]:calendar[-1] - pd.Timedelta(days=1)] if len(calendar) > 1 else rates.iloc[:0]
    if required.isna().any().any():
        raise ValueError("A holding day has no usable rate; backward filling is forbidden")
    integrated = (rates / 100 / 365).fillna(0).cumsum().shift(1, fill_value=0)
    values = integrated.reindex(calendar)
    if values.isna().any().any():
        raise ValueError("Accrual requires daily midnight rate timestamps")
    return values.diff().fillna(0)


def pair_differentials(closes, rates):
    return pd.concat({pair: monthly_diff(pair, rates) for pair in closes}, axis=1).reindex(closes.index)


def c1_weights(closes, rates):
    validate_times(closes)
    validate_rates(rates)
    usable = rates.reindex(closes.index)
    ready = usable.notna().all(axis=1) & (closes.index >= primary_start)
    targets = pd.DataFrame(np.nan, index=closes.index, columns=list(monthly_sources))
    for position in np.flatnonzero(first_in_period(closes.index, "month") & ready):
        ranked = usable.iloc[position].sort_index().sort_values(kind="stable")
        targets.iloc[position] = 0.0
        targets.loc[closes.index[position], ranked.index[:2]] = -0.25
        targets.loc[closes.index[position], ranked.index[-2:]] = 0.25
    if not ready.any():
        raise ValueError("No fully observed C1 signal dates")
    return targets.ffill().fillna(0).loc[ready.index[ready][0]:]


def c2_weights(closes, rates, *, always_long=False):
    validate_rates(rates)
    _, volatility = m1_features(closes)
    gaps = pair_differentials(closes, rates)
    # Preserve the inclusive decimal boundary despite binary subtraction error.
    side = np.sign(gaps).where(gaps.abs() >= 0.25 - 1e-12, 0)
    if always_long:
        side = gaps * 0 + 1
    ready = gaps.notna().all(axis=1) & volatility.gt(0).all(axis=1) & (closes.index >= primary_start)
    target = (side * .10 / volatility / 10).clip(-.30, .30)
    schedule = pd.Series(first_in_period(closes.index, "week"), index=closes.index)
    held = target.where(ready & schedule, axis=0).ffill().fillna(0)
    if not ready.any():
        raise ValueError("Insufficient usable rates and volatility history for C2")
    return held.loc[ready.index[ready][0]:]


def add_carry(frame, rates, *, currency_book=False):
    """Reuse Task 11's spot accounting; its one financing charge becomes markup."""
    names = [name.removeprefix("weight_") for name in frame if name.startswith("weight_")]
    accrued = rate_accrual(rates, frame.index)
    result = frame.rename(columns={name: name.replace("financing", "markup", 1)
                                   for name in frame if name.startswith("financing")}).copy()
    result["spot"] = frame.gross
    for name in names:
        difference = accrued[name] - accrued.usd if currency_book else accrued[name[:3]] - accrued[name[3:]]
        carry = frame[f"weight_{name}"].shift(1).fillna(0) * difference
        result[f"spot_{name}"] = frame[f"gross_{name}"]
        result[f"carry_{name}"] = carry
        result[f"gross_{name}"] = frame[f"gross_{name}"] + carry
        result[f"net_{name}"] = frame[f"net_{name}"] + carry
    result["carry"] = result[[f"carry_{name}" for name in names]].sum(axis=1)
    result["gross"] = result.spot + result.carry
    result["net"] = result.gross - result.markup - result.cost
    return result


def evaluate_currency_book(closes, allocations, rates, cfg=None):
    cfg = Config() if cfg is None else cfg
    prices = currency_prices(closes)
    cost_prices = pd.DataFrame({"usd": 1.0}, index=closes.index)
    unit_costs = {"usd": 0.0}
    for currency, pair in currency_pairs.items():
        cost_prices[currency] = closes[pair]
        unit_costs[currency] = (0.5 * cfg.spread_pips[pair] + 0.5) * pip_size(pair)
    weights = allocations.copy()
    weights["usd"] = 0.0
    original = portfolio_returns(prices, weights, cost_prices, unit_costs)
    return add_carry(original, rates, currency_book=True)


def evaluate_pair_book(closes, weights, rates, cfg=None):
    cfg = Config() if cfg is None else cfg
    costs = {pair: (0.5 * cfg.spread_pips[pair] + .5) * pip_size(pair) for pair in closes}
    return add_carry(portfolio_returns(closes, weights, closes, costs), rates)


def long_usd_weights(closes):
    pairs = list(currency_pairs.values())
    weights = pd.DataFrame(0.0, index=closes.index, columns=closes.columns)
    for pair in pairs:
        weights[pair] = (1 if pair in inverse_pairs else -1) / 7
    return weights.loc[weights.index >= primary_start]


def blend_returns(c1, m1):
    index = c1.index.intersection(m1.index)
    result = pd.DataFrame(index=index)
    for name in ("spot", "carry", "markup", "cost", "gross", "net", "gross_leverage", "turnover"):
        result[name] = (c1.loc[index, name] + m1.loc[index, name]) / 2
    result["rebalance"] = c1.loc[index, "rebalance"] | m1.loc[index, "rebalance"]
    return result


def drawdown_dates(series):
    values = series.to_numpy(dtype=float)
    equity = np.r_[1., np.cumprod(1 + values)]
    peak = np.maximum.accumulate(equity)
    trough = int(np.argmax(1 - equity / peak))
    beginning = int(np.argmax(equity[:trough + 1]))
    recovery = np.flatnonzero(equity[trough + 1:] >= equity[beginning])

    def timestamp(position):
        return series.index[max(position - 1, 0)].isoformat()

    return {"drawdown_start": timestamp(beginning), "drawdown_trough": timestamp(trough),
            "drawdown_recovery": timestamp(trough + 1 + recovery[0]) if len(recovery) else "unrecovered"}


def carry_statistics(frame, resamples=5000, seed=0):
    row = portfolio_summary(frame.assign(financing=frame.markup), resamples, seed)
    row["annual_markup_drag"] = row.pop("annual_financing_drag")
    for name in ("spot", "carry", "markup", "cost"):
        row[f"annual_{name}_contribution"] = frame[name].mean() * 252 * (-1 if name in ("markup", "cost") else 1)
    for mode in ("gross", "net"):
        row.update({f"{mode}_{key}": value for key, value in drawdown_dates(frame[mode]).items()})
    monthly = (1 + frame.net).groupby([frame.index.year, frame.index.month]).prod() - 1
    worst = monthly.idxmin()
    row.update(worst_month=f"{worst[0]:04d}-{worst[1]:02d}", worst_month_return=monthly.min())
    return row


def currency_contributions(frame, *, currency_book=False):
    result = pd.DataFrame(0.0, index=list(monthly_sources), columns=["spot", "carry", "markup", "cost", "net"])
    names = [name.removeprefix("weight_") for name in frame if name.startswith("weight_")]
    for name in names:
        currencies = [name] if currency_book else [currency for currency in (name[:3], name[3:]) if currency != "usd"]
        for currency in currencies:
            for metric in result:
                result.loc[currency, metric] += frame[f"{metric}_{name}"].mean() * 252 / len(currencies)
    result[["markup", "cost"]] *= -1
    return result


def monthly_holdings(weights, *, currency_book=False):
    if currency_book:
        exposure = weights.copy()
    else:
        exposure = pd.DataFrame(0.0, index=weights.index, columns=list(monthly_sources))
        for pair in weights:
            exposure[pair[:3]] += weights[pair]
            exposure[pair[3:]] -= weights[pair]
    index = exposure.index.tz_localize(None).to_period("M").astype(str)
    average = exposure.groupby(index).mean()
    return average.mask(average.abs() < 1e-12, 0)


def implementation(closes, weights, sharpe, realized_vol, frequency):
    eligible = weights.index.intersection(closes.index[first_in_period(closes.index, frequency)])
    base_prices = pd.DataFrame(index=closes.index)
    for pair in weights:
        base_prices[pair] = 1.0 if pair[:3] == "usd" else closes[currency_pairs[pair[:3]]]
    rows = []
    for balance in (200, 2000):
        units = weights.loc[eligible].abs() * balance / base_prices.loc[eligible]
        for pair in weights:
            active = units[pair].loc[units[pair] > 1e-12]
            rows.append({"balance": balance, "value": pair, "median_units": units[pair].median(),
                         "median_active_units": active.median(), "share_ge_1000": units[pair].ge(1000).mean(),
                         "n_periods": len(units), "expected_annual_usd": balance * realized_vol * sharpe})
    return pd.DataFrame(rows)


def c1_verdict(summary):
    main = summary.loc[(summary.strategy == "C1") & (summary['group'] == "portfolio") & (summary.period == "all")].iloc[0]
    halves = summary.loc[(summary.strategy == "C1") & (summary['group'] == "portfolio") & summary.period.isin(("2010_2017", "2018_2025"))]
    if main.net_sharpe_ci_low > 0:
        return "PASS"
    if main.net_sharpe >= .30 and len(halves) == 2 and (halves.net_total_return > 0).all():
        return "PROMISING"
    return "FAIL"

"""Local policy and interbank rates with explicit observation-availability lags."""

import csv
from pathlib import Path

import numpy as np
import pandas as pd

from .loader import periods


data_dir = Path(__file__).resolve().parents[1] / "data" / "fred"
sources = {"usd": "dff.csv", "eur": "ecbdfr.csv", "gbp": "iudsoia.csv", "jpy": "boj_policy_rate.csv"}
currencies = {"eurusd": ("eur", "usd"), "gbpusd": ("gbp", "usd"), "usdjpy": ("usd", "jpy")}
lock = pd.Timestamp(periods["test"][0], tz="UTC")
monthly_sources = {
    "usd": "ir3tib01usm156n.csv", "eur": "ir3tib01ezm156n.csv", "gbp": "ir3tib01gbm156n.csv",
    "jpy": "ir3tib01jpm156n.csv", "aud": "ir3tib01aum156n.csv", "cad": "ir3tib01cam156n.csv",
    "chf": "ir3tib01chm156n.csv", "nzd": "ir3tib01nzm156n.csv",
}


def _read_source(filename):
    dates, values = [], []
    with (data_dir / filename).open(newline="", encoding="utf-8-sig") as source:
        header = next(csv.reader(source), None)
        if header is None or len(header) != 2 or header[0] != "observation_date":
            raise ValueError(f"Unexpected rate columns in {filename}")
        for line in source:
            date = pd.Timestamp(line.partition(",")[0].strip().strip('"'), tz="UTC")
            # Gate before price/value parsing: locked and 2026 rates never enter.
            if date >= lock:
                continue
            row = next(csv.reader([line]))
            value = np.nan if row[1].strip() in ("", ".") else float(row[1])
            if not np.isfinite(value) and not np.isnan(value):
                raise ValueError(f"Non-finite rate in {filename}")
            dates.append(date)
            values.append(value)
    series = pd.Series(values, index=pd.DatetimeIndex(dates, tz="UTC"), dtype=float).sort_index()
    if series.index.has_duplicates:
        raise ValueError(f"Duplicate observation dates in {filename}")
    return series


def load_rates():
    """Percent rates on every calendar day, ffill only, then a one-day lag.

    BoJ announcement rows are steps, usable on the following day. Leading
    missing observations remain NaN. No API can unlock rates after the FX lock.
    Only the four named sources are read; the stale monthly Japan file is ignored.
    """
    observed = pd.concat({currency: _read_source(filename) for currency, filename in sources.items()}, axis=1, sort=True)
    if observed.empty:
        raise ValueError("No rate observations before the lock")
    calendar = pd.date_range(observed.index.min(), lock - pd.Timedelta(days=1), freq="1D", tz="UTC")
    result = observed.reindex(calendar).ffill().shift(1)
    result.index.name = "observation_date"
    return result


def diff(pair, rates=None):
    """Base minus quote in percentage points, using already-lagged daily rates."""
    pair = pair.lower()
    if pair not in currencies:
        raise ValueError(f"Unknown pair: {pair}")
    rates = load_rates() if rates is None else rates
    base, quote = currencies[pair]
    return (rates[base] - rates[quote]).rename(pair)


def load_monthly_rates():
    """Task 12 interbank percent rates, usable from the following month only.

    This separate API leaves the earlier daily-policy-rate API unchanged.
    Locked observations are discarded by _read_source before value parsing.
    """
    project = Path(__file__).resolve().parents[1]
    observed = {}
    for currency, filename in monthly_sources.items():
        source = (data_dir / filename).resolve()
        if source.is_relative_to(project / "hist"):
            raise ValueError("Monthly rates cannot read hist/")
        series = _read_source(filename)
        if series.empty or not series.index.is_month_start.all():
            raise ValueError(f"Expected monthly first-day observations in {filename}")
        expected = pd.date_range(series.index[0], series.index[-1], freq="MS")
        if not series.index.equals(expected):
            raise ValueError(f"Missing observation month in {filename}")
        series.index = series.index + pd.offsets.MonthBegin(1)
        observed[currency] = series
    usable = pd.concat(observed, axis=1, sort=True)
    calendar = pd.date_range(pd.Timestamp("2010-01-01", tz="UTC"), lock - pd.Timedelta(days=1), freq="1D")
    # Keep legitimate pre-2010 history for ffill, but never fill from the future.
    result = usable.reindex(usable.index.union(calendar)).ffill().reindex(calendar)
    result.index.name = "timestamp"
    return result


def monthly_diff(pair, rates=None):
    """Base-minus-quote percentage points for the ten registered FX pairs."""
    from .instruments import pip_size

    pair = pair.lower()
    pip_size(pair)
    rates = load_monthly_rates() if rates is None else rates
    return (rates[pair[:3]] - rates[pair[3:]]).rename(pair)

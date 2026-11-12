"""Load UTC OHLC bars, indexed by when their prices become available."""

import csv
from pathlib import Path

import numpy as np
import pandas as pd


data_dir = Path(__file__).resolve().parents[1] / "download"
fresh_data_dir = Path(__file__).resolve().parents[1] / "fresh" / "download"
pairs = ("eurusd", "gbpusd", "usdjpy")
fresh_pairs = (*pairs, "audusd", "usdcad", "usdchf", "nzdusd", "eurjpy", "gbpjpy", "eurgbp")
timeframes = {"15m": "m15", "1h": "h1", "m15": "m15", "h1": "h1"}
ohlc = {"open": "first", "high": "max", "low": "min", "close": "last"}
periods = {
    "train": ("2019-01-01", "2025-01-01"),
    "val": ("2025-01-01", "2025-11-01"),
    "test": ("2025-11-01", "2026-11-01"),
}


def _read_pair(pair, final):
    if data_dir.resolve() == fresh_data_dir.resolve():
        raise ValueError("Fresh and existing data roots must be separate")
    files = sorted(data_dir.glob(f"{pair}-m15-bid-*.csv"))
    if len(files) != 1:
        raise ValueError(f"Expected one CSV for {pair}/m15, found {len(files)}")

    duration = pd.Timedelta("15min")
    start_ms = pd.Timestamp(periods["train"][0], tz="UTC").value // 1_000_000
    end = periods["test"][1] if final else periods["test"][0]
    end_ms = pd.Timestamp(end, tz="UTC").value // 1_000_000
    duration_ms = duration.value // 1_000_000
    rows = []
    with files[0].open(newline="") as source:
        reader = csv.reader(source)
        if next(reader, None) != ["timestamp", *ohlc]:
            raise ValueError(f"Unexpected CSV columns in {files[0].name}")
        for line in source:
            timestamp = int(line.partition(",")[0])
            # Gate before parsing price fields, including in unsorted files.
            if start_ms <= timestamp + duration_ms < end_ms:
                rows.append(next(csv.reader([line])))

    df = pd.DataFrame(rows, columns=["timestamp", *ohlc])
    df.index = pd.DatetimeIndex(
        pd.to_datetime(df.pop("timestamp").astype("int64"), unit="ms", utc=True),
        name="timestamp",
    )
    if df.index.has_duplicates:
        raise ValueError(f"Duplicate timestamps in {files[0].name}")
    df = df.astype("float64").sort_index()
    if not np.isfinite(df.to_numpy()).all():
        raise ValueError(f"Non-finite OHLC prices in {files[0].name}")
    flat = df.eq(df["open"], axis=0).all(axis=1)
    df = df.loc[~flat & (df.index.dayofweek < 5)].copy()
    # CSV timestamps mark openings; a bar is usable only at its close.
    df.index = df.index + duration
    return df.loc[df.index.dayofweek < 5]


def _resample(df, frequency):
    if df.empty:
        return df.copy()
    result = df.resample(frequency, label="right", closed="right").agg(ohlc)
    result = result.dropna(subset=list(ohlc))
    # Do not publish the final bucket until its closing boundary is observed.
    return result.loc[result.index <= df.index[-1]]


def load_pair(pair, tf, *, final=False):
    """Read 15m/1h bars with UTC closing timestamps and no flat/weekend bars.

    By default, prices from the test period are excluded before numeric parsing.
    Explicit final=True unlocks data through October 2026. The m15/h1 spelling
    used in filenames is also accepted. All 1h bars come from m15; native h1
    files are never read.
    """
    pair = pair.lower()
    if pair not in pairs:
        raise ValueError(f"Unknown pair: {pair}")
    if tf not in timeframes:
        raise ValueError(f"Unknown timeframe: {tf}")
    df = _read_pair(pair, final)
    if timeframes[tf] == "h1":
        df = _resample(df, "1h")
    return df.loc[df.index.dayofweek < 5]


def split(df, which, *, final=False):
    """Return one UTC calendar split, with an exclusive upper boundary.

    Test access requires final=True even if the input was already unlocked.
    """
    if which not in periods:
        raise ValueError(f"Unknown split: {which}")
    if which == "test" and not final:
        raise ValueError("The test split is locked; explicit final=True is required")
    start, end = (pd.Timestamp(date, tz="UTC") for date in periods[which])
    return df.loc[(df.index >= start) & (df.index < end)].copy()


def daily_bars(hourly):
    """Resample closing-time 1h bars to completed UTC days, labelled at midnight.

    Missing intraday bars do not invalidate an otherwise completed calendar day.
    A day's bar becomes available at the following midnight; the trailing day is
    omitted unless its midnight boundary has been reached by the input data.
    """
    if not isinstance(hourly.index, pd.DatetimeIndex) or str(hourly.index.tz) != "UTC":
        raise ValueError("Daily resampling requires a UTC DatetimeIndex")
    if hourly.index.has_duplicates or not hourly.index.is_monotonic_increasing:
        raise ValueError("Daily resampling requires unique, sorted timestamps")
    return _resample(hourly, "1D")


def fresh_end(pair):
    """Exclusive closing-time boundary, independent of the existing split API."""
    if pair not in fresh_pairs:
        raise ValueError(f"Unknown fresh pair: {pair}")
    return pd.Timestamp("2019-01-01" if pair in pairs else "2025-11-01", tz="UTC")


def _fresh_file(pair):
    if fresh_data_dir.resolve() == data_dir.resolve():
        raise ValueError("Fresh and existing data roots must be separate")
    files = sorted(fresh_data_dir.glob(f"{pair}-m15-bid-*.csv"))
    if len(files) != 1:
        raise ValueError(f"Expected one fresh CSV for {pair}/m15, found {len(files)}")
    return files[0]


def _read_fresh(pair):
    source_file = _fresh_file(pair)
    duration = pd.Timedelta("15min")
    duration_ms = duration.value // 1_000_000
    start_ms = pd.Timestamp("2010-01-01", tz="UTC").value // 1_000_000
    end_ms = fresh_end(pair).value // 1_000_000
    rows = []
    with source_file.open(newline="") as source:
        if next(csv.reader(source), None) != ["timestamp", *ohlc]:
            raise ValueError(f"Unexpected CSV columns in {source_file.name}")
        for line_number, line in enumerate(source, start=2):
            try:
                timestamp = int(line.partition(",")[0])
                # Read no excluded price fields, even if the file is unsorted.
                if not start_ms <= timestamp + duration_ms < end_ms:
                    continue
                values = next(csv.reader([line]))
                if len(values) != 5 or timestamp % duration_ms:
                    raise ValueError("Expected five fields and a 15m-aligned timestamp")
                rows.append([timestamp, *map(float, values[1:])])
            except (ValueError, csv.Error) as error:
                raise ValueError(f"Malformed {source_file.name}, line {line_number}: {error}") from error
    if not rows:
        raise ValueError(f"No eligible rows in {source_file.name}")
    raw = pd.DataFrame(rows, columns=["timestamp", *ohlc])
    raw.index = pd.DatetimeIndex(pd.to_datetime(raw.pop("timestamp"), unit="ms", utc=True), name="timestamp")
    values = raw.to_numpy()
    valid = (np.isfinite(values).all() and (values > 0).all() and
             (raw.high >= raw[["open", "close", "low"]].max(axis=1)).all() and
             (raw.low <= raw[["open", "close", "high"]].min(axis=1)).all())
    if not valid:
        raise ValueError(f"Malformed OHLC prices in {source_file.name}")
    duplicates = int(raw.index.duplicated().sum())
    if duplicates:
        raise ValueError(f"Duplicate timestamps in {source_file.name}: {duplicates}")
    raw = raw.sort_index()
    flat = raw.eq(raw.open, axis=0).all(axis=1)
    cleaned = raw.loc[~flat & (raw.index.dayofweek < 5)].copy()
    cleaned.index += duration
    cleaned = cleaned.loc[cleaned.index.dayofweek < 5].copy()
    if cleaned.empty:
        raise ValueError(f"No usable bars in {source_file.name}")
    cleaned.attrs = {"data_root": "fresh", "pair": pair, "end": fresh_end(pair).isoformat()}
    report = {"file": source_file.name, "pair": pair, "rows": len(raw), "clean_rows": len(cleaned),
              "first": raw.index[0] + duration, "last": raw.index[-1] + duration,
              "duplicates": duplicates, "flat_share": float(flat.mean()),
              "max_gap_h": _non_weekend_gap(cleaned.index)}
    return cleaned, report


def _non_weekend_gap(index):
    if len(index) < 2:
        return 0.0
    times = index.tz_localize(None).to_numpy(dtype="datetime64[ns]")
    days = times.astype("datetime64[D]")
    hours = (times - days) / np.timedelta64(1, "h")
    coordinate = np.busday_count("1970-01-01", days) * 24 + np.where(np.is_busday(days), hours, 0)
    return float(np.diff(coordinate).max())


def fresh_data_report():
    """Validate every eligible file before any strategy is evaluated."""
    reports = []
    print("Fresh m15 data report: eligible raw rows/closing times; gap measured after cleaning, UTC weekends subtracted.", flush=True)
    print(f"{'pair':<7} {'rows':>7} {'clean':>7} {'first_utc':<25} {'last_utc':<25} {'dup':>4} {'flat_pct':>9} {'gap_h':>8}", flush=True)
    for pair in fresh_pairs:
        try:
            _, row = _read_fresh(pair)
        except (OSError, ValueError) as error:
            print(f"STOP: {error}", flush=True)
            raise
        reports.append(row)
        print(f"{pair:<7} {row['rows']:>7} {row['clean_rows']:>7} {row['first'].isoformat():<25} "
              f"{row['last'].isoformat():<25} {row['duplicates']:>4} {row['flat_share']:>8.3%} {row['max_gap_h']:>8.2f} {row['file']}", flush=True)
    return pd.DataFrame(reports)


def fresh_hourly(quarters):
    """Resample already-validated fresh closing-time m15 data, preserving provenance."""
    pair = quarters.attrs.get("pair")
    if quarters.attrs.get("data_root") != "fresh" or pair not in fresh_pairs:
        raise ValueError("Hourly fresh resampling requires fresh-loader data")
    if len(quarters) and quarters.index[-1] >= fresh_end(pair):
        raise ValueError("Fresh closing-time boundary exceeded")
    result = _resample(quarters, "1h")
    result = result.loc[result.index.dayofweek < 5].copy()
    result.attrs = quarters.attrs.copy()
    return result


def load_fresh_pair(pair, tf):
    """Load a separate, continuous fresh fold; there is no final/test override."""
    pair = pair.lower()
    fresh_end(pair)
    if tf not in timeframes:
        raise ValueError(f"Unknown timeframe: {tf}")
    quarters, _ = _read_fresh(pair)
    return fresh_hourly(quarters) if timeframes[tf] == "h1" else quarters


def load_continuous_pair(pair, tf):
    """Task 11's 2010--October 2025 series, with an audited original-pair join.

    This explicit API never unlocks test data or changes either earlier loader.
    """
    pair = pair.lower()
    if pair not in fresh_pairs or tf not in timeframes:
        raise ValueError("Unknown continuous pair or timeframe")
    project = Path(__file__).resolve().parents[1]
    roots = [fresh_data_dir] + ([data_dir] if pair in pairs else [])
    for root in roots:
        resolved = root.resolve()
        if any(resolved.is_relative_to(project / name) for name in ("hist", "data")):
            raise ValueError("Continuous data cannot read hist/ or data/")
        for source in root.glob(f"{pair}-m15-bid-*.csv"):
            if not source.resolve().is_relative_to(resolved):
                raise ValueError("Input CSV must remain inside its data root")
    quarters = load_fresh_pair(pair, "15m")
    join = {}
    if pair in pairs:
        recent = load_pair(pair, "15m")
        boundary = pd.Timestamp("2019-01-01", tz="UTC")
        if quarters.empty or recent.empty or quarters.index[-1] >= boundary or recent.index[0] < boundary:
            raise ValueError("Invalid 2019 source boundary")
        overlap = len(quarters.index.intersection(recent.index))
        gap = recent.index[0] - quarters.index[-1]
        if overlap or gap <= pd.Timedelta(0) or gap > pd.Timedelta(hours=48, minutes=15):
            raise ValueError(f"Invalid 2019 join for {pair}: overlap={overlap}, gap={gap}")
        join = {"join_overlap": overlap, "join_gap_hours": gap.total_seconds() / 3600,
                "join_before": quarters.index[-1].isoformat(), "join_after": recent.index[0].isoformat()}
        quarters = pd.concat([quarters, recent])
    if not quarters.index.is_unique or not quarters.index.is_monotonic_increasing:
        raise ValueError("Continuous data must have unique sorted timestamps")
    if quarters.index[-1] >= pd.Timestamp("2025-11-01", tz="UTC"):
        raise ValueError("The continuous series cannot load the locked test period")
    result = _resample(quarters, "1h") if timeframes[tf] == "h1" else quarters
    result = result.loc[result.index.dayofweek < 5].copy()
    result.attrs = {"data_root": "continuous", "pair": pair, **join}
    return result

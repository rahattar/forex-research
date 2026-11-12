"""Period-isolated bars, completed higher time frames, and source-keyed caches."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import loader


project = Path(__file__).resolve().parents[1]
cache_dir = project / "cache" / "lab"
timeframes = ("15m", "1h", "4h", "1d")


def bounds(period):
    if period not in ("lab", "confirm"):
        raise ValueError("Period must be lab or confirm")
    return tuple(pd.Timestamp(value, tz="UTC") for value in
                 (("2010-01-01", "2020-01-01") if period == "lab" else ("2020-01-01", "2025-11-01")))


def period_bars(frame, period):
    start, end = bounds(period)
    return frame.loc[(frame.index >= start) & (frame.index < end)].copy()


def validate(frame, period):
    start, end = bounds(period)
    index = frame.index
    if (not isinstance(index, pd.DatetimeIndex) or str(index.tz) != "UTC" or
            not index.is_unique or not index.is_monotonic_increasing):
        raise ValueError("Expected unique sorted UTC closing timestamps")
    if len(frame) and (index[0] < start or index[-1] >= end):
        raise ValueError("Lab data cross their period boundary")
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError("Non-finite lab bars")


def higher_bars(quarters, timeframe):
    if timeframe not in timeframes:
        raise ValueError("Unknown lab timeframe")
    if timeframe == "15m" or quarters.empty:
        return quarters.copy()
    if timeframe != "1d":
        result = quarters.resample(timeframe, closed="right", label="right").agg(loader.ohlc).dropna()
    else:
        # Form wall-clock 17:00 boundaries before localizing, including across DST.
        local = quarters.index.tz_convert("America/New_York").tz_localize(None)
        close = local.normalize() + pd.Timedelta(hours=17)
        close += pd.to_timedelta((local > close).astype(int), unit="D")
        labels = close.tz_localize("America/New_York").tz_convert("UTC")
        result = quarters.groupby(labels).agg(loader.ohlc)
        result.index.name = quarters.index.name
    return result.loc[result.index <= quarters.index[-1]].copy()


def visible(values, times):
    """Only published higher-frame values can be carried to a decision close."""
    return values.reindex(times, method="ffill")


@dataclass
class LabData:
    pair: str
    period: str
    frames: dict
    s: pd.Series
    vol_ratio: pd.Series

    @property
    def quarters(self):
        return self.frames["15m"]

    def expected_move(self, hours):
        return self.quarters.close * self.s * np.sqrt(hours)


def build_data(quarters, pair, period="lab"):
    validate(quarters, period)
    quarters = quarters.copy()
    quarters.index = quarters.index.as_unit("ns")
    frames = {tf: higher_bars(quarters, tf) for tf in timeframes}
    r2 = np.log(frames["1h"].close).diff().pow(2)
    scale = np.sqrt(r2.ewm(span=240, adjust=False, min_periods=240).mean())
    ratio = np.sqrt(r2.ewm(span=24, adjust=False).mean() / r2.ewm(span=480, adjust=False).mean())
    return LabData(pair, period, frames, visible(scale, quarters.index), visible(ratio, quarters.index))


def source_fingerprint(pair):
    """Inspect only source metadata here; all FX parsing stays in the loader."""
    roots = [loader.fresh_data_dir] + ([loader.data_dir] if pair in loader.pairs else [])
    metadata = []
    for root in roots:
        paths = sorted(root.glob(pair + "-m15-bid-*.csv"))
        if len(paths) != 1:
            raise ValueError("Expected one source file for " + pair)
        path = paths[0]
        stat = path.stat()
        metadata.append((str(path.resolve()), stat.st_size, stat.st_mtime_ns))
    return metadata


def cache_key(pair, period):
    payload = dict(version=1, pair=pair, period=period, sources=source_fingerprint(pair))
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("ascii")).hexdigest()


def save_cache(path, data):
    arrays = {}
    for tf, frame in data.frames.items():
        arrays[tf + "_time"] = frame.index.as_unit("ns").asi8
        arrays[tf + "_ohlc"] = frame[list(loader.ohlc)].to_numpy()
    arrays.update(s=data.s.to_numpy(), vol_ratio=data.vol_ratio.to_numpy())
    np.savez_compressed(path, **arrays)


def read_cache(path, pair, period):
    frames = {}
    with np.load(path, allow_pickle=False) as arrays:
        for tf in timeframes:
            ticks = arrays[tf + "_time"]
            start, end = bounds(period)
            if len(ticks) and (ticks.min() < start.value or ticks.max() >= end.value):
                raise ValueError("Cached data cross their period boundary")
            frame = pd.DataFrame(arrays[tf + "_ohlc"], columns=list(loader.ohlc),
                                 index=pd.to_datetime(ticks, unit="ns", utc=True).rename("timestamp"))
            validate(frame, period)
            frames[tf] = frame
        index = frames["15m"].index
        return LabData(pair, period, frames, pd.Series(arrays["s"], index=index),
                       pd.Series(arrays["vol_ratio"], index=index))


def load_data(pair, period="lab", *, use_cache=True):
    bounds(period)
    if pair not in loader.fresh_pairs:
        raise ValueError("Unknown lab pair")
    path = cache_dir / (pair + "_" + period + "_" + cache_key(pair, period) + ".npz") if use_cache else None
    if path is not None and path.exists():
        return read_cache(path, pair, period)
    # The period cut is the first operation after the authorized loader returns.
    quarters = period_bars(loader.load_continuous_pair(pair, "15m"), period)
    data = build_data(quarters, pair, period)
    if path is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        save_cache(path, data)
    return data

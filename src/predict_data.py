"""Cutoff-safe external observations and causal Task 15 input alignment."""

import builtins
from contextlib import contextmanager
import csv
import io
from pathlib import Path
import re

import numpy as np
import pandas as pd

from . import loader, multi_asset_data


project = Path(__file__).resolve().parents[1]
fred_dir = project / "data" / "fred"
etfs = ("spy", "efa", "eem", "ief", "tlt", "hyg", "gld", "dbc")
freds = ("vixcls", "dcoilbrenteu", "dgs2")
lock = "2025-11-01"


def before_lock(line):
    """Compare the raw ISO prefix before calling any date or numeric parser."""
    prefix = line.partition(",")[0].strip().strip('"')[:10]
    if not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", prefix):
        raise ValueError("Expected an ISO observation-date prefix")
    return prefix < lock


@contextmanager
def filtered_etf_reads():
    """Prefilter the eight named ETFs for the unchanged legacy ETF reader."""
    allowed = {(multi_asset_data.etf_dir / (name + ".csv")).resolve() for name in etfs}
    originals = builtins.open, io.open

    def wrap(original):
        def checked(filename, *args, **kwargs):
            if not isinstance(filename, (str, Path)) or Path(filename).resolve() not in allowed:
                return original(filename, *args, **kwargs)
            mode = kwargs.get("mode", args[0] if args else "r")
            if "r" not in mode:
                return original(filename, *args, **kwargs)
            with original(filename, *args, **kwargs) as source:
                if "b" in mode:
                    source = io.StringIO(source.read().decode("utf-8-sig"))
                header = source.readline()
                text = header + "".join(line for line in source if before_lock(line))
            return io.BytesIO(text.encode("ascii")) if "b" in mode else io.StringIO(text)
        return checked

    builtins.open, io.open = wrap(originals[0]), wrap(originals[1])
    try:
        yield
    finally:
        builtins.open, io.open = originals


def read_etf(name):
    if name not in etfs:
        raise ValueError("ETF not registered for Task 15")
    with filtered_etf_reads():
        frame, report, malformed = multi_asset_data._read_etf(name)
    if malformed:
        raise ValueError("Malformed ETF: " + name)
    return frame["Adj Close"].rename(name), {
        "file": name + ".csv", "rows": len(frame), "first": frame.index[0],
        "last": frame.index[-1], "missing": 0,
    }


def read_fred(name):
    if name not in freds:
        raise ValueError("FRED series not registered for Task 15")
    dates, values = [], []
    with (fred_dir / (name + ".csv")).open(newline="", encoding="utf-8-sig") as source:
        header = next(csv.reader(source), None)
        if not header or len(header) != 2 or header[0] != "observation_date" or header[1].lower() != name:
            raise ValueError("Malformed FRED header: " + name)
        for line in source:
            if not before_lock(line):
                continue
            fields = next(csv.reader([line]))
            if len(fields) != 2:
                raise ValueError("Malformed FRED field count: " + name)
            date = pd.Timestamp(fields[0])
            if date.tzinfo is not None or date != date.normalize():
                raise ValueError("Expected a calendar observation date")
            missing = fields[1].strip() in ("", ".")
            value = np.nan if missing else float(fields[1])
            if not missing and not np.isfinite(value):
                raise ValueError("Non-finite FRED observation: " + name)
            dates.append(date)
            values.append(value)
    series = pd.Series(values, index=pd.DatetimeIndex(dates, name="date"), name=name, dtype=float).sort_index()
    if series.empty or series.index.has_duplicates or not series.notna().any():
        raise ValueError("Empty or duplicate FRED observations: " + name)
    return series, {"file": name + ".csv", "rows": len(series), "first": series.index[0],
                    "last": series.index[-1], "missing": int(series.isna().sum())}


def load_inputs():
    bars, external, reports = {}, {}, []
    print("Data audit (FX closing times UTC; ETF dates New York; external dates observation dates)")
    for pair in loader.fresh_pairs:
        frame = loader.load_continuous_pair(pair, "1h")
        if frame.empty or frame.index[-1].strftime("%Y-%m-%d") >= lock:
            raise ValueError("FX data are empty or cross the lock")
        bars[pair] = frame
        print("%-7s rows=%7d first=%s last=%s" % (pair, len(frame), frame.index[0].isoformat(), frame.index[-1].isoformat()))
    for name in (*etfs, *freds):
        series, report = read_etf(name) if name in etfs else read_fred(name)
        external[name] = series
        reports.append(report)
        print("%-16s rows=%7d first=%s last=%s missing=%d" %
              (report["file"], report["rows"], report["first"].date(), report["last"].date(), report["missing"]))
    return bars, external, pd.DataFrame(reports)


def external_inputs(series, times):
    """Return observation-date G3 tables and their strictly previous-date mapping."""
    dates = times.tz_convert("UTC").tz_localize(None).normalize()
    observations, mapped = {}, []
    for name in (*etfs, *freds):
        observed = series[name].dropna().sort_index()
        if not observed.index.is_unique or (len(observed) and observed.index[-1].strftime("%Y-%m-%d") >= lock):
            raise ValueError("External observations cross the lock or contain duplicates")
        if name != "dgs2" and (observed <= 0).any():
            raise ValueError("Log input must be positive: " + name)
        x = observed if name == "dgs2" else np.log(observed)
        sd = x.diff().rolling(60, min_periods=60).std(ddof=1)
        table = pd.DataFrame({name + "_" + str(k): x.diff(k) / (sd * np.sqrt(k)) for k in (1, 5)})
        observations[name] = table
        positions = table.index.searchsorted(dates, side="left") - 1
        values = np.full((len(times), 2), np.nan)
        valid = positions >= 0
        values[valid] = table.to_numpy()[positions[valid]]
        mapped.append(pd.DataFrame(values, index=times, columns=table.columns))
    return observations, pd.concat(mapped, axis=1).replace([np.inf, -np.inf], np.nan)

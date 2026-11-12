"""Local ETF total-return prices on New York dates, with an immutable cutoff."""

import builtins
from contextlib import contextmanager
import csv
import io
from pathlib import Path

import numpy as np
import pandas as pd


project = Path(__file__).resolve().parents[1]
etf_dir = project / "data" / "etf"
cash_file = project / "data" / "fred" / "dtb3.csv"
tickers = ("spy", "iwm", "efa", "eem", "ief", "tlt", "lqd", "hyg", "gld", "dbc", "vnq", "fxe", "fxy", "uup")
cutoff = pd.Timestamp("2025-11-01")
price_columns = ("Open", "High", "Low", "Close", "Adj Close")
required_columns = ("Date", *price_columns, "Volume", "Dividends", "Stock Splits")


@contextmanager
def restricted_asset_reads():
    """The evaluation reads only its named 15 files under data/, never hist/."""
    allowed = {(etf_dir / f"{ticker}.csv").resolve() for ticker in tickers} | {cash_file.resolve()}
    originals = builtins.open, io.open

    def guard(original):
        def checked(filename, *args, **kwargs):
            if isinstance(filename, (str, Path)):
                source = Path(filename).resolve()
                if source.is_relative_to(project / "hist"):
                    raise ValueError("Multi-asset evaluation cannot read hist/")
                if source.is_relative_to(project / "data") and source not in allowed:
                    raise ValueError("Multi-asset evaluation reads only its 15 named data files")
            return original(filename, *args, **kwargs)
        return checked

    builtins.open, io.open = guard(originals[0]), guard(originals[1])
    try:
        yield
    finally:
        builtins.open, io.open = originals


def new_york_date(value):
    instant = pd.Timestamp(value)
    if pd.isna(instant):
        raise ValueError("Missing ETF date")
    if instant.tzinfo is not None:
        instant = instant.tz_convert("America/New_York").tz_localize(None)
    return instant.normalize()


def validate_dates(frame):
    index = frame.index
    if not isinstance(index, pd.DatetimeIndex) or index.tz is not None or not index.equals(index.normalize()):
        raise ValueError("Expected New York calendar dates without time-of-day")
    if not index.is_unique or not index.is_monotonic_increasing:
        raise ValueError("Expected unique sorted calendar dates")
    if len(index) and index[-1] >= cutoff:
        raise ValueError("Multi-asset data cannot include the locked period")


def _read_etf(ticker):
    filename = etf_dir / f"{ticker}.csv"
    dates, rows = [], []
    with filename.open(newline="", encoding="utf-8-sig") as source:
        header = next(csv.reader(source), None)
        if (not header or header[0] != "Date" or len(set(header)) != len(header) or
                not set(required_columns).issubset(header) or set(header) - set(required_columns) - {"Capital Gains"}):
            raise ValueError(f"Malformed columns in {filename.name}")
        positions = [header.index(name) for name in required_columns[1:]]
        for line_number, line in enumerate(source, start=2):
            try:
                date = new_york_date(line.partition(",")[0].strip().strip('"'))
                # No excluded price, dividend, split or volume field is parsed.
                if date >= cutoff:
                    continue
                fields = next(csv.reader([line]))
                if len(fields) != len(header):
                    raise ValueError("Wrong field count")
                rows.append([float(fields[position]) for position in positions])
                dates.append(date)
            except (ValueError, csv.Error) as error:
                raise ValueError(f"Malformed {filename.name}, line {line_number}: {error}") from error
    if not rows:
        raise ValueError(f"No eligible rows in {filename.name}")
    frame = pd.DataFrame(rows, index=pd.DatetimeIndex(dates, name="date"), columns=required_columns[1:]).sort_index()
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError(f"Non-finite values in {filename.name}")
    invalid_price = frame[list(price_columns)].le(0).any(axis=1)
    duplicate = int(frame.index.duplicated().sum())
    report = {"file": filename.name, "rows": len(frame), "first": frame.index[0], "last": frame.index[-1],
              "duplicates": duplicate, "non_positive": int(invalid_price.sum()),
              "above_25pct": int(frame["Adj Close"].pct_change(fill_method=None).abs().gt(.25).sum()), "blanks": 0}
    malformed = (duplicate > 0 or invalid_price.any() or
                 (frame[["Volume", "Dividends", "Stock Splits"]] < 0).any().any() or
                 (frame.High < frame[["Open", "Close", "Low"]].max(axis=1) - 1e-8).any() or
                 (frame.Low > frame[["Open", "Close", "High"]].min(axis=1) + 1e-8).any())
    return frame, report, bool(malformed)


def _read_cash():
    dates, values = [], []
    with cash_file.open(newline="", encoding="utf-8-sig") as source:
        header = next(csv.reader(source), None)
        if not header or len(header) != 2 or header[0] != "observation_date":
            raise ValueError("Malformed cash-rate columns")
        for line_number, line in enumerate(source, start=2):
            try:
                date = pd.Timestamp(line.partition(",")[0].strip().strip('"')).normalize()
                if date.tzinfo is not None or pd.isna(date):
                    raise ValueError("Expected a cash observation calendar date")
                if date >= cutoff:
                    continue
                fields = next(csv.reader([line]))
                if len(fields) != 2:
                    raise ValueError("Wrong field count")
                value = np.nan if fields[1].strip() in ("", ".") else float(fields[1])
                if not np.isfinite(value) and fields[1].strip() not in ("", "."):
                    raise ValueError("Non-finite cash rate")
                dates.append(date)
                values.append(value)
            except (ValueError, csv.Error) as error:
                raise ValueError(f"Malformed {cash_file.name}, line {line_number}: {error}") from error
    series = pd.Series(values, index=pd.DatetimeIndex(dates, name="date"), name="rate", dtype=float).sort_index()
    if series.empty or not series.notna().any():
        raise ValueError("No valid cash-rate observations before the lock")
    duplicate = int(series.index.duplicated().sum())
    report = {"file": cash_file.name, "rows": len(series), "first": series.index[0], "last": series.index[-1],
              "duplicates": duplicate, "non_positive": np.nan, "above_25pct": np.nan, "blanks": int(series.isna().sum())}
    return series, report, duplicate > 0


def load_inputs():
    """Print every file's eligible-prefix audit, aborting before any evaluation."""
    print("Data report: dates are New York calendar dates; returns use Adj Close. Cash has no price-return checks.", flush=True)
    print(f"{'file':<12} {'rows':>6} {'first':<10} {'last':<10} {'dup':>4} {'nonpositive':>11} {'absret>25%':>11} {'blanks':>7}", flush=True)
    frames, reports = {}, []
    try:
        for name in (*tickers, "cash"):
            frame, report, malformed = _read_cash() if name == "cash" else _read_etf(name)
            reports.append(report)
            count = lambda value: "-" if pd.isna(value) else str(int(value))
            print(f"{report['file']:<12} {report['rows']:>6} {report['first']:%Y-%m-%d} {report['last']:%Y-%m-%d} "
                  f"{report['duplicates']:>4} {count(report['non_positive']):>11} {count(report['above_25pct']):>11} {report['blanks']:>7}", flush=True)
            if malformed:
                raise ValueError(f"Malformed observations in {report['file']}")
            frames[name] = frame
    except (OSError, ValueError) as error:
        print(f"STOP: {error}", flush=True)
        raise
    return {ticker: frames[ticker] for ticker in tickers}, frames["cash"], pd.DataFrame(reports)


def cash_returns(observed, calendar):
    validate_dates(observed)
    if calendar[-1] >= cutoff:
        raise ValueError("Cash calendar crosses the lock")
    usable = observed.copy()
    usable.index = usable.index + pd.Timedelta(days=1)
    result = usable.reindex(usable.index.union(calendar)).ffill().reindex(calendar) / 100 / 252
    if result.isna().any():
        raise ValueError("Cash history cannot be backward-filled")
    return result.rename("cash_return")

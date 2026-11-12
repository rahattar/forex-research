import numpy as np
import pandas as pd
import pytest

from scripts.data_report import largest_gap_hours
from src import loader


@pytest.fixture(scope="module", params=[(p, t) for p in loader.pairs for t in ("15m", "1h")])
def market_data(request):
    return loader.load_pair(*request.param)


@pytest.fixture
def write_csv(tmp_path, monkeypatch):
    monkeypatch.setattr(loader, "data_dir", tmp_path)

    def write(times, prices=None, pair="gbpusd", tf="m15"):
        index = pd.to_datetime(times, utc=True)
        if prices is None:
            prices = [[1.0, 2.0, 0.5, 1.5] for _ in times]
        frame = pd.DataFrame(prices, columns=list(loader.ohlc))
        frame.insert(0, "timestamp", index.as_unit("ms").asi8)
        frame.to_csv(tmp_path / f"{pair}-{tf}-bid-fixture.csv", index=False)

    return write


def test_flat_and_weekend_bars_removed(market_data):
    assert not market_data.empty
    assert str(market_data.index.tz) == "UTC"
    assert not market_data.eq(market_data["open"], axis=0).all(axis=1).any()
    assert (market_data.index.dayofweek < 5).all()


def test_unique_sorted_timestamps(market_data):
    assert market_data.index.is_unique
    assert market_data.index.is_monotonic_increasing


def test_real_test_period_stays_locked(market_data):
    assert (market_data.index < pd.Timestamp("2025-11-01", tz="UTC")).all()


@pytest.mark.parametrize("pair", loader.pairs)
def test_resampled_price_bounds(pair):
    hourly = loader.load_pair(pair, "1h")
    assert (hourly["high"] >= hourly[["open", "close"]].max(axis=1)).all()
    assert (hourly["low"] <= hourly[["open", "close"]].min(axis=1)).all()


def test_eurusd_hourly_equals_resampled_m15():
    quarter_hourly = loader.load_pair("eurusd", "15m")
    expected = quarter_hourly.resample("1h", label="right", closed="right").agg({
        "open": "first", "high": "max", "low": "min", "close": "last",
    }).dropna()
    expected = expected.loc[
        (expected.index <= quarter_hourly.index[-1]) & (expected.index.dayofweek < 5)
    ]
    pd.testing.assert_frame_equal(loader.load_pair("eurusd", "1h"), expected)


@pytest.mark.parametrize("frequency", ["15min", "1h"])
def test_split_boundaries(frequency):
    step = pd.Timedelta(frequency)
    train_start = pd.Timestamp("2019-01-01", tz="UTC")
    val_start = pd.Timestamp("2025-01-01", tz="UTC")
    test_start = pd.Timestamp("2025-11-01", tz="UTC")
    test_end = pd.Timestamp("2026-11-01", tz="UTC")
    index = pd.DatetimeIndex([
        train_start - step, train_start, val_start - step, val_start,
        test_start - step, test_start, test_end - step, test_end,
    ])
    df = pd.DataFrame({"value": range(len(index))}, index=index)
    train = loader.split(df, "train")
    val = loader.split(df, "val")
    test = loader.split(df, "test", final=True)
    assert train.index.tolist() == [train_start, val_start - step]
    assert val.index.tolist() == [val_start, test_start - step]
    assert test.index.tolist() == [test_start, test_end - step]
    assert val.index[0] - train.index[-1] == step
    assert train.index.intersection(val.index).empty
    assert val.index.intersection(test.index).empty
    pd.testing.assert_frame_equal(
        pd.concat([train, val]), df.loc[(index >= train_start) & (index < test_start)]
    )


def test_test_split_requires_explicit_final():
    with pytest.raises(ValueError, match="locked.*final=True"):
        loader.split(pd.DataFrame(), "test")


def test_loading_excludes_locked_prices_before_parsing(write_csv):
    write_csv(
        ["2025-10-31 12:00", "2025-11-03 12:00", "2025-10-31 12:15"],
        [[1, 2, 0.5, 1.5], ["locked"] * 4, [1.5, 3, 1, 2]],
    )
    loaded = loader.load_pair("gbpusd", "15m")
    assert loaded.index.tolist() == list(pd.to_datetime(
        ["2025-10-31 12:15", "2025-10-31 12:30"], utc=True
    ))


def test_final_loading_on_synthetic_data_only(write_csv):
    write_csv(["2025-10-31 12:00", "2025-11-03 12:00", "2026-11-02 12:00"])
    assert len(loader.load_pair("gbpusd", "15m")) == 1
    unlocked = loader.load_pair("gbpusd", "15m", final=True)
    assert len(unlocked) == 2
    assert len(loader.split(unlocked, "test", final=True)) == 1


def test_flat_weekend_filter_and_sort_on_synthetic_data(write_csv):
    write_csv(
        ["2025-01-06 10:15", "2025-01-05 22:00", "2025-01-06 10:00", "2025-01-06 09:45"],
        [[1, 2, 0.5, 1.5], [1, 2, 0.5, 1.5], [1, 2, 0.5, 1.5], [1, 1, 1, 1]],
    )
    df = loader.load_pair("gbpusd", "15m")
    assert df.index.tolist() == list(pd.to_datetime(
        ["2025-01-06 10:15", "2025-01-06 10:30"], utc=True
    ))


@pytest.mark.parametrize("prices", [None, [[1, 1, 1, 1], [1, 1, 1, 1]]])
def test_duplicate_timestamps_raise(write_csv, prices):
    write_csv(["2025-01-06 10:00"] * 2, prices)
    with pytest.raises(ValueError, match="Duplicate timestamps"):
        loader.load_pair("gbpusd", "15m")


@pytest.mark.parametrize("pair", loader.pairs)
def test_hourly_aggregation_and_no_lookahead(write_csv, pair):
    times = pd.date_range("2025-01-06 09:00", periods=8, freq="15min", tz="UTC")
    prices = [[10 + i, 20 + i, 5 + i, 11 + i] for i in range(8)]
    write_csv(times, prices, pair=pair)
    baseline = loader.load_pair(pair, "1h")
    cutoff = pd.Timestamp("2025-01-06 10:00", tz="UTC")
    assert baseline.index.tolist() == [cutoff, cutoff + pd.Timedelta("1h")]
    assert baseline.loc[cutoff].tolist() == [10, 23, 5, 14]
    prices[4] = [100, 1000, 0, 999]
    write_csv(times, prices, pair=pair)
    changed = loader.load_pair(pair, "1h")
    pd.testing.assert_frame_equal(baseline.loc[:cutoff], changed.loc[:cutoff])
    assert changed.loc[cutoff + pd.Timedelta("1h"), "high"] == 1000
    write_csv(times[:4], prices[:4], pair=pair)
    pd.testing.assert_frame_equal(
        baseline.loc[:cutoff], loader.load_pair(pair, "1h"), check_freq=False
    )


def test_native_hourly_file_is_never_read(write_csv):
    times = pd.date_range("2025-01-06 09:00", periods=4, freq="15min", tz="UTC")
    write_csv(times, pair="eurusd")
    native = loader.data_dir / "eurusd-h1-bid-fixture.csv"
    native.write_text("invalid native h1 file must not be read\n")
    hourly = loader.load_pair("eurusd", "1h")
    assert hourly.index.tolist() == [pd.Timestamp("2025-01-06 10:00", tz="UTC")]
    assert hourly.iloc[0].tolist() == [1, 2, 0.5, 1.5]


def test_partial_final_hour_is_not_published(write_csv):
    times = pd.date_range("2025-01-06 09:00", periods=5, freq="15min", tz="UTC")
    write_csv(times)
    assert loader.load_pair("gbpusd", "1h").index.tolist() == [pd.Timestamp("2025-01-06 10:00", tz="UTC")]


def test_daily_bars_use_only_completed_days():
    index = pd.date_range("2025-01-06 01:00", periods=36, freq="1h", tz="UTC")
    hourly = pd.DataFrame({
        "open": np.arange(36) + 10,
        "high": np.arange(36) + 20,
        "low": np.arange(36) + 5,
        "close": np.arange(36) + 11,
    }, index=index)
    daily = loader.daily_bars(hourly)
    midnight = pd.Timestamp("2025-01-07", tz="UTC")
    assert daily.index.tolist() == [midnight]
    assert daily.iloc[0].tolist() == [10, 43, 5, 34]
    changed = hourly.copy()
    changed.loc[changed.index > midnight, "high"] = 1000
    pd.testing.assert_frame_equal(daily, loader.daily_bars(changed))
    pd.testing.assert_frame_equal(daily, loader.daily_bars(hourly.loc[:midnight]))
    assert loader.daily_bars(hourly.loc[:midnight - pd.Timedelta("1h")]).empty


def test_daily_empty_input():
    empty = pd.DataFrame(columns=list(loader.ohlc), index=pd.DatetimeIndex([], tz="UTC"))
    pd.testing.assert_frame_equal(loader.daily_bars(empty), empty)


@pytest.mark.parametrize("times, expected", [
    ([], 0),
    (["2025-01-03 21:00"], 0),
    (["2025-01-03 21:00", "2025-01-06 01:00"], 4),
    (["2025-01-03 21:00", "2025-01-07 01:00"], 28),
    (["2025-01-06 10:00", "2025-01-06 10:15"], 0.25),
])
def test_gap_report_subtracts_weekends(times, expected):
    frame = pd.DataFrame(index=pd.to_datetime(times, utc=True))
    assert largest_gap_hours(frame) == expected

import numpy as np
import pandas as pd
import pytest

from scripts import test2_trend as trend
from scripts import test3_volatility as volatility
from src.instruments import Config
from src.loader import daily_bars, split


def bars(index, close):
    close = np.asarray(close)
    return pd.DataFrame({"open": close - 0.00002, "high": close + 0.0003,
                         "low": close - 0.0002, "close": close}, index=index)


@pytest.fixture
def hourly():
    index = pd.date_range("2024-01-01T01:00Z", "2025-02-03T12:00Z", freq="1h")
    index = index[index.dayofweek < 5]
    changes = np.random.default_rng(42).normal(0.00001, 0.0001, len(index))
    frame = bars(index, 1.1 + changes.cumsum())
    frame["high"] += np.arange(len(index)) % 24 * 0.00001
    return frame


@pytest.mark.parametrize("direction", [-1, 0, 1])
def test_daily_trend_definition_and_200_completed_days(direction):
    index = pd.date_range("2024-01-01T01:00Z", periods=201 * 24, freq="1h")
    frame = bars(index, 1.1 + direction * np.arange(len(index)) * 0.00001)
    state = trend.daily_trend(frame)
    assert state.iloc[:199].isna().all()
    assert (state.iloc[199:] == direction).all()
    boundary = state.index[199]
    assert trend.daily_trend(frame.loc[:boundary - pd.Timedelta("1h")]).notna().sum() == 0
    at_completion = trend.daily_trend(frame.loc[:boundary])
    assert at_completion.notna().sum() == 1
    assert at_completion.iloc[-1] == direction


def test_daily_state_cannot_change_when_future_bars_change(hourly):
    cutoff = pd.Timestamp("2024-12-03T12:00Z")
    baseline = trend.daily_trend(hourly)
    changed = hourly.copy()
    changed.loc[changed.index > cutoff, :] += 100
    altered = trend.daily_trend(changed)
    assert baseline.loc[:cutoff].notna().any()
    pd.testing.assert_series_equal(baseline.loc[:cutoff], altered.loc[:cutoff])
    pd.testing.assert_series_equal(baseline.loc[:cutoff], trend.daily_trend(hourly.loc[:cutoff]))


def test_daily_bar_completion_excludes_trailing_day(hourly):
    cutoff = pd.Timestamp("2025-01-07T12:00Z")
    prefix = hourly.loc[:cutoff]
    daily = daily_bars(prefix)
    assert daily.index[-1] == cutoff.normalize()
    first_day_hours = prefix.loc[(prefix.index > cutoff.normalize() - pd.Timedelta("1D")) &
                                 (prefix.index <= cutoff.normalize())]
    assert daily.iloc[-1]["high"] == first_day_hours["high"].max()
    assert daily.iloc[-1]["close"] == first_day_hours["close"].iloc[-1]
    assert not (daily.index > cutoff).any()


def test_trend_entry_uses_first_close_strictly_after_completion():
    index = pd.date_range("2025-01-06T00:00Z", periods=4, freq="1h")
    frame = bars(index, [1.1, 1.1001, 1.1004, 1.1005])
    signals = pd.Series([1], index=index[:1])
    sample = trend.signal_returns(frame, signals, 1, "eurusd", "val", Config())
    assert sample.index.tolist() == [index[1]]
    assert sample["gross"].iloc[0] == pytest.approx(3)
    assert sample["net"].iloc[0] == pytest.approx(2)


def test_trend_entry_and_endpoint_do_not_cross_splits():
    index = pd.date_range("2024-12-31T22:00Z", periods=5, freq="1h")
    frame = bars(index, [1.1, 1.1001, 1.2, 1.2001, 1.2004])
    signals = pd.Series(1, index=index[[0, 2]])
    assert trend.signal_returns(frame, signals, 1, "eurusd", "train", Config()).empty
    sample = trend.signal_returns(frame, signals, 1, "eurusd", "val", Config())
    assert sample.index.tolist() == [index[3]]
    assert sample["gross"].iloc[0] == pytest.approx(3)


def test_hourly_pullbacks_use_current_daily_state_and_delay_entry(monkeypatch):
    index = pd.date_range("2025-01-06T00:00Z", periods=7, freq="1h")
    frame = bars(index, 1.1 + np.arange(7) * 0.0001)
    monkeypatch.setattr(trend, "pairs", ("eurusd",))
    monkeypatch.setattr(trend, "horizons", (1,))
    monkeypatch.setattr(trend, "spans", (24,))
    monkeypatch.setattr(trend, "thresholds", (1.0,))
    monkeypatch.setattr(trend, "load_pair", lambda pair, tf: frame)
    monkeypatch.setattr(trend, "daily_trend", lambda hourly: pd.Series([1, -1], index=index[[0, 2]]))
    monkeypatch.setattr(trend, "residual_features", lambda close, span: pd.DataFrame(
        {"z": [0, -2, 2, 2, -2, -2, -2]}, index=index))
    grid = trend.build_grid()
    row = grid.loc[(grid["pair"] == "eurusd") & (grid["rule"] == "pullback") &
                   (grid["split"] == "val")].iloc[0]
    assert row.n_trades == 3
    assert row.mean_gross_pips == pytest.approx(-1 / 3)


def test_pooled_statistics_weight_trades_and_thin_each_pair():
    samples = [pd.DataFrame({"gross": [4, 6, 8], "net": [3, 5, 7]}),
               pd.DataFrame({"gross": [12], "net": [10]})]
    stats = trend.pooled_statistics(samples, h=2)
    assert stats["n_trades"] == 4
    assert stats["mean_gross_pips"] == 7.5
    assert stats["mean_net_pips"] == 6.25
    assert stats["n_nonoverlap"] == 3
    assert stats["mean_net_nonoverlap_pips"] == pytest.approx(20 / 3)


@pytest.mark.parametrize("cutoff", ["2024-12-03T12:00Z", "2025-01-07T12:00Z"])
def test_volatility_forecasts_do_not_use_future_bars(hourly, cutoff):
    cutoff = pd.Timestamp(cutoff)
    baseline = volatility.volatility_forecasts(hourly, "eurusd")
    changed = hourly.copy()
    changed.loc[changed.index > cutoff, :] += 100
    altered = volatility.volatility_forecasts(changed, "eurusd")
    assert baseline.loc[cutoff].notna().all()
    pd.testing.assert_frame_equal(baseline.loc[:cutoff], altered.loc[:cutoff])
    pd.testing.assert_frame_equal(
        baseline.loc[:cutoff], volatility.volatility_forecasts(hourly.loc[:cutoff], "eurusd")
    )


def test_seasonal_profile_is_frozen_for_validation(hourly):
    original = volatility.seasonal_factor(hourly)
    changed = hourly.copy()
    val = changed.index >= pd.Timestamp("2025-01-01", tz="UTC")
    changed.loc[val, "high"] += 100
    pd.testing.assert_series_equal(original, volatility.seasonal_factor(changed))
    factors = original.loc[val].groupby(original.loc[val].index.hour).nunique()
    assert (factors == 1).all()
    assert original.loc[val].nunique() > 1


def test_next_24_bar_range_excludes_signal_bar_and_requires_full_horizon():
    index = pd.date_range("2025-01-06T00:00Z", periods=50, freq="1h")
    frame = bars(index, 1.1 + np.arange(50) * 0.0001)
    frame.iloc[0, frame.columns.get_loc("high")] = 1000
    target = volatility.future_range(frame, "eurusd")
    expected = (frame["high"].iloc[1:25].max() - frame["low"].iloc[1:25].min()) / 0.0001
    assert target.iloc[0] == pytest.approx(expected)
    assert target.iloc[-24:].isna().all()
    past = volatility.past_range(frame, "eurusd")
    assert past.iloc[:23].isna().all()
    assert past.iloc[23] > 1_000_000


def test_volatility_training_metrics_ignore_validation_changes(hourly):
    baseline = volatility.pair_report("eurusd", hourly)
    changed = hourly.copy()
    changed.loc[changed.index >= pd.Timestamp("2025-01-01", tz="UTC"), :] += 100
    altered = volatility.pair_report("eurusd", changed)
    pd.testing.assert_frame_equal(baseline.loc[baseline["split"] == "train"],
                                  altered.loc[altered["split"] == "train"])
    train = split(hourly, "train")
    assert volatility.future_range(train, "eurusd").iloc[-24:].isna().all()


def test_forecast_metrics_and_strict_exceedance():
    stats = volatility.forecast_metrics(pd.Series([2, 4, 6]), pd.Series([3, 3, 3]), naive_mae=2)
    assert stats["mae_pips"] == pytest.approx(5 / 3)
    assert stats["r_squared"] == pytest.approx(-3 / 8)
    assert stats["mae_ratio"] == pytest.approx(5 / 6)
    assert stats["exceed_1x"] == pytest.approx(2 / 3)
    assert stats["exceed_1_5x"] == pytest.approx(1 / 3)
    assert stats["exceed_2x"] == 0


@pytest.mark.parametrize("module, expected_rows, expected_count", [
    (trend, 272, 84), (volatility, 42, 21),
])
def test_scripts_use_default_loader_and_write_expected_grids(module, expected_rows, expected_count,
                                                           hourly, monkeypatch, tmp_path, capsys):
    calls = []

    def load(pair, tf, **kwargs):
        assert tf == "1h"
        assert kwargs == {}
        calls.append(pair)
        return hourly

    destination = tmp_path / "grid.csv"
    monkeypatch.setattr(module, "load_pair", load)
    monkeypatch.setattr(module, "result_path", destination)
    module.main()
    assert calls == list(module.pairs)
    grid = pd.read_csv(destination)
    assert len(grid) == expected_rows
    assert set(grid["split"]) == {"train", "val"}
    if module is trend:
        trials = grid.loc[grid["pair"].isin(trend.pairs) & grid["rule"].isin(trend.rules)]
        assert len(trials[["pair", "rule", "span", "h", "k"]].drop_duplicates()) == expected_count
        assert len(trend.compact_summary(grid)) == 10
        assert len(trend.compact_summary(grid, pooled=True, count=1)) == 2
    else:
        assert len(grid[["pair", "model"]].drop_duplicates()) == expected_count
        assert np.allclose(grid.loc[grid["model"] == "naive", "mae_ratio"], 1)
        assert (grid.groupby(["pair", "split"])["n"].nunique() == 1).all()
    assert str(expected_count) in capsys.readouterr().out

import builtins
import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import run_fresh_confirm as runner
from scripts import run_signal_eval as original_runner
from scripts.run_variants import variants
from src import loader
from src.costs import swap_pips, swap_usd
from src.engine import EngineConfig, PreparedMarket, Signal
from src.instruments import Config, pip_size, pip_value_usd_per_lot
from src.signal_eval import FreshMarket, evaluate_signals


@pytest.fixture
def fresh_files(tmp_path, monkeypatch):
    root = tmp_path / "fresh"
    root.mkdir()
    monkeypatch.setattr(loader, "fresh_data_dir", root)

    def write(pair, times, values=None):
        times = pd.to_datetime(times, utc=True)
        if values is None:
            values = [[1.0, 1.1, 0.9, 1.01]] * len(times)
        frame = pd.DataFrame(values, columns=list(loader.ohlc))
        frame.insert(0, "timestamp", times.as_unit("ms").asi8)
        file = root / f"{pair}-m15-bid-fixture.csv"
        frame.to_csv(file, index=False)
        return file

    return write


@pytest.mark.parametrize("pair", loader.fresh_pairs)
def test_all_ten_pip_sizes_and_cost_settings(pair):
    assert pip_size(pair) == (0.01 if pair.endswith("jpy") else 0.0001)
    assert pip_size(pair.upper()) == pip_size(pair)
    expected = dict(eurusd=1.0, gbpusd=1.2, usdjpy=1.0, audusd=1.3, usdcad=1.7, usdchf=1.6,
                    nzdusd=1.8, eurjpy=1.8, gbpjpy=2.8, eurgbp=1.5)
    cfg = Config()
    assert cfg.spread_pips[pair] == expected[pair]
    assert cfg.slippage_pips == 1 and cfg.swap_pips_per_night == -0.2


@pytest.mark.parametrize("pair", loader.fresh_pairs)
def test_closing_time_cutoff_gates_before_price_parsing(pair, fresh_files):
    end = loader.fresh_end(pair)
    valid = end - pd.Timedelta(hours=3)
    while valid.dayofweek >= 5:
        valid -= pd.Timedelta(days=1)
    file = fresh_files(pair, [valid])
    with file.open("a") as out:
        # This candle closes exactly at the boundary and must not be parsed.
        out.write(f"{(end - pd.Timedelta(minutes=15)).value // 1_000_000},forbidden-price-fields\n")
        out.write(f"{pd.Timestamp('2026-01-01', tz='UTC').value // 1_000_000},forbidden-price-fields\n")
    bars = loader.load_fresh_pair(pair, "15m")
    assert bars.index.tolist() == [valid + pd.Timedelta(minutes=15)]
    assert (bars.index < end).all()
    assert bars.index[-1] < pd.Timestamp("2025-11-01", tz="UTC")
    assert bars.attrs["data_root"] == "fresh" and bars.attrs["pair"] == pair
    with pytest.raises(TypeError):
        loader.load_fresh_pair(pair, "15m", final=True)


def test_fresh_cleaning_and_completed_resampling(fresh_files):
    times = ["2018-01-08T09:00Z", "2018-01-08T09:15Z", "2018-01-08T09:30Z", "2018-01-08T09:45Z",
             "2018-01-08T10:00Z", "2018-01-08T10:15Z", "2018-01-06T10:00Z"]
    values = [[1, 1.1, .9, 1.01], [1.01, 1.2, .8, 1.02], [1.02, 1.15, .7, 1.03], [1.03, 1.1, .9, 1.04],
              [1.04, 1.1, .9, 1.05], [1.0] * 4, [1, 1.1, .9, 1.01]]
    fresh_files("eurusd", times[::-1], values[::-1])
    quarters = loader.load_fresh_pair("eurusd", "15m")
    assert len(quarters) == 5
    assert quarters.index.is_unique and quarters.index.is_monotonic_increasing
    assert (quarters.index.dayofweek < 5).all()
    hourly = loader.load_fresh_pair("eurusd", "1h")
    assert hourly.index.tolist() == [pd.Timestamp("2018-01-08T10:00Z")]
    assert hourly.iloc[0].tolist() == [1, 1.2, .7, 1.04]
    pd.testing.assert_frame_equal(hourly, loader.fresh_hourly(quarters))
    assert loader.daily_bars(hourly).empty
    with pytest.raises(ValueError, match="fresh-loader"):
        loader.fresh_hourly(pd.DataFrame())


@pytest.mark.parametrize("problem", ["missing", "header", "prices", "duplicate", "timestamp", "fields"])
def test_validation_stops_before_any_strategy_on_bad_file(problem, fresh_files, monkeypatch, capsys):
    file = fresh_files("eurusd", ["2018-01-08T09:00Z"])
    if problem == "missing":
        file.unlink()
    elif problem == "header":
        file.write_text("time,o,h,l,c\n")
    elif problem == "prices":
        fresh_files("eurusd", ["2018-01-08T09:00Z"], [[1, .5, .9, 1.01]])
    elif problem == "duplicate":
        fresh_files("eurusd", ["2018-01-08T09:00Z"] * 2)
    elif problem == "timestamp":
        file.write_text("timestamp,open,high,low,close\ninvalid,1,2,.5,1\n")
    elif problem == "fields":
        file.write_text("timestamp,open,high,low,close\n1515402000000,1,2,.5\n")

    def forbidden(*args, **kwargs):
        raise AssertionError("Strategy evaluation must wait for every file to validate")

    monkeypatch.setattr(runner, "load_fresh_pair", forbidden)
    monkeypatch.setattr(runner, "evaluate_signals", forbidden)
    with pytest.raises(ValueError):
        runner.main()
    assert "STOP:" in capsys.readouterr().out


def test_fresh_and_existing_roots_never_fall_back_or_coincide(fresh_files, tmp_path, monkeypatch):
    existing = tmp_path / "existing"
    existing.mkdir()
    monkeypatch.setattr(loader, "data_dir", existing)
    fresh_files("eurusd", ["2018-01-08T09:00Z"])
    with pytest.raises(ValueError, match="found 0"):
        loader.load_pair("eurusd", "15m")
    fresh = loader.fresh_data_dir
    (fresh / "eurusd-m15-bid-fixture.csv").rename(existing / "eurusd-m15-bid-fixture.csv")
    with pytest.raises(ValueError, match="found 0"):
        loader.load_fresh_pair("eurusd", "15m")
    monkeypatch.setattr(loader, "fresh_data_dir", existing)
    for load in (loader.load_pair, loader.load_fresh_pair):
        with pytest.raises(ValueError, match="separate"):
            load("eurusd", "15m")


def test_continuous_fresh_market_cannot_be_mixed_with_existing_market(fresh_files):
    fresh_files("audusd", ["2018-12-31T22:45Z", "2019-01-01T00:45Z", "2019-01-01T01:00Z"])
    quarter = loader.load_fresh_pair("audusd", "15m")
    hourly = loader.fresh_hourly(quarter)
    market = FreshMarket({"audusd": hourly}, {"audusd": quarter})
    assert market.times[0] < runner.boundary < market.times[-1]
    assert market.which == "fresh"
    with pytest.raises(ValueError, match="Fresh data"):
        PreparedMarket({"audusd": hourly}, {"audusd": quarter}, "train")
    wrong = quarter.copy()
    wrong.attrs = {}
    with pytest.raises(ValueError, match="provenance"):
        FreshMarket({"audusd": hourly}, {"audusd": wrong})
    wrong = quarter.copy()
    wrong.attrs["pair"] = "eurusd"
    with pytest.raises(ValueError, match="provenance"):
        FreshMarket({"audusd": hourly}, {"audusd": wrong})
    with pytest.raises(ValueError, match="locked"):
        PreparedMarket({"audusd": hourly}, {"audusd": quarter}, "test")


@pytest.mark.parametrize("pair", loader.fresh_pairs)
def test_real_fresh_files_obey_hard_cutoffs(pair):
    quarters = loader.load_fresh_pair(pair, "15m")
    hourly = loader.fresh_hourly(quarters)
    for frame in (quarters, hourly, loader.daily_bars(hourly)):
        assert frame.index[-1] < loader.fresh_end(pair)
        assert frame.index[-1] < runner.end
        assert frame.index[0] >= runner.start


@pytest.mark.parametrize("pair", loader.fresh_pairs)
@pytest.mark.parametrize("side", ["long", "short"])
def test_fresh_r_path_needs_no_usd_conversion(pair, side, fresh_files, monkeypatch):
    from src import instruments, signal_eval

    def forbidden(*args, **kwargs):
        raise AssertionError("No USD conversion or lots in fresh signal R")

    monkeypatch.setattr(instruments, "pip_value_usd_per_lot", forbidden)
    monkeypatch.setattr(signal_eval, "pip_value_usd_per_lot", forbidden)
    size = pip_size(pair)
    price = 150 if pair.endswith("jpy") else 1.1
    direction = 1 if side == "long" else -1
    stop = price - direction * 10 * size
    first = [price, price + size, price - size, price]
    second = [price, price + 11 * size, price - 11 * size, price]
    fresh_files(pair, ["2018-01-08T09:45Z", "2018-01-08T10:00Z"], [first, second])
    quarter = loader.load_fresh_pair(pair, "15m")
    hourly = loader.fresh_hourly(quarter)
    market = FreshMarket({pair: hourly}, {pair: quarter})
    signal = Signal(pair, side, stop, price + direction * 50 * size, "fixture")
    result = evaluate_signals(market, lambda data: [signal])
    assert len(result.trades) == 1
    assert result.trades.iloc[0].r_strategy == pytest.approx(-1, abs=1e-10)


def test_swap_in_pips_preserves_existing_usd_helper():
    cfg = Config()
    opened, closed = pd.Timestamp("2018-01-09T21:00Z"), pd.Timestamp("2018-01-11T23:00Z")
    accrual = swap_pips(opened, closed, cfg)
    assert accrual == -1  # Tuesday + triple Wednesday + Thursday.
    for pair, price in (("eurusd", 1.1), ("gbpusd", 1.4), ("usdjpy", 110)):
        assert swap_usd(pair, 1, opened, closed, price, cfg) / pip_value_usd_per_lot(pair, price) == pytest.approx(accrual)


def test_primary_rule_uses_only_pooled_b2_and_strict_lower_bound():
    sample = pd.DataFrame({"variant": ["B2", "C2", "B2"], "group": ["all", "all", "block"],
                           "d_ci_low": [0., 10., 10.]})
    assert runner.primary_result(sample) == "FAIL"
    sample.loc[0, "d_ci_low"] = .00001
    assert runner.primary_result(sample) == "PASS"
    sample.loc[0, "d_ci_low"] = np.nan
    assert runner.primary_result(sample) == "FAIL"
    assert [v[0] for v in variants] == ["A", "B1.5", "B2", "B3", "C1.5", "C2", "C3"]
    assert runner.bootstrap_resamples == 5000 and runner.bootstrap_seed == 0


def test_fresh_runner_uses_only_separate_root(fresh_files, tmp_path, monkeypatch, capsys):
    project = Path(__file__).resolve().parents[1]
    for pair in loader.fresh_pairs:
        fresh_files(pair, ["2018-01-08T09:00Z", "2018-01-08T09:45Z"])
    original_open, original_io_open = builtins.open, io.open

    def check_path(filename):
        if isinstance(filename, (str, Path)):
            source = Path(filename).resolve()
            assert not source.is_relative_to(project / "download")
            assert not source.is_relative_to(project / "data")

    def guarded_open(filename, *args, **kwargs):
        check_path(filename)
        return original_open(filename, *args, **kwargs)

    def guarded_io_open(filename, *args, **kwargs):
        check_path(filename)
        return original_io_open(filename, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", guarded_open)
    monkeypatch.setattr(io, "open", guarded_io_open)
    monkeypatch.setattr(runner, "results_dir", tmp_path)
    monkeypatch.setattr(runner, "bootstrap_resamples", 5)
    runner.main()
    trades = pd.read_csv(tmp_path / "fresh_signal_trades.csv")
    summary = pd.read_csv(tmp_path / "fresh_signal_summary.csv")
    assert trades.empty
    assert len(summary) == 209
    assert len(summary.loc[summary['group'] == "all"]) == 7
    output = capsys.readouterr().out
    assert "Primary hypothesis B2: FAIL" in output
    assert output.index("Fresh m15 data report") < output.index("Pooled fresh confirmation")


def test_task9_full_results_reproduce_unchanged(tmp_path, monkeypatch, capsys):
    """Integration regression against the saved, pre-confirmation Task 9 results."""
    project = Path(__file__).resolve().parents[1]
    expected_trades = pd.read_csv(project / "results" / "signal_trades.csv")
    expected_summary = pd.read_csv(project / "results" / "signal_summary.csv")
    (tmp_path / "variants_trades.csv").write_bytes((project / "results" / "variants_trades.csv").read_bytes())
    original_open, original_io_open = builtins.open, io.open

    def check_path(filename):
        if isinstance(filename, (str, Path)):
            source = Path(filename).resolve()
            assert not source.is_relative_to(project / "fresh")
            assert not source.is_relative_to(project / "data")

    def guarded_open(filename, *args, **kwargs):
        check_path(filename)
        return original_open(filename, *args, **kwargs)

    def guarded_io_open(filename, *args, **kwargs):
        check_path(filename)
        return original_io_open(filename, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", guarded_open)
    monkeypatch.setattr(io, "open", guarded_io_open)
    monkeypatch.setattr(original_runner, "results_dir", tmp_path)
    original_runner.main()
    actual_trades = pd.read_csv(tmp_path / "signal_trades.csv")
    actual_summary = pd.read_csv(tmp_path / "signal_summary.csv")
    pd.testing.assert_frame_equal(actual_trades, expected_trades, rtol=1e-10, atol=1e-11)
    pd.testing.assert_frame_equal(actual_summary, expected_summary, rtol=1e-10, atol=1e-11)
    for name in ("signal_trades.csv", "signal_summary.csv"):
        assert (tmp_path / name).read_bytes() == (project / "results" / name).read_bytes()
    assert len(actual_trades) == 8276

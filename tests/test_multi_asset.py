import builtins
import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import run_daily_carry as task12
from scripts import run_multi_asset as runner
from src import multi_asset_data as data
from src.multi_asset import (account_returns, evaluate, features, month_ends, normalize_weights,
                             paired_bootstrap, target_weights, trend_verdict)


def prices_fixture(n=800):
    index = pd.bdate_range("2000-01-03", periods=n)
    changes = np.random.default_rng(19).normal(.0002, .009, size=(n, len(data.tickers)))
    return pd.DataFrame(np.exp(changes.cumsum(axis=0)) * 100, index=index, columns=data.tickers)


@pytest.fixture
def input_files(tmp_path, monkeypatch):
    root = tmp_path / "etf"
    root.mkdir()
    cash = tmp_path / "dtb3.csv"
    monkeypatch.setattr(data, "etf_dir", root)
    monkeypatch.setattr(data, "cash_file", cash)

    def write(ticker, dates=None, prices=None):
        dates = pd.to_datetime(dates if dates is not None else ["2025-10-29", "2025-10-30", "2025-10-31"])
        prices = np.asarray(prices if prices is not None else np.arange(len(dates)) + 100, dtype=float)
        frame = pd.DataFrame({"Date": dates.tz_localize("America/New_York").astype(str),
                              "Open": prices, "High": prices + 1, "Low": prices - 1, "Close": prices,
                              "Adj Close": prices / 2, "Volume": 100, "Dividends": 0., "Stock Splits": 0.,
                              "Capital Gains": "ignored"})
        filename = root / f"{ticker}.csv"
        frame.to_csv(filename, index=False)
        return filename

    for ticker in data.tickers:
        write(ticker)
    cash.write_text("observation_date,value\n2025-10-28,2\n2025-10-29,3\n2025-10-30,4\n2025-10-31,5\n")
    return write, cash


def test_new_york_date_cutoff_precedes_numeric_parsing(input_files):
    write, cash_file = input_files
    filename = write("spy", ["2025-10-30"])
    with filename.open("a") as source:
        source.write("2025-11-01 01:00:00+00:00,100,101,99,100,50,100,0,0,ignored\n")
        source.write("2025-10-31 23:30:00-08:00,FORBIDDEN-PRICES\n")
        source.write("2026-01-02 00:00:00-05:00,NEVER-PARSE\n")
    frame, _, malformed = data._read_etf("spy")
    assert not malformed
    assert frame.index.tolist() == [pd.Timestamp("2025-10-30"), pd.Timestamp("2025-10-31")]
    with cash_file.open("a") as source:
        source.write("2025-11-01,FORBIDDEN-RATE\n2026-01-01,NEVER-PARSE\n")
    observed, _, _ = data._read_cash()
    before = data.cash_returns(observed, frame.index)
    observed.loc["2025-10-31"] = 999.
    pd.testing.assert_series_equal(before, data.cash_returns(observed, frame.index))
    with pytest.raises(ValueError, match="lock"):
        data.cash_returns(observed, pd.to_datetime(["2025-10-31", "2025-11-01"]))


def test_capital_gains_ignored_adjusted_returns_reported(input_files, capsys):
    write, _ = input_files
    write("spy", prices=[100, 126, 125])
    frames, _, report = data.load_inputs()
    assert frames["spy"]["Adj Close"].iloc[0] == 50
    assert "Capital Gains" not in frames["spy"]
    assert report.set_index("file").loc["spy.csv", "above_25pct"] == 1
    assert "spy.csv" in capsys.readouterr().out


@pytest.mark.parametrize("problem", ["missing", "header", "duplicate", "nonpositive", "badnumber", "missingcash"])
def test_missing_or_malformed_file_aborts_before_evaluation(problem, input_files, monkeypatch, capsys):
    write, cash_file = input_files
    filename = write("spy")
    if problem == "missing":
        filename.unlink()
    elif problem == "header":
        filename.write_text("Date,Close\n2025-10-30,100\n")
    elif problem == "duplicate":
        with filename.open("a") as stream:
            stream.write(filename.read_text().splitlines()[1] + "\n")
    elif problem == "nonpositive":
        frame = pd.read_csv(filename)
        frame.loc[0, "Adj Close"] = 0
        frame.to_csv(filename, index=False)
    elif problem == "badnumber":
        filename.write_text(filename.read_text().replace("100.0", "BAD", 1))
    else:
        cash_file.unlink()

    def forbidden(*args, **kwargs):
        raise AssertionError("No strategy may run before all files validate")

    monkeypatch.setattr(runner, "evaluate", forbidden)
    with pytest.raises((ValueError, OSError)):
        runner.main()
    assert "STOP:" in capsys.readouterr().out


def test_cash_ffill_next_calendar_day_and_no_backfill(input_files):
    _, filename = input_files
    filename.write_text("observation_date,value\n2019-01-03,.5\n2019-01-04,1\n2019-01-07,\n2019-01-08,3\n")
    observed, _, _ = data._read_cash()
    calendar = pd.to_datetime(["2019-01-04", "2019-01-07", "2019-01-08", "2019-01-09"])
    actual = data.cash_returns(observed, calendar)
    np.testing.assert_allclose(actual, np.array([.5, 1, 1, 3]) / 100 / 252)
    with pytest.raises(ValueError, match="backward"):
        data.cash_returns(observed, pd.to_datetime(["2019-01-03", "2019-01-04"]))


def test_signal_and_next_close_execution_use_no_later_prices():
    prices = prices_fixture()
    cash = pd.Series(.03 / 252, index=prices.index)
    signal_position = np.flatnonzero(month_ends(prices.index))[17]
    signal = prices.index[signal_position]
    execution = prices.index[signal_position + 1]
    original, _, info, _ = target_weights(prices, cash)
    changed = prices.copy()
    changed.loc[changed.index > signal] *= np.linspace(2, 5, len(prices.columns))
    altered, _, after, _ = target_weights(changed, cash)
    for name in original:
        pd.testing.assert_frame_equal(original[name].loc[:execution], altered[name].loc[:execution])
        # The signal is visible at month-end, and executes the following close.
        assert original[name].loc[signal].equals(original[name].loc[prices.index[signal_position - 1]])
    pd.testing.assert_frame_equal(info["vol"].loc[:signal], after["vol"].loc[:signal])


def test_eligibility_uses_full_252_return_history_and_staggered_inceptions():
    prices = prices_fixture()
    prices.loc[prices.index[:120], "hyg"] = np.nan
    info = features(prices, pd.Series(.02 / 252, index=prices.index))
    assert not info["eligible"].iloc[251].any()
    assert info["eligible"].iloc[252].sum() == 13
    assert not info["eligible"].hyg.iloc[371]
    assert info["eligible"].hyg.iloc[372]


def test_trend_uses_compounded_cash_not_only_spot_return():
    prices = prices_fixture()
    growth = .00001 + np.sin(np.arange(len(prices))) * .000005
    prices[:] = np.exp(growth.cumsum())[:, None] * np.ones((1, len(prices.columns)))
    info = features(prices, pd.Series(.0002, index=prices.index))
    assert info["long_score"].iloc[252:].eq(0).all().all()
    assert info["signed_score"].iloc[252:].eq(-1).all().all()


def test_gross_scaling_cash_and_all_book_dates():
    examples = pd.DataFrame([[2, .5], [2, -.5], [-.8, -.8]], columns=["a", "b"])
    actual = normalize_weights(examples)
    np.testing.assert_allclose(actual, [[.8, .2], [.8, -.2], [-.5, -.5]])
    assert actual.abs().sum(axis=1).le(1 + 1e-12).all()
    prices = prices_fixture()
    frames, weights, _, _, _ = evaluate(prices, pd.Series(.02 / 252, index=prices.index))
    for name, weight in weights.items():
        assert weight.abs().sum(axis=1).le(1 + 1e-12).all()
        assert frames[name].index.equals(frames["S1"].index)
    assert weights["T1"].ge(0).all().all()
    np.testing.assert_allclose(frames["T2"].cash_weight, 1 - weights["T2"].shift(1).fillna(0).sum(axis=1))


def test_execution_cost_cash_and_short_fee_hand_calculation():
    index = pd.date_range("2019-01-02", periods=3, freq="B")
    prices = pd.DataFrame({"spy": [100., 110., 121.]}, index=index)
    cash = pd.Series(.001, index=index)
    weights = pd.DataFrame({"spy": [0., .5, .5]}, index=index)
    result = account_returns(prices, cash, weights)
    # Buy 50% at the second close: cost .5*5bp, then earn .5*(10%-.1%).
    np.testing.assert_allclose(result.trading_cost, [0, .00025, 0])
    np.testing.assert_allclose(result.excess, [0, -.00025, .0495])
    np.testing.assert_allclose(result.total, [.001, .00075, .0505])
    short = account_returns(prices, cash, -weights, short_fee=.005)
    assert short.short_cost.iloc[2] == pytest.approx(.5 * .005 / 252)
    assert short.cash_weight.iloc[2] == 1.5
    assert short.excess.iloc[2] == pytest.approx(-.0495 - .5 * .005 / 252)


def test_paired_month_bootstrap_matches_explicit_daily_paths():
    index = pd.bdate_range("2019-01-01", periods=85)
    total = pd.DataFrame({"S1": np.sin(np.arange(85)) * .01 + .0002,
                          "T1": np.cos(np.arange(85)) * .007 + .0001, "cash": .0001}, index=index)
    excess = total.sub(.0001)
    sharpe, dd = paired_bootstrap(total, excess, 200, 7)
    repeated = paired_bootstrap(total, excess, 200, 7)
    np.testing.assert_allclose(sharpe, repeated[0], equal_nan=True)
    np.testing.assert_allclose(dd, repeated[1])
    groups = [np.flatnonzero(index.month == month) for month in pd.unique(index.month)]
    selections = np.random.default_rng(7).integers(0, len(groups), size=(200, len(groups)))
    expected_sharpe, expected_dd = [], []
    for draw in selections:
        positions = np.concatenate([groups[month] for month in draw])
        values = excess.iloc[positions].to_numpy()
        std = values.std(axis=0, ddof=1)
        expected_sharpe.append(np.divide(values.mean(axis=0) * np.sqrt(252), std,
                                         out=np.full(3, np.nan), where=std > 0))
        equity = np.vstack([np.ones(3), (1 + total.iloc[positions].to_numpy()).cumprod(axis=0)])
        expected_dd.append((1 - equity / np.maximum.accumulate(equity, axis=0)).max(axis=0))
    np.testing.assert_allclose(sharpe, expected_sharpe, atol=1e-12, equal_nan=True)
    np.testing.assert_allclose(dd, expected_dd, atol=1e-12)


@pytest.mark.parametrize("lower, point, t1dd, s1dd, expected", [
    (.001, -.2, .2, .1, "ADDS VALUE"), (0, -.1, .075, .1, "REDUCES RISK ONLY"),
    (-.1, -.1001, .05, .1, "NO BENEFIT"), (-.1, .1, .08, .1, "NO BENEFIT"),
    (-.1, .1, 0, 0, "NO BENEFIT"),
])
def test_exact_pre_registered_verdict(lower, point, t1dd, s1dd, expected):
    sample = pd.DataFrame({"group": ["comparison"], "period": ["all"], "sharpe_difference_ci_low": [lower],
                           "sharpe_difference": [point], "t1_drawdown": [t1dd], "s1_drawdown": [s1dd]})
    assert trend_verdict(sample) == expected


def test_prohibited_input_directories_and_locked_frames():
    project = Path(__file__).resolve().parents[1]
    with data.restricted_asset_reads():
        for opener in (builtins.open, io.open):
            with pytest.raises(ValueError, match="hist"):
                opener(project / "hist" / "not_read.csv")
            with pytest.raises(ValueError, match="15 named"):
                opener(project / "data" / "fred" / "dff.csv")
    prices = prices_fixture()
    prices.index = pd.bdate_range("2025-10-01", periods=len(prices))
    with pytest.raises(ValueError, match="locked"):
        features(prices, pd.Series(.0001, index=prices.index))


def test_task12_full_outputs_reproduce_unchanged(tmp_path, monkeypatch, capsys):
    project = Path(__file__).resolve().parents[1]
    for name in ("daily_m1_returns.csv", "daily_m2_returns.csv", "daily_d1_trades.csv", "daily_summary.csv"):
        (tmp_path / name).write_bytes((project / "results" / name).read_bytes())
    monkeypatch.setattr(task12, "results_dir", tmp_path)
    task12.main()
    for name in ("carry_c1_returns.csv", "carry_c2_returns.csv", "carry_summary.csv"):
        assert (tmp_path / name).read_bytes() == (project / "results" / name).read_bytes()
    assert capsys.readouterr().out.encode() == (project / "results" / "carry_report.txt").read_bytes()

import builtins
import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import run_daily_momentum as runner
from scripts import run_fresh_confirm as fresh_runner
from src import loader
from src.daily_breakout import (breakout_features, evaluate_breakout, simulate_breakout,
                                trade_bootstrap)
from src.daily_momentum import (currency_prices, daily_panel, end, evaluate_m1, evaluate_m2, first_in_period,
                                implementability, m1_features, m1_verdict, m1_weights, m2_weights,
                                portfolio_bootstrap, portfolio_returns, restricted_reads)
from src.engine import Signal
from src.signal_eval import mirror_signal


def close_fixture(n=720):
    index = pd.bdate_range("2010-01-04", periods=n, tz="UTC")
    rng = np.random.default_rng(17)
    changes = rng.normal(.0001, .006, size=(n, 10))
    levels = [1.2, 1.4, 110, .8, 1.2, 1.1, .7, 130, 155, .9]
    return pd.DataFrame(np.exp(np.cumsum(changes, axis=0)) * levels, index=index, columns=loader.fresh_pairs)


def hourly_fixture(n=220):
    index = pd.date_range("2010-01-04T01:00Z", periods=n * 24, freq="1h")
    index = index[index.dayofweek < 5]
    phase = np.arange(len(index))
    close = 1.2 + phase * .00002 + np.sin(phase / 70) * .003
    return pd.DataFrame({"open": close, "high": close + .0001, "low": close - .0001, "close": close}, index=index)


def test_m1_features_weights_and_returns_are_causal_and_lagged():
    closes = close_fixture()
    weights, _, _, _ = m1_weights(closes)
    cutoff = weights.index[117]
    changed = closes.copy()
    changed.loc[changed.index > cutoff] *= np.linspace(2, 10, 10)
    original = m1_features(closes)
    altered = m1_features(changed)
    for a, b in zip(original, altered):
        pd.testing.assert_frame_equal(a.loc[:cutoff], b.loc[:cutoff])
    pd.testing.assert_frame_equal(weights.loc[:cutoff], m1_weights(changed)[0].loc[:cutoff])
    baseline = evaluate_m1(closes)
    actual = evaluate_m1(changed)
    pd.testing.assert_frame_equal(baseline.loc[:cutoff], actual.loc[:cutoff])
    returns = closes.pct_change(fill_method=None).reindex(weights.index)
    expected = (weights.shift(1).fillna(0) * returns).sum(axis=1)
    pd.testing.assert_series_equal(baseline.gross, expected, check_names=False)
    assert (weights.abs() <= .3).all().all()
    changed_dates = weights.diff().abs().sum(axis=1).gt(1e-14)
    schedule = pd.Series(first_in_period(closes.index, "week"), index=closes.index)
    assert schedule.reindex(weights.index)[changed_dates].all()


def test_only_completed_daily_bars_can_change_a_weight_or_d1_signal():
    hourly = hourly_fixture(430)
    cutoff = pd.Timestamp("2011-02-01T00:00Z")
    before = hourly.loc[:cutoff].copy()
    partial = hourly.loc[:cutoff + pd.Timedelta(hours=12)].copy()
    partial.loc[partial.index > cutoff] *= 100
    frames_before = {pair: before.copy() for pair in loader.fresh_pairs}
    frames_partial = {pair: partial.copy() for pair in loader.fresh_pairs}
    first = daily_panel(frames_before)
    second = daily_panel(frames_partial)
    pd.testing.assert_frame_equal(first, second)
    pd.testing.assert_frame_equal(m1_weights(first)[0], m1_weights(second)[0])
    pd.testing.assert_frame_equal(breakout_features(before), breakout_features(partial))


def test_m2_rankings_and_weights_are_causal():
    closes = close_fixture()
    weights, _ = m2_weights(closes)
    cutoff = weights.index[85]
    changed = closes.copy()
    changed.loc[changed.index > cutoff] *= np.linspace(.1, 5, 10)
    pd.testing.assert_frame_equal(weights.loc[:cutoff], m2_weights(changed)[0].loc[:cutoff])
    pd.testing.assert_frame_equal(evaluate_m2(closes).loc[:cutoff], evaluate_m2(changed).loc[:cutoff])
    active = weights.loc[weights.abs().sum(axis=1) > 0]
    assert (active.eq(.25).sum(axis=1) == 2).all()
    assert (active.eq(-.25).sum(axis=1) == 2).all()
    assert active.sum(axis=1).abs().max() < 1e-12


@pytest.mark.parametrize("pair, currency", [("usdjpy", "jpy"), ("usdcad", "cad"), ("usdchf", "chf")])
def test_usd_base_currency_returns_use_exact_inversion(pair, currency):
    closes = close_fixture(3)
    closes[pair] = [100, 110, 99]
    prices = currency_prices(closes)
    returns = prices.pct_change(fill_method=None)
    assert returns[currency].iloc[1] == pytest.approx(100 / 110 - 1)
    assert returns[currency].iloc[2] == pytest.approx(110 / 99 - 1)
    assert returns.usd.iloc[1:].eq(0).all()


def test_cost_financing_and_next_day_application_hand_calculation():
    index = pd.to_datetime(["2019-01-04T00:00Z", "2019-01-05T00:00Z", "2019-01-08T00:00Z"])
    prices = pd.DataFrame({"eurusd": [1.0, 1.1, 1.21]}, index=index)
    weights = pd.DataFrame({"eurusd": [.2, -.1, -.1]}, index=index)
    # EURUSD: (.5*1 pip spread + .5 pip)*.0001=.0001 price units per unit turnover.
    result = portfolio_returns(prices, weights, prices, {"eurusd": .0001})
    np.testing.assert_allclose(result.gross, [0, .02, -.01])
    np.testing.assert_allclose(result.cost, [.2 * .0001, .3 * .0001 / 1.1, 0])
    np.testing.assert_allclose(result.financing, [0, .2 * .0075 / 365, .1 * .0075 * 3 / 365])
    np.testing.assert_allclose(result.net, result.gross - result.cost - result.financing)
    np.testing.assert_allclose(result.turnover, [.2, .3, 0])


def test_m2_usd_leg_needs_no_position_or_financing():
    result = evaluate_m2(close_fixture())
    assert result.weight_usd.eq(0).all()
    assert result.gross_usd.eq(0).all() and result.cost_usd.eq(0).all() and result.financing_usd.eq(0).all()
    active = result.gross_leverage.loc[result.gross_leverage > 0]
    assert active.between(.75, 1).all()


def test_continuous_join_no_overlap_and_audited_new_year_closure(monkeypatch):
    old = pd.DataFrame({"open": [1.0], "high": [1.2], "low": [.9], "close": [1.1]},
                       index=pd.to_datetime(["2018-12-31T22:00Z"]))
    recent = old.set_axis(pd.to_datetime(["2019-01-01T22:15Z"]))
    calls = []

    def fresh(pair, tf):
        calls.append(("fresh", pair, tf))
        return old.copy()

    def existing(pair, tf, **kwargs):
        assert not kwargs
        calls.append(("existing", pair, tf))
        return recent.copy()

    monkeypatch.setattr(loader, "load_fresh_pair", fresh)
    monkeypatch.setattr(loader, "load_pair", existing)
    result = loader.load_continuous_pair("eurusd", "15m")
    assert result.index.is_unique and result.index.is_monotonic_increasing
    assert result.attrs["join_overlap"] == 0 and result.attrs["join_gap_hours"] == 24.25
    assert calls == [("fresh", "eurusd", "15m"), ("existing", "eurusd", "15m")]
    recent.index = old.index
    with pytest.raises(ValueError, match="boundary|join"):
        loader.load_continuous_pair("eurusd", "15m")
    recent.index = pd.to_datetime(["2019-01-04T00:00Z"])
    with pytest.raises(ValueError, match="join"):
        loader.load_continuous_pair("eurusd", "15m")


def test_real_2019_joins_and_all_ten_hard_cutoffs():
    with restricted_reads():
        for pair in loader.fresh_pairs:
            hourly = loader.load_continuous_pair(pair, "1h")
            assert hourly.index[0] >= pd.Timestamp("2010-01-01", tz="UTC")
            assert hourly.index[-1] < end
            assert hourly.index.is_unique and hourly.index.is_monotonic_increasing
            if pair in loader.pairs:
                assert hourly.attrs["join_overlap"] == 0
                assert hourly.attrs["join_gap_hours"] == 24.25


@pytest.mark.parametrize("pair", loader.fresh_pairs)
def test_continuous_loader_does_not_parse_locked_prices(pair, tmp_path, monkeypatch):
    fresh_root, original_root = tmp_path / "fresh", tmp_path / "original"
    fresh_root.mkdir()
    original_root.mkdir()
    monkeypatch.setattr(loader, "fresh_data_dir", fresh_root)
    monkeypatch.setattr(loader, "data_dir", original_root)

    def write(root, time):
        ms = pd.Timestamp(time).value // 1_000_000
        file = root / f"{pair}-m15-bid-fixture.csv"
        file.write_text(f"timestamp,open,high,low,close\n{ms},1,1.2,.9,1.1\n"
                        f"{(end - pd.Timedelta(minutes=15)).value // 1_000_000},LOCKED-PRICES\n")

    write(fresh_root, "2018-12-31T21:45Z" if pair in loader.pairs else "2025-10-30T21:45Z")
    if pair in loader.pairs:
        write(original_root, "2019-01-01T22:00Z")
    result = loader.load_continuous_pair(pair, "15m")
    assert result.index[-1] < end


def test_blocked_directories_and_symlinks_cannot_be_opened(tmp_path, monkeypatch):
    project = Path(__file__).resolve().parents[1]
    with restricted_reads():
        for name in ("hist", "data"):
            blocked = project / name / "not_read.csv"
            for opener in (builtins.open, io.open):
                with pytest.raises(ValueError, match="cannot access"):
                    opener(blocked)
            link = tmp_path / name
            link.symlink_to(blocked)
            with pytest.raises(ValueError, match="cannot access"):
                link.open()
    monkeypatch.setattr(loader, "fresh_data_dir", project / "hist")
    with pytest.raises(ValueError, match="cannot read"):
        loader.load_continuous_pair("audusd", "1h")


def test_daily_research_helpers_reject_test_period():
    closes = close_fixture()
    closes.index = pd.bdate_range("2025-10-01", periods=len(closes), tz="UTC")
    for function in (m1_features, m2_weights):
        with pytest.raises(ValueError, match="locked"):
            function(closes)


@pytest.mark.parametrize("side", ["long", "short"])
def test_d1_mirror_entry_stop_and_clean_stop_r(side):
    hourly = pd.DataFrame({"open": [1.1, 1.1, 1.1], "high": [1.10001, 1.1011, 1.10001],
                           "low": [1.09999, 1.0989, 1.09999], "close": [1.1] * 3},
                          index=pd.date_range("2019-01-07T01:00Z", periods=3, freq="1h"))
    features = pd.DataFrame({"exit_long": False, "exit_short": False}, index=hourly.index)
    direction = 1 if side == "long" else -1
    signal = Signal("eurusd", side, 1.1 - direction * .001, np.inf if side == "long" else -np.inf, "D1")
    mirror = mirror_signal(signal, 1.1)
    first = simulate_breakout(hourly, features, "eurusd", signal, 0)
    second = simulate_breakout(hourly, features, "eurusd", mirror, 0)
    assert first["entry_time"] == second["entry_time"]
    assert first["side"] != second["side"]
    assert first["stop_distance_pips"] == pytest.approx(second["stop_distance_pips"])
    assert first["initial_risk_pips"] == pytest.approx(second["initial_risk_pips"])
    assert first["r"] == pytest.approx(-1, abs=1e-10)


def test_d1_own_channel_exit_no_target_breakeven_or_time_stop():
    index = pd.date_range("2019-01-07T01:00Z", periods=100, freq="1h")
    hourly = pd.DataFrame({"open": 1.1, "high": 1.2, "low": 1.0999, "close": 1.1}, index=index)
    features = pd.DataFrame({"exit_long": [False, True], "exit_short": [False, False]},
                            index=pd.to_datetime(["2019-01-07T00:00Z", "2019-01-10T00:00Z"]))
    signal = Signal("eurusd", "long", 1.0, np.inf, "D1")
    first = simulate_breakout(hourly, features, "eurusd", signal, 0)
    assert first["exit_time"] == pd.Timestamp("2019-01-10T01:00Z")
    assert first["exit_reason"] == "channel" and first["hold_days"] == 3
    assert first["stop"] == 1.0
    # The opposite side's stop is 1.2; remove touches to test its independent channel.
    hourly.high = 1.1001
    second = simulate_breakout(hourly, features, "eurusd", mirror_signal(signal, 1.1), 0)
    assert second["exit_reason"] == "end_of_data" and second["exit_time"] > first["exit_time"]


def test_d1_gap_stop_precedes_same_hour_daily_exit():
    index = pd.date_range("2019-01-07T23:00Z", periods=3, freq="1h")
    hourly = pd.DataFrame([[1.1, 1.1001, 1.0999, 1.1], [1.1, 1.1001, 1.0999, 1.1], [1.08, 1.09, 1.07, 1.08]],
                          columns=["open", "high", "low", "close"], index=index)
    features = pd.DataFrame({"exit_long": True, "exit_short": False}, index=pd.to_datetime(["2019-01-08T00:00Z"]))
    signal = Signal("eurusd", "long", 1.09, np.inf, "D1")
    trade = simulate_breakout(hourly, features, "eurusd", signal, 0)
    assert trade["exit_reason"] == "stop"
    assert trade["exit_fill"] == pytest.approx(1.0799)


def test_d1_features_no_lookahead_and_pair_occupancy():
    hourly = hourly_fixture()
    cutoff = hourly.index[len(hourly) // 2].normalize()
    changed = hourly.copy()
    changed.loc[changed.index > cutoff] *= 3
    pd.testing.assert_frame_equal(breakout_features(hourly).loc[:cutoff], breakout_features(changed).loc[:cutoff])
    before = evaluate_breakout(hourly, "eurusd")
    after = evaluate_breakout(changed, "eurusd")
    columns = ["signal_time", "entry_time", "side", "entry_bid", "initial_stop_bid", "initial_risk_pips"]
    pd.testing.assert_frame_equal(before.loc[before.entry_time <= cutoff, columns].reset_index(drop=True),
                                  after.loc[after.entry_time <= cutoff, columns].reset_index(drop=True))
    assert not before.empty
    assert (before.entry_time > before.signal_time).all()
    assert (before.entry_time.iloc[1:].reset_index(drop=True) >= before.exit_time.iloc[:-1].reset_index(drop=True)).all()


def test_atr_is_wilder_seeded_with_twenty_completed_true_ranges():
    features = breakout_features(hourly_fixture())
    daily = loader.daily_bars(hourly_fixture())
    tr = pd.concat([daily.high - daily.low, (daily.high - daily.close.shift(1)).abs(),
                    (daily.low - daily.close.shift(1)).abs()], axis=1).max(axis=1)
    assert features.atr.iloc[:19].isna().all()
    assert features.atr.iloc[19] == pytest.approx(tr.iloc[:20].mean())
    assert features.atr.iloc[20] == pytest.approx((features.atr.iloc[19] * 19 + tr.iloc[20]) / 20)


def test_monthly_portfolio_bootstrap_matches_hand_draws_and_fixed_seed():
    index = pd.bdate_range("2019-01-01", periods=80, tz="UTC")
    frame = pd.DataFrame({"gross": np.sin(np.arange(80)) * .01, "net": np.sin(np.arange(80)) * .01 - .0001}, index=index)
    result = portfolio_bootstrap(frame, 250, 9)
    assert result == portfolio_bootstrap(frame, 250, 9)
    months = [group for _, group in frame.groupby(frame.index.month)]
    rng = np.random.default_rng(9)
    draws = rng.integers(0, len(months), size=(250, len(months)))
    sharpes = []
    for selected in draws:
        values = np.concatenate([months[i].net.to_numpy() for i in selected])
        sharpes.append(values.mean() / values.std(ddof=1) * np.sqrt(252))
    assert result["net_sharpe_ci_low"] == pytest.approx(np.percentile(sharpes, 2.5))
    assert result["net_sharpe_ci_high"] == pytest.approx(np.percentile(sharpes, 97.5))


def test_monthly_trade_bootstrap_fixed_seed_and_joint_pairs():
    trades = pd.DataFrame({"entry_time": pd.to_datetime(["2019-01-03T01:00Z", "2019-01-04T01:00Z", "2019-02-05T01:00Z"]),
                           "pair": ["eurusd", "gbpusd", "eurusd"], "r_strategy": [1., 3., -1.], "d": [.5, 1.5, -.5]})
    first = trade_bootstrap(trades, 100, 7)
    assert first == trade_bootstrap(trades, 100, 7)
    assert first["n_months"] == 190
    assert first["r_ci_low"] == pytest.approx(first["d_ci_low"] * 2)


@pytest.mark.parametrize("lower, sharpe, first_half, second_half, expected", [
    (.00001, -.1, -1, -1, "PASS"), (0, .30, .01, .02, "PROMISING"),
    (-.1, .29, .1, .1, "FAIL"), (-.1, .3, -.1, .1, "FAIL"), (-.1, .3, .1, 0, "FAIL"),
])
def test_exact_primary_rule(lower, sharpe, first_half, second_half, expected):
    summary = pd.DataFrame({"strategy": ["M1"] * 3, "group": ["portfolio"] * 3,
                            "period": ["all", "2010_2017", "2018_2025"], "net_sharpe_ci_low": [lower, 0, 0],
                            "net_sharpe": [sharpe] * 3, "net_total_return": [0, first_half, second_half]})
    assert m1_verdict(summary) == expected


def test_position_units_and_sharpe_implied_dollars():
    closes = close_fixture()
    positions = implementability(closes, .4)
    assert len(positions) == 40
    assert positions.share_ge_1000.between(0, 1).all()
    for row in positions.itertuples(index=False):
        assert row.expected_annual_usd == pytest.approx(row.balance * row.target_vol * .4)
    assert positions.loc[(positions.balance == 200) & (positions.target_vol == .1), "median_units"].max() < 1000


def test_task10_full_results_reproduce_unchanged(tmp_path, monkeypatch, capsys):
    project = Path(__file__).resolve().parents[1]
    monkeypatch.setattr(fresh_runner, "results_dir", tmp_path)
    with restricted_reads():
        fresh_runner.main()
    for name in ("fresh_signal_trades.csv", "fresh_signal_summary.csv"):
        assert (tmp_path / name).read_bytes() == (project / "results" / name).read_bytes()

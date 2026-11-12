from functools import partial

import numpy as np
import pandas as pd
import pytest

from scripts import run_variants as runner
from src.engine import EngineConfig, PreparedMarket, VisibleData, run_engine
from src.strategies import MatchedRandom, build_features, matched_entries, swing_signal, trend_swing, trend_swing_15m


def view(time="2025-01-06T09:00Z", side="long", close=1.1, hourly=True, state=None,
         high=None, low=None, previous_high=1.1001, previous_low=1.0999):
    direction = 1 if side == "long" else -1
    row = {"state": direction if state is None else state, "z": -2 * direction,
           "range_pips": 20, "close": 1.1, "previous_close": 1.1 - direction * 0.0001,
           "ready": True, "swing_low": 1.099, "swing_high": 1.101}
    bar = (close, max(close, 1.1001) if high is None else high,
           min(close, 1.0999) if low is None else low, close)
    previous = (1.1, previous_high, previous_low, 1.1)
    return VisibleData(pd.Timestamp(time), {"eurusd": close}, {"eurusd": row}, {},
                       {"eurusd": bar}, {"eurusd": previous}, ("eurusd",) if hourly else ())


def market_frames(end="2025-01-06T16:00Z", pairs=("eurusd",)):
    index = pd.date_range("2025-01-06T09:00Z", end, freq="15min")
    quarters = {}
    for pair in pairs:
        close = 1.1 if pair == "eurusd" else 1.25
        quarters[pair] = pd.DataFrame({"open": close, "high": close + 0.0001,
                                       "low": close - 0.0001, "close": close}, index=index)
    hourly = {pair: frame.loc[frame.index.minute == 0].copy() for pair, frame in quarters.items()}
    return hourly, quarters


@pytest.mark.parametrize("side", ["long", "short"])
def test_swing_stop_uses_exactly_12_completed_bars_and_no_future(side):
    index = pd.date_range("2025-01-06T00:00Z", periods=36, freq="1h")
    hourly = pd.DataFrame({"open": 1.1, "high": 1.101, "low": 1.099, "close": 1.1}, index=index)
    hourly.iloc[0, hourly.columns.get_loc("low")] = 0.5
    hourly.iloc[0, hourly.columns.get_loc("high")] = 2
    cutoff = index[24]
    hourly.loc[cutoff, ["low", "high"]] = [1.0989, 1.1011]
    before = build_features(hourly, "eurusd")
    altered = hourly.copy()
    altered.loc[altered.index > cutoff, ["low", "high"]] = [0.1, 100]
    after = build_features(altered, "eurusd")
    pd.testing.assert_series_equal(before.loc[cutoff], after.loc[cutoff])
    data = view(str(cutoff), side)
    data.features["eurusd"].update({"swing_low": before.loc[cutoff, "swing_low"],
                                    "swing_high": before.loc[cutoff, "swing_high"]})
    signal = trend_swing(data, rr=2)[0]
    expected = hourly.low.iloc[13:25].min() - 0.0002 if side == "long" else hourly.high.iloc[13:25].max() + 0.0002
    assert signal.stop == pytest.approx(expected)
    assert signal.stop == pytest.approx(1.0987 if side == "long" else 1.1013)


@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize("ratio, accepted", [(0.299, False), (0.3, True), (1, True), (1.5, True), (1.501, False), (-1, False)])
def test_stop_distance_skip_rule_includes_boundaries(side, ratio, accepted):
    direction = 1 if side == "long" else -1
    stop = 1.1 - direction * ratio * 0.01
    signal = swing_signal("eurusd", side, 1.1, stop, 100, 2, EngineConfig(), "fixture")
    assert (signal is not None) == accepted


@pytest.mark.parametrize("side", ["long", "short"])
def test_15m_trigger_cannot_enter_arming_candle_and_keeps_structural_stop(side):
    strategy = trend_swing_15m(2)
    arm_view = view(side=side, previous_high=1.0998, previous_low=1.1002)
    assert strategy(arm_view) == []
    original = strategy.armed["eurusd"]
    assert original.stop == pytest.approx(1.0988 if side == "long" else 1.1012)
    close = 1.1002 if side == "long" else 1.0998
    trigger = view("2025-01-06T09:15Z", side, close, hourly=False)
    trigger.features["eurusd"].update({"swing_low": 1.09, "swing_high": 1.12})
    signals = strategy(trigger)
    assert len(signals) == 1
    assert signals[0].stop == original.stop
    assert signals[0].side == side
    assert not strategy.armed


@pytest.mark.parametrize("time, fires", [("2025-01-06T13:00Z", True), ("2025-01-06T13:15Z", False)])
def test_trigger_four_hour_expiry_and_repeated_signals_do_not_extend_it(time, fires):
    strategy = trend_swing_15m(2)
    strategy(view())
    strategy(view("2025-01-06T10:00Z"))
    assert strategy.armed["eurusd"].expires == pd.Timestamp("2025-01-06T13:00Z")
    signals = strategy(view(time, close=1.1002, hourly=False))
    assert bool(signals) == fires
    assert not strategy.armed


@pytest.mark.parametrize("new_state", [-1, 0])
def test_trigger_disarms_on_opposite_or_flat_trend(new_state):
    strategy = trend_swing_15m(2)
    strategy(view())
    assert not strategy(view("2025-01-06T09:15Z", close=1.1002, hourly=False, state=new_state))
    assert not strategy.armed


@pytest.mark.parametrize("side", ["long", "short"])
def test_stop_breach_disarms_even_when_the_same_candle_triggers(side):
    strategy = trend_swing_15m(2)
    strategy(view(side=side))
    stop = strategy.armed["eurusd"].stop
    trigger = view("2025-01-06T09:15Z", side, 1.1002 if side == "long" else 1.0998,
                   hourly=False, low=stop if side == "long" else 1.0997,
                   high=stop if side == "short" else 1.1003)
    assert strategy(trigger) == []
    assert not strategy.armed


def test_trigger_rechecks_entry_distance_without_moving_stop():
    strategy = trend_swing_15m(2)
    strategy(view())
    # The initial stop is valid, but this trigger's entry would be >1.5 forecasts away.
    assert strategy(view("2025-01-06T09:15Z", close=1.103, hourly=False)) == []
    assert not strategy.armed


def test_completed_quarter_bar_trigger_is_causal_and_engine_can_enter_between_hours():
    hourly, quarters = market_frames()
    quarter = quarters["eurusd"]
    quarter.loc[pd.Timestamp("2025-01-06T09:15Z"), ["high", "close"]] = [1.1003, 1.1002]
    row = view().features["eurusd"]
    features = {"eurusd": pd.DataFrame([row] * len(hourly["eurusd"]), index=hourly["eurusd"].index)}
    market = PreparedMarket(hourly, quarters, "val", features)
    altered_quarters = {"eurusd": quarter.copy()}
    cutoff = pd.Timestamp("2025-01-06T09:15Z")
    altered_quarters["eurusd"].loc[quarter.index > cutoff] += 100
    altered = PreparedMarket(hourly, altered_quarters, "val", features)

    def prefix_signals(prepared):
        strategy = trend_swing_15m(2)
        signals = []
        for i, time in enumerate(prepared.times):
            if time > cutoff:
                break
            signals.extend((time, signal) for signal in strategy(prepared.quarter_context(i)))
        return signals

    expected = prefix_signals(market)
    assert len(expected) == 1 and expected[0][0] == cutoff
    assert prefix_signals(altered) == expected
    result = run_engine(market, trend_swing_15m(2), EngineConfig(max_hold_hours=0.25))
    assert result.trades.iloc[0].entry_time == cutoff
    assert result.trades.iloc[0].setup == "trend_swing_15m"


@pytest.mark.parametrize("side", ["long", "short"])
def test_exact_1_5_cost_inclusive_r_passes_existing_gate(side):
    hourly, quarters = market_frames()
    candidate = swing_signal("eurusd", side, 1.1, 1.0988 if side == "long" else 1.1012,
                             20, 1.5, EngineConfig(), "fixture")
    quarters["eurusd"].loc[pd.Timestamp("2025-01-06T09:15Z")] = (
        [1.1, candidate.target + 0.00001, 1.1, candidate.target] if side == "long" else
        [1.1, 1.1, candidate.target - 0.00001, candidate.target]
    )
    market = PreparedMarket(hourly, quarters, "val")
    strategy = lambda data: [candidate] if data.time == pd.Timestamp("2025-01-06T09:00Z") else []
    result = run_engine(market, strategy, EngineConfig(swap_pips_per_night=0))
    assert result.trades.iloc[0].exit_reason == "target"
    assert result.trades.iloc[0].r_multiple == pytest.approx(1.5)


def source_trades():
    return pd.DataFrame([
        {"entry_time": pd.Timestamp("2025-01-06T09:15Z"), "pair": "eurusd", "side": "long",
         "entry_bid": 1.1, "initial_stop": 1.099, "stop": 1.1001, "target": 1.103},
        {"entry_time": pd.Timestamp("2025-01-06T13:15Z"), "pair": "eurusd", "side": "short",
         "entry_bid": 1.1, "initial_stop": 1.1011, "stop": 1.1, "target": 1.0971},
    ])


def test_matched_random_replays_entry_times_and_initial_distances_deterministically():
    template = matched_entries(source_trades())
    np.testing.assert_allclose(template.stop_pips, [10, 10])
    np.testing.assert_allclose(template.target_pips, [30, 30])
    hourly, quarters = market_frames()
    market = PreparedMarket(hourly, quarters, "val")
    cfg = EngineConfig(max_hold_hours=1)
    first = run_engine(market, MatchedRandom(template, 0), cfg)
    same = run_engine(market, MatchedRandom(template, 0), cfg)
    pd.testing.assert_frame_equal(first.trades, same.trades)
    assert first.trades.entry_time.tolist() == template.entry_time.tolist()
    replayed = matched_entries(first.trades)
    np.testing.assert_allclose(replayed[["stop_pips", "target_pips"]], template[["stop_pips", "target_pips"]])
    assert MatchedRandom(template, 0).schedule != MatchedRandom(template, 1).schedule


def test_matched_random_keeps_account_gates_and_never_reschedules():
    hourly, quarters = market_frames(pairs=("eurusd", "gbpusd"))
    template = pd.DataFrame([
        {"entry_time": pd.Timestamp("2025-01-06T09:15Z"), "pair": "eurusd", "stop_pips": 10, "target_pips": 30},
        {"entry_time": pd.Timestamp("2025-01-06T13:15Z"), "pair": "gbpusd", "stop_pips": 10, "target_pips": 30},
    ])
    market = PreparedMarket(hourly, quarters, "val")
    replay = run_engine(market, MatchedRandom(template, 0), record_events=True)
    assert len(replay.trades) == 1
    assert replay.trades.entry_time.iloc[0] == template.entry_time.iloc[0]
    assert replay.summary["skipped_usd_direction"] == 1


def test_pass_rule_uses_only_pre_registered_rank_on_both_splits():
    rows = []
    for name, ranks in (("A", (90, 90)), ("B", (99, 89.5)), ("C", (99, np.nan))):
        for which, rank in zip(("train", "val"), ranks):
            rows.append({"variant": name, "split": which, "expectancy_rank": rank, "balance_rank": 0})
    rows.append({"variant": "D", "split": "train", "expectancy_rank": 99, "balance_rank": 100})
    result = runner.apply_pass_rule(pd.DataFrame(rows))
    assert set(result.loc[result.pass_both_splits, "variant"]) == {"A"}


def test_runner_has_only_seven_variants_and_never_loads_test(tmp_path, monkeypatch, capsys):
    assert len(runner.variants) == 7
    assert len(runner.benchmark_seeds) == 200
    assert [name for name, _, _ in runner.variants] == ["A", "B1.5", "B2", "B3", "C1.5", "C2", "C3"]
    index = pd.to_datetime(["2024-12-30T09:00Z", "2024-12-30T10:00Z", "2025-01-06T09:00Z", "2025-01-06T10:00Z"])
    calls = []

    def load(pair, tf, **kwargs):
        assert kwargs == {}
        calls.append((pair, tf))
        price = 150 if pair == "usdjpy" else 1.1
        return pd.DataFrame({"open": price, "high": price, "low": price, "close": price}, index=index)

    monkeypatch.setattr(runner, "load_pair", load)
    monkeypatch.setattr(runner, "benchmark_seeds", (0, 1))
    monkeypatch.setattr(runner, "results_dir", tmp_path)
    runner.main()
    assert len(calls) == 6
    summary = pd.read_csv(tmp_path / "variants_summary.csv")
    assert len(summary) == 14 and not summary.pass_both_splits.any()
    assert set(summary["split"]) == {"train", "val"}
    assert len(pd.read_csv(tmp_path / "variants_random.csv")) == 28
    assert "Passing variants: none" in capsys.readouterr().out


def test_other_rr_values_are_rejected():
    with pytest.raises(ValueError, match="Only rr"):
        trend_swing(view(), 4)
    with pytest.raises(ValueError, match="Only rr"):
        trend_swing_15m(4)

from dataclasses import replace
from functools import partial

import numpy as np
import pandas as pd
import pytest

from scripts import run_engine as runner
from src.costs import swap_usd
from src.engine import EngineConfig, Position, PreparedMarket, Signal, move_to_breakeven, run_engine, usd_direction
from src.strategies import build_features, construct_signal, random_entry, signal_probabilities, trend_pullback


prices = {"eurusd": 1.1, "gbpusd": 1.25, "usdjpy": 150.0}


def frames(start="2025-01-06T09:00Z", end="2025-01-06T17:00Z", pairs=("eurusd",)):
    index = pd.date_range(start, end, freq="15min")
    index = index[index.dayofweek < 5]
    quarters = {}
    for pair in pairs:
        price = prices[pair]
        wiggle = 0.001 if pair == "usdjpy" else 0.00001
        quarters[pair] = pd.DataFrame({"open": price, "high": price + wiggle,
                                       "low": price - wiggle, "close": price}, index=index)
    return quarters


def prepare(quarters, which="val", features=None):
    hourly = {pair: frame.loc[frame.index.minute == 0].copy() for pair, frame in quarters.items()}
    return PreparedMarket(hourly, quarters, which, features)


def signal(pair="eurusd", side="long", setup="fixture"):
    direction = 1 if side == "long" else -1
    distance = 0.1 if pair == "usdjpy" else 0.001
    return Signal(pair, side, prices[pair] - direction * distance,
                  prices[pair] + direction * 3 * distance, setup)


def planned(plan):
    normalized = {pd.Timestamp(time): signals for time, signals in plan.items()}
    return lambda data: normalized.get(data.time, [])


def execute(quarters, plan, cfg=None):
    return run_engine(prepare(quarters), planned(plan), cfg, record_events=True)


def entries(result):
    return [event for event in result.events if event["kind"] == "entry"]


def skips(result, reason):
    return [event for event in result.events if event["kind"] == "skip" and event["reason"] == reason]


def test_new_trade_cannot_exit_using_its_entry_bar():
    quarters = frames()
    quarters["eurusd"].loc[pd.Timestamp("2025-01-06T09:00Z"), "low"] = 1.09
    cfg = EngineConfig(max_hold_hours=1)
    result = execute(quarters, {"2025-01-06T09:00Z": [signal()]}, cfg)
    row = result.trades.iloc[0]
    assert row.exit_reason == "time"
    assert row.exit_time == pd.Timestamp("2025-01-06T10:00Z")
    assert row.entry_fill == pytest.approx(1.1001)


def test_short_bid_levels_convert_to_ask_exactly_once():
    quarters = frames()
    quarters["eurusd"].loc[pd.Timestamp("2025-01-06T09:15Z"), "high"] = 1.101
    result = execute(quarters, {"2025-01-06T09:00Z": [signal(side="short")]})
    row = result.trades.iloc[0]
    assert row.initial_stop == pytest.approx(1.1011)
    assert row.target == pytest.approx(1.0971)
    assert row.exit_reason == "stop"
    assert row.exit_time == pd.Timestamp("2025-01-06T09:15Z")
    assert row.exit_fill == pytest.approx(1.1012)
    assert row.pnl_usd == pytest.approx(-row.risk_usd)


def test_15m_stop_first_and_gap_fill():
    quarters = frames()
    quarters["eurusd"].loc[pd.Timestamp("2025-01-06T09:15Z")] = [1.098, 1.104, 1.097, 1.1]
    result = execute(quarters, {"2025-01-06T09:00Z": [signal()]})
    assert result.trades.iloc[0].exit_reason == "stop"
    assert result.trades.iloc[0].exit_fill == pytest.approx(1.0979)
    assert result.trades.iloc[0].r_multiple < -1


@pytest.mark.parametrize("side", ["long", "short"])
def test_breakeven_uses_executable_quotes_and_activates_next_bar(side):
    quarters = frames()
    if side == "long":
        quarters["eurusd"].loc[pd.Timestamp("2025-01-06T09:15Z")] = [1.1, 1.1015, 1.0999, 1.1008]
        quarters["eurusd"].loc[pd.Timestamp("2025-01-06T09:30Z")] = [1.1008, 1.101, 1.1, 1.1005]
        expected_stop, expected_fill = 1.1001, 1.1
    else:
        quarters["eurusd"].loc[pd.Timestamp("2025-01-06T09:15Z")] = [1.1, 1.10005, 1.0986, 1.099]
        quarters["eurusd"].loc[pd.Timestamp("2025-01-06T09:30Z")] = [1.099, 1.1000, 1.0988, 1.0995]
        expected_stop, expected_fill = 1.1, 1.1001
    result = execute(quarters, {"2025-01-06T09:00Z": [signal(side=side)]})
    row = result.trades.iloc[0]
    assert row.breakeven
    assert row.stop == pytest.approx(expected_stop)
    assert row.exit_fill == pytest.approx(expected_fill)
    assert row.exit_time == pd.Timestamp("2025-01-06T09:30Z")
    moves = [event for event in result.events if event["kind"] == "breakeven"]
    assert len(moves) == 1
    assert moves[0]["time"] == pd.Timestamp("2025-01-06T09:15Z")
    assert moves[0]["stop"] == moves[0]["entry_fill"]


@pytest.mark.parametrize("side, stop", [("long", 1.1005), ("short", 1.0995)])
def test_breakeven_never_widens_a_better_stop(side, stop):
    position = Position("eurusd", side, "fixture", 1.1, 1.1, stop, 1.2, stop,
                        0.01, 1, 0.001, pd.Timestamp("2025-01-06T09:00Z"))
    assert not move_to_breakeven(position)
    assert position.stop == stop


def test_swap_balance_reconciliation_and_realized_drawdown():
    quarters = frames("2025-01-07T21:00Z", "2025-01-08T23:30Z")
    cfg = EngineConfig(max_hold_hours=26)
    result = execute(quarters, {"2025-01-07T21:00Z": [signal()]}, cfg)
    row = result.trades.iloc[0]
    expected_swap = swap_usd("eurusd", row.lots, row.entry_time, row.exit_time, row.exit_fill, cfg)
    assert row.swap == pytest.approx(expected_swap)
    assert row.swap == pytest.approx(-0.8 * 10 * row.lots)
    expected_pnl = (row.exit_fill - row.entry_fill) / 0.0001 * 10 * row.lots + expected_swap
    assert row.pnl_usd == pytest.approx(expected_pnl)
    assert cfg.start_balance + result.trades.pnl_usd.sum() == pytest.approx(result.summary["ending_balance"])
    np.testing.assert_allclose(result.trades.balance_after, cfg.start_balance + result.trades.pnl_usd.cumsum())
    assert result.summary["max_drawdown_usd"] == pytest.approx(-expected_pnl)
    assert result.summary["max_drawdown_pct"] == pytest.approx(-expected_pnl / 200)


def test_min_lot_skip_and_no_trades_summary():
    too_large = Signal("eurusd", "long", 1.05, 1.3, "wide")
    result = execute(frames(), {"2025-01-06T09:00Z": [too_large]})
    assert result.trades.empty
    assert result.summary["skipped_min_lot"] == 1
    assert result.summary["ending_balance"] == 200
    assert np.isnan(result.summary["expectancy_r"])


def test_size_compounds_from_current_balance_and_reconciles_multiple_trades():
    quarters = frames()
    quarters["eurusd"].loc[pd.Timestamp("2025-01-06T09:15Z"), "high"] = 1.1031
    quarters["eurusd"].loc[pd.Timestamp("2025-01-06T10:15Z"), "low"] = 1.0985
    result = execute(quarters, {"2025-01-06T09:00Z": [signal()], "2025-01-06T10:00Z": [signal()]},
                     EngineConfig(min_gap_hours=0))
    first, second = result.trades.iloc[0], result.trades.iloc[1]
    assert second.lots > first.lots
    assert second.risk_usd <= first.balance_after * 0.05
    assert result.summary["ending_balance"] == pytest.approx(200 + result.trades.pnl_usd.sum())
    np.testing.assert_allclose(result.trades.balance_after, 200 + result.trades.pnl_usd.cumsum())
    assert result.summary["max_drawdown_usd"] == pytest.approx(-second.pnl_usd)


def test_time_stop_uses_elapsed_hours_and_first_quote_after_weekend():
    quarters = frames("2025-01-10T19:00Z", "2025-01-13T12:00Z")
    result = execute(quarters, {"2025-01-10T19:00Z": [signal()]})
    row = result.trades.iloc[0]
    assert row.exit_reason == "time"
    assert row.exit_time == pd.Timestamp("2025-01-13T00:00Z")
    assert result.summary["average_hold_hours"] == 53


def test_daily_two_losses_halt_and_midnight_reset():
    quarters = frames(end="2025-01-07T12:00Z")
    for time in ("2025-01-06T09:15Z", "2025-01-06T10:15Z", "2025-01-07T09:15Z"):
        quarters["eurusd"].loc[pd.Timestamp(time), "low"] = 1.0985
    plan = {time: [signal()] for time in ("2025-01-06T09:00Z", "2025-01-06T10:00Z",
                                         "2025-01-06T11:00Z", "2025-01-07T09:00Z")}
    result = execute(quarters, plan, EngineConfig(min_gap_hours=0))
    assert [e["time"] for e in entries(result)] == list(pd.to_datetime([
        "2025-01-06T09:00Z", "2025-01-06T10:00Z", "2025-01-07T09:00Z"]))
    assert len(skips(result, "halt")) == 1
    assert result.summary["daily_halts"] == 1
    assert result.summary["weekly_halts"] == 0


def test_daily_percentage_halt_without_two_losses():
    quarters = frames()
    quarters["eurusd"].loc[pd.Timestamp("2025-01-06T09:15Z"), "low"] = 1.0985
    cfg = EngineConfig(risk_pct=0.16, combined_open_risk_pct=1, min_gap_hours=0, daily_max_losses=99)
    result = execute(quarters, {"2025-01-06T09:00Z": [signal()], "2025-01-06T10:00Z": [signal()]}, cfg)
    assert result.summary["daily_halts"] == 1
    assert len(entries(result)) == 1
    assert len(skips(result, "halt")) == 1


def test_weekly_halt_survives_day_reset_and_resets_monday():
    quarters = frames(end="2025-01-13T12:00Z")
    for time in ("2025-01-06T09:15Z", "2025-01-06T10:15Z", "2025-01-13T09:15Z"):
        quarters["eurusd"].loc[pd.Timestamp(time), "low"] = 1.0985
    plan = {time: [signal()] for time in ("2025-01-06T09:00Z", "2025-01-06T10:00Z",
                                         "2025-01-06T11:00Z", "2025-01-07T09:00Z", "2025-01-13T09:00Z")}
    cfg = EngineConfig(min_gap_hours=0, daily_max_losses=99, daily_loss_pct=1, weekly_loss_pct=0.08)
    result = execute(quarters, plan, cfg)
    assert len(entries(result)) == 3
    assert entries(result)[-1]["time"] == pd.Timestamp("2025-01-13T09:00Z")
    assert len(skips(result, "halt")) == 2
    assert result.summary["weekly_halts"] == 1


def test_global_minimum_entry_gap():
    quarters = frames()
    quarters["eurusd"].loc[pd.Timestamp("2025-01-06T09:15Z"), "low"] = 1.0985
    plan = {f"2025-01-06T{hour:02d}:00Z": [signal()] for hour in (9, 10, 12, 13)}
    result = execute(quarters, plan, EngineConfig(daily_max_losses=99))
    assert [e["time"].hour for e in entries(result)] == [9, 13]
    assert len(skips(result, "gap")) == 2


def test_combined_open_risk_cap_uses_current_balance():
    quarters = frames(pairs=("eurusd", "gbpusd"))
    cfg = EngineConfig(min_gap_hours=0, combined_open_risk_pct=0.07)
    result = execute(quarters, {"2025-01-06T09:00Z": [signal(), signal("gbpusd", "short")]}, cfg)
    assert len(entries(result)) == 1
    assert len(skips(result, "open_risk")) == 1
    assert all(e["open_risk"] <= e["balance"] * cfg.combined_open_risk_pct for e in entries(result))


def test_per_pair_limit():
    result = execute(frames(), {"2025-01-06T09:00Z": [signal()], "2025-01-06T10:00Z": [signal()]},
                     EngineConfig(min_gap_hours=0))
    assert len(entries(result)) == 1
    assert len(skips(result, "pair")) == 1


@pytest.mark.parametrize("pair, side", [("gbpusd", "long"), ("usdjpy", "short")])
def test_usd_direction_limit(pair, side):
    quarters = frames(pairs=("eurusd", pair))
    result = execute(quarters, {"2025-01-06T09:00Z": [signal(), signal(pair, side)]}, EngineConfig(min_gap_hours=0))
    assert len(entries(result)) == 1
    assert len(skips(result, "usd_direction")) == 1
    assert usd_direction("eurusd", "long") == usd_direction(pair, side)


def test_opposite_usd_directions_allowed_and_max_positions_configurable():
    quarters = frames(pairs=("eurusd", "gbpusd"))
    plan = {"2025-01-06T09:00Z": [signal(), signal("gbpusd", "short")]}
    cfg = EngineConfig(min_gap_hours=0)
    result = execute(quarters, plan, cfg)
    assert len(entries(result)) == 2
    limited = execute(quarters, plan, replace(cfg, max_open_positions=1))
    assert len(entries(limited)) == 1
    assert len(skips(limited, "max_positions")) == 1


def test_friday_cutoff_and_next_week_entry():
    quarters = frames("2025-01-10T19:00Z", "2025-01-13T12:00Z")
    quarters["eurusd"].loc[pd.Timestamp("2025-01-10T19:15Z"), "low"] = 1.0985
    plan = {time: [signal()] for time in ("2025-01-10T19:00Z", "2025-01-10T20:00Z", "2025-01-13T09:00Z")}
    result = execute(quarters, plan, EngineConfig(min_gap_hours=0))
    assert [e["time"] for e in entries(result)] == list(pd.to_datetime(["2025-01-10T19:00Z", "2025-01-13T09:00Z"]))
    assert len(skips(result, "friday")) == 1


def test_reward_risk_gate_skips_without_adjusting_target():
    invalid_rr = replace(signal(), target=1.101)
    result = execute(frames(), {"2025-01-06T09:00Z": [invalid_rr]})
    assert result.trades.empty
    assert len(skips(result, "reward_risk")) == 1
    assert invalid_rr.target == 1.101


def test_target_construction_is_two_cost_inclusive_r():
    cfg = EngineConfig(swap_pips_per_night=0)
    for side in ("long", "short"):
        candidate = construct_signal("eurusd", side, 1.1, 20, cfg, "fixture")
        quarters = frames()
        quarters["eurusd"].loc[pd.Timestamp("2025-01-06T09:15Z")] = (
            [1.1, candidate.target + 0.00001, 1.1, candidate.target] if side == "long" else
            [1.1, 1.1, candidate.target - 0.00001, candidate.target]
        )
        result = execute(quarters, {"2025-01-06T09:00Z": [candidate]}, cfg)
        assert result.trades.iloc[0].exit_reason == "target"
        assert result.trades.iloc[0].r_multiple == pytest.approx(2)


def ready_market():
    quarters = frames()
    index = quarters["eurusd"].index[quarters["eurusd"].index.minute == 0]
    features = {"eurusd": pd.DataFrame({"state": 1, "z": -2, "range_pips": 20,
                                        "close": 1.1, "previous_close": 1.0999, "ready": True}, index=index)}
    return prepare(quarters, features=features)


def test_random_benchmark_is_deterministic_for_fixed_seed():
    market = ready_market()
    rates, counts, opportunities = signal_probabilities(market)
    assert rates == {"eurusd": 1}
    assert counts == opportunities
    first = run_engine(market, random_entry(rates, 0), record_events=True)
    same = run_engine(market, random_entry(rates, 0), record_events=True)
    different = run_engine(market, random_entry(rates, 1), record_events=True)
    pd.testing.assert_frame_equal(first.trades, same.trades)
    assert first.events == same.events
    assert entries(first)[0]["side"] != entries(different)[0]["side"]


def test_future_data_cannot_change_features_or_a_decision():
    index = pd.date_range("2024-01-01T01:00Z", "2025-02-01T12:00Z", freq="1h")
    index = index[index.dayofweek < 5]
    rng = np.random.default_rng(7)
    close = 1.1 + np.cumsum(rng.normal(0.000003, 0.0002, len(index)))
    hourly = pd.DataFrame({"open": close, "high": close + 0.0002, "low": close - 0.0002, "close": close}, index=index)
    features = build_features(hourly, "eurusd")
    candidates = features.loc[features.ready & (((features.state == 1) & (features.z < -1) &
                             (features.close > features.previous_close)) | ((features.state == -1) &
                             (features.z > 1) & (features.close < features.previous_close)))]
    assert not candidates.empty
    time = candidates.index[0]
    market = PreparedMarket({"eurusd": hourly}, {"eurusd": hourly}, "train", {"eurusd": features})
    view = next(view for view in market.contexts.values() if view.time == time)
    before = trend_pullback(view)
    assert before
    changed = hourly.copy()
    changed.loc[changed.index > time] += 100
    changed_features = build_features(changed, "eurusd")
    altered = PreparedMarket({"eurusd": changed}, {"eurusd": changed}, "train", {"eurusd": changed_features})
    after_view = next(view for view in altered.contexts.values() if view.time == time)
    assert trend_pullback(after_view) == before
    pd.testing.assert_series_equal(features.loc[time], changed_features.loc[time])
    assert view.history("eurusd").index[-1] == time
    assert after_view.history("eurusd").index[-1] == time


def test_engine_rejects_test_split_and_test_period_input():
    quarters = frames()
    with pytest.raises(ValueError, match="test is locked"):
        prepare(quarters, "test")
    locked = frames("2025-11-03T09:00Z", "2025-11-03T12:00Z")
    with pytest.raises(ValueError, match="locked test period"):
        prepare(locked)


def test_runner_uses_default_loader_and_fresh_accounts(tmp_path, monkeypatch, capsys):
    from_years = pd.to_datetime(["2024-12-30T09:00Z", "2024-12-30T10:00Z",
                                "2025-01-06T09:00Z", "2025-01-06T10:00Z"])
    calls = []

    def load(pair, tf, **kwargs):
        assert not kwargs
        calls.append((pair, tf))
        value = prices[pair]
        return pd.DataFrame({"open": value, "high": value, "low": value, "close": value}, index=from_years)

    monkeypatch.setattr(runner, "load_pair", load)
    monkeypatch.setattr(runner, "benchmark_seeds", (0, 1))
    monkeypatch.setattr(runner, "results_dir", tmp_path)
    runner.main()
    assert len(calls) == 6
    summary = pd.read_csv(tmp_path / "engine_summary.csv")
    assert len(summary) == 6
    assert set(summary["split"]) == {"train", "val"}
    assert (summary.ending_balance == 200).all()
    for which in ("train", "val"):
        assert pd.read_csv(tmp_path / f"engine_trades_{which}.csv").empty
    assert "Test remains locked" in capsys.readouterr().out

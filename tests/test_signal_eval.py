import builtins
import io
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import run_signal_eval as runner
from src import engine, sizing
from src.engine import EngineConfig, PreparedMarket, Signal, run_engine, trade_columns
from src.signal_eval import (basic_statistics, block_bootstrap, engine_consistency, evaluate_signals,
                             mirror_signal, simulate_trade, weekly_blocks)


prices = {"eurusd": 1.1, "gbpusd": 1.25, "usdjpy": 150.0}


def frames(start="2025-01-06T09:00Z", end="2025-01-06T13:00Z", pairs=("eurusd",)):
    index = pd.date_range(start, end, freq="15min")
    index = index[index.dayofweek < 5]
    quarters = {}
    for pair in pairs:
        price = prices[pair]
        wiggle = 0.001 if pair == "usdjpy" else 0.00001
        quarters[pair] = pd.DataFrame({"open": price, "high": price + wiggle,
                                       "low": price - wiggle, "close": price}, index=index)
    hourly = {pair: frame.loc[frame.index.minute == 0].copy() for pair, frame in quarters.items()}
    return hourly, quarters


def candidate(pair="eurusd", side="long"):
    direction = 1 if side == "long" else -1
    distance = 0.1 if pair == "usdjpy" else 0.001
    return Signal(pair, side, prices[pair] - direction * distance,
                  prices[pair] + direction * 3 * distance, "fixture")


def planned(plan):
    plan = {pd.Timestamp(time): values for time, values in plan.items()}
    return lambda view: plan.get(view.time, [])


def test_one_open_strategy_trade_per_pair():
    hourly, quarters = frames()
    market = PreparedMarket(hourly, quarters, "val")
    plan = {f"2025-01-06T{hour:02d}:00Z": [candidate()] for hour in (9, 10, 11)}
    result = evaluate_signals(market, planned(plan))
    assert len(result.trades) == 1
    assert result.counts == {"signals": 3, "accepted": 1, "ignored_pair": 2}


def test_mirrors_do_not_occupy_pair_or_force_the_strategy_exit():
    hourly, quarters = frames()
    quarters["eurusd"].loc[pd.Timestamp("2025-01-06T09:15Z"), "low"] = 1.09895
    market = PreparedMarket(hourly, quarters, "val")
    result = evaluate_signals(market, planned({"2025-01-06T09:00Z": [candidate()],
                                              "2025-01-06T10:00Z": [candidate()]}))
    assert len(result.trades) == 2
    first, second = result.trades.iloc[0], result.trades.iloc[1]
    assert first.exit_time == pd.Timestamp("2025-01-06T09:15Z")
    assert first.mirror_exit_time > second.entry_time
    assert first.mirror_exit_time != first.exit_time


def test_no_account_gate_or_lot_size_can_skip_a_signal(monkeypatch):
    hourly, quarters = frames(pairs=tuple(prices))
    for time in ("2025-01-06T09:15Z", "2025-01-06T10:15Z"):
        quarters["eurusd"].loc[pd.Timestamp(time), "low"] = 1.09895
        quarters["gbpusd"].loc[pd.Timestamp(time), "low"] = 1.24895
        quarters["usdjpy"].loc[pd.Timestamp(time), "high"] = 150.1005
    market = PreparedMarket(hourly, quarters, "val")
    signals = [candidate("eurusd"), candidate("gbpusd"), candidate("usdjpy", "short")]
    strategy = planned({"2025-01-06T09:00Z": signals, "2025-01-06T10:00Z": signals})
    baseline = evaluate_signals(market, strategy)

    def forbidden(*args, **kwargs):
        raise AssertionError("No account engine or sizing may be called")

    monkeypatch.setattr(sizing, "lot_size", forbidden)
    monkeypatch.setattr(engine, "lot_size", forbidden)
    monkeypatch.setattr(engine.Engine, "run", forbidden)
    restrictive = EngineConfig(start_balance=1e-12, risk_pct=1e-12, min_lot=10000, lot_step=10000,
                               daily_max_losses=1, daily_loss_pct=1e-12, weekly_loss_pct=1e-12,
                               min_gap_hours=1e9, max_open_positions=1, combined_open_risk_pct=1e-12)
    actual = evaluate_signals(market, strategy, restrictive)
    assert len(actual.trades) == 6
    pd.testing.assert_frame_equal(baseline.trades, actual.trades)
    assert not any("balance" in column or "lots" in column for column in actual.trades)


@pytest.mark.parametrize("pair, side", [(pair, side) for pair in prices for side in ("long", "short")])
def test_clean_stop_is_minus_one_r_and_mirror_distances_match(pair, side):
    hourly, quarters = frames(pairs=(pair,))
    signal = candidate(pair, side)
    timestamp = pd.Timestamp("2025-01-06T09:15Z")
    if side == "long":
        quarters[pair].loc[timestamp, "low"] = signal.stop - (0.00001 if pair != "usdjpy" else 0.001)
    else:
        quarters[pair].loc[timestamp, "high"] = signal.stop + (0.00001 if pair != "usdjpy" else 0.001)
    market = PreparedMarket(hourly, quarters, "val")
    result = evaluate_signals(market, planned({"2025-01-06T09:00Z": [signal]}))
    row = result.trades.iloc[0]
    assert row.exit_reason == "stop"
    assert row.r_strategy == pytest.approx(-1, abs=1e-10)
    assert row.entry_time == row.mirror_entry_time
    assert row.pair == row.mirror_pair
    assert row.side != row.mirror_side
    assert row.stop_distance_pips == pytest.approx(row.mirror_stop_distance_pips)
    assert row.target_distance_pips == pytest.approx(row.mirror_target_distance_pips)
    assert row.initial_risk_pips == pytest.approx(row.mirror_initial_risk_pips)
    assert row.d == pytest.approx((row.r_strategy - row.r_mirror) / 2)
    assert row.coinflip_r == pytest.approx((row.r_strategy + row.r_mirror) / 2)
    assert row.r_strategy == pytest.approx(row.coinflip_r + row.d)


def test_future_prices_cannot_change_decisions_at_or_before_t():
    hourly, quarters = frames()
    cutoff = pd.Timestamp("2025-01-06T10:00Z")

    def strategy(view):
        history = view.history("eurusd")
        assert history.index[-1] <= view.time
        if view.time.hour not in (9, 10, 11):
            return []
        original = candidate()
        shift = view.quotes["eurusd"] - prices["eurusd"]
        return [replace(original, stop=original.stop + shift, target=original.target + shift)]

    cfg = EngineConfig(max_hold_hours=0.25)
    market = PreparedMarket(hourly, quarters, "val")
    baseline = evaluate_signals(market, strategy, cfg).trades
    altered_quarters = {"eurusd": quarters["eurusd"].copy()}
    altered_quarters["eurusd"].loc[altered_quarters["eurusd"].index > cutoff] += 100
    altered_hourly = {"eurusd": hourly["eurusd"].copy()}
    altered_hourly["eurusd"].loc[altered_hourly["eurusd"].index > cutoff] += 100
    altered = evaluate_signals(PreparedMarket(altered_hourly, altered_quarters, "val"), strategy, cfg).trades
    columns = ["entry_time", "pair", "side", "entry_bid", "entry_fill", "initial_stop", "target", "initial_risk_pips"]
    before = baseline.loc[baseline.entry_time <= cutoff, columns].reset_index(drop=True)
    after = altered.loc[altered.entry_time <= cutoff, columns].reset_index(drop=True)
    assert len(before) == 2
    pd.testing.assert_frame_equal(before, after)


def test_friday_is_only_time_gate_and_final_quote_signal_is_counted():
    hourly, quarters = frames("2025-01-10T19:00Z", "2025-01-10T21:00Z")
    result = evaluate_signals(PreparedMarket(hourly, quarters, "val"), planned({
        "2025-01-10T19:00Z": [candidate()], "2025-01-10T20:00Z": [candidate()],
    }), EngineConfig(max_hold_hours=0.25))
    assert len(result.trades) == 1 and result.counts["ignored_friday"] == 1
    hourly, quarters = frames(end="2025-01-06T10:00Z")
    result = evaluate_signals(PreparedMarket(hourly, quarters, "val"),
                              planned({"2025-01-06T10:00Z": [candidate()]}))
    row = result.trades.iloc[0]
    assert row.exit_reason == "end_of_split" and row.hold_hours == 0
    assert row.entry_time == row.exit_time
    assert row.price_pips == pytest.approx(-1)


def test_exit_priority_gap_and_next_bar_breakeven_match_existing_rules():
    hourly, quarters = frames()
    quarters["eurusd"].loc[pd.Timestamp("2025-01-06T09:15Z")] = [1.1, 1.1015, 1.0999, 1.1008]
    quarters["eurusd"].loc[pd.Timestamp("2025-01-06T09:30Z")] = [1.1008, 1.101, 1.1, 1.1005]
    market = PreparedMarket(hourly, quarters, "val")
    path = simulate_trade(market, candidate(), 0)
    assert path["breakeven"]
    assert path["exit_time"] == pd.Timestamp("2025-01-06T09:30Z")
    assert path["stop"] == pytest.approx(path["entry_fill"])
    assert path["exit_fill"] == pytest.approx(1.1)
    quarters["eurusd"].loc[pd.Timestamp("2025-01-06T09:15Z")] = [1.098, 1.104, 1.097, 1.1]
    path = simulate_trade(PreparedMarket(hourly, quarters, "val"), candidate(), 0)
    assert path["exit_reason"] == "stop" and path["r"] < -1
    assert path["exit_fill"] == pytest.approx(1.0979)


def test_swap_converts_to_pips_without_sizing():
    hourly, quarters = frames("2025-01-07T21:00Z", "2025-01-08T23:30Z")
    path = simulate_trade(PreparedMarket(hourly, quarters, "val"), candidate(), 0, EngineConfig(max_hold_hours=26))
    assert path["exit_reason"] == "time"
    assert path["swap_pips"] == pytest.approx(-0.8)  # Tuesday once, Wednesday three times.
    assert path["net_pips"] == pytest.approx(path["price_pips"] - 0.8)


def test_weekly_bootstrap_is_paired_trade_weighted_and_deterministic():
    entries = pd.to_datetime(["2025-01-06T09:00Z"] * 3 + ["2025-01-13T09:00Z"] + ["2025-01-20T09:00Z"] * 2)
    sample = pd.DataFrame({"entry_time": entries, "r_strategy": [1, 2, -1, -1, 1, 1],
                           "d": [0.1, 0.2, -0.1, -0.2, 0.6, 0.6]})
    start, end = pd.Timestamp("2025-01-06T00:00Z"), pd.Timestamp("2025-02-03T00:00Z")
    blocks = weekly_blocks(sample, start, end)
    assert blocks.n.tolist() == [3, 1, 2, 0]
    first = block_bootstrap(sample, start, end, resamples=200, seed=17)
    assert first == block_bootstrap(sample, start, end, resamples=200, seed=17)
    draws = np.random.default_rng(17).integers(0, 4, size=(200, 4))
    n = blocks.n.to_numpy()[draws].sum(axis=1)
    values = blocks[["r_strategy", "d"]].to_numpy()[draws].sum(axis=1)[n > 0] / n[n > 0, None]
    intervals = np.percentile(values, [2.5, 97.5], axis=0)
    assert first["r_ci_low"] == pytest.approx(intervals[0, 0])
    assert first["d_ci_high"] == pytest.approx(intervals[1, 1])
    weekly = np.array([0.2, -0.2, 1.2, 0])
    assert first["weekly_d_t_stat"] == pytest.approx(weekly.mean() / (weekly.std(ddof=1) / 2))
    assert first["bootstrap_valid"] == (n > 0).sum()


def test_cumulative_r_drawdown_uses_exit_order():
    sample = pd.DataFrame({"entry_time": pd.to_datetime(["2025-01-06T09:00Z", "2025-01-06T10:00Z"]),
                           "exit_time": pd.to_datetime(["2025-01-06T12:00Z", "2025-01-06T11:00Z"]),
                           "pair": ["eurusd", "gbpusd"], "r_strategy": [-3.0, 2.0], "r_mirror": [0, 0],
                           "d": [-1.5, 1], "coinflip_r": [-1.5, 1], "cost_r": [0.1, 0.2],
                           "exit_reason": ["stop", "target"], "mirror_exit_reason": ["time", "time"]})
    assert basic_statistics(sample)["max_drawdown_r"] == 3


def test_engine_common_paths_reconcile_after_usdjpy_unit_conversion():
    hourly, quarters = frames(pairs=("eurusd", "usdjpy"))
    quarters["eurusd"].loc[pd.Timestamp("2025-01-06T09:15Z"), "low"] = 1.09895
    quarters["usdjpy"].loc[pd.Timestamp("2025-01-06T09:15Z"), "high"] = 150.1005
    market = PreparedMarket(hourly, quarters, "val")
    strategy = planned({"2025-01-06T09:00Z": [candidate(), candidate("usdjpy", "short")]})
    cfg = EngineConfig(min_gap_hours=0, max_per_usd_direction=2)
    signal = evaluate_signals(market, strategy, cfg).trades.assign(variant="A", split="val")
    reference = run_engine(market, strategy, cfg).trades.assign(variant="A", split="val")
    check = engine_consistency(signal, reference, cfg).set_index("period").loc["val"]
    assert check.engine_matches == 2
    assert check.engine_max_abs_r_diff > 0.0001
    assert check.engine_max_abs_r_diff_fx_normalized < 1e-10
    assert check.engine_exit_path_mismatches == 0


def test_test_period_input_is_rejected():
    hourly, quarters = frames("2025-11-03T09:00Z", "2025-11-03T11:00Z")
    with pytest.raises(ValueError, match="locked"):
        PreparedMarket(hourly, quarters, "val")
    hourly, quarters = frames()
    with pytest.raises(ValueError, match="locked"):
        PreparedMarket(hourly, quarters, "test")


def test_script_cannot_load_test_or_read_fresh_or_data(tmp_path, monkeypatch, capsys):
    project = Path(__file__).resolve().parents[1]
    index = pd.to_datetime(["2024-12-30T09:00Z", "2024-12-30T10:00Z", "2025-01-06T09:00Z", "2025-01-06T10:00Z"])
    calls = []

    def load(pair, tf, **kwargs):
        assert tf in ("1h", "15m") and not kwargs
        calls.append((pair, tf))
        value = prices[pair]
        return pd.DataFrame({"open": value, "high": value, "low": value, "close": value}, index=index)

    pd.DataFrame(columns=trade_columns + ["variant", "split", "rr"]).to_csv(tmp_path / "variants_trades.csv", index=False)
    original_open, original_io_open = builtins.open, io.open

    def check_path(filename):
        if isinstance(filename, (str, Path)):
            path = Path(filename).resolve()
            assert not path.is_relative_to(project / "fresh")
            assert not path.is_relative_to(project / "data")

    def guarded_open(filename, *args, **kwargs):
        check_path(filename)
        return original_open(filename, *args, **kwargs)

    def guarded_io_open(filename, *args, **kwargs):
        check_path(filename)
        return original_io_open(filename, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", guarded_open)
    monkeypatch.setattr(io, "open", guarded_io_open)
    monkeypatch.setattr(runner, "load_pair", load)
    monkeypatch.setattr(runner, "results_dir", tmp_path)
    monkeypatch.setattr(runner, "bootstrap_resamples", 20)
    runner.main()
    assert len(calls) == 6
    summary = pd.read_csv(tmp_path / "signal_summary.csv")
    assert len(summary) == 301
    main = summary.loc[summary['group'] == "all"]
    assert len(main) == 21
    assert set(main.period) == {"train", "val", "pooled"}
    assert not main.pooled_d_ci_above_zero.any()
    assert pd.read_csv(tmp_path / "signal_trades.csv").empty
    assert "fresh/ and data/ are never read" in capsys.readouterr().out

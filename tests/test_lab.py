"""Fast synthetic checks for period isolation, every block, fills, and ledgers."""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts import lab as runner
from src import lab_blocks as blocks
from src import lab_data as data
from src import lab_score as score
from src import lab_sim as sim
from src.engine import EngineConfig, Signal
from src.signal_eval import simulate_trade as reference_trade


def quarters(n=3600, start="2011-02-14", seed=3):
    index = pd.date_range(start, periods=n, freq="15min", tz="UTC", name="timestamp")
    rng = np.random.default_rng(seed)
    close = 1.1 * np.exp(rng.normal(0, .0002, n).cumsum())
    opening = np.r_[close[0], close[:-1]]
    return pd.DataFrame({"open": opening, "high": np.maximum(opening, close) + .0002,
                         "low": np.minimum(opening, close) - .0002, "close": close}, index=index)


def idea(**changes):
    result = dict(name="synthetic", description="Synthetic idea", max_hold_hours=1,
                  entry=dict(type="random", probability=1., seed=0), stop=dict(type="pips", value=10),
                  target=dict(type="r", multiple=2))
    result.update(changes)
    return result


def compiled(frame=None, config=None, **changes):
    frame = quarters(10, "2011-01-03 09:00") if frame is None else frame
    return blocks.CompiledIdea(data.build_data(frame, "eurusd"), idea(**changes), config)


level_blocks = [dict(type="swing", timeframe="1h", k=3), dict(type="swing", timeframe="15m", k=2),
                dict(type="swing", timeframe="1d", k=1), dict(type="rolling", timeframe="4h", n=3),
                dict(type="prev_day"), dict(type="prev_week"), dict(type="round", step_pips=100)]


@pytest.mark.parametrize("block", level_blocks)
def test_level_no_lookahead_and_higher_frame_visibility(block):
    frame = quarters()
    t = frame.index[2701]
    full = blocks.Context(data.build_data(frame, "eurusd"))
    truncated = blocks.Context(data.build_data(frame.loc[:t], "eurusd"))
    np.testing.assert_allclose(full.level(block)[:2702], truncated.level(block), equal_nan=True)
    for tf in data.timeframes:
        pd.testing.assert_frame_equal(full.data.frames[tf].loc[:t], truncated.data.frames[tf])


@pytest.mark.parametrize("entry", [dict(type="break", direction="with", level=level_blocks[0]),
                                    dict(type="break", direction="against", level=level_blocks[0]),
                                    dict(type="bounce", level=level_blocks[1]), dict(type="trigger"),
                                    dict(type="random", probability=.4, seed=7)])
def test_entry_no_lookahead(entry):
    frame = quarters()
    bias = dict(type="trend", timeframe="1h", fast=3, slow=8)
    full = compiled(frame, entry=entry, bias=bias)
    truncated = compiled(frame.iloc[:2702], entry=entry, bias=bias)
    for a, b in zip(full.candidates(), truncated.candidates()):
        np.testing.assert_array_equal(a[:2702], b)


@pytest.mark.parametrize("category,block", [
    ("bias", dict(type="trend", timeframe="4h", fast=3, slow=8)),
    ("bias", dict(type="sma", timeframe="1d", n=3)),
    ("light", dict(type="hours", start=3, end=18)), ("light", dict(type="cost", max_cost_r=.15)),
    ("light", dict(type="vol_ratio", min=.5, max=2.)),
    ("light", dict(type="trend_agree", timeframe="4h", fast=3, slow=8)),
    ("stop", dict(type="level", level=level_blocks[0], buffer=.1)),
    ("stop", dict(type="vol", multiple=1.2, hours=4)), ("stop", dict(type="pips", value=10)),
    ("target", dict(type="r", multiple=2)), ("target", dict(type="level", level=level_blocks[0])),
    ("target", dict(type="none")),
])
def test_bias_light_stop_target_no_lookahead(category, block):
    frame = quarters()
    contexts = [blocks.Context(data.build_data(source, "eurusd")) for source in (frame, frame.iloc[:2702])]
    for side in (1, -1):
        results = []
        for context in contexts:
            if category == "bias":
                value = context.bias(block)
            elif category == "light":
                value = blocks.lights[block["type"]](context, block, side)
            elif category == "stop":
                value = blocks.stops[block["type"]](context, block, side)
            else:
                stop = blocks.pips_stop(context, dict(value=10), side)
                value = blocks.targets[block["type"]](context, block, side, stop)
            results.append(value)
        if np.isscalar(results[0]):
            assert results[0] == results[1]
        else:
            np.testing.assert_allclose(results[0][:2702], results[1], equal_nan=True)


@pytest.mark.parametrize("rule", [dict(type="breakeven", at_r=1.),
                                  dict(type="trail_level", level=level_blocks[0], buffer=.1),
                                  dict(type="trail_vol", multiple=1., hours=4)])
def test_manage_no_lookahead_and_never_loosen(rule):
    frame = quarters()
    full, truncated = compiled(frame, manage=[rule]), compiled(frame.iloc[:2702], manage=[rule])
    for side in (1, -1):
        state = sim.PathState(side, 1.1, .001, 1.08 if side == 1 else 1.12, 1.1 + side * .003, 1.1 + side * .002)
        other = deepcopy(state)
        for index in range(2500, 2702):
            old = state.stop
            assert sim.apply_manage(full, index, state) == sim.apply_manage(truncated, index, other)
            assert side * (state.stop - old) >= 0


def test_swings_strict_confirmation_and_rolling_hand_example():
    frame = quarters(5)
    frame["low"] = [1.04, 1.03, 1.01, 1.02, 1.04]
    frame["high"] = [1.08, 1.09, 1.10, 1.09, 1.08]
    context = blocks.Context(data.build_data(frame, "eurusd"))
    swing = context.level(dict(type="swing", timeframe="15m", k=1))
    assert np.isnan(swing[:3]).all()
    np.testing.assert_allclose(swing[3:], [[1.01, 1.10], [1.01, 1.10]])
    slow = context.level(dict(type="swing", timeframe="15m", k=2))
    assert np.isnan(slow[:4]).all()
    np.testing.assert_allclose(slow[4], [1.01, 1.10])
    rolling = context.level(dict(type="rolling", timeframe="15m", n=3))
    assert np.isnan(rolling[:2]).all()
    np.testing.assert_allclose(rolling[2:], [[1.01, 1.10]] * 3)
    tied = frame.copy()
    tied.loc[tied.index[1], "low"] = 1.01
    assert np.isnan(blocks.Context(data.build_data(tied, "eurusd")).level(dict(type="swing", timeframe="15m", k=1))[:, 0]).all()


def test_daily_ny_17_dst_prev_day_and_prev_week():
    source = data.build_data(quarters(3600), "eurusd")
    assert set(source.frames["1d"].index.tz_convert("America/New_York").hour) == {17}
    assert set(source.frames["4h"].index.hour).issubset({0, 4, 8, 12, 16, 20})
    assert source.frames["1d"].index[0].utcoffset().total_seconds() == 0
    daily_times = pd.bdate_range("2011-01-03", periods=10) + pd.Timedelta(hours=17)
    daily_times = daily_times.tz_localize("America/New_York").tz_convert("UTC")
    source = data.build_data(quarters(1500, "2011-01-03"), "eurusd")
    source.frames["1d"] = pd.DataFrame({"low": np.arange(10) + 1., "high": np.arange(10) + 101.}, index=daily_times)
    context = blocks.Context(source)
    at = context.times.get_loc(pd.Timestamp("2011-01-07 22:00", tz="UTC"))
    np.testing.assert_allclose(context.level(dict(type="prev_day"))[at], [5., 105.])
    np.testing.assert_allclose(context.level(dict(type="prev_week"))[at], [1., 105.])
    assert np.isnan(context.level(dict(type="prev_week"))[at - 1]).all()


def test_round_levels_are_strict():
    frame = quarters(3)
    frame["close"] = [1.1, 1.1001, 1.0999]
    context = blocks.Context(data.build_data(frame, "eurusd"))
    values = context.level(dict(type="round", step_pips=100))
    np.testing.assert_allclose(values, [[1.09, 1.11], [1.10, 1.11], [1.09, 1.10]])
    assert (values[:, 0] < frame.close).all() and (values[:, 1] > frame.close).all()


def fixed_levels(context, support=1., resistance=1.2):
    block = dict(type="rolling", timeframe="15m", n=1)
    context._levels[json.dumps(block, sort_keys=True)] = np.tile([support, resistance], (len(context.times), 1))
    return block


def test_break_bounce_trigger_and_ambiguous_count_hand_examples():
    frame = quarters(4)
    frame["close"] = [1.1, 1.21, .99, 1.1]
    context = blocks.Context(data.build_data(frame, "eurusd"))
    level = fixed_levels(context)
    buy, sell = blocks.break_entry(context, dict(level=level, direction="with"))
    np.testing.assert_array_equal(buy, [False, True, False, False])
    np.testing.assert_array_equal(sell, [False, False, True, False])
    opposite = blocks.break_entry(context, dict(level=level, direction="against"))
    np.testing.assert_array_equal(opposite[0], sell)
    frame = quarters(2)
    frame.iloc[1] = [1.1, 1.3, .9, 1.1]
    candidate = compiled(frame, entry=dict(type="bounce", level=level))
    fixed_levels(candidate.context)
    trades, counts = sim.evaluate_pair(candidate)
    assert trades.empty and counts == {"signals": 2, "ambiguous": 2}
    frame = quarters(3)
    frame.iloc[:] = [[1.1, 1.11, 1.09, 1.1], [1.1, 1.13, 1.1, 1.12], [1.12, 1.12, 1.08, 1.09]]
    context = blocks.Context(data.build_data(frame, "eurusd"))
    buy, sell = blocks.trigger(context, {})
    np.testing.assert_array_equal(buy, [False, True, False])
    np.testing.assert_array_equal(sell, [False, False, True])


def test_biases_lights_stops_and_targets_hand_examples():
    frame = quarters(6, "2011-01-03")
    frame["close"] = [1., 2., 3., 4., 3., 1.]
    context = blocks.Context(data.build_data(frame, "eurusd"))
    np.testing.assert_allclose(blocks.sma(context, dict(timeframe="15m", n=3)), [np.nan, np.nan, 1, 1, -1, -1], equal_nan=True)
    trend = blocks.trend(context, dict(timeframe="15m", fast=1, slow=3))
    np.testing.assert_allclose(trend, [np.nan, np.nan, 1, 1, -1, -1], equal_nan=True)
    np.testing.assert_array_equal(blocks.trend_light(context, dict(timeframe="15m", fast=1, slow=3), 1), trend == 1)
    np.testing.assert_array_equal(blocks.hours_light(context, dict(start=0, end=1), 1), [True, True, True, True, False, False])
    context.data.vol_ratio[:] = [0.5, 1., 2., .8, 1.5, 3.]
    np.testing.assert_array_equal(blocks.vol_light(context, dict(min=.8, max=1.5), 1), [False, True, False, True, True, False])
    context.close[:] = 1.1
    context.e1[:] = .001
    level = fixed_levels(context, 1.09, 1.11)
    for side in (1, -1):
        stop = blocks.pips_stop(context, dict(value=10), side)
        np.testing.assert_allclose(stop, 1.1 - side * .001)
        np.testing.assert_allclose(blocks.vol_stop(context, dict(multiple=2, hours=4), side), 1.1 - side * .004)
        np.testing.assert_allclose(blocks.level_stop(context, dict(level=level, buffer=.1), side), 1.1 - side * .0101)
        np.testing.assert_allclose(blocks.r_target(context, dict(multiple=2), side, stop), 1.1 + side * .0025)
        np.testing.assert_allclose(blocks.level_target(context, dict(level=level), side, stop), 1.1 + side * .01)
        assert np.isinf(blocks.no_target(context, {}, side, stop)).all()
    candidate = compiled(lights=[dict(type="cost", max_cost_r=.15)])
    assert candidate.signal(0, 1)[1] == "light cost"
    candidate = compiled(stop=dict(type="pips", value=2))
    assert candidate.signal(0, 1)[1] is None
    candidate = compiled(stop=dict(type="pips", value=1.9))
    assert candidate.signal(0, 1)[1] == "stop too close"


def test_manage_hand_examples():
    frame = quarters(3)
    candidate = compiled(frame, manage=[dict(type="trail_vol", multiple=2, hours=4)])
    candidate.managers[0]["distance"][:] = .004
    state = sim.PathState(1, 1.1, .001, 1.08, 1.12, 1.12)
    assert sim.apply_manage(candidate, 0, state) == pytest.approx(1.116)
    state = sim.PathState(-1, 1.1, .001, 1.12, 1.08, 1.08)
    assert sim.apply_manage(candidate, 0, state) == pytest.approx(1.0841)
    assert blocks.breakeven(dict(at_r=1), 0, state) == 1.1
    assert blocks.trail_level(dict(values={1: np.array([1.09])}), 0, sim.PathState(1, 1.1, .001, 1.08, 1.1, 1.1)) == 1.09


@pytest.mark.parametrize("side", [1, -1])
@pytest.mark.parametrize("outcome", ["stop", "target", "time", "breakeven", "gap"])
def test_simulator_matches_signal_eval(side, outcome):
    frame = quarters(9, "2011-01-03 09:00")
    frame.loc[:, :] = [1.1, 1.10001, 1.09999, 1.1]
    if outcome == "stop":
        frame.iloc[1, frame.columns.get_loc("low" if side == 1 else "high")] = 1.0985 if side == 1 else 1.1015
    elif outcome == "target":
        frame.iloc[1, frame.columns.get_loc("high" if side == 1 else "low")] = 1.103 if side == 1 else 1.097
    elif outcome == "gap":
        frame.iloc[1] = [1.097, 1.098, 1.096, 1.097] if side == 1 else [1.103, 1.104, 1.102, 1.103]
    elif outcome == "breakeven":
        frame.iloc[1] = [1.1, 1.1015, 1.0998, 1.1008] if side == 1 else [1.1, 1.1005, 1.0985, 1.099]
    candidate = compiled(frame, manage=[dict(type="breakeven", at_r=1)], max_hold_hours=1)
    signal, skipped = candidate.signal(0, side)
    assert skipped is None
    actual = sim.simulate_trade(candidate, signal, 0)
    market = SimpleNamespace(which="train", pair_ids={"eurusd": 0}, times=tuple(frame.index),
                             bars=frame.to_numpy()[:, None, :], last_indices={"eurusd": len(frame) - 1})
    reference = reference_trade(market, Signal("eurusd", "long" if side == 1 else "short", signal["stop_bid"], signal["target_bid"], "synthetic"),
                                0, EngineConfig(max_hold_hours=1))
    for key in ("entry_fill", "exit_fill", "stop", "target", "r", "swap_pips", "hold_hours"):
        assert actual[key] == pytest.approx(reference[key])
    assert actual["exit_reason"] == reference["exit_reason"] and actual["exit_time"] == reference["exit_time"]


def test_moved_stop_next_bar_stop_wins_and_entry_bar_cannot_exit():
    frame = quarters(4, "2011-01-03 09:00")
    frame.loc[:, :] = [1.1, 1.10001, 1.09999, 1.1]
    frame.iloc[0] = [1.1, 2., .1, 1.1]
    frame.iloc[1] = [1.1, 1.1015, 1.0998, 1.1008]
    candidate = compiled(frame, manage=[dict(type="breakeven", at_r=1)])
    signal, _ = candidate.signal(0, 1)
    trace = []
    path = sim.simulate_trade(candidate, signal, 0, trace=trace)
    assert path["exit_time"] == frame.index[2]
    assert trace[0][1] < trace[0][2] and trace[0][3] is None
    assert trace[1][1] == trace[0][2]
    frame.iloc[1] = [1.1, 1.104, 1.098, 1.1]
    candidate = compiled(frame, manage=[dict(type="breakeven", at_r=1)])
    assert sim.simulate_trade(candidate, candidate.signal(0, 1)[0], 0)["exit_reason"] == "stop"


def test_mirror_distances_d_and_coin():
    candidate = compiled()
    signal, _ = candidate.signal(0, 1)
    mirror = sim.mirror_signal(signal, candidate.context.close[0])
    assert mirror["side"] == -1
    assert abs(mirror["stop_bid"] - candidate.context.close[0]) == pytest.approx(abs(signal["stop_bid"] - candidate.context.close[0]))
    assert abs(mirror["target_bid"] - candidate.context.close[0]) == pytest.approx(abs(signal["target_bid"] - candidate.context.close[0]))
    trades, _ = sim.evaluate_pair(candidate)
    np.testing.assert_allclose(trades.d, (trades.r_strategy - trades.r_mirror) / 2)
    np.testing.assert_allclose(trades.coin, (trades.r_strategy + trades.r_mirror) / 2)


def test_pair_busy_and_friday_counts():
    candidate = compiled()
    candidate.candidates = lambda: (np.ones(10, dtype=bool), np.zeros(10, dtype=bool))
    candidate.context.bars[:] = [1.1, 1.10001, 1.09999, 1.1]
    candidate.stop[1][:] = 1.099
    candidate.target[1][:] = 1.1025
    trades, counts = sim.evaluate_pair(candidate)
    assert len(trades) == 4 and counts["pair busy"] == 6
    friday = compiled(quarters(9, "2011-01-07 19:00"))
    friday.candidates = lambda: (np.ones(9, dtype=bool), np.zeros(9, dtype=bool))
    friday.context.bars[:] = [1.1, 1.10001, 1.09999, 1.1]
    friday.stop[1][:] = 1.099
    friday.target[1][:] = 1.1025
    _, counts = sim.evaluate_pair(friday)
    assert counts["trades"] == 1 and counts["pair busy"] == 3 and counts["friday"] == 5


def test_random_baseline_seed_counts_ignored_bias_and_independence():
    first = quarters(20, "2011-01-03 09:00")
    second = quarters(20, "2012-01-03 09:00")
    candidate = compiled(pd.concat([first, second]), bias=dict(type="sma", timeframe="15m", n=2),
                         lights=[dict(type="trend_agree", timeframe="15m", fast=1, slow=2)])
    template = pd.DataFrame({"pair": ["eurusd"] * 12, "entry_time": list(first.index[:6]) + list(second.index[:6])})
    a, audit = sim.random_baseline({"eurusd": candidate}, template, 0)
    b, audit2 = sim.random_baseline({"eurusd": candidate}, template, 0)
    pd.testing.assert_frame_equal(a, b)
    pd.testing.assert_frame_equal(audit, audit2)
    assert audit.requested.tolist() == audit.achieved.tolist() == [6, 6]
    assert set(a.side) == {"long", "short"}
    assert len(a) == 12
    rng = np.random.default_rng(0)
    expected_index, expected_side = int(rng.integers(20)), int(rng.choice([-1, 1]))
    assert a.entry_time.iloc[0] == first.index[expected_index]
    assert a.side.iloc[0] == ("long" if expected_side == 1 else "short")


def test_random_baseline_twenty_failed_draws_are_dropped():
    candidate = compiled(lights=[dict(type="hours", start=23, end=24)])
    template = pd.DataFrame({"pair": ["eurusd"] * 3, "entry_time": candidate.context.times[:3]})
    paths, audit = sim.random_baseline({"eurusd": candidate}, template, 4)
    assert paths.empty and audit.requested.sum() == 3 and audit.achieved.sum() == 0
    assert audit.skipped_draws.sum() == 60


def test_random_seed_order_pair_then_year_and_sides_filter():
    frame = quarters(6, "2011-01-03 09:00")
    values = {}
    for pair in ("eurusd", "usdjpy"):
        values[pair] = blocks.CompiledIdea(data.build_data(frame, pair), idea(entry=dict(type="random", probability=1., seed=0, sides="long")))
    template = pd.DataFrame({"pair": ["eurusd", "usdjpy"], "entry_time": [frame.index[0]] * 2})
    result, audit = sim.random_baseline(dict(reversed(list(values.items()))), template, 0)
    assert result.pair.tolist() == ["eurusd", "usdjpy"]
    rng = np.random.default_rng(0)
    for row in result.itertuples():
        assert row.entry_time == frame.index[rng.integers(len(frame))]
        assert row.side == ("long" if rng.choice([-1, 1]) == 1 else "short")
    assert audit.achieved.sum() == 2


def test_no_target_and_max_hold_default():
    frame = quarters(10)
    frame.loc[:, :] = [1.1, 1.10001, 1.09999, 1.1]
    candidate = compiled(frame, target=dict(type="none"))
    candidate.idea.pop("max_hold_hours")
    path = sim.simulate_trade(candidate, candidate.signal(0, 1)[0], 0)
    assert path["exit_reason"] == "end_of_split" and np.isinf(path["target"])
    mirror = sim.mirror_signal(candidate.signal(0, 1)[0], candidate.context.close[0])
    assert np.isneginf(mirror["target_bid"])


def test_roundtrip_swap_and_clean_stop_risk():
    frame = quarters(109, "2011-01-04 21:00")
    frame.loc[:, :] = [1.1, 1.10001, 1.09999, 1.1]
    candidate = compiled(frame, max_hold_hours=26)
    path = sim.simulate_trade(candidate, candidate.signal(0, 1)[0], 0)
    assert path["swap_pips"] == pytest.approx(-.8)
    frame.iloc[1] = [1.1, 1.1, 1.0989, 1.099]
    candidate = compiled(frame)
    path = sim.simulate_trade(candidate, candidate.signal(0, 1)[0], 0)
    assert path["r"] == pytest.approx(-1)


def test_loader_period_guards_cut_before_features_and_cache(tmp_path, monkeypatch):
    before = quarters(2, "2019-12-31 23:30")
    after = quarters(2, "2020-01-01")
    tail = quarters(2, "2025-10-31 23:30")
    source = pd.concat([before, after, tail])
    calls = []
    def load(pair, tf):
        assert tf == "15m"
        calls.append(pair)
        return source
    monkeypatch.setattr(data.loader, "load_continuous_pair", load)
    monkeypatch.setattr(data, "cache_dir", tmp_path / "cache")
    marker = tmp_path / "source.csv"
    marker.write_text("synthetic", encoding="ascii")
    monkeypatch.setattr(data, "source_fingerprint", lambda pair: [(marker.stat().st_size, marker.stat().st_mtime_ns)])
    original = data.build_data
    def guarded(frame, pair, period):
        if period == "lab":
            assert frame.index.max() < pd.Timestamp("2020-01-01", tz="UTC")
        else:
            assert frame.index.min() >= pd.Timestamp("2020-01-01", tz="UTC")
            assert frame.index.max() < pd.Timestamp("2025-11-01", tz="UTC")
        return original(frame, pair, period)
    monkeypatch.setattr(data, "build_data", guarded)
    lab = data.load_data("eurusd")
    pd.testing.assert_frame_equal(lab.quarters, data.load_data("eurusd").quarters)
    assert len(calls) == 1
    marker.write_text("changed synthetic", encoding="ascii")
    data.load_data("eurusd")
    assert len(calls) == 2
    confirm = data.load_data("eurusd", "confirm")
    assert confirm.quarters.index.min().year == 2020 and confirm.quarters.index.max().year == 2025
    assert len(calls) == 3


@pytest.mark.parametrize("change", [
    lambda x: x.update(extra=1), lambda x: x["entry"].update(extra=1), lambda x: x["stop"].update(type="unknown"),
    lambda x: x.pop("target"), lambda x: x["stop"].pop("value"), lambda x: x["entry"].update(sides="unknown"),
    lambda x: x.update(entry=dict(type="trigger")), lambda x: x.update(lights=[dict(type="hours", start=20, end=4)]),
    lambda x: x.update(stop=dict(type="level", buffer=.1, level=dict(type="swing", timeframe="weekly", k=3))),
    lambda x: x.update(manage=[dict(type="unknown")]), lambda x: x["entry"].update(probability=1.1),
])
def test_strict_file_validation(change):
    value = idea()
    change(value)
    with pytest.raises(ValueError):
        blocks.validate_idea(value)


def write_idea(tmp_path):
    path = tmp_path / "idea.toml"
    path.write_text('name = "synthetic"\ndescription = "Synthetic idea"\nmax_hold_hours = 0.25\n'
                    '[entry]\ntype = "random"\nprobability = 1.0\nseed = 0\n'
                    '[stop]\ntype = "pips"\nvalue = 10\n[target]\ntype = "r"\nmultiple = 2\n', encoding="ascii")
    return path


def test_ledger_identical_scorecard_and_confirm_authorization(tmp_path, monkeypatch, capsys):
    path = write_idea(tmp_path)
    _, digest = blocks.load_idea(path)
    monkeypatch.setattr(runner, "results_dir", tmp_path / "results")
    monkeypatch.setattr(runner, "fresh_pairs", ("eurusd",))
    calls = []
    def load(pair, period):
        calls.append(period)
        return data.build_data(quarters(5, "2011-01-03" if period == "lab" else "2020-01-06"), pair, period)
    monkeypatch.setattr(runner, "load_data", load)
    first, trades, _ = runner.execute(path)
    second, again, _ = runner.execute(path)
    assert first == second
    pd.testing.assert_frame_equal(trades, again)
    ledger = score.read_ledger(runner.results_dir / "lab")
    assert len(ledger) == 2 and ledger.sha256.tolist() == [digest, digest]
    with pytest.raises(ValueError, match="pre-registration"):
        runner.execute(path, "confirm")
    prereg = tmp_path / "prereg.md"
    prereg.write_text("Wrong hash", encoding="ascii")
    with pytest.raises(ValueError, match="sha256"):
        runner.execute(path, "confirm", prereg)
    assert calls == ["lab", "lab"]
    prereg.write_text("Pre-registered idea sha256: " + digest, encoding="ascii")
    runner.execute(path, "confirm", prereg)
    assert calls == ["lab", "lab", "confirm"]
    with pytest.raises(ValueError, match="second confirm"):
        runner.execute(path, "confirm", prereg)
    assert calls == ["lab", "lab", "confirm"]
    assert len(score.read_ledger(runner.results_dir / "confirm")) == 1
    runner.main(["list"])
    assert "synthetic" in capsys.readouterr().out


def test_registries_complete_and_no_unlocks():
    for category, registry in (("level", blocks.levels), ("entry", blocks.entries), ("bias", blocks.biases),
                               ("light", blocks.lights), ("stop", blocks.stops), ("target", blocks.targets), ("manage", blocks.manage)):
        assert set(registry) == set(blocks.schemas[category])
    for name in (data.__file__, blocks.__file__, sim.__file__, score.__file__, runner.__file__):
        source = Path(name).read_text()
        source.encode("ascii")
        assert "final=True" not in source


def test_empty_scorecard_is_defined_without_trades():
    candidate = compiled(lights=[dict(type="hours", start=23, end=24)])
    trades, counts = sim.evaluate_pair(candidate)
    baseline, audits = {}, {}
    for seed in range(5):
        baseline[seed], audits[seed] = sim.random_baseline({"eurusd": candidate}, trades, seed)
    report, summary = score.scorecard(candidate.idea, "0" * 64, "lab", trades, {"eurusd": counts}, baseline, audits)
    assert summary["trades"] == 0 and np.isnan(summary["mean_r"])
    assert "direction skill d = nan" in report


def test_cached_and_uncached_blocks_and_trades_identical(tmp_path):
    source = data.build_data(quarters(), "eurusd")
    path = tmp_path / "cached.npz"
    data.save_cache(path, source)
    cached = data.read_cache(path, "eurusd", "lab")
    value = idea(entry=dict(type="break", direction="with", level=level_blocks[0]),
                 stop=dict(type="vol", multiple=1., hours=4), manage=[dict(type="breakeven", at_r=1.)])
    a, ca = sim.evaluate_pair(blocks.CompiledIdea(source, value))
    b, cb = sim.evaluate_pair(blocks.CompiledIdea(cached, value))
    pd.testing.assert_frame_equal(a, b, check_exact=True)
    assert ca == cb


def test_cache_cannot_mix_periods_before_date_conversion(tmp_path, monkeypatch):
    source = data.build_data(quarters(8, "2020-01-06"), "eurusd", "confirm")
    path = tmp_path / "cached.npz"
    data.save_cache(path, source)
    def forbidden(*args, **kwargs):
        raise AssertionError("An excluded cached date reached the date converter")
    monkeypatch.setattr(data.pd, "to_datetime", forbidden)
    with pytest.raises(ValueError, match="period boundary"):
        data.read_cache(path, "eurusd", "lab")

import pytest

from src.costs import bar_exit, entry_fill_price, exit_fill_price, swap_usd
from src.instruments import Config, pip_size, pip_value_usd_per_lot
from src.sizing import lot_size, trade_pnl_usd


@pytest.mark.parametrize("pair, size, price, value", [
    ("eurusd", 0.0001, 1.1, 10.0),
    ("gbpusd", 0.0001, 1.25, 10.0),
    ("usdjpy", 0.01, 150.0, 1000 / 150),
])
def test_pip_sizes_and_values(pair, size, price, value):
    assert pip_size(pair) == size
    assert pip_value_usd_per_lot(pair, price) == pytest.approx(value)


def test_usdjpy_pip_value_changes_with_price():
    assert pip_value_usd_per_lot("usdjpy", 100) == 10
    assert pip_value_usd_per_lot("usdjpy", 200) == 5


def test_config_defaults_and_independent_spreads():
    cfg = Config()
    assert cfg.spread_pips == {"eurusd": 1.0, "gbpusd": 1.2, "usdjpy": 1.0,
                               "audusd": 1.3, "usdcad": 1.7, "usdchf": 1.6, "nzdusd": 1.8,
                               "eurjpy": 1.8, "gbpjpy": 2.8, "eurgbp": 1.5}
    assert cfg.slippage_pips == 1.0
    assert cfg.swap_pips_per_night == -0.2
    assert cfg.min_lot == cfg.lot_step == 0.01
    assert cfg.start_balance == 200.0
    cfg.spread_pips["eurusd"] = 3
    assert Config().spread_pips["eurusd"] == 1.0


@pytest.mark.parametrize("pair, bid, spread", [
    ("eurusd", 1.1, 0.0001),
    ("gbpusd", 1.25, 0.00012),
    ("usdjpy", 150.0, 0.01),
])
def test_market_fills_use_correct_side_of_spread(pair, bid, spread):
    cfg = Config()
    assert entry_fill_price(pair, "long", bid, cfg) == pytest.approx(bid + spread)
    assert entry_fill_price(pair, "short", bid, cfg) == bid
    assert exit_fill_price(pair, "long", bid, cfg) == bid
    assert exit_fill_price(pair, "short", bid, cfg) == pytest.approx(bid + spread)


def test_short_stop_triggers_on_ask_high():
    bar = (1.0998, 1.1000, 1.0995, 1.0998)
    result = bar_exit("eurusd", bar, "short", 1.1001, 1.0980, Config())
    assert result[0] == "stop"
    assert result[1] == pytest.approx(1.1002)
    no_spread = Config(spread_pips={"eurusd": 0})
    assert bar_exit("eurusd", bar, "short", 1.1001, 1.0980, no_spread) is None
    assert bar_exit("eurusd", bar, "long", 1.0980, 1.1001, Config()) is None


def test_long_stop_triggers_on_bid_low():
    result = bar_exit("eurusd", (1.1, 1.1005, 1.0990, 1.1), "long", 1.0990, 1.1020, Config())
    assert result[0] == "stop"
    assert result[1] == pytest.approx(1.0989)


def test_short_target_requires_ask_low_to_touch():
    untouched = (1.1, 1.1005, 1.0990, 1.1)
    touched = (1.1, 1.1005, 1.0989, 1.1)
    assert bar_exit("eurusd", untouched, "short", 1.1020, 1.09905, Config()) is None
    assert bar_exit("eurusd", touched, "short", 1.1020, 1.09905, Config()) == ("target", 1.09905)


@pytest.mark.parametrize("side, stop, target, fill", [
    ("long", 1.0990, 1.1010, 1.0989),
    ("short", 1.1010, 1.0990, 1.1011),
])
def test_stop_wins_when_both_levels_touched(side, stop, target, fill):
    result = bar_exit("eurusd", (1.1, 1.102, 1.098, 1.1), side, stop, target, Config())
    assert result[0] == "stop"
    assert result[1] == pytest.approx(fill)


@pytest.mark.parametrize("side, stop, target, bar, expected", [
    ("long", 1.099, 1.102, (1.098, 1.100, 1.0975, 1.0995), 1.0979),
    ("short", 1.101, 1.098, (1.102, 1.103, 1.100, 1.101), 1.1022),
    ("long", 1.099, 1.102, (1.100, 1.1005, 1.0985, 1.100), 1.0989),
    ("short", 1.101, 1.098, (1.100, 1.1015, 1.0995, 1.100), 1.1011),
])
def test_gap_stop_fills_and_unchanged_normal_fills(side, stop, target, bar, expected):
    reason, fill = bar_exit("eurusd", bar, side, stop, target, Config())
    assert reason == "stop"
    assert fill == pytest.approx(expected)


@pytest.mark.parametrize("side, stop, target, bar, expected", [
    ("long", 1.099, 1.102, (1.098, 1.103, 1.0975, 1.100), 1.0979),
    ("short", 1.101, 1.098, (1.102, 1.103, 1.097, 1.100), 1.1022),
])
def test_gap_stop_still_wins_when_target_also_touched(side, stop, target, bar, expected):
    reason, fill = bar_exit("eurusd", bar, side, stop, target, Config())
    assert reason == "stop"
    assert fill == pytest.approx(expected)


@pytest.mark.parametrize("side, stop, target, bar", [
    ("long", 1.099, 1.102, (1.103, 1.104, 1.1025, 1.103)),
    ("short", 1.101, 1.098, (1.097, 1.0975, 1.096, 1.097)),
])
def test_gap_beyond_target_fills_at_target_level(side, stop, target, bar):
    assert bar_exit("eurusd", bar, side, stop, target, Config()) == ("target", target)


@pytest.mark.parametrize("side, stop, target, stop_bar, target_bar, direction", [
    ("long", 1.099, 1.102, (1.1, 1.1005, 1.0985, 1.1), (1.1, 1.1025, 1.0995, 1.1), -1),
    ("short", 1.101, 1.098, (1.1, 1.1015, 1.0995, 1.1), (1.1, 1.1005, 1.0975, 1.1), 1),
])
@pytest.mark.parametrize("slippage", [0, 1, 2.5])
def test_slippage_applies_only_to_stops(side, stop, target, stop_bar, target_bar, direction, slippage):
    cfg = Config(slippage_pips=slippage)
    reason, fill = bar_exit("eurusd", stop_bar, side, stop, target, cfg)
    assert reason == "stop"
    assert fill == pytest.approx(stop + direction * slippage * 0.0001)
    assert bar_exit("eurusd", target_bar, side, stop, target, cfg) == ("target", target)


@pytest.mark.parametrize("side, stop, target", [("long", 1.099, 1.101), ("short", 1.101, 1.099)])
def test_bar_with_no_touch_returns_none(side, stop, target):
    assert bar_exit("eurusd", (1.1, 1.1005, 1.0995, 1.1), side, stop, target, Config()) is None


@pytest.mark.parametrize("date, expected", [
    ("2025-01-06", -0.2),
    ("2025-01-07", -0.2),
    ("2025-01-08", -0.6),
    ("2025-01-11", 0),
    ("2025-01-12", 0),
])
def test_weekday_swap_and_triple_wednesday(date, expected):
    cost = swap_usd("eurusd", 0.1, date + "T21:00Z", date + "T22:00Z", 1.1, Config())
    assert cost == pytest.approx(expected)


def test_weekly_swap_has_seven_weighted_nights():
    cost = swap_usd("eurusd", 0.1, "2025-01-06T21:00Z", "2025-01-13T21:00Z", 1.1, Config())
    assert cost == pytest.approx(-1.4)


@pytest.mark.parametrize("opened, closed, expected", [
    ("2025-01-06T21:59Z", "2025-01-06T21:59:59Z", 0),
    ("2025-01-06T21:59Z", "2025-01-06T22:00Z", -0.2),
    ("2025-01-06T22:00Z", "2025-01-06T22:00Z", 0),
    ("2025-01-06T22:00Z", "2025-01-07T22:00Z", -0.2),
])
def test_swap_rollover_boundaries(opened, closed, expected):
    assert swap_usd("eurusd", 0.1, opened, closed, 1.1, Config()) == pytest.approx(expected)


def test_swap_converts_timezones_and_usdjpy_pip_value():
    cost = swap_usd("usdjpy", 0.1, "2025-01-08T16:00-06:00", "2025-01-08T17:00-06:00", 200, Config())
    # Entry is exactly 22:00 UTC, so that Wednesday rollover is excluded.
    assert cost == 0
    cost = swap_usd("usdjpy", 0.1, "2025-01-08T15:00-06:00", "2025-01-08T16:00-06:00", 200, Config())
    assert cost == pytest.approx(-0.3)


@pytest.mark.parametrize("pair, entry, distance", [
    ("eurusd", 1.1, 0.001),
    ("gbpusd", 1.25, 0.001),
    ("usdjpy", 150.0, 0.1),
])
@pytest.mark.parametrize("balance", [17.5, 200, 10000.11])
@pytest.mark.parametrize("direction", [-1, 1])
def test_round_down_never_exceeds_risk(pair, entry, distance, balance, direction):
    cfg = Config()
    stop = entry + direction * distance
    lot, risk = lot_size(balance, 0.05, entry, stop, pair, cfg)
    per_lot = (
        abs(entry - stop) / pip_size(pair) + cfg.spread_pips[pair] + cfg.slippage_pips
    ) * pip_value_usd_per_lot(pair, entry)
    assert risk <= balance * 0.05
    assert risk == pytest.approx(lot * per_lot)
    assert lot / cfg.lot_step == pytest.approx(round(lot / cfg.lot_step))
    if lot:
        assert lot >= cfg.min_lot
        assert (lot + cfg.lot_step) * per_lot > balance * 0.05
    else:
        assert cfg.min_lot * per_lot > balance * 0.05


@pytest.mark.parametrize("pair, entry, stop, expected_lot, expected_risk", [
    ("eurusd", 1.1, 1.099, 0.08, 9.6),
    ("gbpusd", 1.25, 1.249, 0.08, 9.76),
    ("usdjpy", 150, 149.9, 0.12, 9.6),
])
def test_sizing_includes_spread_and_stop_slippage(pair, entry, stop, expected_lot, expected_risk):
    lot, risk = lot_size(200, 0.05, entry, stop, pair, Config())
    assert lot == expected_lot
    assert risk == pytest.approx(expected_risk)


@pytest.mark.parametrize("pair, entry, distance", [
    ("eurusd", 1.1, 0.001), ("gbpusd", 1.25, 0.001), ("usdjpy", 150, 0.1),
])
@pytest.mark.parametrize("side", ["long", "short"])
def test_sized_risk_matches_executed_stop_loss(pair, entry, distance, side):
    cfg = Config()
    direction = 1 if side == "long" else -1
    reference_stop = entry - direction * distance
    lot, risk = lot_size(200, 0.05, entry, reference_stop, pair, cfg)
    spread = cfg.spread_pips[pair] * pip_size(pair)
    # Short exit levels are ask quotes; sizing takes the underlying bid level.
    stop = reference_stop + (spread if side == "short" else 0)
    target = entry + direction * 2 * distance + (spread if side == "short" else 0)
    bar = (entry, max(entry, reference_stop) + distance / 2,
           min(entry, reference_stop) - distance / 2, entry)
    reason, exit_price = bar_exit(pair, bar, side, stop, target, cfg)
    assert reason == "stop"
    entry_price = entry_fill_price(pair, side, entry, cfg)
    loss = -trade_pnl_usd(pair, side, entry_price, exit_price, lot, entry)
    assert loss == pytest.approx(risk)
    assert loss <= 10


def test_exact_step_and_custom_step():
    no_costs = Config(spread_pips={"eurusd": 0}, slippage_pips=0)
    assert lot_size(60, 0.05, 1.1, 1.099, "eurusd", no_costs) == (0.03, 3.0)
    custom = Config(min_lot=0.003, lot_step=0.003)
    lot, risk = lot_size(200, 0.05, 1.1, 1.099, "eurusd", custom)
    assert lot == 0.081
    assert risk == pytest.approx(9.72)


def test_minimum_lot_skip_and_exact_minimum():
    assert lot_size(10, 0.05, 1.1, 1.099, "eurusd", Config()) == (0.0, 0.0)
    assert lot_size(24, 0.05, 1.1, 1.099, "eurusd", Config()) == (0.01, 1.2)


@pytest.mark.parametrize("balance, risk_pct", [(0, 0.05), (200, 0)])
def test_zero_budget_skips(balance, risk_pct):
    assert lot_size(balance, risk_pct, 1.1, 1.099, "eurusd", Config()) == (0.0, 0.0)


# EURUSD long: ask entry 1.1001, stop fill 1.0989 -> -12 pips * $10 * .1 = -$12;
# Wednesday swap -0.2 * 3 * $10 * .1 = -$0.60; total -$12.60.
# GBPUSD short: bid entry 1.2500, ask target 1.2480 -> 20 pips * $10 * .1 = $20;
# Friday swap -0.2 * $10 * .1 = -$0.20; total $19.80.
# USDJPY long: ask entry 150.01, stop fill 149.89 -> -12 pips * ($1000/150) * .1 = -$8;
# Wednesday swap -0.2 * 3 * ($1000/150) * .1 = -$0.40; total -$8.40.
@pytest.mark.parametrize("pair, side, bid, stop, target, bar, date, price, reason, expected_entry, expected_exit, expected_net", [
    ("eurusd", "long", 1.1, 1.099, 1.102, (1.1, 1.1005, 1.0985, 1.1),
     "2025-01-08", 1.1, "stop", 1.1001, 1.0989, -12.6),
    ("gbpusd", "short", 1.25, 1.252, 1.248, (1.25, 1.2505, 1.247, 1.248),
     "2025-01-10", 1.25, "target", 1.25, 1.248, 19.8),
    ("usdjpy", "long", 150, 149.9, 150.2, (150, 150.05, 149.85, 150),
     "2025-01-08", 150, "stop", 150.01, 149.89, -8.4),
])
def test_hand_computed_round_trip(pair, side, bid, stop, target, bar, date, price, reason, expected_entry, expected_exit, expected_net):
    cfg = Config()
    entry = entry_fill_price(pair, side, bid, cfg)
    exit_reason, exit_price = bar_exit(pair, bar, side, stop, target, cfg)
    assert entry == pytest.approx(expected_entry)
    assert exit_reason == reason
    assert exit_price == pytest.approx(expected_exit)
    pnl = trade_pnl_usd(pair, side, entry, exit_price, 0.1, price)
    swap = swap_usd(pair, 0.1, date + "T21:00Z", date + "T23:00Z", price, cfg)
    assert pnl + swap == pytest.approx(expected_net)


@pytest.mark.parametrize("side, expected", [("long", 10), ("short", -10)])
def test_pnl_direction(side, expected):
    assert trade_pnl_usd("eurusd", side, 1.1, 1.101, 0.1, 1.1) == pytest.approx(expected)


def test_invalid_inputs_raise():
    with pytest.raises(ValueError, match="Unknown pair"):
        pip_size("abcxyz")
    with pytest.raises(ValueError, match="positive"):
        pip_value_usd_per_lot("usdjpy", 0)
    with pytest.raises(ValueError, match="fraction"):
        lot_size(200, 5, 1.1, 1.099, "eurusd", Config())
    with pytest.raises(ValueError, match="Side"):
        bar_exit("eurusd", (1, 2, 0.5, 1.5), "buy", 0.9, 1.1, Config())
    with pytest.raises(ValueError, match="timezone-aware"):
        swap_usd("eurusd", 0.1, "2025-01-06 21:00", "2025-01-06 23:00", 1.1, Config())
    with pytest.raises(ValueError, match="precede"):
        swap_usd("eurusd", 0.1, "2025-01-06T23:00Z", "2025-01-06T21:00Z", 1.1, Config())

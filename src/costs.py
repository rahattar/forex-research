"""Pure execution-cost helpers for bid-only bars."""

from math import isfinite

import pandas as pd

from .instruments import pip_size, pip_value_usd_per_lot


def _spread_price(pair, cfg):
    return pip_size(pair) * cfg.spread_pips[pair.lower()]


def _check_side(side):
    if side not in ("long", "short"):
        raise ValueError("Side must be 'long' or 'short'")


def entry_fill_price(pair, side, bid, cfg):
    """A market entry buys at ask or sells at bid, with no entry slippage."""
    _check_side(side)
    spread = _spread_price(pair, cfg)
    return bid + spread if side == "long" else bid


def exit_fill_price(pair, side, bid, cfg):
    """A market exit sells a long at bid or buys back a short at ask."""
    _check_side(side)
    spread = _spread_price(pair, cfg)
    return bid if side == "long" else bid + spread


def bar_exit(pair, bar, side, stop, target, cfg):
    """Return None or (reason, fill) for a bid (open, high, low, close) tuple.

    Stop/target levels are executable quotes: bid for longs, ask for shorts.
    Stops take priority, including when both levels are touched. A stop fills
    at the worse of its level and the opening executable quote, plus adverse
    slippage. Targets always fill at their level, even after a favorable gap.
    """
    _check_side(side)
    opening, high, low, _ = bar
    spread = _spread_price(pair, cfg)
    slippage = pip_size(pair) * cfg.slippage_pips
    if side == "long":
        if low <= stop:
            return "stop", min(opening, stop) - slippage
        if high >= target:
            return "target", target
    else:
        if high + spread >= stop:
            return "stop", max(opening + spread, stop) + slippage
        if low + spread <= target:
            return "target", target
    return None


def swap_pips(opened_at, closed_at, cfg):
    """Charge weekday 22:00 UTC rollovers in (opened_at, closed_at], in pips.

    Wednesday has weight three and weekends have weight zero. Both sides use
    the configured rate.
    Times must be timezone-aware; any timezone is converted to UTC.
    """
    opened = pd.Timestamp(opened_at)
    closed = pd.Timestamp(closed_at)
    if opened.tzinfo is None or closed.tzinfo is None or pd.isna(opened) or pd.isna(closed):
        raise ValueError("Swap times must be valid, timezone-aware timestamps")
    opened, closed = opened.tz_convert("UTC"), closed.tz_convert("UTC")
    if closed < opened:
        raise ValueError("Close time must not precede open time")
    first = opened.normalize() + pd.Timedelta(hours=22)
    rollovers = pd.date_range(first, closed, freq="1D")
    rollovers = rollovers[(rollovers > opened) & (rollovers.dayofweek < 5)]
    nights = sum(3 if time.dayofweek == 2 else 1 for time in rollovers)
    return nights * cfg.swap_pips_per_night


def swap_usd(pair, lot, opened_at, closed_at, price_for_pip_value, cfg):
    """Convert the same pip rollover accrual at the supplied USD pip value."""
    accrual = swap_pips(opened_at, closed_at, cfg)
    if not isfinite(lot) or lot < 0:
        raise ValueError("Lot size must be finite and nonnegative")
    return accrual * pip_value_usd_per_lot(pair, price_for_pip_value) * lot

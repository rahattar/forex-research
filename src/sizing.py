"""Pure lot sizing and USD P&L calculations; no position or account state."""

from decimal import Decimal, ROUND_FLOOR
from math import isfinite

from .instruments import pip_size, pip_value_usd_per_lot


def lot_size(balance, risk_pct, entry, stop, pair, cfg):
    """Return (lots, actual_stop_risk_usd), or (0.0, 0.0) to skip.

    risk_pct is a fraction (0.05 means 5%). entry/stop are bid reference prices,
    not executable fills; their distance receives one spread plus stop slippage.
    USDJPY pip value uses entry. Swap is excluded from the stop-risk budget.
    """
    if not isfinite(balance) or balance < 0:
        raise ValueError("Balance must be finite and nonnegative")
    if not isfinite(risk_pct) or not 0 <= risk_pct <= 1:
        raise ValueError("risk_pct must be a fraction between zero and one")
    pip_value = pip_value_usd_per_lot(pair, entry)
    if not isfinite(stop) or stop <= 0:
        raise ValueError("Stop must be finite and positive")

    # Decimal avoids losing a lot step to binary floating-point rounding.
    budget = Decimal(str(balance * risk_pct))
    distance = abs(Decimal(str(entry)) - Decimal(str(stop))) / Decimal(str(pip_size(pair)))
    distance += Decimal(str(cfg.spread_pips[pair.lower()])) + Decimal(str(cfg.slippage_pips))
    risk_per_lot = distance * Decimal(str(pip_value))
    if risk_per_lot <= 0:
        raise ValueError("Stop risk per lot must be positive")
    step = Decimal(str(cfg.lot_step))
    steps = (budget / risk_per_lot / step).to_integral_value(rounding=ROUND_FLOOR)
    lots = steps * step
    if lots < Decimal(str(cfg.min_lot)):
        return 0.0, 0.0
    return float(lots), float(lots * risk_per_lot)


def trade_pnl_usd(pair, side, entry_fill, exit_fill, lot, price_for_pip_value):
    """P&L from executable fills; costs in fills are not deducted a second time.

    Swap is separate. USDJPY uses the caller's explicit conversion price.
    """
    if side not in ("long", "short"):
        raise ValueError("Side must be 'long' or 'short'")
    for price in (entry_fill, exit_fill):
        if not isfinite(price) or price <= 0:
            raise ValueError("Fill prices must be finite and positive")
    if not isfinite(lot) or lot < 0:
        raise ValueError("Lot size must be finite and nonnegative")
    direction = 1 if side == "long" else -1
    pips = direction * (exit_fill - entry_fill) / pip_size(pair)
    return pips * pip_value_usd_per_lot(pair, price_for_pip_value) * lot

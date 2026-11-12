"""Instrument units and configurable costs for standard 100,000-unit lots."""

from dataclasses import dataclass, field
from math import isfinite


def pip_size(pair):
    pair = pair.lower()
    if pair not in ("eurusd", "gbpusd", "usdjpy", "audusd", "usdcad", "usdchf", "nzdusd", "eurjpy", "gbpjpy", "eurgbp"):
        raise ValueError(f"Unknown pair: {pair}")
    return 0.01 if pair.endswith("jpy") else 0.0001


def pip_value_usd_per_lot(pair, price):
    pip_size(pair)
    if pair.lower() not in ("eurusd", "gbpusd", "usdjpy"):
        raise ValueError("USD pip-value conversion is supported only for the original three pairs")
    if not isfinite(price) or price <= 0:
        raise ValueError("Price must be finite and positive")
    return 1000.0 / price if pair.lower() == "usdjpy" else 10.0


@dataclass
class Config:
    spread_pips: dict[str, float] = field(default_factory=lambda: {
        "eurusd": 1.0, "gbpusd": 1.2, "usdjpy": 1.0,
        "audusd": 1.3, "usdcad": 1.7, "usdchf": 1.6, "nzdusd": 1.8,
        "eurjpy": 1.8, "gbpjpy": 2.8, "eurgbp": 1.5,
    })
    slippage_pips: float = 1.0
    swap_pips_per_night: float = -0.2
    min_lot: float = 0.01
    lot_step: float = 0.01
    start_balance: float = 200.0

    def __post_init__(self):
        for pair, spread in self.spread_pips.items():
            pip_size(pair)
            if not isfinite(spread) or spread < 0:
                raise ValueError("Spreads must be finite and nonnegative")
        if not isfinite(self.slippage_pips) or self.slippage_pips < 0:
            raise ValueError("Slippage must be finite and nonnegative")
        if not isfinite(self.swap_pips_per_night):
            raise ValueError("Swap must be finite")
        for name in ("min_lot", "lot_step", "start_balance"):
            value = getattr(self, name)
            if not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")

"""Pluggable causal signals and a frequency-matched seeded random control."""

import numpy as np
import pandas as pd
from dataclasses import dataclass

from scripts.test1_residual import residual_features
from scripts.test2_trend import daily_trend
from scripts.test3_volatility import past_range

from .engine import EngineConfig, Signal
from .instruments import pip_size


def build_features(hourly, pair):
    """Precompute causal indicators; the engine exposes only the current row."""
    state = daily_trend(hourly).reindex(hourly.index, method="ffill")
    z = residual_features(hourly["close"], span=24)["z"]
    forecast = past_range(hourly, pair).ewm(span=10 * 24, adjust=False).mean()
    features = pd.DataFrame({
        "state": state, "z": z, "range_pips": forecast,
        "close": hourly["close"], "previous_close": hourly["close"].shift(1),
    })
    features["ready"] = features.notna().all(axis=1) & (forecast > 0)
    features["swing_low"] = hourly["low"].rolling(12, min_periods=12).min()
    features["swing_high"] = hourly["high"].rolling(12, min_periods=12).max()
    return features


def construct_signal(pair, side, bid, forecast_pips, cfg, setup, stop_mult=0.75):
    direction = 1 if side == "long" else -1
    distance = stop_mult * forecast_pips * pip_size(pair)
    spread = cfg.spread_pips[pair] * pip_size(pair)
    risk_price = distance + spread + cfg.slippage_pips * pip_size(pair)
    # The executable target earns two initial sizing R before swap/FX conversion.
    target_distance = 2 * risk_price + spread
    return Signal(pair, side, bid - direction * distance, bid + direction * target_distance, setup)


def trend_pullback(data, cfg=None, stop_mult=0.75, k=1.0):
    cfg = EngineConfig() if cfg is None else cfg
    signals = []
    for pair, row in sorted(data.features.items()):
        if not row["ready"]:
            continue
        side = None
        if row["state"] == 1 and row["z"] < -k and row["close"] > row["previous_close"]:
            side = "long"
        elif row["state"] == -1 and row["z"] > k and row["close"] < row["previous_close"]:
            side = "short"
        if side is not None:
            signals.append(construct_signal(pair, side, data.quotes[pair], row["range_pips"],
                                            cfg, "trend_pullback", stop_mult))
    return signals


def signal_probabilities(market, cfg=None):
    """Ex-post raw signal rates per pair/split, before balance/position gates.

    This matches the benchmark's requested frequency, without using trade P&L.
    Both samples exclude Friday's blocked hours and each pair's final bar.
    """
    cfg = EngineConfig() if cfg is None else cfg
    opportunities = dict.fromkeys(market.pairs, 0)
    counts = dict.fromkeys(market.pairs, 0)
    for index, view in market.contexts.items():
        if view.time.dayofweek == 4 and view.time.hour >= cfg.friday_cutoff_hour:
            continue
        eligible = {pair for pair, row in view.features.items()
                    if row["ready"] and index != market.last_indices[pair]}
        for pair in eligible:
            opportunities[pair] += 1
        for signal in trend_pullback(view, cfg):
            if signal.pair in eligible:
                counts[signal.pair] += 1
    rates = {pair: counts[pair] / opportunities[pair] if opportunities[pair] else 0.0 for pair in market.pairs}
    return rates, counts, opportunities


def random_entry(probabilities, seed, cfg=None, stop_mult=0.75):
    """Create a fresh deterministic benchmark; never reuse it across runs."""
    cfg = EngineConfig() if cfg is None else cfg
    rng = np.random.default_rng(seed)

    def strategy(data):
        signals = []
        for pair, row in sorted(data.features.items()):
            if not row["ready"] or rng.random() >= probabilities.get(pair, 0):
                continue
            side = "long" if rng.random() < 0.5 else "short"
            signals.append(construct_signal(pair, side, data.quotes[pair], row["range_pips"],
                                            cfg, "random_entry", stop_mult))
        return signals

    return strategy


def _check_rr(rr):
    if rr not in (1.5, 2.0, 3.0):
        raise ValueError("Only rr=1.5, 2.0 and 3.0 are registered")


def swing_signal(pair, side, bid, stop, forecast_pips, rr, cfg, setup):
    """Keep the structural stop fixed; apply inclusive distance bounds at entry."""
    _check_rr(rr)
    direction = 1 if side == "long" else -1
    forecast = forecast_pips * pip_size(pair)
    distance = direction * (bid - stop)
    if not np.isfinite(forecast) or forecast <= 0 or not np.isfinite(distance):
        return None
    ratio = distance / forecast
    if ratio < 0.3 - 1e-12 or ratio > 1.5 + 1e-12:
        return None
    spread = cfg.spread_pips[pair] * pip_size(pair)
    risk_price = distance + spread + cfg.slippage_pips * pip_size(pair)
    target = bid + direction * (rr * risk_price + spread)
    return Signal(pair, side, stop, target, setup)


def trend_swing(data, rr, cfg=None):
    """Same hourly pullback/confirmation; stop uses 12 completed hourly bars."""
    _check_rr(rr)
    cfg = EngineConfig() if cfg is None else cfg
    signals = []
    for candidate in trend_pullback(data, cfg):
        row = data.features[candidate.pair]
        forecast = row["range_pips"]
        buffer = 0.1 * forecast * pip_size(candidate.pair)
        stop = row["swing_low"] - buffer if candidate.side == "long" else row["swing_high"] + buffer
        signal = swing_signal(candidate.pair, candidate.side, data.quotes[candidate.pair], stop,
                              forecast, rr, cfg, "trend_swing")
        if signal is not None:
            signals.append(signal)
    return signals


@dataclass(frozen=True)
class ArmedTrigger:
    side: str
    stop: float
    forecast_pips: float
    armed_at: pd.Timestamp
    expires: pd.Timestamp


class Swing15m:
    frequency = "15m"

    def __init__(self, rr, cfg=None):
        _check_rr(rr)
        self.rr = rr
        self.cfg = EngineConfig() if cfg is None else cfg
        self.armed = {}

    def wants_bar(self, time, hourly):
        return hourly or bool(self.armed)

    def __call__(self, data):
        signals, consumed = [], set()
        for pair, arm in sorted(list(self.armed.items())):
            if data.time > arm.expires:
                del self.armed[pair]
                continue
            if pair not in data.quarter_bars:
                continue
            row = data.features.get(pair)
            direction = 1 if arm.side == "long" else -1
            bar = data.quarter_bars[pair]
            breached = bar[2] <= arm.stop if arm.side == "long" else bar[1] >= arm.stop
            if row is None or row["state"] != direction or breached:
                del self.armed[pair]
                continue
            previous = data.previous_quarter_bars.get(pair)
            if previous is None or data.time <= arm.armed_at:
                continue
            triggered = bar[3] > previous[1] if arm.side == "long" else bar[3] < previous[2]
            if triggered:
                signal = swing_signal(pair, arm.side, bar[3], arm.stop, arm.forecast_pips,
                                      self.rr, self.cfg, "trend_swing_15m")
                if signal is not None:
                    signals.append(signal)
                del self.armed[pair]
                consumed.add(pair)
        if data.hourly_pairs:
            for signal in trend_swing(data, self.rr, self.cfg):
                pair = signal.pair
                if pair in data.hourly_pairs and pair not in self.armed and pair not in consumed:
                    self.armed[pair] = ArmedTrigger(signal.side, signal.stop, data.features[pair]["range_pips"],
                                                   data.time, data.time + pd.Timedelta(hours=4))
        return signals


def trend_swing_15m(rr, cfg=None):
    """New pending triggers are not refreshed by subsequent hourly signals."""
    return Swing15m(rr, cfg)


def matched_entries(trades, cfg=None):
    """Recover bid-reference distances at entry, never the later BE stop."""
    cfg = EngineConfig() if cfg is None else cfg
    rows = []
    for trade in trades.sort_values(["entry_time", "pair"]).itertuples(index=False):
        size = pip_size(trade.pair)
        adjustment = cfg.spread_pips[trade.pair] * size if trade.side == "short" else 0
        rows.append({
            "entry_time": trade.entry_time, "pair": trade.pair,
            "stop_pips": abs(trade.entry_bid - (trade.initial_stop - adjustment)) / size,
            "target_pips": abs((trade.target - adjustment) - trade.entry_bid) / size,
        })
    return pd.DataFrame(rows, columns=["entry_time", "pair", "stop_pips", "target_pips"])


class MatchedRandom:
    frequency = "scheduled"

    def __init__(self, template, seed):
        self.template = template.sort_values(["entry_time", "pair"]).reset_index(drop=True).copy()
        sides = np.random.default_rng(seed).choice(("long", "short"), size=len(self.template))
        self.schedule = {}
        for row, side in zip(self.template.itertuples(index=False), sides):
            self.schedule.setdefault(row.entry_time, []).append((row.pair, side, row.stop_pips, row.target_pips))
        self.decision_times = frozenset(self.schedule)

    def __call__(self, data):
        signals = []
        for pair, side, stop_pips, target_pips in self.schedule.get(data.time, []):
            bid = data.quotes[pair]
            direction = 1 if side == "long" else -1
            size = pip_size(pair)
            signals.append(Signal(pair, side, bid - direction * stop_pips * size,
                                  bid + direction * target_pips * size, "matched_random"))
        return signals


def matched_random(trades, seed, cfg=None):
    return MatchedRandom(matched_entries(trades, cfg), seed)

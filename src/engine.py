"""Hourly decisions with chronological 15-minute execution and realized balance."""

from collections import Counter
from dataclasses import dataclass, field
from math import isfinite
from types import MappingProxyType

import numpy as np
import pandas as pd

from .costs import bar_exit, entry_fill_price, exit_fill_price, swap_usd
from .instruments import Config, pip_size, pip_value_usd_per_lot
from .loader import periods, split
from .sizing import lot_size, trade_pnl_usd


@dataclass
class EngineConfig(Config):
    risk_pct: float = 0.05
    max_hold_hours: float = 48
    daily_max_losses: int = 2
    daily_loss_pct: float = 0.15
    weekly_loss_pct: float = 0.25
    min_gap_hours: float = 4
    max_open_positions: int = 2
    combined_open_risk_pct: float = 0.10
    max_per_pair: int = 1
    max_per_usd_direction: int = 1
    friday_cutoff_hour: int = 20
    min_reward_risk: float = 1.5

    def __post_init__(self):
        super().__post_init__()
        for name in ("risk_pct", "daily_loss_pct", "weekly_loss_pct", "combined_open_risk_pct"):
            if not isfinite(getattr(self, name)) or not 0 < getattr(self, name) <= 1:
                raise ValueError(f"{name} must be a fraction in (0, 1]")
        for name in ("daily_max_losses", "max_open_positions", "max_per_pair", "max_per_usd_direction"):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not isfinite(self.max_hold_hours) or self.max_hold_hours <= 0:
            raise ValueError("max_hold_hours must be positive")
        if not isfinite(self.min_gap_hours) or self.min_gap_hours < 0:
            raise ValueError("min_gap_hours must be nonnegative")
        if not 0 <= self.friday_cutoff_hour <= 24 or self.min_reward_risk < 1.5:
            raise ValueError("Invalid Friday cutoff or reward/risk gate")


@dataclass(frozen=True)
class Signal:
    pair: str
    side: str
    stop: float
    target: float
    setup: str


@dataclass(frozen=True)
class VisibleData:
    time: pd.Timestamp
    quotes: dict
    features: dict
    _hourly: dict = field(repr=False)
    quarter_bars: dict = field(default_factory=dict)
    previous_quarter_bars: dict = field(default_factory=dict)
    hourly_pairs: tuple = ()

    def history(self, pair):
        """A private copy of hourly history through this completed decision bar."""
        return self._hourly[pair].loc[:self.time].copy()


def _validate_frame(frame):
    index = frame.index
    if not isinstance(index, pd.DatetimeIndex) or str(index.tz) != "UTC":
        raise ValueError("Engine data requires a UTC DatetimeIndex")
    if not index.is_unique or not index.is_monotonic_increasing:
        raise ValueError("Engine data must have unique sorted timestamps")
    if len(index) and index[-1] >= pd.Timestamp(periods["test"][0], tz="UTC"):
        raise ValueError("The engine cannot receive the locked test period")


class PreparedMarket:
    """Prepare immutable visible snapshots once, shared across benchmark runs.

    Only train/val are supported. Hourly history before the split is retained
    for indicator warm-up; execution bars and decisions stay inside the split.
    """

    def __init__(self, hourly, quarter_hourly, which, features=None):
        self._check_fold(which)
        if set(hourly) != set(quarter_hourly) or not hourly:
            raise ValueError("Hourly and 15m data must contain the same pairs")
        self.which = which
        self.pairs = tuple(sorted(hourly))
        self.pair_ids = {pair: i for i, pair in enumerate(self.pairs)}
        for frames in (hourly, quarter_hourly, features or {}):
            for pair, frame in frames.items():
                pip_size(pair)
                _validate_frame(frame)
                self._check_source(pair, frame)
        self.hourly = hourly
        quarters = {pair: self._execution_frame(frame, which) for pair, frame in quarter_hourly.items()}
        indices = [frame.index.as_unit("ns").asi8 for frame in quarters.values()]
        ticks = np.unique(np.concatenate(indices))
        index = pd.to_datetime(ticks, unit="ns", utc=True)
        self.times = tuple(index)
        self.index = index
        self.day_keys = index.normalize().as_unit("ns").asi8
        self.week_keys = self.day_keys - index.dayofweek.to_numpy(dtype=np.int64) * 86_400_000_000_000
        self.bars = np.full((len(index), len(self.pairs), 4), np.nan)
        self.previous_bars = np.full_like(self.bars, np.nan)
        self._quarter_cache = {}
        self._feature_rows, self._feature_positions = {}, {}
        for pair, frame in (features or {}).items():
            self._feature_rows[pair] = tuple(MappingProxyType(row) for row in frame.to_dict("records"))
            self._feature_positions[pair] = np.searchsorted(frame.index.as_unit("ns").asi8, ticks, side="right") - 1
        self.last_indices = {}
        decisions = {}
        for pair, frame in quarters.items():
            pair_id = self.pair_ids[pair]
            positions = np.searchsorted(ticks, frame.index.as_unit("ns").asi8)
            values = frame[["open", "high", "low", "close"]].to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise ValueError("Execution prices must be finite")
            self.bars[positions, pair_id] = values
            self.previous_bars[positions[1:], pair_id] = values[:-1]
            self.last_indices[pair] = int(positions[-1]) if len(positions) else -1
            completed = self._execution_frame(hourly[pair], which)
            matching = completed.index.intersection(frame.index)
            for timestamp in matching:
                position = int(np.searchsorted(ticks, timestamp.value))
                decisions.setdefault(position, []).append(pair)
        self.contexts = {}
        for position, decision_pairs in decisions.items():
            timestamp = self.times[position]
            quotes, visible_features = {}, {}
            for pair in sorted(decision_pairs):
                quotes[pair] = float(self.bars[position, self.pair_ids[pair], 3])
                if features is not None and pair in features and timestamp in features[pair].index:
                    row_position = self._feature_positions[pair][position]
                    visible_features[pair] = self._feature_rows[pair][row_position]
            self.contexts[position] = VisibleData(
                timestamp, MappingProxyType(quotes), MappingProxyType(visible_features), hourly,
                hourly_pairs=tuple(sorted(decision_pairs)),
            )
        self.bars.flags.writeable = False
        self.previous_bars.flags.writeable = False

    def _check_fold(self, which):
        if which not in ("train", "val"):
            raise ValueError("The engine supports train and val only; test is locked")

    def _check_source(self, pair, frame):
        if frame.attrs.get("data_root") == "fresh":
            raise ValueError("Fresh data cannot enter the existing train/val market")

    def _execution_frame(self, frame, which):
        return split(frame, which)

    def quarter_context(self, position):
        """Only completed current/previous 15m candles and as-of hourly features."""
        if position not in self._quarter_cache:
            quotes, features, bars, previous = {}, {}, {}, {}
            for pair, pair_id in self.pair_ids.items():
                bar = self.bars[position, pair_id]
                if np.isnan(bar[0]):
                    continue
                quotes[pair] = float(bar[3])
                bars[pair] = tuple(bar)
                prior = self.previous_bars[position, pair_id]
                if not np.isnan(prior[0]):
                    previous[pair] = tuple(prior)
                if pair in self._feature_positions:
                    row_position = self._feature_positions[pair][position]
                    if row_position >= 0:
                        features[pair] = self._feature_rows[pair][row_position]
            hourly = self.contexts.get(position)
            self._quarter_cache[position] = VisibleData(
                self.times[position], MappingProxyType(quotes), MappingProxyType(features), self.hourly,
                MappingProxyType(bars), MappingProxyType(previous), hourly.hourly_pairs if hourly else (),
            )
        return self._quarter_cache[position]


def usd_direction(pair, side):
    if side not in ("long", "short"):
        raise ValueError("Side must be long or short")
    pip_size(pair)
    direction = 1 if side == "long" else -1
    return direction if pair == "usdjpy" else -direction


@dataclass
class Position:
    pair: str
    side: str
    setup: str
    entry_bid: float
    entry_fill: float
    stop: float
    target: float
    initial_stop: float
    lots: float
    risk_usd: float
    risk_price: float
    open_time: pd.Timestamp
    breakeven: bool = False


def move_to_breakeven(position):
    old_stop = position.stop
    if position.side == "long":
        position.stop = max(old_stop, position.entry_fill)
    else:
        position.stop = min(old_stop, position.entry_fill)
    position.breakeven = True
    return position.stop != old_stop


@dataclass
class EngineResult:
    trades: pd.DataFrame
    summary: dict
    events: list


trade_columns = [
    "entry_time", "exit_time", "pair", "side", "setup", "entry_bid", "entry_fill", "exit_fill",
    "initial_stop", "stop", "target", "lots", "risk_usd", "pnl_usd", "r_multiple", "exit_reason",
    "swap", "balance_after", "breakeven",
]


class Engine:
    def __init__(self, market, strategy, cfg=None, record_events=False):
        self.market, self.strategy = market, strategy
        self.cfg = EngineConfig() if cfg is None else cfg
        self.balance = self.cfg.start_balance
        self.positions, self.records, self.events = [], [], []
        self.record_events = record_events
        self.counts = Counter()
        self.last_entry = None
        self.day_key = self.week_key = None
        self.day_start = self.week_start = self.balance
        self.day_losses = self.week_losses = 0
        self.day_halted = self.week_halted = False
        self.peak = self.balance
        self.max_drawdown_usd = self.max_drawdown_pct = 0.0
        self.hold_delta = pd.Timedelta(hours=self.cfg.max_hold_hours)
        self.gap_delta = pd.Timedelta(hours=self.cfg.min_gap_hours)
        self.frequency = getattr(strategy, "frequency", "hourly")
        self.scheduled_indices = None
        if self.frequency == "scheduled":
            indices = market.index.get_indexer(list(strategy.decision_times))
            if (indices < 0).any():
                raise ValueError("Scheduled entry time has no completed 15m quote")
            self.scheduled_indices = set(indices)

    def _event(self, time, kind, **values):
        if self.record_events:
            self.events.append({"time": time, "kind": kind, **values})

    def _reset_periods(self, index, time):
        day, week = self.market.day_keys[index], self.market.week_keys[index]
        if day != self.day_key:
            self.day_key, self.day_start = day, self.balance
            self.day_losses, self.day_halted = 0, False
            self._event(time, "day_reset", balance=self.balance)
        if week != self.week_key:
            self.week_key, self.week_start = week, self.balance
            self.week_losses, self.week_halted = 0, False
            self._event(time, "week_reset", balance=self.balance)

    def _close(self, position, time, fill, reason):
        spread = self.cfg.spread_pips[position.pair] * pip_size(position.pair)
        # USDJPY quote P&L and accumulated swap convert at the exit bid quote.
        conversion = fill if position.side == "long" else fill - spread
        swap = swap_usd(position.pair, position.lots, position.open_time, time, conversion, self.cfg)
        gross = trade_pnl_usd(position.pair, position.side, position.entry_fill, fill,
                              position.lots, conversion)
        pnl = gross + swap
        self.balance += pnl
        self.positions.remove(position)
        self.records.append({
            "entry_time": position.open_time, "exit_time": time, "pair": position.pair,
            "side": position.side, "setup": position.setup, "entry_bid": position.entry_bid,
            "entry_fill": position.entry_fill, "exit_fill": fill, "initial_stop": position.initial_stop,
            "stop": position.stop, "target": position.target, "lots": position.lots,
            "risk_usd": position.risk_usd, "pnl_usd": pnl, "r_multiple": pnl / position.risk_usd,
            "exit_reason": reason, "swap": swap, "balance_after": self.balance,
            "breakeven": position.breakeven,
        })
        self._event(time, "exit", pair=position.pair, reason=reason, pnl=pnl, balance=self.balance)
        if pnl < 0:
            self.day_losses += 1
            self.week_losses += 1
        if not self.day_halted and (self.day_losses >= self.cfg.daily_max_losses or
                                   self.balance <= self.day_start * (1 - self.cfg.daily_loss_pct)):
            self.day_halted = True
            self.counts["daily_halts"] += 1
            self._event(time, "daily_halt", balance=self.balance, losses=self.day_losses)
        if not self.week_halted and self.balance <= self.week_start * (1 - self.cfg.weekly_loss_pct):
            self.week_halted = True
            self.counts["weekly_halts"] += 1
            self._event(time, "weekly_halt", balance=self.balance, losses=self.week_losses)
        self.peak = max(self.peak, self.balance)
        drawdown = self.peak - self.balance
        self.max_drawdown_usd = max(self.max_drawdown_usd, drawdown)
        self.max_drawdown_pct = max(self.max_drawdown_pct, drawdown / self.peak)

    def _skip(self, time, signal, reason):
        self.counts[f"skipped_{reason}"] += 1
        self._event(time, "skip", pair=signal.pair, reason=reason)

    def _enter(self, signal, view, index):
        cfg, time = self.cfg, view.time
        if signal.pair not in view.quotes:
            return self._skip(time, signal, "missing_quote")
        if signal.side not in ("long", "short"):
            raise ValueError("Signal side must be long or short")
        if index == self.market.last_indices[signal.pair]:
            return self._skip(time, signal, "end")
        if self.balance <= 0:
            return self._skip(time, signal, "balance")
        if self.day_halted or self.week_halted:
            return self._skip(time, signal, "halt")
        if time.dayofweek == 4 and time.hour >= cfg.friday_cutoff_hour:
            return self._skip(time, signal, "friday")
        if self.last_entry is not None and time - self.last_entry < self.gap_delta:
            return self._skip(time, signal, "gap")
        if len(self.positions) >= cfg.max_open_positions:
            return self._skip(time, signal, "max_positions")
        if sum(p.pair == signal.pair for p in self.positions) >= cfg.max_per_pair:
            return self._skip(time, signal, "pair")
        usd = usd_direction(signal.pair, signal.side)
        if sum(usd_direction(p.pair, p.side) == usd for p in self.positions) >= cfg.max_per_usd_direction:
            return self._skip(time, signal, "usd_direction")
        bid = view.quotes[signal.pair]
        direction = 1 if signal.side == "long" else -1
        spread = cfg.spread_pips[signal.pair] * pip_size(signal.pair)
        slippage = cfg.slippage_pips * pip_size(signal.pair)
        distance = direction * (bid - signal.stop)
        reward = direction * (signal.target - bid) - spread
        if not all(isfinite(x) and x > 0 for x in (bid, signal.stop, signal.target)) or distance <= 0:
            return self._skip(time, signal, "invalid")
        if reward / (distance + spread + slippage) + 1e-10 < cfg.min_reward_risk:
            return self._skip(time, signal, "reward_risk")
        lots, risk = lot_size(self.balance, cfg.risk_pct, bid, signal.stop, signal.pair, cfg)
        if lots == 0:
            return self._skip(time, signal, "min_lot")
        # Initial allocated risk stays reserved until close, including after BE.
        if sum(p.risk_usd for p in self.positions) + risk > self.balance * cfg.combined_open_risk_pct + 1e-10:
            return self._skip(time, signal, "open_risk")
        quote_adjustment = spread if signal.side == "short" else 0
        stop, target = signal.stop + quote_adjustment, signal.target + quote_adjustment
        fill = entry_fill_price(signal.pair, signal.side, bid, cfg)
        risk_price = risk / (lots * pip_value_usd_per_lot(signal.pair, bid)) * pip_size(signal.pair)
        position = Position(signal.pair, signal.side, signal.setup, bid, fill, stop, target, stop,
                            lots, risk, risk_price, time)
        self.positions.append(position)
        self.last_entry = time
        self._event(time, "entry", pair=signal.pair, side=signal.side, stop=stop, target=target,
                    entry_fill=fill, lots=lots, risk_usd=risk, balance=self.balance,
                    open_risk=sum(p.risk_usd for p in self.positions))

    def run(self):
        for index, time in enumerate(self.market.times):
            if self.frequency == "scheduled":
                view = self.market.quarter_context(index) if index in self.scheduled_indices else None
            elif self.frequency == "15m":
                hourly = index in self.market.contexts
                view = self.market.quarter_context(index) if self.strategy.wants_bar(time, hourly) else None
            else:
                view = self.market.contexts.get(index)
            if not self.positions and view is None:
                continue
            self._reset_periods(index, time)
            for position in sorted(self.positions, key=lambda p: p.pair):
                bar = self.market.bars[index, self.market.pair_ids[position.pair]]
                if np.isnan(bar[0]):
                    continue
                outcome = bar_exit(position.pair, bar, position.side, position.stop, position.target, self.cfg)
                if outcome is None and time - position.open_time >= self.hold_delta:
                    outcome = ("time", exit_fill_price(position.pair, position.side, bar[3], self.cfg))
                if outcome is None and index == self.market.last_indices[position.pair]:
                    outcome = ("end_of_split", exit_fill_price(position.pair, position.side, bar[3], self.cfg))
                if outcome is not None:
                    self._close(position, time, outcome[1], outcome[0])
                    continue
                spread = self.cfg.spread_pips[position.pair] * pip_size(position.pair)
                favorable_quote = bar[1] if position.side == "long" else bar[2] + spread
                direction = 1 if position.side == "long" else -1
                if direction * (favorable_quote - position.entry_fill) >= position.risk_price:
                    old_stop = position.stop
                    if move_to_breakeven(position):
                        self._event(time, "breakeven", pair=position.pair, old_stop=old_stop,
                                    stop=position.stop, entry_fill=position.entry_fill)
            if view is not None:
                signals = self.strategy(view)
                for signal in sorted(signals, key=lambda s: (s.pair, s.side, s.setup)):
                    self._enter(signal, view, index)
        trades = pd.DataFrame(self.records, columns=trade_columns)
        summary = self._summary(trades)
        return EngineResult(trades, summary, self.events)

    def _summary(self, trades):
        pnl, r = trades["pnl_usd"].astype(float), trades["r_multiple"].astype(float)
        gains, losses = pnl.loc[pnl > 0].sum(), -pnl.loc[pnl < 0].sum()
        hold = (trades["exit_time"] - trades["entry_time"]).dt.total_seconds() / 3600 if len(trades) else pd.Series(dtype=float)
        return {
            "split": self.market.which, "trades": len(trades),
            "win_rate": float((pnl > 0).mean()) if len(pnl) else np.nan,
            "average_r": r.mean(), "expectancy_r": r.mean(),
            "profit_factor": gains / losses if losses > 0 else (np.inf if gains > 0 else np.nan),
            "ending_balance": self.balance, "max_drawdown_usd": self.max_drawdown_usd,
            "max_drawdown_pct": self.max_drawdown_pct, "daily_halts": self.counts["daily_halts"],
            "weekly_halts": self.counts["weekly_halts"], "skipped_min_lot": self.counts["skipped_min_lot"],
            "average_hold_hours": hold.mean(), **dict(self.counts),
        }


def run_engine(market, strategy, cfg=None, record_events=False):
    return Engine(market, strategy, cfg, record_events).run()

"""D1 daily channels, fixed ATR stops on hourly bars, and exact mirrors."""

import numpy as np
import pandas as pd

from .costs import bar_exit, entry_fill_price, exit_fill_price, swap_pips
from .daily_momentum import end, month_keys, start, validate_times
from .engine import Signal
from .instruments import Config, pip_size
from .loader import daily_bars
from .signal_eval import mirror_signal


def breakout_features(hourly):
    validate_times(hourly)
    daily = daily_bars(hourly)
    previous = daily.close.shift(1)
    true_range = pd.concat([daily.high - daily.low, (daily.high - previous).abs(),
                            (daily.low - previous).abs()], axis=1).max(axis=1)
    # Wilder recursion seeded with the first 20 true ranges' arithmetic mean.
    atr = np.full(len(daily), np.nan)
    if len(daily) >= 20:
        atr[19] = true_range.iloc[:20].mean()
        for position in range(20, len(daily)):
            atr[position] = (19 * atr[position - 1] + true_range.iloc[position]) / 20
    result = pd.DataFrame({"close": daily.close, "atr": atr,
                           "entry_high": daily.high.rolling(55).max().shift(1),
                           "entry_low": daily.low.rolling(55).min().shift(1),
                           "exit_high": daily.high.rolling(20).max().shift(1),
                           "exit_low": daily.low.rolling(20).min().shift(1)})
    result["side"] = np.select([result.close > result.entry_high, result.close < result.entry_low], [1, -1], default=0)
    result["exit_long"] = result.close < result.exit_low
    result["exit_short"] = result.close > result.exit_high
    return result


def simulate_breakout(hourly, features, pair, signal, entry_index, cfg=None):
    """A D1 path has no target, BE or time stop; active stops precede daily exits."""
    cfg = Config() if cfg is None else cfg
    validate_times(hourly)
    validate_times(features)
    side = signal.side
    direction = 1 if side == "long" else -1
    size = pip_size(pair)
    spread_pips = cfg.spread_pips[pair]
    spread = spread_pips * size
    bars = hourly[["open", "high", "low", "close"]].to_numpy(dtype=float)
    bid = float(bars[entry_index, 3])
    distance = direction * (bid - signal.stop) / size
    if not np.isfinite(distance) or distance <= 0:
        raise ValueError("D1 stop must be on the losing side")
    risk = distance + spread_pips + cfg.slippage_pips
    stop = signal.stop + (spread if side == "short" else 0)
    entry_time = hourly.index[entry_index]
    entry_fill = entry_fill_price(pair, side, bid, cfg)
    target = np.inf if side == "long" else -np.inf
    channel = features.index[features[f"exit_{side}"] & (features.index >= entry_time)]
    # An exit announced at midnight executes at the first strictly later quote.
    channel_index = hourly.index.searchsorted(channel[0], side="right") if len(channel) else len(hourly)
    exit_index = len(hourly) - 1
    reason = "end_of_data"
    fill = exit_fill_price(pair, side, bars[-1, 3], cfg)
    for position in range(entry_index + 1, len(hourly)):
        outcome = bar_exit(pair, bars[position], side, stop, target, cfg)
        if outcome is not None:
            reason, fill = outcome
        elif position >= channel_index:
            reason, fill = "channel", exit_fill_price(pair, side, bars[position, 3], cfg)
        else:
            continue
        exit_index = position
        break
    exit_time = hourly.index[exit_index]
    swap = swap_pips(entry_time, exit_time, cfg)
    price_pips = direction * (fill - entry_fill) / size
    net_pips = price_pips + swap
    return {"entry_time": entry_time, "exit_time": exit_time, "pair": pair, "side": side,
            "entry_bid": bid, "entry_fill": entry_fill, "exit_fill": fill, "initial_stop_bid": signal.stop,
            "stop": stop, "stop_distance_pips": distance, "initial_risk_pips": risk,
            "price_pips": price_pips, "swap_pips": swap, "net_pips": net_pips, "r": net_pips / risk,
            "cost_r": (spread_pips + cfg.slippage_pips) / risk,
            "hold_days": (exit_time - entry_time).total_seconds() / 86400, "exit_reason": reason}


def evaluate_breakout(hourly, pair, cfg=None):
    cfg = Config() if cfg is None else cfg
    features = breakout_features(hourly)
    busy_until, records = None, []
    for time, row in features.iterrows():
        if row.side == 0 or not np.isfinite(row.atr) or row.atr <= 0:
            continue
        if busy_until is not None and time < busy_until:
            continue
        entry = int(hourly.index.searchsorted(time, side="right"))
        if entry >= len(hourly):
            continue
        entry_time = hourly.index[entry]
        bid = hourly.close.iloc[entry]
        direction = int(row.side)
        signal = Signal(pair, "long" if direction == 1 else "short", bid - direction * 2 * row.atr,
                        np.inf if direction == 1 else -np.inf, "daily_breakout")
        original = simulate_breakout(hourly, features, pair, signal, entry, cfg)
        opposite = mirror_signal(signal, bid)
        mirror = simulate_breakout(hourly, features, pair, opposite, entry, cfg)
        record = {key: value for key, value in original.items() if key != "r"}
        record.update({f"mirror_{key}": value for key, value in mirror.items() if key != "r"})
        record.update(signal_time=time, atr=row.atr, r_strategy=original["r"], r_mirror=mirror["r"],
                      d=(original["r"] - mirror["r"]) / 2)
        records.append(record)
        busy_until = original["exit_time"]
    columns = ["entry_time", "exit_time", "pair", "side", "entry_bid", "entry_fill", "exit_fill", "initial_stop_bid",
               "stop", "stop_distance_pips", "initial_risk_pips", "price_pips", "swap_pips", "net_pips",
               "cost_r", "hold_days", "exit_reason"]
    columns += [f"mirror_{column}" for column in columns]
    columns += ["signal_time", "atr", "r_strategy", "r_mirror", "d"]
    trades = pd.DataFrame(records, columns=columns)
    for column in ("entry_time", "exit_time", "mirror_entry_time", "mirror_exit_time", "signal_time"):
        trades[column] = pd.to_datetime(trades[column], utc=True)
    return trades


def trade_statistics(trades):
    result = {"n": len(trades), "mean_r_strategy": trades.r_strategy.mean(), "mean_r_mirror": trades.r_mirror.mean(),
              "mean_d": trades.d.mean(), "mean_cost_r": trades.cost_r.mean(), "average_hold_days": trades.hold_days.mean()}
    for prefix, reason_column, value_column in (("", "exit_reason", "r_strategy"),
                                                ("mirror_", "mirror_exit_reason", "r_mirror")):
        for reason in ("stop", "channel", "end_of_data"):
            values = trades.loc[trades[reason_column] == reason, value_column]
            result[f"{prefix}{reason}_n"] = len(values)
            result[f"{prefix}{reason}_mean_r"] = values.mean()
    return result


def trade_bootstrap(trades, resamples=5000, seed=0):
    calendar = np.arange(start.year * 12 + start.month, end.year * 12 + end.month)
    values = trades[["r_strategy", "d"]].copy()
    values["month"] = month_keys(trades.entry_time)
    values["n"] = 1
    blocks = values.groupby("month")[["r_strategy", "d", "n"]].sum().reindex(calendar, fill_value=0)
    sums = blocks[["r_strategy", "d"]].to_numpy(dtype=float)
    counts = blocks.n.to_numpy(dtype=int)
    rng = np.random.default_rng(seed)
    draws = []
    for offset in range(0, resamples, 250):
        selected = rng.integers(0, len(blocks), size=(min(250, resamples - offset), len(blocks)))
        n = counts[selected].sum(axis=1)[:, None]
        total = sums[selected].sum(axis=1)
        draws.append(np.divide(total, n, out=np.full_like(total, np.nan), where=n > 0))
    draws = np.concatenate(draws)
    valid = np.isfinite(draws).all(axis=1)
    interval = np.percentile(draws[valid], [2.5, 97.5], axis=0) if valid.any() else np.full((2, 2), np.nan)
    return {"r_ci_low": interval[0, 0], "r_ci_high": interval[1, 0], "d_ci_low": interval[0, 1],
            "d_ci_high": interval[1, 1], "bootstrap_resamples": resamples, "bootstrap_seed": seed,
            "n_months": len(blocks), "bootstrap_valid": int(valid.sum())}

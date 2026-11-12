"""Independent bid-bar trade paths, chronological pair occupancy, and controls."""

from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .costs import bar_exit, entry_fill_price, exit_fill_price, swap_pips
from .instruments import pip_size
from .lab_blocks import manage
from .loader import fresh_pairs


@dataclass
class PathState:
    side: int
    entry_fill: float
    risk: float
    stop: float
    best_quote: float
    best_close: float


def mirror_signal(signal, bid):
    side = -signal["side"]
    return dict(side=side, stop_bid=bid - side * abs(bid - signal["stop_bid"]),
                target_bid=bid + side * abs(signal["target_bid"] - bid), risk_price=signal["risk_price"])


def apply_manage(compiled, index, state):
    for rule in compiled.managers:
        proposed = manage[rule["type"]](rule, index, state)
        if np.isfinite(proposed):
            state.stop = max(state.stop, proposed) if state.side == 1 else min(state.stop, proposed)
    return state.stop


def simulate_trade(compiled, signal, entry_index, *, trace=None):
    context, idea = compiled.context, compiled.idea
    pair, cfg, times, bars = context.pair, context.cfg, context.times, context.bars
    side = signal["side"]
    side_name = "long" if side == 1 else "short"
    size, spread = pip_size(pair), context.spread
    bid, opened = float(bars[entry_index, 3]), times[entry_index]
    fill = entry_fill_price(pair, side_name, bid, cfg)
    adjustment = spread if side == -1 else 0
    stop, target = signal["stop_bid"] + adjustment, signal["target_bid"] + adjustment
    stop_pips = side * (bid - signal["stop_bid"]) / size
    risk_pips = stop_pips + cfg.spread_pips[pair] + cfg.slippage_pips
    risk = risk_pips * size
    state = PathState(side, fill, risk, stop, fill, bid)
    deadline = opened.value + int(idea.get("max_hold_hours", 48) * 3_600_000_000_000)
    ticks = context.ticks
    exit_index, reason = entry_index, "end_of_split"
    exit_fill = exit_fill_price(pair, side_name, bid, cfg)
    breakeven = False
    for index in range(entry_index + 1, len(times)):
        bar = bars[index]
        active_stop = state.stop
        outcome = bar_exit(pair, bar, side_name, active_stop, target, cfg)
        if outcome is None and ticks[index] >= deadline:
            outcome = ("time", exit_fill_price(pair, side_name, bar[3], cfg))
        if outcome is None and index == len(times) - 1:
            outcome = ("end_of_split", exit_fill_price(pair, side_name, bar[3], cfg))
        if outcome is not None:
            reason, exit_fill = outcome
            exit_index = index
            if trace is not None:
                trace.append((index, active_stop, state.stop, reason))
            break
        favorable = bar[1] if side == 1 else bar[2] + spread
        state.best_quote = max(state.best_quote, favorable) if side == 1 else min(state.best_quote, favorable)
        state.best_close = max(state.best_close, bar[3]) if side == 1 else min(state.best_close, bar[3])
        for rule in compiled.managers:
            if rule["type"] == "breakeven" and side * (state.best_quote - fill) >= rule["at_r"] * risk:
                breakeven = True
        apply_manage(compiled, index, state)
        if trace is not None:
            trace.append((index, active_stop, state.stop, None))
    closed = times[exit_index]
    swap = swap_pips(opened, closed, cfg)
    price_pips = side * (exit_fill - fill) / size
    return dict(entry_time=opened, exit_time=closed, pair=pair, side=side_name, setup=idea["name"],
                entry_bid=bid, entry_fill=fill, initial_stop_bid=signal["stop_bid"], initial_target_bid=signal["target_bid"],
                initial_stop=stop, stop=state.stop, target=target, exit_fill=exit_fill, exit_reason=reason,
                stop_distance_pips=side * (bid - signal["stop_bid"]) / size,
                target_distance_pips=side * (signal["target_bid"] - bid) / size,
                initial_risk_pips=risk_pips, price_pips=price_pips, swap_pips=swap, net_pips=price_pips + swap,
                r=(price_pips + swap) / risk_pips, hold_hours=(closed - opened).total_seconds() / 3600,
                cost_r=(context.spread + context.slippage) / risk, breakeven=breakeven, exit_index=exit_index)


path_columns = ("entry_time", "exit_time", "pair", "side", "setup", "entry_bid", "entry_fill", "initial_stop_bid",
                "initial_target_bid", "initial_stop", "stop", "target", "exit_fill", "exit_reason", "stop_distance_pips",
                "target_distance_pips", "initial_risk_pips", "price_pips", "swap_pips", "net_pips", "hold_hours", "cost_r", "breakeven")
trade_columns = list(path_columns) + ["r_strategy"] + ["mirror_" + name for name in path_columns] + ["r_mirror", "d", "coin"]


def evaluate_pair(compiled):
    buy, sell = compiled.candidates()
    indices = np.flatnonzero(buy | sell)
    counts, records, busy = Counter(), [], -1
    for index in indices:
        ambiguous = buy[index] and sell[index]
        count = 2 if ambiguous else 1
        counts["signals"] += count
        if ambiguous:
            counts["ambiguous"] += 2
            continue
        if index < busy:
            counts["pair busy"] += 1
            continue
        side = 1 if buy[index] else -1
        signal, skipped = compiled.signal(index, side)
        if skipped:
            counts[skipped] += 1
            continue
        original = simulate_trade(compiled, signal, index)
        mirror = simulate_trade(compiled, mirror_signal(signal, original["entry_bid"]), index)
        row = {key: original[key] for key in path_columns}
        row["r_strategy"] = original["r"]
        row.update({"mirror_" + key: mirror[key] for key in path_columns})
        row.update(r_mirror=mirror["r"], d=(original["r"] - mirror["r"]) / 2, coin=(original["r"] + mirror["r"]) / 2)
        records.append(row)
        counts["trades"] += 1
        busy = original["exit_index"]
    frame = pd.DataFrame(records, columns=trade_columns)
    for column in ("entry_time", "exit_time", "mirror_entry_time", "mirror_exit_time"):
        frame[column] = pd.to_datetime(frame[column], utc=True)
    return frame, dict(counts)


def random_baseline(compiled_by_pair, trades, seed):
    rng = np.random.default_rng(seed)
    records, audit = [], []
    for pair in fresh_pairs:
        if pair not in compiled_by_pair:
            continue
        compiled = compiled_by_pair[pair]
        times = compiled.context.times
        template = trades.loc[trades.pair == pair]
        years = sorted(template.entry_time.dt.year.unique()) if len(template) else []
        for year in years:
            requested = int((template.entry_time.dt.year == year).sum())
            candidates = np.flatnonzero(times.year == year)
            achieved, skipped_draws = 0, 0
            for _ in range(requested):
                for _ in range(20):
                    index = int(candidates[rng.integers(len(candidates))])
                    side = int(rng.choice([-1, 1]))
                    signal, skipped = compiled.signal(index, side, baseline=True)
                    if skipped:
                        skipped_draws += 1
                        continue
                    path = simulate_trade(compiled, signal, index)
                    records.append({key: path[key] for key in ("entry_time", "exit_time", "pair", "side", "r", "exit_reason",
                                                               "stop_distance_pips", "target_distance_pips")})
                    achieved += 1
                    break
            audit.append(dict(seed=seed, pair=pair, year=int(year), requested=requested, achieved=achieved, skipped_draws=skipped_draws))
    columns = ("entry_time", "exit_time", "pair", "side", "r", "exit_reason", "stop_distance_pips", "target_distance_pips")
    return pd.DataFrame(records, columns=columns), pd.DataFrame(audit, columns=["seed", "pair", "year", "requested", "achieved", "skipped_draws"])

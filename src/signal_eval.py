"""Account-free signal paths, exact opposite-side mirrors, and weekly inference."""

from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .costs import bar_exit, entry_fill_price, exit_fill_price, swap_pips
from .engine import EngineConfig, PreparedMarket, Signal, move_to_breakeven
from .instruments import pip_size, pip_value_usd_per_lot
from .loader import fresh_end, periods


class FreshMarket(PreparedMarket):
    """One continuous, explicitly separate fresh fold per pair; no test unlock."""

    def __init__(self, hourly, quarter_hourly, features=None):
        for frames in (hourly, quarter_hourly):
            for pair, frame in frames.items():
                if frame.attrs.get("data_root") != "fresh" or frame.attrs.get("pair") != pair:
                    raise ValueError("FreshMarket requires fresh-loader provenance")
        super().__init__(hourly, quarter_hourly, "fresh", features)

    def _check_fold(self, which):
        if which != "fresh":
            raise ValueError("FreshMarket supports the fresh fold only")

    def _check_source(self, pair, frame):
        if len(frame) and (frame.index[0] < pd.Timestamp("2010-01-01", tz="UTC") or frame.index[-1] >= fresh_end(pair)):
            raise ValueError(f"Fresh closing-time boundary exceeded for {pair}")

    def _execution_frame(self, frame, which):
        return frame


def _check_market(market):
    if market.which not in ("train", "val") and not (isinstance(market, FreshMarket) and market.which == "fresh"):
        raise ValueError("Signal evaluation supports train/val or an explicit FreshMarket; test is locked")


@dataclass
class PathPosition:
    side: str
    entry_fill: float
    stop: float
    breakeven: bool = False


@dataclass
class SignalResult:
    trades: pd.DataFrame
    counts: dict


path_columns = [
    "entry_time", "pair", "side", "setup", "entry_bid", "entry_fill", "initial_stop_bid",
    "initial_target_bid", "initial_stop", "stop", "target", "stop_distance_pips", "target_distance_pips",
    "initial_risk_pips", "exit_time", "exit_fill", "exit_reason", "price_pips", "swap_pips",
    "net_pips", "r", "breakeven", "hold_hours", "cost_r",
]
trade_columns = [column for column in path_columns if column != "r"] + ["r_strategy"]
trade_columns += [f"mirror_{column}" for column in path_columns if column != "r"]
trade_columns += ["r_mirror", "d", "coinflip_r"]


def mirror_signal(signal, bid):
    """Mirror the initial bid-reference distances, independent of later exits."""
    side = "short" if signal.side == "long" else "long"
    direction = 1 if side == "long" else -1
    stop_distance = abs(bid - signal.stop)
    target_distance = abs(signal.target - bid)
    return Signal(signal.pair, side, bid - direction * stop_distance,
                  bid + direction * target_distance, signal.setup)


def simulate_trade(market, signal, entry_index, cfg=None):
    """Simulate one path in pips, with no balance, lot sizing, or account gates.

    The entry candle cannot exit the trade. Active exits precede time/split-end
    exits and BE changes. A signal at the final quote is immediately liquidated
    at that quote, charging its spread, without touching past candle extremes.
    """
    cfg = EngineConfig() if cfg is None else cfg
    _check_market(market)
    if signal.side not in ("long", "short"):
        raise ValueError("Side must be long or short")
    pair_id = market.pair_ids[signal.pair]
    entry_time = market.times[entry_index]
    entry_bid = float(market.bars[entry_index, pair_id, 3])
    direction = 1 if signal.side == "long" else -1
    size = pip_size(signal.pair)
    spread_pips = cfg.spread_pips[signal.pair]
    spread = spread_pips * size
    stop_distance = direction * (entry_bid - signal.stop) / size
    target_distance = direction * (signal.target - entry_bid) / size
    if not all(np.isfinite(value) for value in (entry_bid, stop_distance, target_distance)) or stop_distance <= 0 or target_distance <= 0:
        raise ValueError("Invalid strategy signal or entry quote")
    risk_pips = stop_distance + spread_pips + cfg.slippage_pips
    adjustment = spread if signal.side == "short" else 0
    initial_stop, target = signal.stop + adjustment, signal.target + adjustment
    entry_fill = entry_fill_price(signal.pair, signal.side, entry_bid, cfg)
    position = PathPosition(signal.side, entry_fill, initial_stop)
    deadline = entry_time + pd.Timedelta(hours=cfg.max_hold_hours)
    last = market.last_indices[signal.pair]
    exit_time, exit_fill, reason = entry_time, exit_fill_price(signal.pair, signal.side, entry_bid, cfg), "end_of_split"
    for index in range(entry_index + 1, last + 1):
        bar = market.bars[index, pair_id]
        if np.isnan(bar[0]):
            continue
        time = market.times[index]
        outcome = bar_exit(signal.pair, bar, signal.side, position.stop, target, cfg)
        if outcome is None and time >= deadline:
            outcome = ("time", exit_fill_price(signal.pair, signal.side, bar[3], cfg))
        if outcome is None and index == last:
            outcome = ("end_of_split", exit_fill_price(signal.pair, signal.side, bar[3], cfg))
        if outcome is not None:
            reason, exit_fill = outcome
            exit_time = time
            break
        favorable = bar[1] if signal.side == "long" else bar[2] + spread
        if direction * (favorable - entry_fill) >= risk_pips * size:
            move_to_breakeven(position)
    swap = swap_pips(entry_time, exit_time, cfg)
    price_pips = direction * (exit_fill - entry_fill) / size
    net_pips = price_pips + swap
    return {
        "entry_time": entry_time, "pair": signal.pair, "side": signal.side, "setup": signal.setup,
        "entry_bid": entry_bid, "entry_fill": entry_fill, "initial_stop_bid": signal.stop,
        "initial_target_bid": signal.target, "initial_stop": initial_stop, "stop": position.stop, "target": target,
        "stop_distance_pips": stop_distance, "target_distance_pips": target_distance,
        "initial_risk_pips": risk_pips, "exit_time": exit_time, "exit_fill": exit_fill,
        "exit_reason": reason, "price_pips": price_pips, "swap_pips": swap, "net_pips": net_pips,
        "r": net_pips / risk_pips, "breakeven": position.breakeven,
        "hold_hours": (exit_time - entry_time).total_seconds() / 3600,
        "cost_r": (spread_pips + cfg.slippage_pips) / risk_pips,
    }


def evaluate_signals(market, strategy, cfg=None):
    """Take every signal except Friday entries and an already-occupied pair.

    A future path is evaluated once to obtain the pair's exit time; signals
    strictly before that exit are ignored, exactly as chronological occupancy.
    Mirrors neither share occupancy nor change the original strategy's path.
    """
    cfg = EngineConfig() if cfg is None else cfg
    _check_market(market)
    frequency = getattr(strategy, "frequency", "hourly")
    if frequency not in ("hourly", "15m"):
        raise ValueError("Only the registered hourly/15m strategies are supported")
    busy_until, records, counts = {}, [], Counter()
    for index, time in enumerate(market.times):
        if frequency == "15m":
            if not strategy.wants_bar(time, index in market.contexts):
                continue
            view = market.quarter_context(index)
        else:
            view = market.contexts.get(index)
            if view is None:
                continue
        accepted_now = set()
        for signal in sorted(strategy(view), key=lambda s: (s.pair, s.side, s.setup)):
            counts["signals"] += 1
            if time.dayofweek == 4 and time.hour >= cfg.friday_cutoff_hour:
                counts["ignored_friday"] += 1
                continue
            if signal.pair in accepted_now or (signal.pair in busy_until and time < busy_until[signal.pair]):
                counts["ignored_pair"] += 1
                continue
            original = simulate_trade(market, signal, index, cfg)
            opposite = mirror_signal(signal, original["entry_bid"])
            mirror = simulate_trade(market, opposite, index, cfg)
            row = {key: value for key, value in original.items() if key != "r"}
            row["r_strategy"] = original["r"]
            row.update({f"mirror_{key}": value for key, value in mirror.items() if key != "r"})
            row["r_mirror"] = mirror["r"]
            row["d"] = (original["r"] - mirror["r"]) / 2
            row["coinflip_r"] = (original["r"] + mirror["r"]) / 2
            records.append(row)
            busy_until[signal.pair] = original["exit_time"]
            accepted_now.add(signal.pair)
            counts["accepted"] += 1
    trades = pd.DataFrame(records, columns=trade_columns)
    for column in ("entry_time", "exit_time", "mirror_entry_time", "mirror_exit_time"):
        trades[column] = pd.to_datetime(trades[column], utc=True)
    return SignalResult(trades, dict(counts))


def week_start(times):
    times = pd.DatetimeIndex(times)
    return times.normalize() - pd.to_timedelta(times.dayofweek, unit="D")


def weekly_blocks(trades, start, end):
    """Entry-week blocks contain all pairs; include zero-trade calendar weeks."""
    first = week_start([pd.Timestamp(start)])[0]
    last = week_start([pd.Timestamp(end) - pd.Timedelta(1, "ns")])[0]
    calendar = pd.date_range(first, last, freq="7D")
    values = trades[["r_strategy", "d"]].astype(float).copy()
    values["week"] = week_start(trades.entry_time)
    values["n"] = 1
    return values.groupby("week")[["r_strategy", "d", "n"]].sum().reindex(calendar, fill_value=0)


def block_bootstrap(trades, start, end, resamples=5000, seed=0):
    """Cluster-resample weeks; ratio of summed returns to summed trade counts."""
    if resamples < 1:
        raise ValueError("resamples must be positive")
    blocks = weekly_blocks(trades, start, end)
    sums = blocks[["r_strategy", "d"]].to_numpy(dtype=float)
    counts = blocks.n.to_numpy(dtype=int)
    rng = np.random.default_rng(seed)
    estimates = []
    # Batch draws keep memory bounded without changing RNG order.
    for offset in range(0, resamples, 250):
        indices = rng.integers(0, len(blocks), size=(min(250, resamples - offset), len(blocks)))
        n = counts[indices].sum(axis=1)
        totals = sums[indices].sum(axis=1)
        estimates.append(np.divide(totals, n[:, None], out=np.full_like(totals, np.nan), where=n[:, None] > 0))
    draws = np.concatenate(estimates)
    valid = np.isfinite(draws).all(axis=1)
    intervals = np.percentile(draws[valid], [2.5, 97.5], axis=0) if valid.any() else np.full((2, 2), np.nan)
    weekly_d = blocks.d
    se = weekly_d.std(ddof=1) / np.sqrt(len(weekly_d)) if len(weekly_d) >= 2 else np.nan
    return {
        "r_ci_low": intervals[0, 0], "r_ci_high": intervals[1, 0],
        "d_ci_low": intervals[0, 1], "d_ci_high": intervals[1, 1],
        "weekly_d_t_stat": weekly_d.mean() / se if se > 0 else np.nan,
        "n_weeks": len(blocks), "bootstrap_resamples": resamples,
        "bootstrap_valid": int(valid.sum()), "bootstrap_seed": seed,
    }


def basic_statistics(trades):
    ordered = trades.sort_values(["exit_time", "pair", "entry_time"])
    cumulative = np.r_[0.0, ordered.r_strategy.to_numpy(dtype=float).cumsum()]
    drawdown = (np.maximum.accumulate(cumulative) - cumulative).max()
    row = {
        "n": len(trades), "mean_r_strategy": trades.r_strategy.mean(), "mean_r_mirror": trades.r_mirror.mean(),
        "mean_d": trades.d.mean(), "mean_coinflip_r": trades.coinflip_r.mean(), "mean_cost_r": trades.cost_r.mean(),
        "max_drawdown_r": drawdown,
    }
    for prefix, reason_column, value_column in (("", "exit_reason", "r_strategy"),
                                                ("mirror_", "mirror_exit_reason", "r_mirror")):
        for reason in ("stop", "target", "time", "end_of_split"):
            values = trades.loc[trades[reason_column] == reason, value_column]
            row[f"{prefix}{reason}_n"] = len(values)
            row[f"{prefix}{reason}_mean_r"] = values.mean()
    return row


def summarize(trades, resamples=5000, seed=0, variant_names=None):
    """Main uncertainty rows plus year, pair, side, and pair/side detail rows."""
    rows = []
    for variant in trades.variant.drop_duplicates() if variant_names is None else variant_names:
        all_variant = trades.loc[trades.variant == variant]
        for period in ("train", "val", "pooled"):
            sample = all_variant if period == "pooled" else all_variant.loc[all_variant['split'] == period]
            start, end = ((periods["train"][0], periods["val"][1]) if period == "pooled" else periods[period])
            bounds = (pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC"))
            rows.append({"variant": variant, "period": period, "group": "all", "value": "all",
                         **basic_statistics(sample), **block_bootstrap(sample, *bounds, resamples, seed)})
            for group, column, categories in (("pair", "pair", ("eurusd", "gbpusd", "usdjpy")),
                                               ("side", "side", ("long", "short"))):
                for value in categories:
                    subset = sample.loc[sample[column] == value]
                    rows.append({"variant": variant, "period": period, "group": group, "value": value,
                                 **basic_statistics(subset)})
            for pair in ("eurusd", "gbpusd", "usdjpy"):
                for side in ("long", "short"):
                    subset = sample.loc[(sample.pair == pair) & (sample.side == side)]
                    rows.append({"variant": variant, "period": period, "group": "pair_side", "value": f"{pair}/{side}",
                                 **basic_statistics(subset)})
        for year in range(2019, 2026):
            sample = all_variant.loc[all_variant.entry_time.dt.year == year]
            rows.append({"variant": variant, "period": "pooled", "group": "year", "value": str(year),
                         **basic_statistics(sample)})
    result = pd.DataFrame(rows)
    result["pooled_d_ci_above_zero"] = ((result.period == "pooled") & (result['group'] == "all") & (result.d_ci_low > 0))
    return result


def engine_consistency(trades, engine_trades, cfg=None, variant_names=None):
    """Compare common entries in raw engine R and FX-normalized pip R."""
    cfg = EngineConfig() if cfg is None else cfg
    engine_trades = engine_trades.copy()
    for column in ("entry_time", "exit_time"):
        engine_trades[column] = pd.to_datetime(engine_trades[column], utc=True)
    keys = ["variant", "split", "pair", "entry_time"]
    if trades.duplicated(keys).any() or engine_trades.duplicated(keys).any():
        raise ValueError("Duplicate trade entry keys in consistency check")
    common = trades.merge(engine_trades, on=keys, suffixes=("_signal", "_engine"), validate="one_to_one")
    raw, normalized, path_mismatch = [], [], []
    for row in common.itertuples(index=False):
        spread = cfg.spread_pips[row.pair] * pip_size(row.pair)
        exit_bid = row.exit_fill_engine if row.side_engine == "long" else row.exit_fill_engine - spread
        factor = pip_value_usd_per_lot(row.pair, row.entry_bid_engine) / pip_value_usd_per_lot(row.pair, exit_bid)
        raw.append(abs(row.r_strategy - row.r_multiple))
        normalized.append(abs(row.r_strategy - row.r_multiple * factor))
        path_mismatch.append(row.side_signal != row.side_engine or row.exit_time_signal != row.exit_time_engine or
                             row.exit_reason_signal != row.exit_reason_engine or abs(row.exit_fill_signal - row.exit_fill_engine) > 1e-8)
    common["raw_r_difference"] = raw
    common["normalized_r_difference"] = normalized
    common["path_mismatch"] = path_mismatch
    rows = []
    for variant in trades.variant.drop_duplicates() if variant_names is None else variant_names:
        for period in ("train", "val", "pooled"):
            selected = common.loc[common.variant == variant]
            reference = engine_trades.loc[engine_trades.variant == variant]
            if period != "pooled":
                selected = selected.loc[selected['split'] == period]
                reference = reference.loc[reference['split'] == period]
            rows.append({"variant": variant, "period": period, "engine_trades": len(reference),
                         "engine_matches": len(selected), "engine_max_abs_r_diff": selected.raw_r_difference.max(),
                         "engine_max_abs_r_diff_fx_normalized": selected.normalized_r_difference.max(),
                         "engine_exit_path_mismatches": int(selected.path_mismatch.sum())})
    return pd.DataFrame(rows)

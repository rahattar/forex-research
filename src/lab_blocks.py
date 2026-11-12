"""Strict idea-file validation and the lab's registered array building blocks."""

import hashlib
import json
from pathlib import Path
import re
import tomllib

import numpy as np
import pandas as pd

from .instruments import Config, pip_size
from .lab_data import timeframes, visible


levels, entries, biases, lights, stops, targets, manage = {}, {}, {}, {}, {}, {}, {}
schemas = {
    "level": {"swing": ("timeframe", "k"), "rolling": ("timeframe", "n"),
              "prev_day": (), "prev_week": (), "round": ("step_pips",)},
    "entry": {"break": ("level", "direction"), "bounce": ("level",), "trigger": (), "random": ("probability", "seed")},
    "bias": {"trend": ("timeframe", "fast", "slow"), "sma": ("timeframe", "n")},
    "light": {"hours": ("start", "end"), "cost": ("max_cost_r",), "vol_ratio": ("min", "max"),
              "trend_agree": ("timeframe", "fast", "slow")},
    "stop": {"level": ("level", "buffer"), "vol": ("multiple", "hours"), "pips": ("value",)},
    "target": {"r": ("multiple",), "level": ("level",), "none": ()},
    "manage": {"breakeven": ("at_r",), "trail_level": ("level", "buffer"), "trail_vol": ("multiple", "hours")},
}


def register(registry, name):
    def decorator(function):
        registry[name] = function
        return function
    return decorator


def number(value, minimum=0, *, positive=False, integer=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or
            (integer and not isinstance(value, int)) or value < minimum or (positive and value == minimum)):
        raise ValueError("Invalid numeric block parameter")


def validate_block(block, category):
    if not isinstance(block, dict) or block.get("type") not in schemas[category]:
        raise ValueError("Unknown or missing " + category + " type")
    required = {"type", *schemas[category][block["type"]]}
    optional = {"sides"} if category == "entry" else set()
    if required - block.keys() or block.keys() - required - optional:
        raise ValueError("Missing required or unknown keys in " + category)
    for key, value in block.items():
        if key == "level":
            validate_block(value, "level")
        elif key == "timeframe" and value not in timeframes:
            raise ValueError("Unknown timeframe")
        elif key in ("k", "n", "fast", "slow"):
            number(value, positive=True, integer=True)
        elif key == "seed":
            number(value, integer=True)
        elif key in ("multiple", "hours", "value", "step_pips"):
            number(value, positive=True)
        elif key in ("buffer", "at_r", "max_cost_r", "min", "max", "probability"):
            number(value)
        elif key == "direction" and value not in ("with", "against"):
            raise ValueError("Unknown break direction")
        elif key == "sides" and value not in ("both", "long", "short"):
            raise ValueError("Unknown sides")
    if block["type"] == "hours":
        number(block["start"], integer=True)
        number(block["end"], integer=True)
        if not 0 <= block["start"] < block["end"] <= 24:
            raise ValueError("Invalid hours interval")
    if block["type"] == "vol_ratio" and block["min"] > block["max"]:
        raise ValueError("Invalid volatility-ratio interval")
    if block["type"] == "random" and block["probability"] > 1:
        raise ValueError("Probability must be in [0, 1]")


def validate_idea(idea):
    required = {"name", "description", "entry", "stop", "target"}
    allowed = required | {"max_hold_hours", "bias", "lights", "manage"}
    if not isinstance(idea, dict) or required - idea.keys() or idea.keys() - allowed:
        raise ValueError("Unknown or missing top-level idea keys")
    if not isinstance(idea["name"], str) or not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", idea["name"]):
        raise ValueError("Idea name must be a lowercase filename component")
    if not isinstance(idea["description"], str):
        raise ValueError("Description must be text")
    json.dumps(idea, ensure_ascii=False).encode("ascii")
    number(idea.get("max_hold_hours", 48), positive=True)
    for category in ("entry", "stop", "target", "bias"):
        if category in idea:
            validate_block(idea[category], category)
    for field, category in (("lights", "light"), ("manage", "manage")):
        if not isinstance(idea.get(field, []), list):
            raise ValueError(field + " must be a list")
        for block in idea.get(field, []):
            validate_block(block, category)
    if idea["entry"]["type"] == "trigger" and "bias" not in idea:
        raise ValueError("Trigger requires a bias")
    return idea


def load_idea(path):
    raw = Path(path).read_bytes()
    idea = validate_idea(tomllib.loads(raw.decode("ascii")))
    return idea, hashlib.sha256(raw).hexdigest()


class Context:
    def __init__(self, data, cfg=None):
        self.data, self.cfg = data, Config() if cfg is None else cfg
        self.pair, self.times = data.pair, data.quarters.index
        self.weekday, self.hour = self.times.dayofweek.to_numpy(), self.times.hour.to_numpy()
        self.ticks = self.times.as_unit("ns").asi8
        self.bars = data.quarters[["open", "high", "low", "close"]].to_numpy(copy=True)
        self.close = self.bars[:, 3]
        self.e1 = data.expected_move(1).to_numpy(copy=True)
        self.spread = self.cfg.spread_pips[self.pair] * pip_size(self.pair)
        self.slippage = self.cfg.slippage_pips * pip_size(self.pair)
        self._levels, self._biases = {}, {}

    def level(self, block):
        key = json.dumps(block, sort_keys=True)
        if key not in self._levels:
            self._levels[key] = levels[block["type"]](self, block)
        return self._levels[key]

    def bias(self, block):
        key = json.dumps(block, sort_keys=True)
        if key not in self._biases:
            self._biases[key] = biases[block["type"]](self, block)
        return self._biases[key]


def aligned(context, frame):
    return visible(frame, context.times).to_numpy()


@register(levels, "swing")
def swing(context, block):
    frame, k = context.data.frames[block["timeframe"]], block["k"]
    published = pd.DataFrame(np.nan, index=frame.index, columns=["support", "resistance"])
    if len(frame) >= 2 * k + 1:
        for column, source, lower in (("support", "low", True), ("resistance", "high", False)):
            windows = np.lib.stride_tricks.sliding_window_view(frame[source].to_numpy(), 2 * k + 1)
            pivot = windows[:, k]
            others = np.concatenate([windows[:, :k], windows[:, k + 1:]], axis=1)
            confirmed = pivot < others.min(axis=1) if lower else pivot > others.max(axis=1)
            published.iloc[2 * k:, published.columns.get_loc(column)] = np.where(confirmed, pivot, np.nan)
    return aligned(context, published.ffill())


@register(levels, "rolling")
def rolling(context, block):
    frame, n = context.data.frames[block["timeframe"]], block["n"]
    return aligned(context, pd.DataFrame({"support": frame.low.rolling(n).min(), "resistance": frame.high.rolling(n).max()}))


@register(levels, "prev_day")
def prev_day(context, block):
    return aligned(context, context.data.frames["1d"][["low", "high"]])


@register(levels, "prev_week")
def prev_week(context, block):
    frame = context.data.frames["1d"]
    local = frame.index.tz_convert("America/New_York").tz_localize(None)
    weekdays = local.dayofweek < 5
    local = local[weekdays]
    friday = local.normalize() - pd.to_timedelta(local.dayofweek, unit="D") + pd.Timedelta(days=4, hours=17)
    labels = friday.tz_localize("America/New_York").tz_convert("UTC")
    weeks = frame.loc[weekdays].groupby(labels).agg({"low": "min", "high": "max"})
    return aligned(context, weeks.loc[weeks.index <= context.times[-1]]) if len(context.times) else np.empty((0, 2))


@register(levels, "round")
def round_levels(context, block):
    step = block["step_pips"] * pip_size(context.pair)
    support = (np.ceil(context.close / step) - 1) * step
    resistance = (np.floor(context.close / step) + 1) * step
    support = np.where(support >= context.close, support - step, support)
    resistance = np.where(resistance <= context.close, resistance + step, resistance)
    return np.column_stack([support, resistance])


def previous(values):
    return np.concatenate([np.full((1, values.shape[1]), np.nan), values[:-1]], axis=0) if len(values) else values.copy()


@register(entries, "break")
def break_entry(context, block):
    level = previous(context.level(block["level"]))
    close = context.close
    prior = np.r_[np.nan, close[:-1]]
    down = (close < level[:, 0]) & (prior >= level[:, 0])
    up = (close > level[:, 1]) & (prior <= level[:, 1])
    return (up, down) if block["direction"] == "with" else (down, up)


@register(entries, "bounce")
def bounce(context, block):
    level = previous(context.level(block["level"]))
    return ((context.bars[:, 2] <= level[:, 0]) & (context.close > level[:, 0]),
            (context.bars[:, 1] >= level[:, 1]) & (context.close < level[:, 1]))


@register(entries, "trigger")
def trigger(context, block):
    prior = previous(context.bars)
    return context.close > prior[:, 1], context.close < prior[:, 2]


@register(entries, "random")
def random_entry(context, block):
    rng = np.random.default_rng(block["seed"])
    draws = rng.random((len(context.times), 2))
    selected = draws[:, 0] < block["probability"]
    direction = np.where(draws[:, 1] < .5, -1, 1)
    return selected & (direction == 1), selected & (direction == -1)


@register(biases, "trend")
def trend(context, block):
    close = context.data.frames[block["timeframe"]].close
    value = np.sign(close.ewm(span=block["fast"], adjust=False).mean() - close.ewm(span=block["slow"], adjust=False).mean())
    value.iloc[:block["slow"] - 1] = np.nan
    return aligned(context, value)


@register(biases, "sma")
def sma(context, block):
    close = context.data.frames[block["timeframe"]].close
    return aligned(context, np.sign(close - close.rolling(block["n"]).mean()))


@register(lights, "hours")
def hours_light(context, block, side):
    return (context.times.hour >= block["start"]) & (context.times.hour < block["end"])


@register(lights, "cost")
def cost_light(context, block, side):
    return block["max_cost_r"]


@register(lights, "vol_ratio")
def vol_light(context, block, side):
    ratio = context.data.vol_ratio.to_numpy()
    return (ratio >= block["min"]) & (ratio <= block["max"])


@register(lights, "trend_agree")
def trend_light(context, block, side):
    bias = context.bias(dict(type="trend", timeframe=block["timeframe"], fast=block["fast"], slow=block["slow"]))
    return bias == side


@register(stops, "level")
def level_stop(context, block, side):
    level = context.level(block["level"])[:, 0 if side == 1 else 1]
    buffer = block["buffer"] * context.e1 if block["buffer"] else 0
    return level - side * buffer


@register(stops, "vol")
def vol_stop(context, block, side):
    return context.close - side * block["multiple"] * context.e1 * np.sqrt(block["hours"])


@register(stops, "pips")
def pips_stop(context, block, side):
    return context.close - side * block["value"] * pip_size(context.pair)


@register(targets, "r")
def r_target(context, block, side, stop):
    risk = side * (context.close - stop) + context.spread + context.slippage
    return context.close + side * (block["multiple"] * risk + context.spread)


@register(targets, "level")
def level_target(context, block, side, stop):
    return context.level(block["level"])[:, 1 if side == 1 else 0]


@register(targets, "none")
def no_target(context, block, side, stop):
    return np.full(len(context.times), side * np.inf)


@register(manage, "breakeven")
def breakeven(rule, index, state):
    return state.entry_fill if state.side * (state.best_quote - state.entry_fill) >= rule["at_r"] * state.risk else state.stop


@register(manage, "trail_level")
def trail_level(rule, index, state):
    return rule["values"][state.side][index]


@register(manage, "trail_vol")
def trail_vol(rule, index, state):
    return state.best_close - state.side * rule["distance"][index] + (rule["spread"] if state.side == -1 else 0)


class CompiledIdea:
    def __init__(self, data, idea, cfg=None):
        self.context = Context(data, cfg)
        self.idea = validate_idea(idea)
        self.bias = self.context.bias(idea["bias"]) if "bias" in idea else None
        self.stop, self.target, self.masks = {}, {}, {}
        self.cost_limits = []
        for side in (1, -1):
            self.stop[side] = stops[idea["stop"]["type"]](self.context, idea["stop"], side)
            self.target[side] = targets[idea["target"]["type"]](self.context, idea["target"], side, self.stop[side])
            masks = []
            for block in idea.get("lights", []):
                if block["type"] == "cost":
                    if side == 1:
                        self.cost_limits.append(lights["cost"](self.context, block, side))
                else:
                    masks.append((block["type"], np.asarray(lights[block["type"]](self.context, block, side))))
            self.masks[side] = masks
        self.managers = []
        for block in idea.get("manage", []):
            rule = dict(block)
            if rule["type"] == "trail_level":
                stop_block = dict(type="level", level=block["level"], buffer=block["buffer"])
                rule["values"] = {side: level_stop(self.context, stop_block, side) + (self.context.spread if side == -1 else 0)
                                  for side in (1, -1)}
            if rule["type"] == "trail_vol":
                rule["distance"] = block["multiple"] * self.context.e1 * np.sqrt(block["hours"])
                rule["spread"] = self.context.spread
            self.managers.append(rule)

    def candidates(self):
        buy, sell = entries[self.idea["entry"]["type"]](self.context, self.idea["entry"])
        if self.idea["entry"]["type"] == "random" and self.bias is not None:
            selected = buy | sell
            buy, sell = selected & (self.bias == 1), selected & (self.bias == -1)
        return np.asarray(buy), np.asarray(sell)

    def signal(self, index, side, *, baseline=False):
        context = self.context
        if context.weekday[index] == 4 and context.hour[index] >= 20:
            return None, "friday"
        if not baseline:
            if self.bias is not None and self.bias[index] != side:
                return None, "bias"
            sides = self.idea["entry"].get("sides", "both")
            if (sides == "long" and side != 1) or (sides == "short" and side != -1):
                return None, "sides"
        stop, target, bid = self.stop[side][index], self.target[side][index], context.close[index]
        if not np.isfinite(stop):
            return None, "stop unavailable"
        distance = side * (bid - stop)
        if distance <= 0:
            return None, "stop wrong side"
        # Account for the rounding of bid minus a constructed stop at equality.
        if distance + 2 * np.spacing(abs(bid)) < 2 * context.spread:
            return None, "stop too close"
        if np.isnan(target) or side * (target - bid) <= 0:
            return None, "target unavailable or wrong side"
        risk = distance + context.spread + context.slippage
        for kind, mask in self.masks[side]:
            if baseline and kind == "trend_agree":
                continue
            if not mask[index]:
                return None, "light " + kind
        if any((context.spread + context.slippage) / risk > maximum for maximum in self.cost_limits):
            return None, "light cost"
        return {"side": side, "stop_bid": stop, "target_bid": target, "risk_price": risk}, None

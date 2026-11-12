"""Nine pre-registered rates signals/horizons; train/val only, no engine."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_engine import percentile_rank
from src.costs import entry_fill_price, exit_fill_price
from src.instruments import Config, pip_size
from src.loader import load_pair, pairs, periods, split
from src.rates import diff, load_rates, lock


signals = ("level", "chg60", "chg120")
horizons = (1, 4, 8)
years = tuple(range(2020, 2026))
benchmark_seeds = tuple(range(200))
markup_pp = 0.5
results_dir = Path(__file__).resolve().parents[1] / "results"


def rate_signals(differential):
    differential = differential.loc[differential.index < lock]
    result = pd.DataFrame({"level": np.sign(differential)})
    for days in (60, 120):
        previous = differential.reindex(differential.index - pd.Timedelta(days=days)).to_numpy()
        change = differential - previous
        # Include exactly 0.10 pp despite binary subtraction rounding.
        active = change.abs() + 1e-12 >= 0.10
        result[f"chg{days}"] = np.sign(change).where(active, 0).where(change.notna())
    return result


def carry_pips(pair, side, entry_time, exit_time, entry_price, rates, markup=markup_pp):
    """Simple ACT/365 daily accrual on [entry, exit), including weekend days.

    Each UTC day's available rate gets its actual holding-day fraction. Markup
    is subtracted for both sides; conversion uses the executable entry price.
    """
    if side not in ("long", "short"):
        raise ValueError("Side must be long or short")
    entry, exit = pd.Timestamp(entry_time), pd.Timestamp(exit_time)
    if entry.tzinfo is None or exit.tzinfo is None or exit < entry:
        raise ValueError("Carry requires ordered timezone-aware times")
    entry, exit = entry.tz_convert("UTC"), exit.tz_convert("UTC")
    if entry >= lock or exit >= lock:
        raise ValueError("Carry cannot enter the locked period")
    if entry == exit:
        return 0.0
    calendar = pd.date_range(entry.normalize(), (exit - pd.Timedelta(1, "ns")).normalize(), freq="1D")
    daily = diff(pair, rates.reindex(calendar))
    if daily.isna().any():
        return np.nan
    starts = calendar.where(calendar >= entry, entry)
    ends = (calendar + pd.Timedelta(days=1)).where(calendar + pd.Timedelta(days=1) <= exit, exit)
    weights = (ends - starts).total_seconds().to_numpy() / 86_400
    direction = 1 if side == "long" else -1
    annual_pct_days = ((direction * daily.to_numpy() - markup) * weights).sum()
    return float(annual_pct_days / 100 * entry_price / 365 / pip_size(pair))


def monday_opportunities(hourly, pair, which, h, rates, cfg):
    """First close strictly after Monday; endpoint exactly h weeks after entry."""
    part = split(hourly, which)
    start, end = (pd.Timestamp(date, tz="UTC") for date in periods[which])
    mondays = pd.date_range(start, end, freq="W-MON", inclusive="left")
    positions = part.index.searchsorted(mondays, side="right")
    rows = []
    for monday_number, (monday, position) in enumerate(zip(mondays, positions)):
        if position >= len(part) or monday < part.index[0].normalize():
            continue
        entry = part.index[position]
        exit = entry + pd.Timedelta(weeks=h)
        if entry >= monday + pd.Timedelta(weeks=1) or exit >= end or exit >= lock or exit not in part.index:
            continue
        entry_bid, exit_bid = part.loc[entry, "close"], part.loc[exit, "close"]
        row = {"pair": pair, "split": which, "h": h, "monday": monday, "monday_number": monday_number,
               "entry_time": entry, "exit_time": exit, "entry_bid": entry_bid, "exit_bid": exit_bid}
        for side, direction in (("long", 1), ("short", -1)):
            entry_fill = entry_fill_price(pair, side, entry_bid, cfg)
            exit_fill = exit_fill_price(pair, side, exit_bid, cfg)
            price = direction * (exit_fill - entry_fill) / pip_size(pair)
            carry = carry_pips(pair, side, entry, exit, entry_fill, rates)
            row[f"{side}_price_pips"] = price
            row[f"{side}_carry_pips"] = carry
            row[f"{side}_net_pips"] = price + carry
        if all(np.isfinite(row[f"{side}_net_pips"]) for side in ("long", "short")):
            rows.append(row)
    columns = ["pair", "split", "h", "monday", "monday_number", "entry_time", "exit_time", "entry_bid", "exit_bid"]
    columns += [f"{side}_{metric}_pips" for side in ("long", "short") for metric in ("price", "carry", "net")]
    result = pd.DataFrame(rows, columns=columns)
    for column in ("monday", "entry_time", "exit_time"):
        result[column] = pd.to_datetime(result[column], utc=True)
    return result


def select_trades(opportunities, name, signal_table):
    result = opportunities.copy()
    if name == "always_long":
        direction = np.ones(len(result))
    elif name == "always_short":
        direction = -np.ones(len(result))
    else:
        direction = signal_table[name].reindex(pd.DatetimeIndex(result.monday)).to_numpy()
    keep = np.isfinite(direction) & (direction != 0)
    result = result.loc[keep].copy()
    direction = direction[keep]
    result["signal"] = name
    result["side"] = np.where(direction > 0, "long", "short")
    for metric in ("price", "carry", "net"):
        result[f"{metric}_pips"] = np.where(direction > 0, result[f"long_{metric}_pips"], result[f"short_{metric}_pips"])
    return result


def nonoverlapping(sample, h):
    """Anchor on every h-th calendar Monday, then exclude delayed-entry overlaps."""
    indices = []
    for _, pair_rows in sample.groupby("pair", sort=True):
        last_exit = None
        for row in pair_rows.sort_values("monday").itertuples():
            if row.monday_number % h == 0 and (last_exit is None or row.entry_time >= last_exit):
                indices.append(row.Index)
                last_exit = row.exit_time
    return sample.loc[indices]


def statistics(sample, h):
    independent = nonoverlapping(sample, h)["net_pips"].astype(float)
    n = len(independent)
    se = independent.std(ddof=1) / np.sqrt(n) if n >= 2 else np.nan
    return {
        "n_trades": len(sample), "mean_price_pips": sample.price_pips.mean(),
        "mean_carry_pips": sample.carry_pips.mean(), "mean_net_pips": sample.net_pips.mean(),
        "hit_rate": (sample.net_pips > 0).mean() if len(sample) else np.nan,
        "n_nonoverlap": n, "mean_net_nonoverlap_pips": independent.mean(),
        "t_stat": independent.mean() / se if se > 0 else np.nan,
    }


def matched_random_means(sample, seeds=benchmark_seeds):
    """Same pair/Monday/non-flat observations; only side changes, with p=1/2."""
    ordered = sample.sort_values(["pair", "monday"])
    long = ordered.long_net_pips.to_numpy(dtype=float)
    short = ordered.short_net_pips.to_numpy(dtype=float)
    values = []
    for seed in seeds:
        sides = np.random.default_rng(seed).random(len(ordered)) < 0.5
        values.append(np.where(sides, long, short).mean() if len(ordered) else np.nan)
    return pd.Series(values, index=pd.Index(seeds, name="seed"), name="mean_net_pips")


def year_statistics(trades):
    rows = []
    for name in signals:
        for h in horizons:
            sample = trades.loc[(trades.signal == name) & (trades.h == h)]
            row = {"signal": name, "h": h}
            for year in years:
                yearly = sample.loc[sample.entry_time.dt.year == year, "net_pips"]
                row[f"net_{year}"] = yearly.mean()
                row[f"n_{year}"] = len(yearly)
            row["positive_years"] = sum(row[f"net_{year}"] > 0 for year in years)
            rows.append(row)
    return pd.DataFrame(rows)


def apply_pass_rule(grid, year_table):
    result = grid.copy()
    for year in years:
        result[f"net_{year}"] = np.nan
    result["positive_years"] = np.nan
    result["pass_rule"] = False
    for row in year_table.itertuples(index=False):
        mask = (result.pair == "pooled") & (result.signal == row.signal) & (result.h == row.h)
        pooled = result.loc[mask].set_index("split")
        passed = (set(pooled.index) == {"train", "val"} and
                  pooled.loc["train", "random_rank"] >= 95 and
                  pooled.loc["val", "random_rank"] >= 90 and row.positive_years >= 4)
        result.loc[mask, "positive_years"] = row.positive_years
        for year in years:
            result.loc[mask, f"net_{year}"] = getattr(row, f"net_{year}")
        result.loc[mask, "pass_rule"] = bool(passed)
    return result


def build_reports(rates, hourly, cfg=None):
    cfg = Config() if cfg is None else cfg
    rates = rates.loc[rates.index < lock]
    rows, trade_frames, random_frames = [], [], []
    for h in horizons:
        for which in ("train", "val"):
            pooled = {name: [] for name in (*signals, "always_long", "always_short")}
            for pair in pairs:
                table = rate_signals(diff(pair, rates))
                opportunities = monday_opportunities(hourly[pair], pair, which, h, rates, cfg)
                for name in pooled:
                    sample = select_trades(opportunities, name, table)
                    pooled[name].append(sample)
                    trade_frames.append(sample)
                    rows.append({"pair": pair, "signal": name, "h": h, "split": which,
                                 "random_rank": np.nan, **statistics(sample, h)})
            for name, samples in pooled.items():
                sample = pd.concat(samples, ignore_index=True)
                stats = statistics(sample, h)
                row = {"pair": "pooled", "signal": name, "h": h, "split": which,
                       "random_rank": np.nan, **stats}
                if name in signals:
                    random = matched_random_means(sample, benchmark_seeds)
                    row["random_rank"] = percentile_rank(random, stats["mean_net_pips"])
                    row["random_seeds"] = len(random)
                    random_frames.append(random.reset_index().assign(signal=name, h=h, split=which))
                rows.append(row)
    trades = pd.concat(trade_frames, ignore_index=True)
    year_table = year_statistics(trades)
    grid = apply_pass_rule(pd.DataFrame(rows), year_table)
    return grid, year_table, trades, pd.concat(random_frames, ignore_index=True)


def print_tables(grid, year_table):
    print("Pooled rates signals: price already includes one spread; carry includes the 0.5 pp annual markup.")
    print(f"{'signal':<6} {'wk':>2} {'n_tr/va':>9} {'price_tr/va':>16} {'carry_tr/va':>16} {'net_tr/va':>16} "
          f"{'hit_tr/va':>13} {'t_tr/va':>13} {'rank_tr/va':>13} {'+years':>6} {'pass':>5}")
    for name in signals:
        for h in horizons:
            both = grid.loc[(grid.pair == "pooled") & (grid.signal == name) & (grid.h == h)].set_index("split")
            train, val = both.loc["train"], both.loc["val"]
            cells = [f"{int(train.n_trades)}/{int(val.n_trades)}"]
            cells += [f"{train[col]:.2f}/{val[col]:.2f}" for col in ("mean_price_pips", "mean_carry_pips", "mean_net_pips")]
            hit = f"{train.hit_rate:.1%}/{val.hit_rate:.1%}"
            t = f"{train.t_stat:.2f}/{val.t_stat:.2f}"
            rank = f"{train.random_rank:.1f}/{val.random_rank:.1f}"
            print(f"{name:<6} {h:>2} {cells[0]:>9} {cells[1]:>16} {cells[2]:>16} {cells[3]:>16} "
                  f"{hit:>13} {t:>13} {rank:>13} {int(train.positive_years):>4}/6 {str(bool(train.pass_rule)):>5}")
    print("Per-pair mean net pips, train/val (full n/price/carry/hit/t statistics are in test4_grid.csv):")
    print(f"{'signal':<6} {'wk':>2} {'eurusd':>19} {'gbpusd':>19} {'usdjpy':>19}")
    for name in signals:
        for h in horizons:
            cells = []
            for pair in pairs:
                both = grid.loc[(grid.pair == pair) & (grid.signal == name) & (grid.h == h)].set_index("split")
                cells.append(f"{both.loc['train', 'mean_net_pips']:.2f}/{both.loc['val', 'mean_net_pips']:.2f}")
            print(f"{name:<6} {h:>2} " + " ".join(f"{cell:>19}" for cell in cells))
    print("Unconditional baselines: price/carry/net pips; L/S = always long/short:")
    print(f"{'pair':<7} {'wk':>2} {'train_L':>25} {'train_S':>25} {'val_L':>25} {'val_S':>25}")
    for pair in pairs:
        for h in horizons:
            cells = []
            for which in ("train", "val"):
                for name in ("always_long", "always_short"):
                    row = grid.loc[(grid.pair == pair) & (grid.signal == name) & (grid.h == h) & (grid['split'] == which)].iloc[0]
                    cells.append(f"{row.mean_price_pips:.2f}/{row.mean_carry_pips:.2f}/{row.mean_net_pips:.2f}")
            print(f"{pair:<7} {h:>2} " + " ".join(f"{cell:>25}" for cell in cells))
    print("Year-by-year pooled mean net pips by entry year; 2025 is partial validation:")
    print(f"{'signal':<6} {'wk':>2} " + " ".join(f"{year:>10}" for year in years) + f" {'+years':>6}")
    for row in year_table.itertuples(index=False):
        values = " ".join(f"{getattr(row, f'net_{year}'):>10.2f}" for year in years)
        print(f"{row.signal:<6} {row.h:>2} {values} {row.positive_years:>4}/6")
    passed = grid.loc[grid.pass_rule, ["signal", "h"]].drop_duplicates()
    labels = [f"{row.signal}/{row.h}w" for row in passed.itertuples(index=False)]
    print("Pass rule: pooled rank >=95 train AND >=90 val AND >=4 positive years. Passing combinations: " +
          (", ".join(labels) if labels else "none") + ".")
    print(f"Total pre-registered combinations: {len(signals) * len(horizons)} (27 pair-level settings); baselines are descriptive.")


def main():
    rates = load_rates()
    hourly = {pair: load_pair(pair, "1h") for pair in pairs}
    grid, year_table, trades, random = build_reports(rates, hourly)
    results_dir.mkdir(parents=True, exist_ok=True)
    for name, table in (("test4_grid.csv", grid), ("test4_years.csv", year_table),
                        ("test4_trades.csv", trades), ("test4_random.csv", random)):
        table.to_csv(results_dir / name, index=False, float_format="%.12g")
    print_tables(grid, year_table)
    print(f"{len(benchmark_seeds)} matched seeds per combination/split; same Mondays and non-flat observations, random side only.")
    print("Calendar-day rates are lagged one day; locked rates and all 2026 observations are excluded.")
    print("Saved results/test4_{grid,years,trades,random}.csv. FX test remains locked.")


if __name__ == "__main__":
    main()

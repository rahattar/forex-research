"""Deterministic scorecards and append-only exploration/confirmation ledgers."""

import csv
from datetime import datetime, timezone
from pathlib import Path
import re

import numpy as np
import pandas as pd

from .lab_data import bounds
from .loader import fresh_pairs
from .signal_eval import block_bootstrap, weekly_blocks


ledger_columns = ("time", "name", "sha256", "trades", "mean_r", "d", "t_d", "coin", "random_r")


def weekly_t(values):
    sd = values.std(ddof=1)
    return float(values.mean() / (sd / np.sqrt(len(values)))) if len(values) >= 2 and sd > 0 else np.nan


def longest_losing_streak(trades):
    longest, current = 0, 0
    for value in trades.sort_values(["exit_time", "pair", "entry_time"]).r_strategy:
        current = current + 1 if value < 0 else 0
        longest = max(longest, current)
    return longest


def scorecard(idea, digest, period, trades, counts, baseline, audits):
    start, end = bounds(period)
    uncertainty = block_bootstrap(trades, start, end, resamples=5000, seed=0)
    blocks = weekly_blocks(trades, start, end)
    r = trades.r_strategy
    wins, losses = r.loc[r > 0], r.loc[r < 0]
    per_seed = [baseline[seed].r.mean() for seed in range(5)]
    win_seed = [(baseline[seed].r > 0).mean() if len(baseline[seed]) else np.nan for seed in range(5)]
    summary = dict(trades=len(trades), mean_r=r.mean(), d=trades.d.mean(), t_d=uncertainty["weekly_d_t_stat"],
                   coin=trades.coin.mean(), random_r=float(np.mean(per_seed)), random_win=float(np.mean(win_seed)))
    lines = ["Idea: " + idea["name"], "Description: " + idea["description"], "SHA256: " + digest,
             "Period: %s [%s, %s)" % (period, start.isoformat(), end.isoformat()),
             "Results from the lab are not claims.", "", "Counts by pair",
             "pair       signals   skipped    trades trades/year"]
    years = (end - start).total_seconds() / (365.25 * 86400)
    for pair in fresh_pairs:
        if pair not in counts:
            continue
        c = counts[pair]
        n, signals = c.get("trades", 0), c.get("signals", 0)
        lines.append("%-7s %10d %9d %9d %11.3f" % (pair, signals, signals - n, n, n / years))
    signals = sum(value.get("signals", 0) for value in counts.values())
    lines.append("%-7s %10d %9d %9d %11.3f" % ("all", signals, signals - len(trades), len(trades), len(trades) / years))
    reasons = sorted(set().union(*(set(c) - {"signals", "trades"} for c in counts.values())))
    pair_order = [pair for pair in fresh_pairs if pair in counts]
    lines += ["", "Skipped by reason and pair", "%-32s %s %9s" %
              ("reason", " ".join("%9s" % pair for pair in pair_order), "all")]
    for reason in reasons:
        lines.append("%-32s %s %9d" %
                     (reason, " ".join("%9d" % counts[pair].get(reason, 0) for pair in pair_order),
                      sum(c.get(reason, 0) for c in counts.values())))
    if not reasons:
        lines.append("none")
    payoff = wins.mean() / abs(losses.mean()) if len(wins) and len(losses) else np.nan
    lines += ["", "Strategy",
              "trades=%d mean_r=%.6f [%.6f, %.6f] weekly_t=%.4f" %
              (len(trades), r.mean(), uncertainty["r_ci_low"], uncertainty["r_ci_high"], weekly_t(blocks.r_strategy)),
              "win_rate=%.6f mean_win=%.6f mean_loss=%.6f payoff=%.6f skewness=%.6f" %
              ((r > 0).mean() if len(r) else np.nan, wins.mean(), losses.mean(), payoff, r.skew()),
              "median_hold_hours=%.4f longest_losing_streak=%d mean_cost_r=%.6f" %
              (trades.hold_hours.median(), longest_losing_streak(trades), trades.cost_r.mean()), "Exit reasons"]
    for reason in ("stop", "target", "time", "end_of_split"):
        sample = trades.loc[trades.exit_reason == reason]
        lines.append("%-14s n=%7d share=%.6f mean_r=%.6f" %
                     (reason, len(sample), len(sample) / len(trades) if len(trades) else np.nan, sample.r_strategy.mean()))
    lines += ["", "Mirror",
              "mean_r_mirror=%.6f d=%.6f [%.6f, %.6f] weekly_t_d=%.4f coin=%.6f" %
              (trades.r_mirror.mean(), summary["d"], uncertainty["d_ci_low"], uncertainty["d_ci_high"], summary["t_d"], summary["coin"]),
              "", "Random baseline (independent trades; no bias or trend_agree)",
              "seed requested achieved skipped_draws     mean_r   win_rate"]
    for seed in range(5):
        audit = audits[seed]
        lines.append("%4d %9d %8d %13d %10.6f %10.6f" %
                     (seed, audit.requested.sum(), audit.achieved.sum(), audit.skipped_draws.sum(), per_seed[seed], win_seed[seed]))
    lines.append("mean over seeds: mean_r=%.6f win_rate=%.6f" % (summary["random_r"], summary["random_win"]))
    lines += ["", "Breakdown by pair", "pair          n      mean_r           d"]
    for pair in fresh_pairs:
        if pair in counts:
            sample = trades.loc[trades.pair == pair]
            lines.append("%-7s %7d %11.6f %11.6f" % (pair, len(sample), sample.r_strategy.mean(), sample.d.mean()))
    lines += ["", "Breakdown by entry year", "year          n      mean_r           d"]
    for year in range(start.year, (end - pd.Timedelta(1, "ns")).year + 1):
        sample = trades.loc[trades.entry_time.dt.year == year]
        lines.append("%4d %10d %11.6f %11.6f" % (year, len(sample), sample.r_strategy.mean(), sample.d.mean()))
    lines += ["", "direction skill d = %.6f (t = %.4f); exits and costs alone give %.6f" %
              (summary["d"], summary["t_d"], summary["random_r"])]
    report = "\n".join(lines) + "\n"
    report.encode("ascii")
    return report, summary


def read_ledger(root):
    path = Path(root) / "ledger.csv"
    if not path.exists():
        return pd.DataFrame(columns=ledger_columns)
    frame = pd.read_csv(path, dtype={"time": str, "name": str, "sha256": str})
    if list(frame.columns) != list(ledger_columns):
        raise ValueError("Malformed lab ledger")
    return frame


def authorize_confirm(root, prereg, digest):
    if prereg is None or not Path(prereg).is_file():
        raise ValueError("Confirm requires an existing pre-registration")
    text = Path(prereg).read_text(encoding="ascii")
    if not re.search(r"\b" + re.escape(digest) + r"\b", text):
        raise ValueError("Pre-registration must contain the idea sha256")
    if digest in set(read_ledger(root).sha256) or (Path(root) / ".claims" / (digest + ".lock")).exists():
        raise ValueError("A second confirm of the same sha256 is refused")


def claim_confirm(root, digest):
    """Reserve the one permitted evaluation before any sealed bars are loaded."""
    folder = Path(root) / ".claims"
    folder.mkdir(parents=True, exist_ok=True)
    try:
        with (folder / (digest + ".lock")).open("x", encoding="ascii") as stream:
            stream.write(digest + "\n")
    except FileExistsError as error:
        raise ValueError("A second confirm of the same sha256 is refused") from error


def write_run(root, idea, digest, trades, report, summary):
    root = Path(root)
    folder = root / idea["name"]
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "report.txt").write_text(report, encoding="ascii")
    trades.to_csv(folder / "trades.csv", index=False, float_format="%.12g")
    path = root / "ledger.csv"
    exists = path.exists()
    with path.open("a", newline="", encoding="ascii") as stream:
        writer = csv.DictWriter(stream, fieldnames=ledger_columns)
        if not exists:
            writer.writeheader()
        row = dict(time=datetime.now(timezone.utc).isoformat(), name=idea["name"], sha256=digest)
        row.update({key: summary[key] for key in ledger_columns[3:]})
        writer.writerow(row)


def print_ledger(root):
    frame = read_ledger(root)
    if frame.empty:
        print("No lab runs.")
        return
    print(frame.to_string(index=False, float_format=lambda value: "%.6f" % value))

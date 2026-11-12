"""Run, list, or explicitly confirm one strictly validated trading idea."""

import argparse
from pathlib import Path
import sys
import time

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.lab_blocks import CompiledIdea, load_idea
from src.lab_data import load_data
from src.lab_score import authorize_confirm, claim_confirm, print_ledger, scorecard, write_run
from src.lab_sim import evaluate_pair, random_baseline, trade_columns
from src.loader import fresh_pairs


project = Path(__file__).resolve().parents[1]
results_dir = project / "results"


def execute(path, period="lab", prereg=None):
    started = time.monotonic()
    idea, digest = load_idea(path)
    root = results_dir / period
    if period == "confirm":
        authorize_confirm(root, prereg, digest)
        claim_confirm(root, digest)
    elif period != "lab":
        raise ValueError("Unknown period")
    compiled, counts, records = {}, {}, []
    for pair in fresh_pairs:
        data = load_data(pair, period)
        compiled[pair] = CompiledIdea(data, idea)
        trades, counts[pair] = evaluate_pair(compiled[pair])
        records.append(trades)
        print("%s: %s trades=%d elapsed=%.2fs" % (idea["name"], pair, len(trades), time.monotonic() - started), file=sys.stderr, flush=True)
    trades = pd.concat(records, ignore_index=True) if records else pd.DataFrame(columns=trade_columns)
    for column in ("entry_time", "exit_time", "mirror_entry_time", "mirror_exit_time"):
        trades[column] = pd.to_datetime(trades[column], utc=True)
    baseline, audits = {}, {}
    for seed in range(5):
        baseline[seed], audits[seed] = random_baseline(compiled, trades, seed)
        print("%s: baseline seed=%d achieved=%d elapsed=%.2fs" %
              (idea["name"], seed, len(baseline[seed]), time.monotonic() - started), file=sys.stderr, flush=True)
    report, summary = scorecard(idea, digest, period, trades, counts, baseline, audits)
    write_run(root, idea, digest, trades, report, summary)
    print(report, end="")
    print("Elapsed: %.2fs" % (time.monotonic() - started))
    return report, trades, summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list")
    for name in ("run", "confirm"):
        command = commands.add_parser(name)
        command.add_argument("idea")
        if name == "confirm":
            command.add_argument("--prereg", required=True)
    args = parser.parse_args(argv)
    if args.command == "list":
        print_ledger(results_dir / "lab")
    else:
        execute(args.idea, "lab" if args.command == "run" else "confirm", getattr(args, "prereg", None))


if __name__ == "__main__":
    main()

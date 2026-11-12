"""Run the pre-registered Task 15 evaluation once, without altering earlier work."""

import os

# Bound parallel resources without changing any registered estimator setting.
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/forex_predict_matplotlib")

from contextlib import redirect_stdout
import io
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import sklearn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import predict as model
from src.predict_data import load_inputs


project = Path(__file__).resolve().parents[1]
result_dir = project / "results"
leak_tests = ("test_no_lookahead_all_inputs", "test_external_observation_not_used_on_its_date",
             "test_walk_forward_purge_selection_and_causal_forecasts",
             "test_tree_walk_forward_future_changes_do_not_change_forecasts")


def sample_specs():
    first, middle, end = (pd.Timestamp(value, tz="UTC") for value in ("2014-01-01", "2020-01-01", "2025-10-31 23:59:59"))
    result = [("all", first, end, None), ("2014-2019", first, middle - pd.Timedelta(seconds=1), None),
              ("2020-2025", middle, end, None)]
    result += [(pair, first, end, i) for i, pair in enumerate(model.fresh_pairs)]
    result += [(str(year), pd.Timestamp("%d-01-01" % year, tz="UTC"),
                min(end, pd.Timestamp("%d-12-31 23:59:59" % year, tz="UTC")), None) for year in range(2014, 2026)]
    return result


def weeks_between(first, last):
    return pd.date_range(model.week_start(pd.DatetimeIndex([first]))[0],
                         model.week_start(pd.DatetimeIndex([last]))[0], freq="7D")


def summarize(trades, quality, variant, name, h, source, forecasts):
    rows, stats, weekly_output = [], {}, []
    for sample, first, last, pair in sample_specs():
        sample_trades = trades.loc[(trades.index >= first) & (trades.index <= last)]
        if pair is not None:
            sample_trades = sample_trades.loc[sample_trades.pair == pair]
        quality_mask = np.isfinite(forecasts) & (source.index >= first) & (source.index <= last)
        if pair is not None:
            quality_mask &= source.pair.to_numpy() == pair
        f, z = forecasts[quality_mask], source["z" + str(h)].to_numpy()[quality_mask]
        sample_quality = dict(scored_rows=int(quality_mask.sum()), pearson=model.correlation(f, np.clip(z, -5, 5)),
                              spearman=model.correlation(pd.Series(f).rank().to_numpy(), pd.Series(z).rank().to_numpy()),
                              mean_cost=float(sample_trades.cost.mean()),
                              break_even_correlation=float(sample_trades.cost.mean() / .8))
        for statistic, value in sample_quality.items():
            rows.append(dict(variant=variant, model=name, horizon=h, rule="ALL", sample=sample,
                             statistic=statistic, value=value, ci_low=np.nan, ci_high=np.nan))
        weeks = weeks_between(first, last)
        for rule in ("ALL", "SEL"):
            weekly = model.weekly_table(sample_trades, rule, weeks, pairs=range(10) if pair is None else [pair])
            stats[sample, rule] = model.inference(weekly)
            if rule == "ALL":
                s = stats[sample, rule]
                rows.append(dict(variant=variant, model=name, horizon=h, rule=rule, sample=sample,
                                 statistic="d", value=s["gross"], ci_low=s["gross_low"], ci_high=s["gross_high"]))
            for statistic, value in stats[sample, rule].items():
                if statistic.endswith(("_low", "_high")):
                    continue
                rows.append(dict(variant=variant, model=name, horizon=h, rule=rule, sample=sample,
                                 statistic=statistic, value=value,
                                 ci_low=stats[sample, rule].get(statistic + "_low", np.nan),
                                 ci_high=stats[sample, rule].get(statistic + "_high", np.nan)))
            if sample == "all":
                output = weekly.reset_index()
                output["variant"], output["model"], output["horizon"], output["rule"] = variant, name, h, rule
                output["pair"] = output.pair.map(dict(enumerate(model.fresh_pairs)))
                weekly_output.append(output)
    for rule in ("ALL", "SEL"):
        for suffix, sample in (("first", "2014-2019"), ("second", "2020-2025")):
            for metric in ("net", "gross"):
                stats["all", rule][metric + "_" + suffix] = stats[sample, rule][metric]
    return stats, rows, weekly_output


def interval(stats, metric):
    return "%.5f [%.5f,%.5f]" % (stats[metric], stats[metric + "_low"], stats[metric + "_high"])


def primary_tables(cells):
    lines = ["Primary forecast quality and verdicts", "model   h scored_rows  Pearson Spearman mean_cost breakeven verdict"]
    for name, h in sorted(cells):
        cell = cells[name, h]
        q = cell["quality"]
        lines.append("%-5s %3d %11d %8.5f %8.5f %9.5f %9.5f %s" %
                     (name, h, q["scored_rows"], q["pearson"], q["spearman"], q["mean_cost"], q["break_even_correlation"],
                      model.verdict(cell["stats"]["all", "ALL"], cell["stats"]["all", "SEL"])))
    for rule in ("ALL", "SEL"):
        lines += ["", "Rule " + rule,
                  "model   h       n mean_gross [95% CI]             t_gross mean_net [95% CI]               t_net net_14_19 net_20_25"]
        for name, h in sorted(cells):
            s = cells[name, h]["stats"]["all", rule]
            lines.append("%-5s %3d %7d %-33s %7.3f %-33s %7.3f %9.5f %9.5f" %
                         (name, h, s["n"], interval(s, "gross"), s["t_gross"], interval(s, "net"), s["t_net"],
                          s["net_first"], s["net_second"]))
    return lines


def small_table(cells, samples=("all",)):
    lines = ["model   h sample         n_all    gross_all      net_all t_net_all   n_sel      net_sel t_net_sel"]
    for name, h in sorted(cells):
        for sample in samples:
            a, b = (cells[name, h]["stats"][sample, rule] for rule in ("ALL", "SEL"))
            lines.append("%-5s %3d %-11s %8d %12.5f %12.5f %9.3f %7d %12.5f %9.3f" %
                         (name, h, sample, a["n"], a["gross"], a["net"], a["t_net"], b["n"], b["net"], b["t_net"]))
    return lines


def save_chart(series):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return "Chart skipped: matplotlib is missing."
    figure, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    for h, axis in zip(model.horizons, axes):
        for name, color in (("m1", "tab:blue"), ("m2", "tab:orange")):
            sample = series[name, h]
            axis.plot(sample.index, sample.gross.cumsum(), color=color, label=name + " gross")
            axis.plot(sample.index, sample.net.cumsum(), color=color, linestyle="--", label=name + " net")
        axis.set_ylabel("%dh cumulative units" % h)
        axis.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
        axis.grid(alpha=.25)
    figure.subplots_adjust(right=.80, hspace=.12)
    figure.savefig(result_dir / "predict_cumulative.png", dpi=150)
    plt.close(figure)
    return "Chart: results/predict_cumulative.png"


def main():
    started = time.monotonic()
    outputs = ("predict_report.txt", "predict_summary.csv", "predict_weekly.csv", "predict_forecasts_24h.csv", "predict_cumulative.png")
    if any((result_dir / name).exists() for name in outputs):
        raise RuntimeError("Task 15 outputs already exist; refusing to rerun or overwrite them")

    def progress(message):
        print("elapsed=%8.1fs %s" % (time.monotonic() - started, message), file=sys.stderr, flush=True)

    environment = "Environment: python=%s numpy=%s pandas=%s scikit-learn=%s" % (
        platform.python_version(), np.__version__, pd.__version__, sklearn.__version__)
    progress(environment)
    audit = io.StringIO()
    with redirect_stdout(audit):
        bars, external, _ = load_inputs()
    progress("data audit complete")
    table, columns = model.build_table(bars, external)
    del bars, external
    progress("input table built once: %d rows" % len(table))
    check = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/test_predict.py", "-k",
                            " or ".join(leak_tests)], cwd=project, text=True, capture_output=True)
    if check.returncode:
        raise RuntimeError("L1 FAIL; evaluation not started\n" + check.stdout + check.stderr)
    progress("L1 PASS")
    counts = []
    for h in model.horizons:
        mask = model.usable(table, columns["primary"], h)
        scored = mask & (table.index.year >= 2014)
        counts.append("%3dh eligible=%d scored=%d grid_decisions=%d" %
                      (h, mask.sum(), scored.sum(), (scored & table["grid" + str(h)].to_numpy()).sum()))

    selection_lines, summary_rows, weeklies, forecasts24, primary, report_only, placebo = [], [], [], [], {}, {}, {}
    final_models, chart_series = {}, {}

    def evaluate(variant, name, h, feature_names, source):
        progress("begin %s %s h=%d" % (variant, name, h))
        selection = model.select_once(source, feature_names, h, name,
                                      progress=lambda message: progress(variant + " " + message))
        for row in selection["records"]:
            choice = str(row["choice"]) if name == "m1" else "%d/%d" % row["choice"]
            selection_lines.append("%-7s %-3s %3d %-7s %-10s %9.6f %s" %
                                   (variant, name, h, row["pair"], choice, row["correlation"], row["cut"].isoformat()))
        forecasts, fitted, _ = model.walk_forward(source, feature_names, h, name, selection,
                                                  progress=lambda message: progress(variant + " " + message))
        trades, quality = model.score(source, forecasts, h)
        stats, rows, weekly = summarize(trades, quality, variant, name, h, source, forecasts)
        summary_rows.extend(rows)
        weeklies.extend(weekly)
        if variant == "primary" and name == "m1":
            final_models[h] = fitted
        if variant == "primary":
            chart_series[name, h] = trades.loc[trades.traded_all].groupby(level=0)[["gross", "net"]].sum()
        if variant == "primary" and h == 24:
            saved = trades.copy()
            saved["model"] = name
            saved["pair"] = saved.pair.map(dict(enumerate(model.fresh_pairs)))
            forecasts24.append(saved.reset_index(names="time"))
        progress("completed %s %s h=%d" % (variant, name, h))
        return {"quality": quality, "stats": stats}

    for h in model.horizons:
        for name in ("m1", "m2"):
            primary[name, h] = evaluate("primary", name, h, columns["primary"], table)
    eligible24 = model.usable(table, columns["primary"], 24)
    placebo_targets = np.full(len(table), np.nan)
    placebo_targets[eligible24] = table.z24.to_numpy()[eligible24] * model.placebo_signs(table.index[eligible24])
    changed = table.assign(z24=placebo_targets)
    for name in ("m1", "m2"):
        placebo[name, 24] = evaluate("placebo", name, 24, columns["primary"], changed)
    del changed
    gate_pass = all(abs(placebo[name, 24]["stats"]["all", "ALL"]["t_gross"]) < 3 for name in ("m1", "m2"))
    if gate_pass:
        for h in model.horizons:
            for name in ("m1", "m2"):
                report_only["r1", name, h] = evaluate("r1", name, h, columns["r1_" + name], table)
            report_only["r2", "m1", h] = evaluate("r2", "m1", h, columns["g1"], table)

    lines = [environment, "", audit.getvalue().rstrip(), "", *counts, "", "Selection (capacity is depth/iterations)",
             "variant model   h pair    choice      holdout_r cut_utc", *selection_lines, "", "Leak gates",
             "L1 PASS: " + ", ".join(leak_tests), "L2 placebo (ALL gross, absolute weekly t < 3)",
             "model   h    trades    d_gross   weekly_t result"]
    for name in ("m1", "m2"):
        s = placebo[name, 24]["stats"]["all", "ALL"]
        lines.append("%-5s %3d %9d %10.6f %10.4f %s" %
                     (name, 24, s["n"], s["gross"], s["t_gross"], "PASS" if abs(s["t_gross"]) < 3 else "FAIL"))
    lines += ["", *primary_tables(primary), ""]
    ranking = {"NO PREDICTABILITY": 0, "PREDICTABLE, BELOW COST": 1, "TRADABLE": 2}
    verdicts = {key: model.verdict(value["stats"]["all", "ALL"], value["stats"]["all", "SEL"]) for key, value in primary.items()}
    best_rank = max(ranking[value] for value in verdicts.values())
    best = ["%s/%dh" % key for key, value in verdicts.items() if ranking[value] == best_rank]
    overall = next(value for value in verdicts.values() if ranking[value] == best_rank)
    lines.append("Overall verdict: %s; best cell(s): %s" % (overall, ", ".join(best)))
    if not gate_pass:
        lines.append("EVALUATION VOID: L2 failed. Verdicts above are not valid; report-only fits not run.")
    for variant in ("r1", "r2"):
        lines += ["", "%s report-only" % variant.upper(),
                  *small_table({(name, h): cell for (v, name, h), cell in report_only.items() if v == variant})]
    lines += ["", "R3 report-only: primary per half, pair, and year",
              *small_table(primary, tuple(sample for sample, *_ in sample_specs() if sample != "all")), "",
              "R4 report-only: last-refit M1 standardized coefficients, mean absolute value over pairs",
              "  h feature                 mean_abs_beta"]
    for h in model.horizons:
        betas = []
        for fitted in final_models[h].values():
            beta = np.zeros(len(columns["primary"]))
            beta[fitted.keep] = fitted.beta
            betas.append(beta)
        values = np.mean(np.abs(betas), axis=0)
        for i in np.argsort(-values, kind="stable")[:10]:
            lines.append("%3d %-24s %.8f" % (h, columns["primary"][i], values[i]))
    lines += ["", "Fixed caveat: 1h and some 4h gross bid-only reversal may reflect spread changes, not mid-price skill.",
              "Static spreads may understate costs in thin hours.", "", "Deviations:",
              "Execution environment changed to ~/fxenv/bin/python by user instruction; no packages installed.",
              "FRED files supplied by user after sandbox download failed; original download command retained in data/readme.md.",
              "No evaluation rerun.", "", "Configuration count: 6 decision-bearing cells, 9 report-only fits, 2 placebo fits."]
    weekly = pd.concat(weeklies, ignore_index=True)
    lines += ["", save_chart(chart_series)]
    pd.DataFrame(summary_rows).to_csv(result_dir / "predict_summary.csv", index=False, float_format="%.10g")
    weekly[["week", "variant", "model", "horizon", "rule", "pair", "n", "gross_sum", "cost_sum", "net_sum"]].to_csv(
        result_dir / "predict_weekly.csv", index=False, float_format="%.10g")
    pd.concat(forecasts24, ignore_index=True)[["time", "pair", "model", "forecast", "target", "close", "s", "cost", "gross", "net", "traded_sel"]].to_csv(
        result_dir / "predict_forecasts_24h.csv", index=False, float_format="%.10g")
    report = "\n".join(lines) + "\n"
    report.encode("ascii")
    (result_dir / "predict_report.txt").write_text(report, encoding="ascii")
    print(report, end="")
    progress("evaluation finished")


if __name__ == "__main__":
    main()

"""Operating characteristics of the task 16 verdict rule on synthetic month-end returns.

Usage: python scripts/sim_month_end_power.py
One basket return per strong month, in units of the expected 4h move (sd 1, fat tails).
Verdicts as in prereg_month_end.md:
  ALIVE            : t(net, all) >= 2.0 and t(net, late half) >= 1.0
  EXISTED, DECAYED : not ALIVE and t(gross, early half) >= 2.0
  NOT FOUND        : otherwise
"""

import numpy as np


early, late, cost, sd, reps = 56, 54, 0.075, 1.0, 20000


def t_stat(values):
    return values.mean(axis=1) / (values.std(axis=1, ddof=1) / np.sqrt(values.shape[1]))


def main():
    rng = np.random.default_rng(16)
    print(f"{early} early and {late} late strong months, cost {cost}, sd {sd}, {reps} histories per row")
    print(f"{'true gross effect early / late':<32} {'ALIVE':>7} {'DECAYED':>8} {'NOT FOUND':>10}")
    for a, b in ((0, 0), (0.15, 0.15), (0.30, 0.30), (0.45, 0.45), (0.30, 0), (0.45, 0), (0.60, 0), (0.45, 0.15)):
        noise = rng.standard_t(5, size=(reps, early + late)) * np.sqrt(3 / 5) * sd
        gross = noise + np.r_[np.full(early, a), np.full(late, b)]
        net = gross - cost
        alive = (t_stat(net) >= 2.0) & (t_stat(net[:, early:]) >= 1.0)
        decayed = ~alive & (t_stat(gross[:, :early]) >= 2.0)
        print(f"{a:>10.2f} / {b:<19.2f} {alive.mean():>7.2f} {decayed.mean():>8.2f} {(~alive & ~decayed).mean():>10.2f}")


if __name__ == "__main__":
    main()

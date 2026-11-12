"""Can the registered models detect a planted signal of break-even size? Synthetic data only.

Usage: python scripts/sim_predict_power.py
Rows are time-ordered; features are iid normal; the target is noise plus a planted part.
Walk-forward: fit on the past, predict the next block, as in prereg_predictability.md.
Reported: out-of-sample correlation of forecast and target, direction skill
d = mean(sign(forecast) * target) and its t-statistic.
"""

import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor


rows, features, blocks = 480_000, 40, 6
alphas = (1e2, 1e3, 1e4, 1e5, 1e6)
capacities = ({"max_depth": 3, "max_iter": 100}, {"max_depth": 6, "max_iter": 300})


def planted(x, kind, rho):
    if kind == "none":
        return np.zeros(len(x))
    if kind == "linear":
        return rho * x[:, 0]
    if kind == "spread":      # the same total correlation spread over ten weak inputs
        return rho / np.sqrt(10) * x[:, :10].sum(axis=1)
    if kind == "interaction":  # invisible to a linear model
        return rho * np.sign(x[:, 1]) * x[:, 2]
    raise ValueError(kind)


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else 0.0


def ridge_fit(x, y, alpha):
    return np.linalg.solve(x.T @ x + alpha * np.eye(x.shape[1]), x.T @ y)


def ridge_forecast(x_train, y_train, x_test):
    cut = int(len(x_train) * 0.8)
    best = max(alphas, key=lambda a: corr(x_train[cut:] @ ridge_fit(x_train[:cut], y_train[:cut], a), y_train[cut:]))
    return x_test @ ridge_fit(x_train, y_train, best)


def tree_fit(x, y, capacity):
    model = HistGradientBoostingRegressor(learning_rate=0.05, min_samples_leaf=2000, early_stopping=False,
                                          random_state=0, **capacity)
    return model.fit(x, y)


def tree_forecast(x_train, y_train, x_test):
    cut = int(len(x_train) * 0.8)
    best = max(capacities, key=lambda c: corr(tree_fit(x_train[:cut], y_train[:cut], c).predict(x_train[cut:]), y_train[cut:]))
    return tree_fit(x_train, y_train, best).predict(x_test)


def run(kind, rho, rng):
    x = rng.standard_normal((rows, features))
    noise = rng.standard_t(5, size=rows) * np.sqrt(3 / 5)
    y = planted(x, kind, rho) + noise
    size = rows // blocks
    out = {}
    for name, forecast in (("ridge", ridge_forecast), ("trees", tree_forecast)):
        f, z = [], []
        for k in range(2, blocks):          # first two blocks are the initial training window
            train, test = slice(0, k * size), slice(k * size, (k + 1) * size)
            f.append(forecast(x[train], np.clip(y[train], -5, 5), x[test]))
            z.append(y[test])
        f, z = np.concatenate(f), np.concatenate(z)
        gain = np.sign(f) * z
        out[name] = (corr(f, z), gain.mean(), gain.mean() / (gain.std(ddof=1) / np.sqrt(len(gain))))
    return out


def main():
    rng = np.random.default_rng(15)
    print(f"{rows} rows, {features} features, {blocks} blocks; out-of-sample rows {rows // blocks * (blocks - 2)}")
    print(f"{'planted signal':<22} {'model':<6} {'oos corr':>9} {'d':>8} {'t':>7}")
    for kind, rho in (("none", 0.0), ("linear", 0.03), ("linear", 0.06), ("spread", 0.06),
                      ("interaction", 0.06), ("interaction", 0.12)):
        start = time.time()
        for name, (c, d, t) in run(kind, rho, rng).items():
            print(f"{kind + ' ' + format(rho, '.2f'):<22} {name:<6} {c:>+9.4f} {d:>+8.4f} {t:>+7.2f}")
        print(f"   ({time.time() - start:.0f}s)", flush=True)


if __name__ == "__main__":
    main()

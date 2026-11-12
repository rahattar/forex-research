"""Fixed ETF risk/trend rules, next-close execution, and paired month inference."""

import numpy as np
import pandas as pd

from .daily_carry import drawdown_dates
from .daily_momentum import return_statistics
from .multi_asset_data import tickers, validate_dates


strategies = ("S1", "T1", "T2", "S2", "S3", "cash")
split_date = pd.Timestamp("2017-01-01")


def price_panel(frames):
    if set(frames) != set(tickers):
        raise ValueError("All fourteen registered ETFs are required")
    calendar = frames["spy"].index
    for ticker, frame in frames.items():
        validate_dates(frame)
        expected = calendar[calendar >= frame.index[0]]
        if not frame.index.equals(expected):
            raise ValueError(f"Missing or extra trading dates after inception for {ticker}")
    return pd.concat({ticker: frames[ticker]["Adj Close"] for ticker in tickers}, axis=1).reindex(calendar)


def month_ends(index):
    """Month ends come from the supplied SPY trading calendar, never prices."""
    keys = index.year * 12 + index.month
    return np.r_[keys[:-1] != keys[1:], True]


def normalize_weights(weights):
    """All strategies obey a gross notional cap of one."""
    return weights.div(weights.abs().sum(axis=1).clip(lower=1), axis=0)


def features(prices, cash):
    validate_dates(prices)
    validate_dates(cash)
    if not prices.index.equals(cash.index) or cash.isna().any():
        raise ValueError("A usable cash return is required on every trading date")
    returns = prices.pct_change(fill_method=None)
    vol = returns.ewm(span=60, adjust=False, min_periods=60).std() * np.sqrt(252)
    excess = {}
    for horizon in (21, 63, 252):
        cash_growth = np.expm1(np.log1p(cash).rolling(horizon, min_periods=horizon).sum())
        excess[horizon] = (prices / prices.shift(horizon) - 1).sub(cash_growth, axis=0)
    eligible = prices.notna().cumsum().ge(253) & prices.shift(252).notna() & prices.notna()
    if (eligible & (~np.isfinite(vol) | vol.le(0))).any().any():
        raise ValueError("An eligible ETF has undefined or zero volatility")
    long_score = sum(value.gt(0).astype(float) for value in excess.values()) / 3
    signed_score = sum(np.sign(value) for value in excess.values()) / 3
    return {"returns": returns, "vol": vol, "eligible": eligible,
            "long_score": long_score, "signed_score": signed_score}


def target_weights(prices, cash):
    info = features(prices, cash)
    count = info["eligible"].sum(axis=1)
    schedule = pd.Series(month_ends(prices.index), index=prices.index) & count.ge(8)
    positions = np.flatnonzero(schedule.to_numpy())
    executable = positions[positions + 1 < len(prices)]
    if not len(executable):
        raise ValueError("No executable month-end signal with at least eight eligible ETFs")
    risk = (.10 / info["vol"]).div(count.replace(0, np.nan), axis=0).where(info["eligible"], 0)
    zero = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    candidates = {"S1": risk, "T1": risk * info["long_score"], "T2": risk * info["signed_score"],
                  "S2": zero.copy(), "S3": zero.copy(), "cash": zero.copy()}
    candidates["S2"]["spy"], candidates["S2"]["ief"] = .6, .4
    candidates["S3"]["spy"] = 1.
    first = prices.index[executable[0] + 1]
    weights = {}
    for name, candidate in candidates.items():
        targets = normalize_weights(candidate).where(schedule, axis=0)
        # Signal at t, execute at t+1 close, earn asset returns from t+2.
        weights[name] = targets.shift(1).ffill().fillna(0).loc[first:]
    executions = schedule.shift(1, fill_value=False).loc[first:]
    return weights, executions, info, prices.index[executable[0]]


def account_returns(prices, cash, weights, *, short_fee=0.0):
    validate_dates(weights)
    returns = prices.pct_change(fill_method=None).reindex(weights.index)
    held = weights.shift(1).fillna(0)
    if (returns.isna() & held.ne(0)).any().any():
        raise ValueError("A held ETF has a missing return")
    returns = returns.fillna(0)
    cash = cash.reindex(weights.index)
    changes = weights.diff()
    changes.iloc[0] = weights.iloc[0]
    trading = changes.abs() * .0005
    borrowing = held.clip(upper=0).abs() * short_fee / 252
    asset_total = held * returns - trading - borrowing
    asset_excess = held.mul(returns.sub(cash, axis=0)) - trading - borrowing
    cash_weight = 1 - held.sum(axis=1)
    result = pd.DataFrame({"total": cash + asset_excess.sum(axis=1), "excess": asset_excess.sum(axis=1),
                           "cash_return": cash, "cash_weight": cash_weight,
                           "gross_weight": held.abs().sum(axis=1), "turnover": changes.abs().sum(axis=1),
                           "trading_cost": trading.sum(axis=1), "short_cost": borrowing.sum(axis=1)}, index=weights.index)
    result["cost"] = result.trading_cost + result.short_cost
    result["cash_contribution"] = cash_weight * cash
    for ticker in prices:
        result[f"weight_{ticker}"] = weights[ticker]
        result[f"held_{ticker}"] = held[ticker]
        result[f"total_{ticker}"] = asset_total[ticker]
        result[f"excess_{ticker}"] = asset_excess[ticker]
    return result


def evaluate(prices, cash):
    weights, executions, info, first_signal = target_weights(prices, cash)
    frames = {name: account_returns(prices, cash, weights[name], short_fee=.005 if name == "T2" else 0)
              for name in strategies}
    for frame in frames.values():
        frame["rebalance"] = executions
    return frames, weights, executions, info, first_signal


def month_blocks(total, excess):
    """Daily moments and log-equity path summaries for exact month composition."""
    keys = total.index.year * 12 + total.index.month
    arrays = {name: [] for name in ("n", "sum", "squares", "log_total", "minimum", "maximum", "drawdown")}
    for key in pd.unique(keys):
        mask = keys == key
        values = excess.loc[mask].to_numpy(dtype=float)
        path = np.vstack([np.zeros((1, len(total.columns))), np.log1p(total.loc[mask].to_numpy()).cumsum(axis=0)])
        arrays["n"].append(len(values))
        arrays["sum"].append(values.sum(axis=0))
        arrays["squares"].append((values ** 2).sum(axis=0))
        arrays["log_total"].append(path[-1])
        arrays["minimum"].append(path.min(axis=0))
        arrays["maximum"].append(path.max(axis=0))
        arrays["drawdown"].append((np.maximum.accumulate(path, axis=0) - path).max(axis=0))
    return {name: np.asarray(values) for name, values in arrays.items()}


def paired_bootstrap(total, excess, resamples=5000, seed=0):
    """Identical sampled months for all books, including paired drawdown paths."""
    if total.empty or not total.index.equals(excess.index) or not total.columns.equals(excess.columns) or resamples < 1:
        raise ValueError("Bootstrap inputs must be aligned and nonempty")
    blocks = month_blocks(total, excess)
    months, n_strategies = len(blocks["n"]), len(total.columns)
    rng = np.random.default_rng(seed)
    sharpe_draws, drawdowns = [], []
    for offset in range(0, resamples, 250):
        size = min(250, resamples - offset)
        draws = rng.integers(0, months, size=(size, months))
        n = blocks["n"][draws].sum(axis=1)[:, None]
        summed = blocks["sum"][draws].sum(axis=1)
        variance = np.maximum((blocks["squares"][draws].sum(axis=1) - summed ** 2 / n) / (n - 1), 0)
        std = np.sqrt(variance)
        sharpes = np.divide(summed / n * np.sqrt(252), std, out=np.full_like(std, np.nan), where=std > 0)
        current = np.zeros((size, n_strategies))
        peak, worst = current.copy(), current.copy()
        for month in draws.T:
            # Compose log-equity paths without materializing thousands of days.
            worst = np.maximum(worst, np.maximum(blocks["drawdown"][month], peak - current - blocks["minimum"][month]))
            peak = np.maximum(peak, current + blocks["maximum"][month])
            current += blocks["log_total"][month]
        sharpe_draws.append(sharpes)
        drawdowns.append(-np.expm1(-worst))
    return np.concatenate(sharpe_draws), np.concatenate(drawdowns)


def interval(values):
    valid = np.asarray(values)[np.isfinite(values)]
    return np.percentile(valid, [2.5, 97.5]) if len(valid) else (np.nan, np.nan)


def portfolio_statistics(frame):
    total = return_statistics(frame.total)
    excess = return_statistics(frame.excess)
    years = (1 + frame.total).groupby(frame.index.year).prod() - 1
    return {"n": len(frame), "first": frame.index[0].date().isoformat(), "last": frame.index[-1].date().isoformat(),
            "annual_total_return": total["annual_return"], "annual_excess_return": excess["annual_return"],
            "annual_arithmetic_excess": frame.excess.mean() * 252, "total_vol": total["annual_vol"],
            "excess_vol": excess["annual_vol"], "excess_sharpe": excess["sharpe"],
            "max_drawdown": total["max_drawdown"], **drawdown_dates(frame.total),
            "worst_year": int(years.idxmin()), "worst_year_return": years.min(),
            "average_cash_weight": frame.cash_weight.mean(), "annual_turnover": frame.turnover.mean() * 252,
            "annual_trading_cost": frame.trading_cost.mean() * 252, "annual_short_cost": frame.short_cost.mean() * 252,
            "annual_cost": frame.cost.mean() * 252}


def summarize(frames, weights, executions, resamples=5000, seed=0):
    rows = []
    calendar = frames["S1"].index
    for period, mask in (("all", np.ones(len(calendar), dtype=bool)),
                         ("before_2017", calendar < split_date), ("from_2017", calendar >= split_date)):
        total = pd.DataFrame({name: frames[name].total.loc[mask] for name in strategies})
        excess = pd.DataFrame({name: frames[name].excess.loc[mask] for name in strategies})
        sharpes, drawdowns = paired_bootstrap(total, excess, resamples, seed)
        statistics = {}
        for column, name in enumerate(strategies):
            row = {"strategy": name, "group": "portfolio", "period": period, "value": "all",
                   **portfolio_statistics(frames[name].loc[mask])}
            row["sharpe_ci_low"], row["sharpe_ci_high"] = interval(sharpes[:, column])
            row.update(bootstrap_resamples=resamples, bootstrap_seed=seed, n_months=len(pd.unique(total.index.to_period("M"))))
            rows.append(row)
            statistics[name] = row
        t1, s1 = strategies.index("T1"), strategies.index("S1")
        difference = {"strategy": "T1-S1", "group": "comparison", "period": period, "value": "all",
                      "sharpe_difference": statistics["T1"]["excess_sharpe"] - statistics["S1"]["excess_sharpe"],
                      "t1_drawdown": statistics["T1"]["max_drawdown"], "s1_drawdown": statistics["S1"]["max_drawdown"],
                      "drawdown_difference": statistics["T1"]["max_drawdown"] - statistics["S1"]["max_drawdown"],
                      "bootstrap_resamples": resamples, "bootstrap_seed": seed}
        difference["sharpe_difference_ci_low"], difference["sharpe_difference_ci_high"] = interval(sharpes[:, t1] - sharpes[:, s1])
        difference["drawdown_difference_ci_low"], difference["drawdown_difference_ci_high"] = interval(drawdowns[:, t1] - drawdowns[:, s1])
        difference["relative_drawdown_reduction"] = 1 - difference["t1_drawdown"] / difference["s1_drawdown"] if difference["s1_drawdown"] > 0 else np.nan
        rows.append(difference)
        for name in ("S1", "T1"):
            frame = frames[name].loc[mask]
            dates = frame.index.intersection(executions.index[executions])
            for ticker in tickers:
                rows.append({"strategy": name, "group": "asset", "period": period, "value": ticker,
                             "annual_total_contribution": frame[f"total_{ticker}"].mean() * 252,
                             "annual_excess_contribution": frame[f"excess_{ticker}"].mean() * 252,
                             "held_share_months": weights[name].loc[dates, ticker].gt(0).mean(), "n_months": len(dates)})
            rows.append({"strategy": name, "group": "asset", "period": period, "value": "cash",
                         "annual_total_contribution": frame.cash_contribution.mean() * 252, "annual_excess_contribution": 0.})
    for name, frame in frames.items():
        for year, values in frame.total.groupby(frame.index.year):
            rows.append({"strategy": name, "group": "year", "period": "all", "value": str(year),
                         "n": len(values), "total_return": (1 + values).prod() - 1})
        overall = next(row for row in rows if row["strategy"] == name and row["group"] == "portfolio" and row["period"] == "all")
        for balance in (2000, 100000):
            rows.append({"strategy": name, "group": "annual_usd", "period": "all", "value": "all", "balance": balance,
                         "annual_usd": balance * overall["annual_excess_return"]})
            if name in ("S1", "T1"):
                for ticker in tickers:
                    rows.append({"strategy": name, "group": "position", "period": "all", "value": ticker, "balance": balance,
                                 "median_position_usd": weights[name].loc[executions, ticker].median() * balance})
    return pd.DataFrame(rows)


def trend_verdict(summary):
    row = summary.loc[(summary['group'] == "comparison") & (summary.period == "all")].iloc[0]
    if row.sharpe_difference_ci_low > 0:
        return "ADDS VALUE"
    if row.s1_drawdown > 0 and row.t1_drawdown <= .75 * row.s1_drawdown and row.sharpe_difference >= -.10:
        return "REDUCES RISK ONLY"
    return "NO BENEFIT"

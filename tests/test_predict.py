"""Synthetic-only checks for the frozen fitted-predictability experiment."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts import run_predict
from src import multi_asset_data
from src import predict as p
from src import predict_data as data
from src.costs import swap_pips
from src.instruments import Config


def bars(n=1400, seed=1):
    index = pd.date_range("2011-01-03", periods=n, freq="h", tz="UTC")
    rng = np.random.default_rng(seed)
    close = np.exp(rng.normal(0, .001, n).cumsum())
    return pd.DataFrame({"open": close * .9999, "high": close * 1.001,
                         "low": close * .999, "close": close}, index=index)


def externals():
    index = pd.bdate_range("2009-01-01", "2011-12-30")
    rng = np.random.default_rng(14)
    return {name: pd.Series(np.exp(rng.normal(0, .01, len(index)).cumsum()), index=index)
            for name in (*data.etfs, *data.freds)}


def test_no_lookahead_all_inputs():
    source = {pair: bars(seed=i + 1) for i, pair in enumerate(p.fresh_pairs)}
    other = externals()
    cutoff = source["eurusd"].index[950]
    before = {pair: frame.loc[:cutoff] for pair, frame in source.items()}
    other_before = {name: series.loc[:cutoff.tz_localize(None).normalize()] for name, series in other.items()}
    dollar, truncated_dollar = p.inputs_g2(source), p.inputs_g2(before)
    for pair in source:
        pd.testing.assert_frame_equal(p.inputs_g1(source[pair]).loc[:cutoff], p.inputs_g1(before[pair]))
        pd.testing.assert_frame_equal(dollar[pair].loc[:cutoff], truncated_dollar[pair])
    _, full = data.external_inputs(other, source["eurusd"].index)
    _, truncated = data.external_inputs(other_before, before["eurusd"].index)
    pd.testing.assert_frame_equal(full.loc[:cutoff], truncated)
    table, names = p.build_table(source, other)
    shortened, _ = p.build_table(before, other_before)
    pd.testing.assert_frame_equal(table.loc[:cutoff, names["primary"]], shortened[names["primary"]])
    assert len(names["primary"]) == 42 and len(names["g1"]) == 11
    assert p.inputs_g1(source["eurusd"]).iloc[:480].isna().all().all()


def test_external_observation_not_used_on_its_date():
    source = externals()
    date = source["spy"].index[350]
    times = pd.DatetimeIndex([date.tz_localize("UTC"), (date + pd.Timedelta(days=1)).tz_localize("UTC")])
    observations, mapped = data.external_inputs(source, times)
    assert mapped["spy_1"].iloc[0] == observations["spy"].loc[:date].iloc[-2, 0]
    assert mapped["spy_1"].iloc[1] == observations["spy"].loc[date, "spy_1"]
    changed = {name: series.copy() for name, series in source.items()}
    changed["spy"].loc[date:] *= 2
    _, alternate = data.external_inputs(changed, times)
    assert mapped["spy_1"].iloc[0] == alternate["spy_1"].iloc[0]


def test_dollar_factor_union_and_signs():
    source = {"eurusd": bars(), "usdjpy": bars(seed=2).drop(bars().index[800])}
    components = pd.concat([-np.log(source["eurusd"].close).diff() / p.hourly_scale(source["eurusd"]),
                            np.log(source["usdjpy"].close).diff() / p.hourly_scale(source["usdjpy"])], axis=1)
    expected = components.mean(axis=1).rolling(4).sum() / 2
    result = p.inputs_g2(source)
    np.testing.assert_allclose(result["eurusd"].usd_4, expected, equal_nan=True)
    np.testing.assert_allclose(result["eurusd"].signed_usd_4, -expected, equal_nan=True)


@pytest.mark.parametrize("h,hours", [(1, tuple(range(20))), (4, (0, 4, 8, 12, 16)), (24, (8,))])
def test_targets_exact_endpoint_guard_and_grids(h, hours):
    frame = bars()
    result = p.targets(frame, h)
    grid = result.loc[result.grid]
    assert set(grid.index.hour) == set(hours)
    assert (grid.index.hour <= 20).all()
    assert ((grid.index + pd.Timedelta(hours=h)).hour <= 20).all()
    assert not result.loc[result.index.hour == 21, "eligible"].any()
    t = frame.index[600]
    t = t.normalize() + pd.Timedelta(hours=8)
    absent = p.targets(frame.drop(t + pd.Timedelta(hours=h)), h)
    assert not absent.loc[t, "eligible"]
    assert np.isnan(absent.loc[t, "target"])
    if h == 1:
        t = t.normalize() + pd.Timedelta(hours=19)
        assert result.loc[t, "eligible"]
        assert not result.loc[t + pd.Timedelta(hours=1), "eligible"]


def model_table(n=1200, seed=4):
    rng = np.random.default_rng(seed)
    times = pd.date_range("2010-01-01", "2015-12-30", periods=n, tz="UTC").floor("h")
    x = rng.normal(size=n)
    y = .5 * x + rng.normal(size=n)
    return pd.DataFrame({"a": x, "constant": 2., "pair": 0, "z1": y,
                         "eligible1": True, "grid1": True, "cost1": .1, "s": .01, "close": 1.}, index=times)


def test_walk_forward_purge_selection_and_causal_forecasts():
    table = model_table()
    names = ["a", "constant"]
    selected = p.select_once(table, names, 1, "m1")
    initial = table.index[(table.index + pd.Timedelta(hours=1)) <= pd.Timestamp("2014-01-01", tz="UTC")]
    assert selected["cut"] == initial[int(np.ceil(.8 * len(initial))) - 1]
    assert all(row["last_training_endpoint"] <= row["cut"] for row in selected["records"])
    forecasts, _, audit = p.walk_forward(table, names, 1, "m1", selected, years=(2014, 2015))
    assert all(row["last_training_endpoint"] <= row["boundary"] for row in audit)
    assert all(row["first_scored_time"].year == row["last_scored_time"].year == row["year"] for row in audit)
    t = table.index[table.index.year == 2014][40]
    changed = table.loc[:t].copy()
    alternate = p.select_once(changed, names, 1, "m1")
    assert alternate == selected
    truncated, _, _ = p.walk_forward(changed, names, 1, "m1", alternate, years=(2014,))
    np.testing.assert_allclose(forecasts[:len(changed)], truncated, equal_nan=True)
    altered = table.copy()
    altered.loc[altered.index > t, ["a", "z1"]] *= 100
    modified, _, _ = p.walk_forward(altered, names, 1, "m1", selected, years=(2014,))
    np.testing.assert_allclose(forecasts[:len(changed)], modified[:len(changed)], equal_nan=True)


def test_ridge_hand_solution_training_scaling_and_intercept():
    x = np.array([[1., .1], [2., .1], [3., .1]])
    y = np.array([1., 2., 3.])
    fitted = p.fit_ridge(x, y, 2.)
    # Sample SD is 1; centered X'X=2, X'y=2, so beta=2/(2+2)=0.5.
    np.testing.assert_array_equal(fitted.keep, [True, False])
    np.testing.assert_allclose(fitted.scale[0], 1.)
    np.testing.assert_allclose(fitted.beta, [.5])
    np.testing.assert_allclose(p.forecast_ridge(fitted, [[4., 999.]]), [3.])
    assert fitted.intercept == 2.
    other = p.fit_ridge(x, y + 1, 2.)
    np.testing.assert_allclose(other.beta, fitted.beta)
    assert other.intercept == 3.
    constant = p.fit_ridge(np.ones((3, 2)), y, 1e6)
    np.testing.assert_allclose(p.forecast_ridge(constant, np.zeros((2, 2))), [2., 2.])
    clipped = p.fit_ridge(np.ones((3, 2)), [-100, 0, 20], 1e2)
    assert clipped.intercept == 0


def test_selection_undefined_ties_strongest_and_no_later_targets():
    table = model_table()
    table["z1"] = 1.
    selected = p.select_once(table, ["a"], 1, "m1")
    assert selected["choices"][0] == 1e6
    assert np.isnan(selected["records"][0]["correlation"])
    changed = table.copy()
    changed.loc[changed.index.year >= 2014, "z1"] = 1000
    again = p.select_once(changed, ["a"], 1, "m1")
    assert again["choices"] == selected["choices"]


def test_trees_deterministic_and_pair_categorical():
    rng = np.random.default_rng(23)
    x = rng.normal(size=(5000, 2))
    pair = np.arange(len(x)) % 10
    y = rng.normal(size=len(x))
    first, second = (p.fit_trees(x, y, pair, (3, 100)) for _ in range(2))
    np.testing.assert_array_equal(first.is_categorical_, [False, False, True])
    np.testing.assert_array_equal(p.forecast_trees(first, x[:50], pair[:50]), p.forecast_trees(second, x[:50], pair[:50]))
    assert first.min_samples_leaf == 2000 and first.early_stopping is False


def test_tree_walk_forward_future_changes_do_not_change_forecasts():
    table = model_table(6000)
    selection = {"choices": {None: (3, 100)}}
    forecast, _, audit = p.walk_forward(table, ["a"], 1, "m2", selection, years=(2014,))
    t = table.index[table.index.year == 2014][50]
    truncated = table.loc[:t].copy()
    changed, _, _ = p.walk_forward(truncated, ["a"], 1, "m2", selection, years=(2014,))
    np.testing.assert_allclose(forecast[:len(truncated)], changed, equal_nan=True)
    assert audit[0]["last_training_endpoint"] <= audit[0]["boundary"]


def test_selection_fewer_than_two_holdout_rows_is_undefined():
    table = model_table(10)
    # Initial window has seven rows, of which six are at or before the 80% cut.
    table = table.loc[table.index.year < 2014]
    selected = p.select_once(table, ["a"], 1, "m1")
    assert selected["records"][0]["holdout_rows"] == 1
    assert selected["choices"][0] == 1e6


def test_spearman_is_unclipped_and_pearson_clipped():
    table = model_table(6)
    table["z1"] = [-20., -10., -2., 2., 10., 20.]
    forecasts = np.arange(6, dtype=float)
    _, quality = p.score(table, forecasts, 1)
    assert quality["spearman"] == pytest.approx(1.)
    assert quality["pearson"] == pytest.approx(p.correlation(forecasts, np.clip(table.z1, -5, 5)))


@pytest.mark.parametrize("time,h,swap", [("2015-01-06 08:00", 4, 0), ("2015-01-06 20:00", 4, -.2),
                                         ("2015-01-07 08:00", 24, -.6)])
def test_cost_expected_move_hand_examples(time, h, swap):
    times = pd.DatetimeIndex([time], tz="UTC")
    # e=.001*sqrt(h)*1/.0001=10*sqrt(h) pips; spread=1 pip.
    actual = p.expected_cost(times, h, [1.], [.001], "eurusd")[0]
    assert actual == pytest.approx((1 - swap) / (10 * np.sqrt(h)))
    assert swap_pips(times[0], times[0] + pd.Timedelta(hours=h), Config()) == pytest.approx(swap)


def test_vectorized_swap_equals_existing_costs_all_days_and_horizons():
    times = pd.date_range("2015-01-05", periods=7 * 24, freq="h", tz="UTC")
    for h in p.horizons:
        actual = p.expected_cost(times, h, np.ones(len(times)), np.full(len(times), .0001 / np.sqrt(h)), "eurusd")
        expected = [1 - swap_pips(t, t + pd.Timedelta(hours=h), Config()) for t in times]
        np.testing.assert_allclose(actual, expected)


def test_rules_all_and_sel_and_unclipped_gross():
    table = model_table(10)
    table["z1"] = 10.
    forecasts = np.array([0, .05, -.05, .1, -.1, 1, -1, 0, .099, -.101])
    trades, quality = p.score(table, forecasts, 1)
    np.testing.assert_array_equal(trades.traded_all, forecasts != 0)
    np.testing.assert_array_equal(trades.traded_sel, np.abs(forecasts) >= .1)
    np.testing.assert_array_equal(trades.gross, np.sign(forecasts) * 10)
    assert np.isnan(quality["pearson"])


def test_weekly_zero_weeks_t_and_ratio_bootstrap():
    times = pd.DatetimeIndex(["2014-01-07 08:00", "2014-01-21 08:00"], tz="UTC")
    trades = pd.DataFrame({"pair": [0, 0], "gross": [2., -1.], "cost": [.1, .1], "net": [1.9, -1.1],
                           "traded_all": True, "traded_sel": False}, index=times)
    weeks = pd.date_range("2014-01-06", periods=4, freq="7D", tz="UTC")
    weekly = p.weekly_table(trades, "ALL", weeks, pairs=[0])
    np.testing.assert_array_equal(weekly.n, [1, 0, 1, 0])
    expected_t = np.mean([2., 0, -1., 0]) / (np.std([2., 0, -1., 0], ddof=1) / 2)
    assert p.weekly_t(weekly.gross_sum) == pytest.approx(expected_t)
    result = p.inference(weekly, seed=12, resamples=200)
    assert result == p.inference(weekly, seed=12, resamples=200)
    indices = p.bootstrap_indices(4, 12, 200)
    counts = weekly.n.to_numpy()[indices].sum(axis=1)
    gross = weekly.gross_sum.to_numpy()[indices].sum(axis=1)
    interval = np.quantile(gross[counts > 0] / counts[counts > 0], [.025, .975])
    np.testing.assert_allclose([result["gross_low"], result["gross_high"]], interval)
    assert result["gross"] == .5
    assert np.isnan(p.weekly_t([1., 1.])) and np.isnan(p.weekly_t([1.]))
    empty = p.inference(p.weekly_table(trades, "SEL", weeks, pairs=[0]), resamples=20)
    assert empty["n"] == 0 and np.isnan(empty["net_low"]) and np.isnan(empty["t_net"])


def rule_stats(**kwargs):
    result = dict(n=500, net=.01, t_net=3., net_first=.01, net_second=.01,
                  gross=.02, t_gross=3., gross_first=.01, gross_second=.01)
    result.update(kwargs)
    return result


@pytest.mark.parametrize("field,value", [("net", 0), ("t_net", 2.999), ("n", 499),
                                        ("net_first", 0), ("net_second", 0), ("t_net", np.nan)])
def test_verdict_each_tradable_failure(field, value):
    failed = rule_stats(**{field: value})
    assert p.verdict(failed, failed) == "PREDICTABLE, BELOW COST"


@pytest.mark.parametrize("field,value", [("gross", 0), ("t_gross", 2.999), ("gross_first", 0),
                                        ("gross_second", 0), ("t_gross", np.nan)])
def test_verdict_each_predictable_failure(field, value):
    failed = rule_stats(net=-1, **{field: value})
    assert p.verdict(failed, failed) == "NO PREDICTABILITY"


def test_verdict_all_or_sel_can_pass():
    good, bad = rule_stats(), rule_stats(net=-1, gross=-1)
    assert p.verdict(good, bad) == p.verdict(bad, good) == "TRADABLE"
    assert p.verdict(bad, bad) == "NO PREDICTABILITY"


def test_placebo_week_shared_all_pairs_and_reproducible():
    times = pd.date_range("2010-01-01", periods=400, freq="D", tz="UTC").repeat(10)
    signs = p.placebo_signs(times)
    np.testing.assert_array_equal(signs, p.placebo_signs(times))
    weeks = p.week_start(times)
    assert pd.Series(signs).groupby(weeks).nunique().max() == 1
    unique = weeks.unique().sort_values()
    expected = np.random.default_rng(0).choice([-1, 1], size=len(unique))
    np.testing.assert_array_equal(pd.Series(signs, index=weeks).groupby(level=0).first(), expected)


def test_readers_skip_locked_rows_before_any_parser(tmp_path, monkeypatch):
    monkeypatch.setattr(multi_asset_data, "etf_dir", tmp_path)
    monkeypatch.setattr(data, "fred_dir", tmp_path)
    (tmp_path / "spy.csv").write_text("Date,Open,High,Low,Close,Adj Close,Volume,Dividends,Stock Splits\n"
                                     "2025-10-30 00:00:00-04:00,100,101,99,100,100,1,0,0\n"
                                     "2025-11-03 00:00:00-05:00,DO-NOT-PARSE\n", encoding="ascii")
    (tmp_path / "vixcls.csv").write_text("observation_date,VIXCLS\n2025-10-29,20\n2025-10-30,.\n2025-11-03,DO-NOT-PARSE\n", encoding="ascii")
    original = pd.Timestamp

    def guarded(value, *args, **kwargs):
        if isinstance(value, str):
            assert value[:10] < "2025-11-01"
        return original(value, *args, **kwargs)

    namespace = SimpleNamespace(**{name: getattr(pd, name) for name in dir(pd) if not name.startswith("__")})
    namespace.Timestamp = guarded
    monkeypatch.setattr(multi_asset_data, "pd", namespace)
    monkeypatch.setattr(data, "pd", namespace)
    etf, _ = data.read_etf("spy")
    fred, report = data.read_fred("vixcls")
    assert len(etf) == 1 and len(fred) == 2 and report["missing"] == 1


def test_fx_only_via_continuous_loader_and_no_unlock(monkeypatch):
    seen = []

    def load(pair, tf):
        seen.append((pair, tf))
        return bars(10)

    monkeypatch.setattr(data.loader, "load_continuous_pair", load)
    report = dict(file="synthetic", rows=10, first=pd.Timestamp("2010-01-01"), last=pd.Timestamp("2010-02-01"), missing=0)
    monkeypatch.setattr(data, "read_etf", lambda name: (pd.Series([1.]), report))
    monkeypatch.setattr(data, "read_fred", lambda name: (pd.Series([1.]), report))
    source, _, _ = data.load_inputs()
    assert len(source) == 10 and seen == [(pair, "1h") for pair in p.fresh_pairs]
    for filename in (Path(p.__file__), Path(data.__file__), Path(run_predict.__file__)):
        assert "final=True" not in filename.read_text()


def test_planted_linear_signal_and_fixed_seed_null():
    rng = np.random.default_rng(77)
    x = rng.normal(size=(30000, 4))
    noise = rng.normal(size=len(x))
    planted = .3 * x[:, 0] + np.sqrt(1 - .3 ** 2) * noise
    fitted = p.fit_ridge(x[:15000], planted[:15000], 1e2)
    forecast = p.forecast_ridge(fitted, x[15000:])
    assert p.correlation(forecast, planted[15000:]) >= .15
    null_model = p.fit_ridge(x[:15000], noise[:15000], 1e2)
    null_forecast = p.forecast_ridge(null_model, x[15000:])
    gross = np.sign(null_forecast) * noise[15000:]
    weekly = gross.reshape(150, 100).sum(axis=1)
    assert abs(p.weekly_t(weekly)) < 3

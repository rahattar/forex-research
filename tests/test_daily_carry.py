import builtins
import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import run_daily_momentum as task11
from src import rates as rate_loader
from src.daily_carry import (add_carry, c1_verdict, c1_weights, c2_weights, currency_contributions,
                             evaluate_currency_book, evaluate_pair_book, monthly_holdings,
                             rate_accrual, restricted_carry_reads)
from src.daily_momentum import evaluate_m1, evaluate_m2, first_in_period, m1_features
from src.instruments import Config, pip_size
from tests.test_daily_momentum import close_fixture


@pytest.fixture
def monthly_files(tmp_path, monkeypatch):
    monkeypatch.setattr(rate_loader, "data_dir", tmp_path)
    months = pd.date_range("2010-01-01", "2025-10-01", freq="MS")
    values = dict(usd=5., eur=3., gbp=6., jpy=.5, aud=4., cad=4., chf=1., nzd=5.5)

    def write(currency, series=None, tail=""):
        if series is None:
            series = pd.Series(values[currency], index=months)
        rows = "".join(f"{date:%Y-%m-%d},{'' if pd.isna(value) else value}\n" for date, value in series.items())
        (tmp_path / rate_loader.monthly_sources[currency]).write_text("observation_date,value\n" + rows + tail)

    for currency in values:
        write(currency)
    (tmp_path / "dff.csv").write_text("must never read the daily policy rate file")
    return write, months


def test_next_month_availability_blank_ffill_and_no_backfill(monthly_files):
    write, months = monthly_files
    usd = pd.Series(5., index=months)
    usd.loc["2020-03-01"] = 2.
    usd.loc["2020-04-01"] = np.nan
    usd.loc["2020-05-01"] = 3.
    write("usd", usd)
    rates = rate_loader.load_monthly_rates()
    assert rates.loc[:"2010-01-31"].isna().all().all()
    assert rates.loc["2010-02-01", "usd"] == 5
    assert rates.loc["2020-03-31", "usd"] == 5
    assert rates.loc["2020-04-01", "usd"] == 2
    assert rates.loc["2020-05-01":"2020-05-31", "usd"].eq(2).all()
    assert rates.loc["2020-06-01", "usd"] == 3
    assert list(rates) == list(rate_loader.monthly_sources)
    assert rates.index[-1] == pd.Timestamp("2025-10-31", tz="UTC")


def test_locked_observations_not_parsed_and_october_cannot_affect_results(monthly_files):
    write, months = monthly_files
    first = rate_loader.load_monthly_rates()
    series = pd.Series(999., index=months)
    series.iloc[:-1] = .5
    write("jpy", series, tail="2025-11-01,FORBIDDEN-RATE\n2026-01-01,NEVER-PARSE\n")
    second = rate_loader.load_monthly_rates()
    pd.testing.assert_frame_equal(first, second)


def test_observation_cannot_change_positions_before_following_month(monthly_files):
    write, months = monthly_files
    before = rate_loader.load_monthly_rates()
    series = pd.Series(4., index=months)
    series.loc["2011-06-01":] = 20.
    write("aud", series)
    after = rate_loader.load_monthly_rates()
    boundary = pd.Timestamp("2011-07-01", tz="UTC")
    pd.testing.assert_frame_equal(before.loc[before.index < boundary], after.loc[after.index < boundary])
    closes = close_fixture()
    for function in (c1_weights, c2_weights):
        first = function(closes, before)
        second = function(closes, after)
        pd.testing.assert_frame_equal(first.loc[first.index < boundary], second.loc[second.index < boundary])
    assert c1_weights(closes, before).loc["2011-08-01", "aud"] != c1_weights(closes, after).loc["2011-08-01", "aud"]


def test_carry_integrates_old_and_new_rates_across_weekend(monthly_files):
    write, months = monthly_files
    usd = pd.Series(5., index=months)
    usd.loc["2011-06-01":] = 8.
    write("usd", usd)
    rates = rate_loader.load_monthly_rates()
    index = pd.to_datetime(["2011-06-30T00:00Z", "2011-07-01T00:00Z", "2011-07-04T00:00Z"])
    accrual = rate_accrual(rates, index)
    assert accrual.usd.iloc[0] == 0
    assert accrual.usd.iloc[1] == pytest.approx(.05 / 365)
    assert accrual.usd.iloc[2] == pytest.approx(.08 * 3 / 365)


@pytest.mark.parametrize("pair, gap", [("eurusd", -2.), ("usdjpy", 4.5), ("usdcad", 1.), ("eurgbp", -3.)])
@pytest.mark.parametrize("side", [1, -1])
def test_pair_carry_sign_cost_markup_hand_calculation(pair, gap, side, monthly_files):
    rates = rate_loader.load_monthly_rates()
    index = pd.to_datetime(["2011-01-05T00:00Z", "2011-01-10T00:00Z"])
    price = 100. if pair.endswith("jpy") else 1.25
    closes = pd.DataFrame({pair: price}, index=index)
    weights = pd.DataFrame({pair: side * .2}, index=index)
    frame = evaluate_pair_book(closes, weights, rates)
    # Five calendar days at 20% notional: signed gap in pp /100 *5/365 *.2.
    carry = side * .2 * gap / 100 * 5 / 365
    markup = .2 * .0075 * 5 / 365
    initial_cost = .2 * (.5 * Config().spread_pips[pair] + .5) * pip_size(pair) / price
    assert frame.carry.iloc[1] == pytest.approx(carry, abs=1e-14)
    assert frame.markup.iloc[1] == pytest.approx(markup)
    assert frame.cost.iloc[0] == pytest.approx(initial_cost)
    assert frame.net.iloc[0] == pytest.approx(-initial_cost)
    assert frame.net.iloc[1] == pytest.approx(carry - markup)
    assert frame.gross.iloc[1] == pytest.approx(carry)


def test_currency_ranking_and_usd_cash_leg(monthly_files):
    write, months = monthly_files
    values = dict(usd=6., eur=5., gbp=4., aud=3., cad=2., chf=1., nzd=.5, jpy=0.)
    for currency, value in values.items():
        write(currency, pd.Series(value, index=months))
    rates = rate_loader.load_monthly_rates()
    closes = close_fixture()
    allocations = c1_weights(closes, rates)
    active = allocations.loc[allocations.abs().sum(axis=1) > 0]
    assert active.usd.eq(.25).all() and active.eur.eq(.25).all()
    assert active.jpy.eq(-.25).all() and active.nzd.eq(-.25).all()
    assert active.eq(.25).sum(axis=1).eq(2).all()
    assert active.eq(-.25).sum(axis=1).eq(2).all()
    frame = evaluate_currency_book(closes, allocations, rates)
    assert frame.weight_usd.eq(0).all() and frame.net_usd.eq(0).all()
    assert frame.gross_leverage.iloc[1:].eq(.75).all()
    assert frame.carry_usd.eq(0).all()
    assert monthly_holdings(allocations, currency_book=True).usd.eq(.25).all()


def test_c2_deadband_volatility_cap_and_rebalance(monthly_files):
    write, months = monthly_files
    write("eur", pd.Series(5.2499, index=months))
    write("gbp", pd.Series(4.75, index=months))
    closes = close_fixture()
    rates = rate_loader.load_monthly_rates()
    weights = c2_weights(closes, rates)
    assert weights.eurusd.eq(0).all()
    assert weights.gbpusd.le(0).all() and weights.gbpusd.lt(0).any()
    assert weights.abs().le(.3).all().all()
    write("eur", pd.Series(5.25, index=months))
    weights = c2_weights(closes, rate_loader.load_monthly_rates())
    _, vol = m1_features(closes)
    schedule = closes.index[first_in_period(closes.index, "week")]
    eligible = weights.index.intersection(schedule)
    np.testing.assert_allclose(weights.loc[eligible, "eurusd"], (.1 / vol.loc[eligible, "eurusd"] / 10).clip(upper=.3))
    changed = weights.diff().abs().sum(axis=1).gt(1e-12)
    assert weights.index[changed].isin(schedule).all()
    rates["usd"], rates["eur"] = .1, .35
    assert c2_weights(closes, rates).eurusd.gt(0).any()


def test_momentum_re_report_changes_only_by_carry_and_one_markup(monthly_files):
    closes = close_fixture()
    rates = rate_loader.load_monthly_rates()
    for original, currency_book in ((evaluate_m1(closes), False), (evaluate_m2(closes), True)):
        actual = add_carry(original, rates, currency_book=currency_book)
        pd.testing.assert_series_equal(actual.spot, original.gross, check_names=False)
        pd.testing.assert_series_equal(actual.markup, original.financing, check_names=False)
        pd.testing.assert_series_equal(actual.cost, original.cost)
        np.testing.assert_allclose(actual.net, original.net + actual.carry)
        for name in original.filter(regex="^weight_"):
            pd.testing.assert_series_equal(actual[name], original[name])


def test_currency_contributions_reconcile_and_holdings_record_usd(monthly_files):
    closes, rates = close_fixture(), rate_loader.load_monthly_rates()
    allocations = c1_weights(closes, rates)
    for frame, currency_book in ((evaluate_currency_book(closes, allocations, rates), True),
                                 (evaluate_pair_book(closes, c2_weights(closes, rates), rates), False)):
        values = currency_contributions(frame, currency_book=currency_book)
        for metric in ("spot", "carry", "net"):
            assert values[metric].sum() == pytest.approx(frame[metric].mean() * 252)
    history = monthly_holdings(allocations, currency_book=True)
    assert list(history) == list(rate_loader.monthly_sources)
    assert history.usd.eq(0).all() and history.gbp.gt(0).all()


def test_unknown_holding_day_rate_cannot_be_backfilled(monthly_files):
    rates = rate_loader.load_monthly_rates()
    with pytest.raises(ValueError, match="backward filling"):
        rate_accrual(rates, pd.to_datetime(["2010-01-15T00:00Z", "2010-02-02T00:00Z"]))
    with pytest.raises(ValueError, match="test lock"):
        rate_accrual(rates, pd.to_datetime(["2025-10-31T00:00Z", "2025-11-01T00:00Z"]))


def test_real_monthly_rates_hard_cutoff_and_known_usd_blank():
    with restricted_carry_reads():
        rates = rate_loader.load_monthly_rates()
    assert rates.index[-1] == pd.Timestamp("2025-10-31", tz="UTC")
    assert not rates.isna().any().any()
    assert rates.loc["2020-04-30", "usd"] == rates.loc["2020-05-01", "usd"]
    assert len(rates.columns) == 8


def test_hist_and_other_real_rate_files_cannot_be_opened():
    project = Path(__file__).resolve().parents[1]
    with restricted_carry_reads():
        for opener in (builtins.open, io.open):
            with pytest.raises(ValueError, match="hist"):
                opener(project / "hist" / "never_read.csv")
            with pytest.raises(ValueError, match="eight monthly"):
                opener(project / "data" / "fred" / "dff.csv")


@pytest.mark.parametrize("lower, sharpe, early, late, expected", [
    (.0001, -.1, -.1, -.1, "PASS"), (0, .30, .01, .01, "PROMISING"),
    (0, .2999, .01, .01, "FAIL"), (-.1, .30, -.01, .01, "FAIL"), (-.1, .3, .01, 0, "FAIL"),
])
def test_c1_exact_verdict_rule(lower, sharpe, early, late, expected):
    rows = pd.DataFrame({"strategy": ["C1"] * 3, "group": ["portfolio"] * 3,
                         "period": ["all", "2010_2017", "2018_2025"], "net_sharpe_ci_low": [lower] * 3,
                         "net_sharpe": [sharpe] * 3, "net_total_return": [0, early, late]})
    assert c1_verdict(rows) == expected


def test_task11_full_outputs_reproduce_unchanged(tmp_path, monkeypatch, capsys):
    project = Path(__file__).resolve().parents[1]
    monkeypatch.setattr(task11, "results_dir", tmp_path)
    with restricted_carry_reads():
        task11.main()
    for name in ("daily_m1_returns.csv", "daily_m2_returns.csv", "daily_d1_trades.csv", "daily_summary.csv"):
        assert (tmp_path / name).read_bytes() == (project / "results" / name).read_bytes()
    assert capsys.readouterr().out.encode() == (project / "results" / "daily_report.txt").read_bytes()

import numpy as np
import pandas as pd
import pytest

from scripts import test4_rates as test4
from src import rates as rate_loader
from src.instruments import Config, pip_size


@pytest.fixture
def rate_files(tmp_path, monkeypatch):
    monkeypatch.setattr(rate_loader, "data_dir", tmp_path)

    def write(currency, rows):
        filename = rate_loader.sources[currency]
        text = "observation_date,value\n" + "".join(f"{date},{value}\n" for date, value in rows)
        (tmp_path / filename).write_text(text)

    for currency, value in (("usd", 5), ("eur", 2), ("gbp", 6), ("jpy", -0.1)):
        write(currency, [("2018-01-01", value)])
    (tmp_path / "japan_monthly_stale_do_not_use.csv").write_text("not a valid CSV; must never be read\n")
    return write


def constant_rates():
    index = pd.date_range("2018-01-01", "2025-10-31", freq="1D", tz="UTC")
    return pd.DataFrame({"usd": 5.0, "eur": 2.0, "gbp": 6.0, "jpy": 0.25}, index=index)


def test_rate_availability_is_next_day_and_forward_fill_only(rate_files):
    rate_files("usd", [("2019-01-01", 1), ("2019-01-03", 2)])
    rate_files("eur", [("2019-01-03", 3)])
    rate_files("gbp", [("2019-01-02", ""), ("2019-01-04", 4), ("2019-01-07", ".")])
    rate_files("jpy", [("2019-01-02", -0.1), ("2019-01-05", 0.05)])
    rates = rate_loader.load_rates()
    assert list(rates.columns) == ["usd", "eur", "gbp", "jpy"]
    assert rates.loc[:"2019-01-01"].isna().all().all()
    assert rates.loc["2019-01-02", "usd"] == 1
    assert rates.loc["2019-01-03", "usd"] == 1
    assert rates.loc["2019-01-04", "usd"] == 2
    assert rates.loc[:"2019-01-03", "eur"].isna().all()
    assert rates.loc["2019-01-04", "eur"] == 3
    assert rates.loc[:"2019-01-04", "gbp"].isna().all()
    assert rates.loc["2019-01-05":"2019-01-09", "gbp"].eq(4).all()
    assert rates.loc["2019-01-05", "jpy"] == -0.1
    assert rates.loc["2019-01-06", "jpy"] == 0.05
    assert (rates.index.to_series().diff().dropna() == pd.Timedelta(days=1)).all()
    assert str(rates.index.tz) == "UTC"


def test_boj_step_table_and_locked_observations_are_not_parsed(rate_files):
    rate_files("jpy", [("2016-01-29", -0.1), ("2024-03-19", 0.05), ("2024-07-31", 0.25),
                       ("2025-01-24", 0.5), ("2025-12-19", "locked"), ("2026-06-16", "never parse")])
    rates = rate_loader.load_rates()
    assert rates.loc[:"2016-01-29", "jpy"].isna().all()
    assert rates.loc["2016-01-30", "jpy"] == -0.1
    assert rates.loc["2024-03-19", "jpy"] == -0.1
    assert rates.loc["2024-03-20", "jpy"] == 0.05
    assert rates.loc["2024-07-31", "jpy"] == 0.05
    assert rates.loc["2024-08-01", "jpy"] == 0.25
    assert rates.loc["2025-01-24", "jpy"] == 0.25
    assert rates.loc["2025-01-25", "jpy"] == 0.5
    assert rates.index.max() == pd.Timestamp("2025-10-31", tz="UTC")


def test_changing_future_rates_cannot_change_a_past_signal(rate_files):
    rate_files("usd", [("2018-01-01", 5), ("2019-03-01", 2), ("2026-01-01", "ignored")])
    before = rate_loader.load_rates()
    before_signal = test4.rate_signals(rate_loader.diff("eurusd", before))
    rate_files("usd", [("2018-01-01", 5), ("2019-03-01", 90), ("2026-01-01", "other garbage")])
    after = rate_loader.load_rates()
    after_signal = test4.rate_signals(rate_loader.diff("eurusd", after))
    pd.testing.assert_frame_equal(before.loc[:"2019-03-01"], after.loc[:"2019-03-01"])
    pd.testing.assert_frame_equal(before_signal.loc[:"2019-03-01"], after_signal.loc[:"2019-03-01"])
    assert before_signal.loc["2019-02-01"].notna().all()
    assert before.loc["2019-03-02", "usd"] != after.loc["2019-03-02", "usd"]


def test_locked_and_2026_rates_cannot_change_any_allowed_rate_signal_or_carry(rate_files):
    rate_files("jpy", [("2018-01-01", -0.1), ("2025-12-19", 0.75), ("2026-06-16", 1)])
    before = rate_loader.load_rates()
    rate_files("jpy", [("2018-01-01", -0.1), ("2025-12-19", "unreadable"), ("2026-06-16", "ignored")])
    after = rate_loader.load_rates()
    pd.testing.assert_frame_equal(before, after)
    for pair in test4.pairs:
        pd.testing.assert_frame_equal(test4.rate_signals(rate_loader.diff(pair, before)),
                                      test4.rate_signals(rate_loader.diff(pair, after)))
        for side in ("long", "short"):
            args = (pair, side, "2025-01-06T01:00Z", "2025-01-13T01:00Z", 1.1)
            assert test4.carry_pips(*args, before) == test4.carry_pips(*args, after)


@pytest.mark.parametrize("change, expected", [(0, 0), (0.0999, 0), (0.1, 1), (-0.1, -1), (-0.0999, 0), (0.1001, 1)])
def test_deadband_and_exact_calendar_lags(change, expected):
    index = pd.date_range("2019-01-01", periods=121, freq="1D", tz="UTC")
    differential = pd.Series(0.2, index=index)
    differential.iloc[-1] += change
    signals = test4.rate_signals(differential)
    assert signals.chg60.iloc[-1] == expected
    assert signals.chg120.iloc[-1] == expected
    assert signals.level.iloc[-1] == 1
    assert signals.chg60.iloc[:60].isna().all()
    assert signals.chg120.iloc[:120].isna().all()


@pytest.mark.parametrize("pair, expected", [("eurusd", -3), ("gbpusd", 1), ("usdjpy", 4.75)])
def test_rate_difference_is_base_minus_quote(pair, expected):
    assert rate_loader.diff(pair, constant_rates()).eq(expected).all()


@pytest.mark.parametrize("pair, side, annual_pct, price", [
    ("eurusd", "long", -3.5, 1.1), ("eurusd", "short", 2.5, 1.1),
    ("gbpusd", "long", 0.5, 1.25), ("gbpusd", "short", -1.5, 1.25),
    ("usdjpy", "long", 4.25, 150), ("usdjpy", "short", -5.25, 150),
])
def test_carry_sign_and_markup_for_every_pair_and_side(pair, side, annual_pct, price):
    actual = test4.carry_pips(pair, side, "2025-01-06T01:00Z", "2025-01-13T01:00Z", price, constant_rates())
    expected = annual_pct / 100 * price / 365 * 7 / pip_size(pair)
    assert actual == pytest.approx(expected)


def test_carry_integrates_daily_rates_and_fractional_utc_days():
    rates = constant_rates()
    rates.loc[:, "eur"] = 9.0  # differential +4 pp
    rates.loc["2025-01-07":, "eur"] = 11.0  # differential +6 pp
    long = test4.carry_pips("eurusd", "long", "2025-01-06T12:00Z", "2025-01-07T12:00Z", 1.1, rates)
    short = test4.carry_pips("eurusd", "short", "2025-01-06T12:00Z", "2025-01-07T12:00Z", 1.1, rates)
    assert long == pytest.approx((3.5 * 0.5 + 5.5 * 0.5) / 100 * 1.1 / 365 / 0.0001)
    assert short == pytest.approx((-4.5 * 0.5 - 6.5 * 0.5) / 100 * 1.1 / 365 / 0.0001)
    rates.loc["2025-01-06", "eur"] = np.nan
    assert np.isnan(test4.carry_pips("eurusd", "long", "2025-01-06T12:00Z", "2025-01-07T12:00Z", 1.1, rates))
    with pytest.raises(ValueError, match="locked"):
        test4.carry_pips("eurusd", "long", "2025-10-31T01:00Z", "2026-01-01T01:00Z", 1.1, rates)


def test_monday_entry_spread_once_and_exact_week_endpoint():
    index = pd.date_range("2025-01-06T00:00Z", "2025-02-03T02:00Z", freq="1h")
    hourly = pd.DataFrame({"close": 1.1 + np.arange(len(index)) * 0.00001}, index=index)
    opportunities = test4.monday_opportunities(hourly, "eurusd", "val", 1, constant_rates(), Config())
    first = opportunities.iloc[0]
    assert first.monday == pd.Timestamp("2025-01-06T00:00Z")
    assert first.entry_time == pd.Timestamp("2025-01-06T01:00Z")
    assert first.exit_time == pd.Timestamp("2025-01-13T01:00Z")
    raw_move = (first.exit_bid - first.entry_bid) / 0.0001
    assert first.long_price_pips == pytest.approx(raw_move - 1)
    assert first.short_price_pips == pytest.approx(-raw_move - 1)
    assert first.long_net_pips == pytest.approx(first.long_price_pips + first.long_carry_pips)


def test_week_endpoint_cannot_cross_split_or_lock():
    index = pd.date_range("2024-12-23T00:00Z", "2025-01-14T00:00Z", freq="1h")
    hourly = pd.DataFrame({"close": 1.1}, index=index)
    train = test4.monday_opportunities(hourly, "eurusd", "train", 1, constant_rates(), Config())
    assert len(train) == 1
    assert (train.exit_time < pd.Timestamp("2025-01-01", tz="UTC")).all()
    index = pd.date_range("2025-10-20T00:00Z", "2025-10-31T23:00Z", freq="1h")
    val = test4.monday_opportunities(pd.DataFrame({"close": 1.1}, index=index), "eurusd", "val", 1, constant_rates(), Config())
    assert len(val) == 1
    assert val.exit_time.iloc[0] < test4.lock


def test_nonoverlapping_calendar_monday_phase_does_not_shift_after_flats():
    monday = pd.date_range("2025-01-06", periods=5, freq="W-MON", tz="UTC")
    sample = pd.DataFrame({"pair": "eurusd", "monday": monday, "monday_number": range(5),
                           "entry_time": monday + pd.Timedelta(hours=1),
                           "exit_time": monday + pd.Timedelta(weeks=2, hours=1), "net_pips": range(5)})
    selected = sample.loc[[0, 1, 3, 4]]  # calendar Monday #2 was flat
    assert test4.nonoverlapping(selected, h=2).monday_number.tolist() == [0, 4]
    sample.loc[0, ["entry_time", "exit_time"]] += pd.Timedelta(hours=22)
    assert test4.nonoverlapping(sample, h=2).monday_number.tolist() == [0, 4]


def test_random_benchmark_keeps_nonflat_mondays_and_is_deterministic():
    sample = pd.DataFrame({"pair": ["eurusd"] * 4, "monday": pd.date_range("2025-01-06", periods=4, freq="W-MON", tz="UTC"),
                           "long_net_pips": [10, 20, 30, 40], "short_net_pips": [-12, -22, -32, -42]})
    table = pd.DataFrame({"level": [1, 0, -1, 0]}, index=pd.DatetimeIndex(sample.monday))
    for side in ("long", "short"):
        sample[f"{side}_price_pips"] = sample[f"{side}_net_pips"]
        sample[f"{side}_carry_pips"] = 0
    selected = test4.select_trades(sample, "level", table)
    assert selected.monday.tolist() == sample.monday.iloc[[0, 2]].tolist()
    first = test4.matched_random_means(selected, seeds=(0, 1, 2))
    same = test4.matched_random_means(selected, seeds=(0, 1, 2))
    pd.testing.assert_series_equal(first, same)
    for seed in first.index:
        mask = np.random.default_rng(seed).random(2) < 0.5
        expected = np.where(mask, selected.long_net_pips, selected.short_net_pips).mean()
        assert first.loc[seed] == expected


def test_pass_rule_uses_all_three_thresholds_and_no_other_selection():
    rows, yearly = [], []
    for h, train_rank, val_rank, positive in ((1, 95, 90, 4), (4, 94.5, 100, 6), (8, 100, 100, 3)):
        for which, rank in (("train", train_rank), ("val", val_rank)):
            rows.append({"pair": "pooled", "signal": "level", "h": h, "split": which, "random_rank": rank})
        yearly.append({"signal": "level", "h": h, "positive_years": positive,
                       **{f"net_{year}": 1 for year in test4.years}})
    grid = test4.apply_pass_rule(pd.DataFrame(rows), pd.DataFrame(yearly))
    assert set(grid.loc[grid.pass_rule, "h"]) == {1}


def test_script_uses_default_loader_ignores_2026_and_saves_all_grids(rate_files, tmp_path, monkeypatch, capsys):
    index = pd.date_range("2024-01-01T00:00Z", "2025-10-31T23:00Z", freq="1h")
    hourly = pd.DataFrame({"close": 1.1 + np.sin(np.arange(len(index)) / 100) * 0.01}, index=index)
    calls = []

    def load(pair, tf, **kwargs):
        assert tf == "1h" and kwargs == {}
        calls.append(pair)
        return hourly

    rate_files("usd", [("2018-01-01", 5), ("2026-01-01", "never parse")])
    monkeypatch.setattr(test4, "load_pair", load)
    monkeypatch.setattr(test4, "results_dir", tmp_path)
    monkeypatch.setattr(test4, "benchmark_seeds", (0, 1))
    test4.main()
    assert calls == list(test4.pairs)
    grid = pd.read_csv(tmp_path / "test4_grid.csv")
    assert len(grid) == 120
    assert set(grid['split']) == {"train", "val"}
    assert len(grid.loc[(grid.pair == "pooled") & grid.signal.isin(test4.signals)]) == 18
    assert len(pd.read_csv(tmp_path / "test4_years.csv")) == 9
    assert len(pd.read_csv(tmp_path / "test4_random.csv")) == 36
    trades = pd.read_csv(tmp_path / "test4_trades.csv", parse_dates=["entry_time", "exit_time"])
    assert (trades.exit_time < test4.lock).all()
    np.testing.assert_allclose(trades.net_pips, trades.price_pips + trades.carry_pips, atol=1e-8)
    assert "Total pre-registered combinations: 9" in capsys.readouterr().out

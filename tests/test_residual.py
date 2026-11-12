import numpy as np
import pandas as pd
import pytest

from scripts import test1_residual as residual
from src.instruments import Config
from src.loader import split


@pytest.fixture
def hourly():
    index = pd.date_range("2024-12-01", "2025-01-16", freq="1h", tz="UTC")
    index = index[index.dayofweek < 5]
    changes = np.random.default_rng(7).normal(0, 0.0005, len(index))
    return pd.DataFrame({"close": 1.1 + changes.cumsum()}, index=index)


@pytest.mark.parametrize("span", residual.spans)
def test_z_score_does_not_change_with_future_prices(hourly, span):
    cutoff = pd.Timestamp("2024-12-30T12:00Z")
    baseline = residual.residual_features(hourly["close"], span)
    changed = hourly["close"].copy()
    changed.loc[changed.index > cutoff] += 1000
    altered = residual.residual_features(changed, span)
    assert baseline.loc[cutoff, "z"] == pytest.approx(altered.loc[cutoff, "z"])
    assert baseline.loc[:cutoff, "z"].notna().any()
    pd.testing.assert_series_equal(baseline.loc[:cutoff, "z"], altered.loc[:cutoff, "z"])
    truncated = residual.residual_features(hourly.loc[:cutoff, "close"], span)
    pd.testing.assert_frame_equal(baseline.loc[:cutoff], truncated)


def test_scale_uses_exactly_240_prior_changes(hourly):
    close = hourly["close"]
    features = residual.residual_features(close, 24)
    position = 300
    expected = close.diff().iloc[position - 240:position].std(ddof=1)
    assert features["scale"].iloc[position] == pytest.approx(expected)
    assert features["z"].iloc[:241].isna().all()
    assert pd.notna(features["z"].iloc[241])
    changed = close.copy()
    changed.iloc[position:] *= 100
    assert residual.residual_features(changed, 24)["scale"].iloc[position] == pytest.approx(expected)


def test_z_is_dimensionless_and_zero_scale_is_undefined(hourly):
    normal = residual.residual_features(hourly["close"], 72)
    scaled = residual.residual_features(hourly["close"] * 100, 72)
    np.testing.assert_allclose(normal["z"], scaled["z"], rtol=1e-9, atol=1e-9, equal_nan=True)
    flat = pd.Series(1.1, index=hourly.index)
    assert residual.residual_features(flat, 24)["z"].isna().all()


def test_forward_move_requires_exact_elapsed_hour():
    index = pd.to_datetime([
        "2025-01-03T21:00Z", "2025-01-06T00:00Z", "2025-01-06T01:00Z",
    ])
    close = pd.Series([1.1, 1.101, 1.1012], index=index)
    move = residual.forward_move(close, 1, "eurusd")
    assert pd.isna(move.iloc[0])
    assert move.iloc[1] == pytest.approx(2)
    assert pd.isna(move.iloc[2])


def test_forward_move_does_not_cross_split_boundary():
    index = pd.to_datetime(["2024-12-31T22:00Z", "2024-12-31T23:00Z", "2025-01-01T00:00Z"])
    hourly = pd.DataFrame({"close": [1.1, 1.1001, 1.2]}, index=index)
    train_close = split(hourly, "train")["close"]
    move = residual.forward_move(train_close, 1, "eurusd")
    assert move.iloc[0] == pytest.approx(1)
    assert pd.isna(move.iloc[-1])


def test_training_statistics_do_not_use_validation_prices(hourly):
    baseline = residual.pair_grid("eurusd", hourly, Config())
    changed = hourly.copy()
    changed.loc[changed.index >= pd.Timestamp("2025-01-01", tz="UTC"), "close"] += 10
    altered = residual.pair_grid("eurusd", changed, Config())
    pd.testing.assert_frame_equal(
        baseline.loc[baseline["split"] == "train"], altered.loc[altered["split"] == "train"]
    )


def test_spearman_handles_ties_without_scipy():
    assert residual.spearman(pd.Series([1, 2, 2, 4]), pd.Series([4, 2, 2, 1])) == pytest.approx(-1)
    assert np.isnan(residual.spearman(pd.Series([1, 1]), pd.Series([2, 3])))


def test_t_stat_uses_nonoverlapping_net_returns():
    gross = pd.Series([3, -1, 5, 4, 1, 6, 2, 5])
    stats = residual.trade_statistics(gross, spread=1, h=2)
    assert stats["n_trades"] == 8
    assert stats["mean_gross_pips"] == 3.125
    assert stats["mean_net_pips"] == 2.125
    assert stats["hit_rate"] == 0.75
    assert stats["n_nonoverlap"] == 4
    assert stats["mean_net_nonoverlap_pips"] == 1.75
    # Selected net returns [2, 4, 0, 1]: mean 1.75, sample variance 8.75/3.
    assert stats["t_stat"] == pytest.approx(1.75 / (np.sqrt(8.75 / 3) / 2))


@pytest.mark.parametrize("gross", [[], [2], [2, 2]])
def test_undefined_t_stat_is_nan(gross):
    assert np.isnan(residual.trade_statistics(pd.Series(gross, dtype=float), spread=1, h=1)["t_stat"])


def test_summary_ranks_on_training_only():
    grid = pd.DataFrame([
        {"pair": "eurusd", "rule": "fade", "span": span, "h": 1, "k": 1,
         "split": which, "n_trades": 20, "mean_net_pips": span if which == "train" else -span,
         "hit_rate": 0.5, "t_stat": 1}
        for span in range(1, 9) for which in ("train", "val")
    ])
    summary = residual.compact_summary(grid)
    assert summary["span"].tolist() == [8, 7, 6, 5, 4]
    assert summary["mean_net_pips_val"].tolist() == [-8, -7, -6, -5, -4]


@pytest.mark.parametrize("z, expected_count", [(1.0, 0), (2.0, 1)])
def test_strict_threshold_and_friday_evening_exclusion(monkeypatch, z, expected_count):
    index = pd.date_range("2025-01-10T19:00Z", periods=4, freq="1h")
    hourly = pd.DataFrame({"close": [1.1, 1.1001, 1.1003, 1.1006]}, index=index)
    monkeypatch.setattr(residual, "spans", (24,))
    monkeypatch.setattr(residual, "horizons", (1,))
    monkeypatch.setattr(residual, "thresholds", (1.0,))

    def features(close, span):
        return pd.DataFrame({"close": close, "residual": [1, 4, 9, 16], "z": z}, index=close.index)

    monkeypatch.setattr(residual, "residual_features", features)
    grid = residual.pair_grid("eurusd", hourly, Config())
    val = grid.loc[grid["split"] == "val"]
    assert (val["n_observations"] == 3).all()
    assert (val["n_trades"] == expected_count).all()


def test_main_uses_default_loader_and_writes_complete_grid(hourly, monkeypatch, tmp_path, capsys):
    calls = []

    def load(pair, tf, **kwargs):
        assert tf == "1h"
        assert kwargs == {}
        calls.append(pair)
        return hourly

    destination = tmp_path / "test1_grid.csv"
    monkeypatch.setattr(residual, "load_pair", load)
    monkeypatch.setattr(residual, "result_path", destination)
    residual.main()
    assert calls == list(residual.pairs)
    grid = pd.read_csv(destination)
    keys = ["pair", "span", "h", "k", "rule"]
    assert len(grid) == 432
    assert len(grid[keys].drop_duplicates()) == 216
    assert set(grid["split"]) == {"train", "val"}
    assert not grid.duplicated(keys + ["split"]).any()
    matched = grid.loc[grid["rule"] == "fade"].merge(
        grid.loc[grid["rule"] == "momentum"], on=["pair", "span", "h", "k", "split"],
        suffixes=("_fade", "_momentum"),
    )
    assert (matched["n_trades_fade"] == matched["n_trades_momentum"]).all()
    np.testing.assert_allclose(
        matched["mean_net_pips_fade"] + matched["mean_net_pips_momentum"],
        -2 * matched["spread_pips_fade"], rtol=1e-8, atol=1e-8,
    )
    assert len(residual.compact_summary(grid)) == 30
    assert "Total combinations tested: 216 (432 rows across train and val)." in capsys.readouterr().out

"""Report train/validation coverage without unlocking the final test period."""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.loader import load_pair, pairs, split


def largest_gap_hours(df):
    """Largest consecutive timestamp gap, subtracting UTC Saturday/Sunday time."""
    if len(df) < 2:
        return 0.0
    times = df.index.tz_localize(None).to_numpy(dtype="datetime64[ns]")
    days = times.astype("datetime64[D]")
    hours = (times - days) / np.timedelta64(1, "h")
    weekday_hours = np.busday_count("1970-01-01", days) * 24
    weekday_hours = weekday_hours + np.where(np.is_busday(days), hours, 0)
    return float(np.diff(weekday_hours).max())


def main():
    print(
        f"{'pair':<7} {'tf':<3} {'split':<5} {'rows':>7} "
        f"{'first_utc':<25} {'last_utc':<25} {'max_gap_h_ex_weekends':>21}"
    )
    for pair in pairs:
        for tf in ("15m", "1h"):
            df = load_pair(pair, tf)
            for which in ("train", "val"):
                part = split(df, which)
                first = part.index[0].isoformat() if len(part) else "-"
                last = part.index[-1].isoformat() if len(part) else "-"
                print(
                    f"{pair:<7} {tf:<3} {which:<5} {len(part):>7} "
                    f"{first:<25} {last:<25} {largest_gap_hours(part):>21.2f}"
                )


if __name__ == "__main__":
    main()

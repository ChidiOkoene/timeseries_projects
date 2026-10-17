"""Minimal demo: engineer features from the tracked sample OHLC CSV."""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running without an editable install: python examples/quickstart_features.py
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from timeseries.features import process_dataframe_daily
from timeseries.paths import EXAMPLES_DIR


def main() -> None:
    path = EXAMPLES_DIR / "sample_xrp_daily.csv"
    df = pd.read_csv(path, parse_dates=["datetime"])
    featured = process_dataframe_daily(df)
    print(f"Rows in: {len(df)}  Rows out (after warm-up drop): {len(featured)}")
    print(featured[["datetime", "close", "RSI", "MACD", "Upper_Band", "Lower_Band"]].tail())


if __name__ == "__main__":
    main()

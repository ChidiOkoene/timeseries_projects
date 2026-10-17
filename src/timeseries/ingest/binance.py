"""Fetch historical klines from Binance using env-based credentials."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from binance.client import Client

from timeseries.paths import ARTIFACTS_DIR, DATA_DIR, ensure_runtime_dirs


def _client_from_env() -> Client:
    api_key = os.environ.get("BINANCE_API_KEY", "")
    api_secret = os.environ.get("BINANCE_API_SECRET", "")
    if not api_key or not api_secret:
        raise EnvironmentError(
            "Set BINANCE_API_KEY and BINANCE_API_SECRET in the environment "
            "(or a local .env file). Do not hardcode credentials."
        )
    return Client(api_key, api_secret)


def fetch_binance_data(
    symbol: str,
    interval: str,
    start_date: str,
    end_date: str,
    output_filename: str | Path | None = None,
) -> pd.DataFrame:
    """
    Fetch historical candlestick data from Binance and optionally save to CSV.

    Parameters
    ----------
    symbol : str
        Trading pair (e.g. ``XRPUSDT``).
    interval : str
        Kline interval from ``Client`` constants (e.g. ``Client.KLINE_INTERVAL_1HOUR``).
    start_date, end_date : str
        Binance date strings such as ``\"1 Jan, 2017\"``.
    output_filename : path-like, optional
        CSV destination. Relative paths are resolved under ``DATA_DIR``.
    """
    ensure_runtime_dirs()
    client = _client_from_env()

    klines = client.get_historical_klines(symbol, interval, start_date, end_date)

    df = pd.DataFrame(
        klines,
        columns=[
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base_asset_volume",
            "taker_buy_quote_asset_volume",
            "ignore",
        ],
    )

    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
    df = df.set_index("open_time")
    # loc[:, cols] stays a DataFrame for the type checker (df[cols] is Series|DataFrame).
    ohlcv = df.loc[:, ["open", "high", "low", "close", "volume"]]
    for col in ohlcv.columns:
        ohlcv[col] = pd.to_numeric(ohlcv[col], errors="coerce")

    if output_filename is not None:
        out = Path(output_filename)
        if not out.is_absolute():
            out = DATA_DIR / out
        out.parent.mkdir(parents=True, exist_ok=True)
        ohlcv.to_csv(out)
        print(f"Data saved to {out}")

    print(ohlcv.head())
    return ohlcv


if __name__ == "__main__":
    ensure_runtime_dirs()
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    fetch_binance_data(
        symbol="XRPUSDT",
        interval=Client.KLINE_INTERVAL_1HOUR,
        start_date="1 Jan, 2017",
        end_date="3 Mar, 2025",
        output_filename="XRPUSDT_hourly_2017_2025.csv",
    )

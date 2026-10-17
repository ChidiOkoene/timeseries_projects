"""Technical indicator features for daily and hourly OHLC frames.

Public seam: OHLC frame → Feature frame (CONTEXT.md vocabulary).
"""

from __future__ import annotations

import pandas as pd


def _close_series(df: pd.DataFrame) -> pd.Series:
    """Narrow ``df['close']`` to Series (pandas stubs also allow DataFrame)."""
    close = df["close"]
    if not isinstance(close, pd.Series):
        raise TypeError("column 'close' must be a Series, not a DataFrame")
    return close


def _calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window=period).mean()
    loss = (-delta.clip(upper=0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    # pandas stubs widen Series arithmetic to float | ndarray | Series
    return pd.Series(rsi, index=close.index, dtype="float64")


def process_dataframe_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a daily Feature frame (SMA/EMA, RSI, MACD, Bollinger, lags).

    Expected columns include at least:
    ``open``, ``high``, ``low``, ``close``, ``volume``.
    Drops ``symbol`` if present. Rows with NaN from rolling windows are dropped.
    """
    df = df.copy()
    df["av_pr"] = (df["low"] + df["high"]) / 2
    df["diff"] = df["close"] - df["open"]
    df["candle"] = (df["open"] < df["close"]).astype(int)

    df["7_day_SMA"] = df["close"].rolling(window=7).mean()
    df["30_day_SMA"] = df["close"].rolling(window=30).mean()
    df["7_day_EMA"] = df["close"].ewm(span=7, adjust=False).mean()
    df["30_day_EMA"] = df["close"].ewm(span=30, adjust=False).mean()

    df["RSI"] = _calculate_rsi(_close_series(df), period=14)

    df["12_day_EMA"] = df["close"].ewm(span=12, adjust=False).mean()
    df["26_day_EMA"] = df["close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = df["12_day_EMA"] - df["26_day_EMA"]
    df["Signal_Line"] = df["MACD"].ewm(span=9, adjust=False).mean()

    window = 20
    df["20_day_SMA"] = df["close"].rolling(window=window).mean()
    df["20_day_STD"] = df["close"].rolling(window=window).std()
    df["Upper_Band"] = df["20_day_SMA"] + (df["20_day_STD"] * 2)
    df["Lower_Band"] = df["20_day_SMA"] - (df["20_day_STD"] * 2)

    df["lag_1"] = df["close"].shift(1)
    df["lag_2"] = df["close"].shift(2)
    df["lag_3"] = df["close"].shift(3)

    df.dropna(inplace=True)
    if "symbol" in df.columns:
        df.drop(columns="symbol", inplace=True)
    return df


def process_dataframe_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build an hourly Feature frame (same indicator family, hour windows).

    Expected columns include at least:
    ``open``, ``high``, ``low``, ``close``, ``volume``.
    """
    df = df.copy()
    df["av_pr"] = (df["low"] + df["high"]) / 2
    df["diff"] = df["close"] - df["open"]
    df["candle"] = (df["open"] < df["close"]).astype(int)

    df["7_hour_SMA"] = df["close"].rolling(window=7).mean()
    df["30_hour_SMA"] = df["close"].rolling(window=30).mean()
    df["7_hour_EMA"] = df["close"].ewm(span=7, adjust=False).mean()
    df["30_hour_EMA"] = df["close"].ewm(span=30, adjust=False).mean()

    df["RSI"] = _calculate_rsi(_close_series(df), period=14)

    df["12_hour_EMA"] = df["close"].ewm(span=12, adjust=False).mean()
    df["26_hour_EMA"] = df["close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = df["12_hour_EMA"] - df["26_hour_EMA"]
    df["Signal_Line"] = df["MACD"].ewm(span=9, adjust=False).mean()

    window = 20
    df["20_hour_SMA"] = df["close"].rolling(window=window).mean()
    df["20_hour_STD"] = df["close"].rolling(window=window).std()
    df["Upper_Band"] = df["20_hour_SMA"] + (df["20_hour_STD"] * 2)
    df["Lower_Band"] = df["20_hour_SMA"] - (df["20_hour_STD"] * 2)

    df["lag_1"] = df["close"].shift(1)
    df["lag_2"] = df["close"].shift(2)
    df["lag_3"] = df["close"].shift(3)

    df.dropna(inplace=True)
    if "symbol" in df.columns:
        df.drop(columns="symbol", inplace=True)
    return df

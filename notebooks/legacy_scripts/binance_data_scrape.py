"""Legacy entrypoint — prefer ``python -m timeseries.ingest.binance``.

Credentials must come from the environment (BINANCE_API_KEY / BINANCE_API_SECRET).
"""

from timeseries.ingest.binance import fetch_binance_data

__all__ = ["fetch_binance_data"]

if __name__ == "__main__":
    from binance.client import Client

    fetch_binance_data(
        symbol="XRPUSDT",
        interval=Client.KLINE_INTERVAL_1HOUR,
        start_date="1 Jan, 2017",
        end_date="3 Mar, 2025",
        output_filename="XRPUSDT_hourly_2017_2025.csv",
    )

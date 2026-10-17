"""``python -m timeseries.ingest`` → Binance fetch CLI."""

from binance.client import Client

from timeseries.ingest.binance import fetch_binance_data

if __name__ == "__main__":
    fetch_binance_data(
        symbol="XRPUSDT",
        interval=Client.KLINE_INTERVAL_1HOUR,
        start_date="1 Jan, 2017",
        end_date="3 Mar, 2025",
        output_filename="XRPUSDT_hourly_2017_2025.csv",
    )

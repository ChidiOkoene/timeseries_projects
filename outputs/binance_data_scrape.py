
from binance.client import Client
import pandas as pd

def fetch_binance_data(symbol, interval, start_date, end_date, output_filename):
    """
    Fetch historical candlestick data from Binance and save to a CSV file.
    
    Parameters:
    - symbol (str): Trading pair symbol (e.g., "XRPUSDT").
    - interval (str): Kline interval from Client constants (e.g., Client.KLINE_INTERVAL_1HOUR).
    - start_date (str): Start date in format "1 Jan, 2017".
    - end_date (str): End date in format "3 Mar, 2025".
    - output_filename (str): Name of the CSV file to save the data.
    
    Returns:
    - df (DataFrame): Pandas DataFrame containing the historical data.
    """
    # Replace these with your Binance API key and secret.
    api_key = 'MALFOiVw1HAZmrIsp9qzPbKYcyz5M24P4gKmAqsLumNOUUh214OHh9E5oMKbsY6X'
    api_secret = 'Y8W3CTDzZWogtKuD6TnWk1seLEjsYDp4txcCkYfPWc6y1tZsZtNYUYxgpooQezx6'
    
    client = Client(api_key, api_secret)
    
    # Fetch historical klines (Binance limits each call to 1000 candles, but
    # get_historical_klines handles pagination automatically)
    klines = client.get_historical_klines(symbol, interval, start_date, end_date)
    
    # Convert the list of klines into a DataFrame.
    df = pd.DataFrame(klines, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ])
    
    # Convert timestamps from milliseconds to datetime.
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
    
    # Set the open_time as the index.
    df.set_index('open_time', inplace=True)
    
    # Select only the desired columns.
    df = df[['open', 'high', 'low', 'close', 'volume']]
    
    # Convert the numeric columns from string to float.
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Save the DataFrame to CSV.
    df.to_csv(output_filename)
    
    print(f"Data saved to {output_filename}")
    print(df.head())
    
    return df

if __name__ == '__main__':
    # Example usage: fetch hourly data for XRP/USDT from 1 Jan, 2017 to 3 Mar, 2025.
    from binance.client import Client  # For the interval constant
    fetch_binance_data(
        symbol="XRPUSDT",
        interval=Client.KLINE_INTERVAL_1HOUR,
        start_date="1 Jan, 2017",
        end_date="3 Mar, 2025",
        output_filename="XRPUSDT_hourly_2017_2025.csv"
    )

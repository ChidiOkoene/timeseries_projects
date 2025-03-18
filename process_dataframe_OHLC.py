

def process_dataframe_daily(df):
    """
    Processes a daily OHLC dataset by adding technical indicator features.
    
    Expected columns include at least:
        'open', 'high', 'low', 'close', 'volume', 'datetime', 'symbol'
    Additional columns (like 'datetime') are preserved (or can be set as index).
    
    Returns:
        A processed DataFrame with features:
            - 'av_pr': average price, (low + high)/2
            - 'diff': difference between close and open
            - 'candle': 1 if bullish (open < close), else 0
            - 7-day SMA and 30-day SMA of 'close'
            - 7-day EMA and 30-day EMA of 'close'
            - RSI computed over a 14-day period
            - MACD and Signal_Line (using 12-day EMA and 26-day EMA)
            - Bollinger Bands (20-day SMA, 20-day STD, Upper_Band, Lower_Band)
            - Lag features: lag_1, lag_2, lag_3 of 'close'
        All rows with NaN (due to rolling windows or shifts) are dropped.
        The 'symbol' column is dropped.
    """
    import numpy as np
    import pandas as pd
    
    # Define helper functions
    def candle(row):
        return 1 if row['open'] < row['close'] else 0

    def diff(row):
        return row['close'] - row['open']

    def ave(row):
        return (row['low'] + row['high']) / 2

    def calculate_rsi(data, period=14):
        delta = data['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    # Compute basic features
    df['av_pr'] = df.apply(ave, axis=1)
    df['diff'] = df.apply(diff, axis=1)
    df['candle'] = df.apply(candle, axis=1)

    # Moving Averages
    df['7_day_SMA'] = df['close'].rolling(window=7).mean()
    df['30_day_SMA'] = df['close'].rolling(window=30).mean()
    df['7_day_EMA'] = df['close'].ewm(span=7, adjust=False).mean()
    df['30_day_EMA'] = df['close'].ewm(span=30, adjust=False).mean()

    # RSI
    df['RSI'] = calculate_rsi(df, period=14)

    # MACD and Signal Line
    df['12_day_EMA'] = df['close'].ewm(span=12, adjust=False).mean()
    df['26_day_EMA'] = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['12_day_EMA'] - df['26_day_EMA']
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # Bollinger Bands
    window = 20
    df['20_day_SMA'] = df['close'].rolling(window=window).mean()
    df['20_day_STD'] = df['close'].rolling(window=window).std()
    df['Upper_Band'] = df['20_day_SMA'] + (df['20_day_STD'] * 2)
    df['Lower_Band'] = df['20_day_SMA'] - (df['20_day_STD'] * 2)

    # Lag Features
    df['lag_1'] = df['close'].shift(1)
    df['lag_2'] = df['close'].shift(2)
    df['lag_3'] = df['close'].shift(3)

    # Drop rows with NaN values (created by rolling windows and shifts)
    df.dropna(inplace=True)

    # Drop unwanted columns (e.g., 'symbol')
    if 'symbol' in df.columns:
        df.drop(columns='symbol', inplace=True)

    # Optionally, set the 'datetime' column as index (if desired)\n    # df.set_index('datetime', inplace=True)
    
    return df


def process_dataframe_hourly(df):
    """
    Processes an hourly OHLC dataset by adding technical indicator features and creating
    a target column for predicting the next hour's close.
    
    Expected columns include at least:
        'open', 'high', 'low', 'close', 'volume', 'datetime', 'symbol'
    Additional columns (like 'datetime') are preserved (or can be set as index).
    
    Returns:
        A processed DataFrame with features:
            - 'av_pr': average price, (low + high)/2
            - 'diff': difference between close and open
            - 'candle': 1 if bullish (open < close), else 0
            - 7-hour SMA and 30-hour SMA of 'close'
            - 7-hour EMA and 30-hour EMA of 'close'
            - RSI computed over a 14-hour period
            - MACD and Signal_Line (using 12-hour EMA and 26-hour EMA, signal line from 9-period EMA)
            - Bollinger Bands (20-hour SMA, 20-hour STD, Upper_Band, Lower_Band)
            - Lag features: lag_1, lag_2, lag_3 of 'close'
            - 'target': next hour's close price (i.e. 'close' shifted by -1)
        All rows with NaN (due to rolling windows, shifts, or target creation) are dropped.
        The 'symbol' column is dropped.
    """
    import numpy as np
    import pandas as pd

    # Helper functions for basic features
    def candle(row):
        return 1 if row['open'] < row['close'] else 0

    def diff(row):
        return row['close'] - row['open']

    def ave(row):
        return (row['low'] + row['high']) / 2

    def calculate_rsi(data, period=14):
        delta = data['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    # Compute basic features
    df['av_pr'] = df.apply(ave, axis=1)
    df['diff'] = df.apply(diff, axis=1)
    df['candle'] = df.apply(candle, axis=1)

    # Moving Averages and EMAs (interpreted as hours)
    df['7_hour_SMA'] = df['close'].rolling(window=7).mean()
    df['30_hour_SMA'] = df['close'].rolling(window=30).mean()
    df['7_hour_EMA'] = df['close'].ewm(span=7, adjust=False).mean()
    df['30_hour_EMA'] = df['close'].ewm(span=30, adjust=False).mean()

    # RSI over a 14-hour period
    df['RSI'] = calculate_rsi(df, period=14)

    # MACD and Signal Line (using 12-hour EMA and 26-hour EMA)
    df['12_hour_EMA'] = df['close'].ewm(span=12, adjust=False).mean()
    df['26_hour_EMA'] = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['12_hour_EMA'] - df['26_hour_EMA']
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # Bollinger Bands (20-hour window)
    window = 20
    df['20_hour_SMA'] = df['close'].rolling(window=window).mean()
    df['20_hour_STD'] = df['close'].rolling(window=window).std()
    df['Upper_Band'] = df['20_hour_SMA'] + (df['20_hour_STD'] * 2)
    df['Lower_Band'] = df['20_hour_SMA'] - (df['20_hour_STD'] * 2)

    # Lag Features: use previous 1, 2, and 3 hours of close prices
    df['lag_1'] = df['close'].shift(1)
    df['lag_2'] = df['close'].shift(2)
    df['lag_3'] = df['close'].shift(3)


    # Drop rows with NaN values created by rolling windows, shifts, or target
    df.dropna(inplace=True)

    # Drop unwanted columns (e.g., 'symbol')
    if 'symbol' in df.columns:
        df.drop(columns='symbol', inplace=True)

    # Optionally, set 'datetime' column as index if it exists (or leave as is)
    # df.set_index('datetime', inplace=True)
    
    return df



# Example usage:
# df_raw = pd.read_csv('your_daily_data.csv', parse_dates=['datetime'])
# df_processed = process_dataframe(df_raw)
# df_processed.to_csv('xrpusdt_daily_dataset_with_features_val.csv', index=False)

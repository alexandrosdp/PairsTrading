import ccxt
import pandas as pd
import os
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
import datetime
import matplotlib.pyplot as plt
import re  # Import regex to extract numbers from timeframe

# --------------------------
# 1. Data Collection Functions
# --------------------------

def fetch_data(symbol, timeframe='1d', since=None, limit=365):
    """
    Fetch OHLCV data from Binance for a given symbol.
    
    Parameters:
        symbol (str): Trading pair symbol in the format 'BASE/QUOTE' (e.g., 'BTC/USDT').
        timeframe (str, optional): Timeframe for the OHLCV data (e.g., '1d', '1h'). Default is '1d'.
        since (int, optional): Timestamp in milliseconds from which to start fetching data. 
                               If None, defaults to fetching data from 'limit' days ago.
        limit (int, optional): Maximum number of data points to fetch. Default is 365.
    
    Returns:
        pd.DataFrame: DataFrame containing OHLCV data with columns ['open', 'high', 'low', 'close', 'volume'] 
                      and a DateTime index. Returns None if data fetching fails.
    """
    exchange = ccxt.binance({
        'enableRateLimit': True,
    })
    # If 'since' is not provided, default to "limit" days ago.
    if since is None:
        since_dt = datetime.datetime.utcnow() - datetime.timedelta(days=limit)
        since = exchange.parse8601(since_dt.strftime('%Y-%m-%dT%H:%M:%S'))
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        return None
    data = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
    data.set_index('timestamp', inplace=True)
    return data



def get_aligned_data(symbols, timeframe='1d', since=None, limit=365):
    """
    Fetch and align closing price data for a list of trading pair symbols.
    
    Parameters:
        symbols (list): List of trading pair symbols (e.g., ['BTC/USDT', 'ETH/USDT']).
        timeframe (str, optional): Timeframe for the OHLCV data. Default is '1d'.
        since (int, optional): Timestamp in milliseconds from which to start fetching data. 
                               If None, defaults to fetching data from 'limit' days ago.
        limit (int, optional): Maximum number of data points to fetch for each symbol. Default is 365.
    
    Returns:
        pd.DataFrame: DataFrame where each column represents the closing prices for a symbol, 
                      aligned on common timestamps (inner join). Returns an empty DataFrame if no data is fetched.
    """

    price_dfs = {}
    for sym in symbols:
        df = fetch_data(sym, timeframe, since, limit)
        if df is not None:
            # Using the 'close' price
            price_dfs[sym] = df['close']
    # Align on the common dates (inner join)
    prices = pd.concat(price_dfs.values(), axis=1, join='inner')
    prices.columns = price_dfs.keys()
    
    return prices

def get_data_by_category(categories_dict, timeframe='1d', since=None, limit=365):
    """
    Fetch and align data for each category in categories_dict.
    
    Parameters:
        categories_dict (dict): e.g. {'Payment': ['BTC/USDT','LTC/USDT'], 'Layer1': ['ETH/USDT', ...], ...}
        timeframe (str): e.g. '1d', '4h'
        since (int): optional 'since' in ms
        limit (int): how many data points to fetch
        
    Returns:
        dict of {category_name: DataFrame} where each DataFrame has aligned closing prices 
        for all symbols in that category.
    """
    category_data = {}
    
    for category_name, symbols in categories_dict.items():
        print(f"Fetching data for category: {category_name} -> {symbols}")
        
        # Use your existing get_aligned_data to fetch and merge
        df_prices = get_aligned_data(symbols, timeframe, since, limit)
        
        # Only store if we have non-empty data
        if not df_prices.empty:
            category_data[category_name] = df_prices
        else:
            print(f"No data fetched for {category_name}, skipping.")
    
    return category_data



def merge_ohlc_closing_prices(directory):
    """
    Reads all merged CSVs from the given directory, extracts closing prices and timestamps, 
    stores them in a single DataFrame, fills missing values, and saves the merged file.

    Parameters:
        directory (str): Path to the directory containing the merged OHLC CSV files.

    Returns:
        pd.DataFrame: A DataFrame with timestamps as the index and closing prices of all cryptos as columns.
    """
    closing_prices = {}

    # Iterate over all CSV files in the directory
    for file in os.listdir(directory):
        if file.endswith(".csv"):  # Ensure we only process CSV files
            file_path = os.path.join(directory, file)

            # Extract the trading pair from the filename (e.g., "BTCUSDT.csv" → "BTC/USDT")
            symbol = file.replace(".csv", "").replace("USDT", "/USDT")

            try:
                # Read CSV file
                df = pd.read_csv(file_path)

                # Ensure required columns exist
                if "Open Time" in df.columns and "Close" in df.columns:
                    df["Open Time"] = pd.to_datetime(df["Open Time"])
                    df = df[["Open Time", "Close"]]
                    df.rename(columns={"Open Time": "timestamp", "Close": symbol}, inplace=True)

                    # Store data in dictionary
                    closing_prices[symbol] = df
                else:
                    print(f"⚠️ Skipping {file} - Required columns missing")

            except Exception as e:
                print(f"❌ Error processing {file}: {e}")

    # Merge all dataframes on timestamp
    if closing_prices:
        # merged_df = list(closing_prices.values())[0]  # Start with the first DataFrame
        # for symbol, df in list(closing_prices.items())[1:]:
        #     merged_df = pd.merge(merged_df, df, on="timestamp", how="outer")  # Merge on timestamp

        # # Sort by timestamp
        # merged_df.sort_values("timestamp", inplace=True)
        # merged_df.reset_index(drop=True, inplace=True)

        # # Check for NaNs and log missing values
        # nan_counts = merged_df.isna().sum()
        # total_nans = nan_counts.sum()

        # Find the latest starting timestamp among all datasets
        common_start_time = max(df['timestamp'].min() for df in closing_prices.values())

        # Filter each dataset to only include data from the common start time onward
        for symbol, df in closing_prices.items():
            closing_prices[symbol] = df[df['timestamp'] >= common_start_time]

        # Merge all dataframes on timestamp
        merged_df = list(closing_prices.values())[0]  # Start with the first DataFrame
        for symbol, df in list(closing_prices.items())[1:]:
            merged_df = pd.merge(merged_df, df, on="timestamp", how="outer")  # Merge on timestamp

        # Sort by timestamp
        merged_df.sort_values("timestamp", inplace=True)
        merged_df.reset_index(drop=True, inplace=True)

        # Check for NaNs and log missing values
        nan_counts = merged_df.isna().sum()
        total_nans = nan_counts.sum()


        if total_nans > 0:
            print(f"⚠️ Detected {total_nans} missing values across the dataset.")

            # Print NaN counts per column
            for column, count in nan_counts.items():
                if count > 0:
                    print(f"   - {column}: {count} missing values")

            # Fill missing values using forward fill (then backward fill for remaining NaNs)
            merged_df.fillna(method='ffill', inplace=True)
            merged_df.fillna(method='bfill', inplace=True)

            # Check if any NaNs remain after filling
            remaining_nans = merged_df.isna().sum().sum()
            if remaining_nans == 0:
                print("✅ All missing values have been successfully filled.")
            else:
                print(f"⚠️ {remaining_nans} missing values remain after filling.")

        else:
            print("✅ No missing values detected.")

        # Save the merged DataFrame as a CSV in the same directory
        output_file = os.path.join(directory, "merged_closing_prices.csv")
        merged_df.to_csv(output_file, index=False)
        print(f"✅ Merged file saved as: {output_file}")

        return merged_df
    else:
        print("⚠️ No valid data found.")
        return None


def resample_data(directory, timeframe, output_directory):

    """
    Converts all 1-minute OHLC data CSVs in a directory to a specified higher timeframe.

    :param directory: Path to the folder containing 1-minute CSV files.
    :param timeframe: The higher timeframe in minutes (e.g., 30 for 30-minute, 60 for 1-hour).
    :param output_directory: Path to save the converted CSV files.
    """

            # T	Minute
            # H	Hourly
            # D	Daily
            # W	Weekly
            # M	Month end
            # MS	Month start
            # Q	Quarter end
            # A	Year end

    # Ensure output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # Get all CSV files in the directory
    csv_files = [f for f in os.listdir(directory) if f.endswith(".csv")]

    if not csv_files:
        print("⚠️ No CSV files found in the directory.")
        return

    print(f"📂 Found {len(csv_files)} CSV files. Converting to {timeframe}-minute timeframe...")

    for file in csv_files:
        file_path = os.path.join(directory, file)
        
        try:
            # Read CSV
            df = pd.read_csv(file_path, parse_dates=["Open Time"], index_col="Open Time")

            # # Ensure 'Open Time' column exists
            # if "Open Time" not in df.columns:
            #     print(f"⚠️ Skipping {file} - 'Open Time' column missing.")
            #     continue

            df_resampled = df.resample(timeframe).agg({
            "Open": "first",  # first value in the 5-minute period
            "High": "max",    # highest value in the 5-minute period
            "Low": "min",     # lowest value in the 5-minute period
            "Close": "last",  # last value in the 5-minute period
            "Volume": "sum"   # sum of volumes over the period
            })

            # Optionally, drop any periods where the market was closed (if any row has a NaN in 'Open')
            df_resampled.dropna(subset=["Open"], inplace=True)

            # Save the new CSV
            df_resampled.reset_index(inplace=True)  # Ensure 'Open Time' is a column

            # # Extract only the numeric part from timeframe (e.g., "5T" → "5min", "1H" → "60min")
            # timeframe_numeric = re.findall(r'\d+', timeframe)  # Extract numbers from timeframe string
            # if not timeframe_numeric:  # Default to "1min" if extraction fails
            #     timeframe_numeric = "1min"
            # else:
            #     timeframe_numeric = f"{timeframe_numeric[0]}min"
            
            # Modify filename by replacing the old timeframe with the new one
            filename_parts = file.split("_")
            filename_parts[-1] = f"{timeframe}.csv"  # Replace last part (e.g., "1m.csv") with new timeframe
            new_filename = "_".join(filename_parts)

            output_path = os.path.join(output_directory, new_filename)



            df_resampled.to_csv(output_path, index=False)
            print(f"✅ Converted {file} -> {output_path}")
            
        except Exception as e:
            print(f"❌ Error processing {file}: {e}")

    print("🎉 Conversion complete!")

# Example usage
# convert_all_to_higher_timeframe("binance_data/1min", 30, "binance_data/30min")



# --------------------------
# 4. Main Execution: Data Gathering, Pair Screening, and Backtesting
# --------------------------

if __name__ == "__main__":

    print("TEST")

    #Define a list of symbols; adjust or extend as needed.
    symbols = ["BTC/USDT", "ETH/USDT", "XRP/USDT", "LTC/USDT", "BCH/USDT"]
    
    # Get aligned historical price data (last 365 days)
    print("Fetching and aligning data...")
    prices = get_aligned_data(symbols, timeframe='1h', limit=1500)
    if prices.empty:
        print("No data fetched. Please check your API connection or symbols.")
        exit()
    else:
        prices.to_csv("data/prices_data_hourly.csv")

#     categories = {
#     'Payment':  ['BTC/USDT','LTC/USDT','BCH/USDT'],
#     'Layer1':   ['ETH/USDT','ADA/USDT','SOL/USDT'],
#     'DeFi':     ['UNI/USDT','AAVE/USDT','COMP/USDT'],
#     # etc.
# }

# # Now fetch data category-by-category
# category_dataframes = get_data_by_category(categories, timeframe='1d', limit=365)

# category_dataframes is a dict: 
# {
#   'Payment':   DataFrame with columns ['BTC/USDT','LTC/USDT','BCH/USDT'],
#   'Layer1':    DataFrame with columns ['ETH/USDT','ADA/USDT','SOL/USDT'],
#   'DeFi':      DataFrame with columns ['UNI/USDT','AAVE/USDT','COMP/USDT'],
#   ...
# }
    
    
# --------------------------
# NOTES & POTENTIAL PITFALLS:
#
# 1. Data Quality:  
#    - Ensure data is complete and time-synchronized. Missing or misaligned data can lead
#      to spurious cointegration results.
#
# 2. API Limitations:  
#    - Exchanges may impose rate limits. Implement caching or delays if you plan to fetch
#      data for many symbols.
#
# 3. Overfitting:  
#    - Be cautious not to over-optimize entry/exit thresholds based solely on historical data.
#
# 4. Market Dynamics:  
#    - Cryptocurrency markets are highly volatile. Relationships may break down quickly,
#      so continuous monitoring and re-calibration of the model is essential.
#
# 5. Execution Risks:  
#    - Slippage, transaction costs, and latency in a live environment can erode theoretical profits.
# --------------------------



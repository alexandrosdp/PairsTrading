import ccxt
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
import datetime
import matplotlib.pyplot as plt

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


# --------------------------
# 4. Main Execution: Data Gathering, Pair Screening, and Backtesting
# --------------------------

if __name__ == "__main__":
    # Define a list of symbols; adjust or extend as needed.
    # symbols = ["BTC/USDT", "ETH/USDT", "XRP/USDT", "LTC/USDT", "BCH/USDT"]
    
    # # Get aligned historical price data (last 365 days)
    # print("Fetching and aligning data...")
    # prices = get_aligned_data(symbols, timeframe='1d', limit=365)
    # if prices.empty:
    #     print("No data fetched. Please check your API connection or symbols.")
    #     exit()
    # else:
    #     prices.to_csv("data/prices_data.csv")

    categories = {
    'Payment':  ['BTC/USDT','LTC/USDT','BCH/USDT'],
    'Layer1':   ['ETH/USDT','ADA/USDT','SOL/USDT'],
    'DeFi':     ['UNI/USDT','AAVE/USDT','COMP/USDT'],
    # etc.
}

# Now fetch data category-by-category
category_dataframes = get_data_by_category(categories, timeframe='1d', limit=365)

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



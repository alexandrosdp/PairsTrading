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


# --------------------------
# 4. Main Execution: Data Gathering, Pair Screening, and Backtesting
# --------------------------

if __name__ == "__main__":
    # Define a list of symbols; adjust or extend as needed.
    symbols = ["BTC/USDT", "ETH/USDT", "XRP/USDT", "LTC/USDT", "BCH/USDT"]
    
    # Get aligned historical price data (last 365 days)
    print("Fetching and aligning data...")
    prices = get_aligned_data(symbols, timeframe='1d', limit=365)
    if prices.empty:
        print("No data fetched. Please check your API connection or symbols.")
        exit()
    else:
        prices.to_csv("data/prices_data.csv")

    
    
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



# -----------------------------------------------------------------------------
# NOTES ON THE ADFULLER FUNCTION
# -----------------------------------------------------------------------------
#
# The adfuller function is used to test whether a time series has a unit root,
# i.e., whether it is non-stationary.
#
# Under the hood, adfuller performs the following steps:
#
# 1. Data Validation & Preparation:
#    - Checks the input series (y) for sufficient length and handles missing values.
#
# 2. Differencing:
#    - Computes the first difference: Δy_t = y_t - y_(t-1).
#
# 3. Model Specification:
#    - Sets up the regression model:
#         Δy_t = α + γ * y_(t-1) + Σ (δ_i * Δy_(t-i)) + ε_t
#      where:
#         α is the constant (if included),
#         γ is the coefficient for the lagged level (y_(t-1)),
#         δ_i are coefficients for the lagged differences (i = 1, 2, ..., p).
#
# 4. Lag Selection:
#    - Determines the number of lagged difference terms to include.
#      If autolag is enabled (e.g., 'AIC'), it selects the lag order that minimizes
#      an information criterion. Otherwise, it uses the provided maxlag.
#
# 5. OLS Regression:
#    - Performs an Ordinary Least Squares (OLS) regression on the model to estimate
#      the coefficients.
#
# 6. Extraction of Test Statistic:
#    - Extracts the t-statistic for the coefficient γ (on y_(t-1)).
#      Under the null hypothesis (γ = 0), the series has a unit root.
#
# 7. Statistical Inference:
#    - Computes the p-value using the distribution of the test statistic under the null.
#    - Provides critical values at common significance levels (e.g., 1%, 5%, 10%).
#
# 8. Return:
#    - Returns a tuple containing:
#         (test statistic, p-value, number of lags used, number of observations,
#          dictionary of critical values, maximized information criterion if applicable)
#
# Example usage:
# result = ts.adfuller(y, maxlag=None, autolag='AIC')


# -----------------------------------------------------------------------------
# NOTES ON THE CINT FUNCTION (Engle-Granger Cointegration Test)
# -----------------------------------------------------------------------------
#
# The coint function tests for cointegration between two time series. The process
# is known as the Engle-Granger two-step method.
#
# Under the hood, coint performs the following steps:
#
# 1. Regression (Long-Run Relationship):
#    - It runs an OLS regression between the two series, typically of the form:
#         y_t = α + β * x_t + ε_t
#
# 2. Residual Extraction:
#    - Calculates the residuals (ε_t) from the regression.
#      These residuals represent the spread between the two series.
#
# 3. Stationarity Testing:
#    - Applies the Augmented Dickey-Fuller (ADF) test on the residuals.
#      The ADF test checks if the residuals are stationary.
#
# 4. Interpretation:
#    - If the residuals are stationary (i.e., the ADF test rejects the null of a unit root),
#      the two series are cointegrated, meaning they share a long-run equilibrium relationship.
#
# 5. Return:
#    - Returns a tuple containing:
#         (ADF test statistic for the residuals, p-value, and additional diagnostic values)
#
# Example usage:
# score, pvalue, _ = ts.coint(series1, series2)

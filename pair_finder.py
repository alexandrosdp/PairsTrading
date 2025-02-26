import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller

import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox

from cointegration_testing import coint_test_modified


# --------------------------
# Pre-Screening & Cointegration Testing
# --------------------------

#TODO: Add Sector Categorization 
#TODO: Add community detection


def create_category_dfs(prices, categories):
    """
    Create a dictionary of DataFrames, where each DataFrame contains the closing prices of assets in a given category.
    
    Parameters:
        prices (pd.DataFrame): DataFrame where each column is a time series of closing prices for a cryptocurrency.
        categories (dict): Dictionary mapping asset symbols to their corresponding categories.
    
    Returns:
        dict: Dictionary of DataFrames, where each key is a category and each value is a DataFrame of closing prices.
    """
    category_dfs = {}
    for category in set(categories.values()):
        category_assets = [symbol for symbol, cat in categories.items() if cat == category]
        category_dfs[category] = prices[category_assets]
        
    return category_dfs


def filter_high_correlation_pairs(prices, threshold=0.8):
    """
    Calculate the correlation matrix for each cryptocurrency time series and select pairs with a correlation above a specified threshold.
    
    Parameters:
        prices (pd.DataFrame): DataFrame where each column is a time series of closing prices for a cryptocurrency.
        threshold (float, optional): Minimum correlation (0 to 1) to consider a pair for cointegration testing. Default is 0.8.
    
    Returns:
        list: List of tuples (symbol1, symbol2, correlation) for pairs with correlation >= threshold.
    """
    corr_matrix = prices.corr()
    pairs = []
    symbols = corr_matrix.columns
    for i in range(len(symbols)):
        for j in range(i+1, len(symbols)):
            corr_val = corr_matrix.iloc[i, j]
            # Using absolute correlation in case of negative correlations, though typically we expect positive correlation.
            if abs(corr_val) >= threshold:
                pairs.append((symbols[i], symbols[j], corr_val))
    return corr_matrix, pairs



# Helper function: splits the price DataFrame into non-overlapping windows.
def split_price_series_into_windows(prices, window_size):
    """
    Splits the price time series into non-overlapping windows of a specified size.
    
    Parameters:
        prices (pd.DataFrame): A DataFrame containing price data, where each column represents
                               a cryptocurrency and the index is a datetime index.
        window_size (int): Number of data points (rows) to include in each window.
    
    Returns:
        list: A list of pd.DataFrame objects, each representing one window of the original data.
    """
    windows = []
    n = len(prices)
    num_windows = n // window_size  # Only complete windows are returned
    
    for i in range(num_windows):
        start_idx = i * window_size             #Eg if windows size = 720 ---> 0 - 720, 720 - 1440, 1440 - 2160
        end_idx = start_idx + window_size
        window = prices.iloc[start_idx:end_idx]
        windows.append(window)
    
    return windows

# Modified cointegration test function that uses multiple windows.
def find_cointegrated_pairs_windows(prices, high_corr_pairs, significance=0.05, window_size=720, min_pass_fraction=0.5):
    """
    Check high-correlation pairs for cointegration across multiple time windows.
    
    The full timeframe (e.g. 1 year of hourly data) is split into smaller windows using the 
    provided window_size. For each candidate pair (from high_corr_pairs), the cointegration test 
    is applied in each window (using coint_test_modified), and the fraction of windows where the 
    p-value is below the significance level is recorded.
    
    Pairs that pass the cointegration test in at least min_pass_fraction of windows are selected.
    The pairs are then ranked by the fraction of windows passed (descending) and by the average
    p-value (ascending) as a tie-breaker.
    
    Parameters:
        prices (pd.DataFrame): DataFrame containing hourly closing prices, with each column representing a symbol.
        high_corr_pairs (list): List of tuples (sym1, sym2, correlation) for high-correlation pairs to test.
        significance (float, optional): Significance level for the cointegration test. Default is 0.05.
        window_size (int, optional): Number of data points per window (e.g., 720 for roughly one month of hourly data).
        min_pass_fraction (float, optional): Minimum fraction of windows in which the cointegration test must pass.
    
    Returns:
        tuple: A tuple containing:
            - cointegrated_pairs (list): List of tuples (sym1, sym2, pass_fraction, avg_pvalue, correlation) for pairs that pass cointegration in enough windows.
            - window_results (dict): A dictionary with keys as (sym1, sym2) and values as the list of p-values across windows.
    """
    # Split the full price DataFrame into windows
    windows = split_price_series_into_windows(prices, window_size)
    num_windows = len(windows)
    
    cointegrated_pairs = []
    window_results = {}
    
    # Loop through each candidate pair from the high-correlation list.
    for sym1, sym2, corr_val in high_corr_pairs:
        pvalues = []
        # Loop through each window and perform the cointegration test.
        for window in windows:
            try:
                S1 = window[sym1]
                S2 = window[sym2]
                # Use your modified cointegration test function.
                # It is assumed to return at least a p-value.
                pvalue, _ = coint_test_modified(S1, S2)
            except Exception as e:
                pvalue = np.nan
            pvalues.append(pvalue)
        
        pvalues_array = np.array(pvalues) #This creates a Boolean array (valid) that is True for every element in pvalues_array that is a finite number (i.e., not NaN or inf). This is important because sometimes a cointegration test might fail or return an invalid result, and you don’t want those to skew your calculations.
        valid = np.isfinite(pvalues_array)
        if valid.sum() == 0:
            pass_fraction = 0
            avg_pvalue = np.nan
        else:
            pass_count = np.sum(pvalues_array[valid] < significance)
            pass_fraction = pass_count / valid.sum()
            avg_pvalue = np.nanmean(pvalues_array[valid])
        
        window_results[(sym1, sym2)] = pvalues
        
        # Select pairs that pass the cointegration test in a sufficient fraction of windows.
        if pass_fraction >= min_pass_fraction:
            cointegrated_pairs.append((sym1, sym2, pass_fraction, avg_pvalue, corr_val))
    
    # Rank the pairs: first by descending pass_fraction, then by ascending average p-value.
    cointegrated_pairs.sort(key=lambda x: (-x[2], x[3]))
    
    # Optionally, print out the results.
    if cointegrated_pairs:
        print("\nCointegrated pairs (across windows):")
        for pair in cointegrated_pairs:
            print(f"{pair[0]} & {pair[1]}: pass fraction = {pair[2]:.2f}, avg p-value = {pair[3]:.4f}, correlation = {pair[4]:.4f}")
    else:
        print("\nNo cointegrated pairs found across the windows.")
    
    return cointegrated_pairs, window_results

# Example usage:
# Assuming `prices` is your DataFrame with 1 year of hourly data and 
# `high_corr_pairs` is a list of candidate pairs (sym1, sym2, correlation)
# For instance:
# high_corr_pairs = [('BTC/USDT', 'ETH/USDT', 0.9), ('XRP/USDT', 'LTC/USDT', 0.85), ... ]
# cointegrated_pairs, window_results = find_cointegrated_pairs(prices, high_corr_pairs, significance=0.05, window_size=720, min_pass_fraction=0.5)




#Normal cointegration method (not using windows)
def find_cointegrated_pairs(prices, high_corr_pairs, significance=0.05):
    """
    Check all pairs of assets for cointegration using the Engle-Granger two-step method.
    
    Parameters:
        prices (pd.DataFrame): DataFrame containing closing prices of assets with each column representing a symbol.
        high_corr_pairs (list): List of tuples (symbol1, symbol2, correlation) for high correlation pairs to test.
        significance (float, optional): Significance level for the cointegration test. Default is 0.05.
    
    Returns:
        tuple: A tuple containing:
            - cointegrated_pairs (list): List of tuples (symbol1, symbol2, p-value, correlation) for pairs 
              where the p-value is below the significance level.
            - pvalue_matrix (pd.DataFrame): DataFrame of p-values for all tested pairs, with coin names as index and columns.
            - residuals_df (pd.DataFrame): DataFrame storing the residuals for each cointegrated pair.
    """
    n = prices.shape[1]
    keys = prices.columns
    # Create a numpy array to store p-values
    pvalue_matrix = np.ones((n, n))

    # Create a dataframe to store the residuals for each cointegrated pair
    residuals_df = pd.DataFrame()

    cointegrated_pairs = []

    for sym1, sym2, corr_val in high_corr_pairs:
        S1 = prices[sym1]
        S2 = prices[sym2]
        pvalue, res_adf = coint_test_modified(S1, S2)

        # Get the index positions of the coins
        i = keys.get_loc(sym1)
        j = keys.get_loc(sym2)
        pvalue_matrix[i, j] = pvalue
        pvalue_matrix[j, i] = pvalue
        
        # If the p-value is less than the significance level, consider the pair cointegrated.
        if pvalue < significance:
            cointegrated_pairs.append((sym1, sym2, pvalue, corr_val))
            # Store the residuals for each cointegrated pair in the dataframe
            residuals_df[sym1 + '_' + sym2] = res_adf

    # Convert the numpy p-value matrix to a DataFrame with appropriate labels.
    pvalue_matrix_df = pd.DataFrame(pvalue_matrix, index=keys, columns=keys)
    
    if cointegrated_pairs:
        print("\nCointegrated pairs (from pre-filtered high-correlation pairs):")
        for pair in cointegrated_pairs:
            print(f"{pair[0]} & {pair[1]}: p-value = {pair[2]:.4f}, correlation = {pair[3]:.4f}")
    else:
        print("\nNo cointegrated pairs found among the high-correlation pairs.")

    return cointegrated_pairs, pvalue_matrix_df, residuals_df

def analyze_residuals(residuals_df, lags):

    """
    Perform Ljung-Box test on the residuals of each cointegrated pair to check for autocorrelation.

    Parameters:
        residuals_df (pd.DataFrame): DataFrame containing residuals for each cointegrated pair.
        lags (int): Number of lags to include in the Ljung-Box test (depends on the time series frequency).
    """

    

    # Plot ACF for each pair's residuals

    # for pair in residuals_df.columns:
    #     plot_acf(residuals_df[pair], lags=20, title=f"ACF of Residuals for {pair}")
    #     plt.show()

    # Perform Ljung-Box test on the residuals for each cointegrated pair
    for pair in residuals_df.columns:
        lb_test = acorr_ljungbox(residuals_df[pair], lags=lags)
        p_value = lb_test['lb_pvalue'].values[0]
        if p_value > 0.05:
            print(f"P-value for Ljung-Box test for pair {pair}: {p_value}")
            print(f"Residuals of pair {pair} are likely white noise (independent).")
        else:
            print(f"P-value for Ljung-Box test for pair {pair}: {p_value}")
            print(f"Residuals of pair {pair} are not white noise (may have autocorrelation).")


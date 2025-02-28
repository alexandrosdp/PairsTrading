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


def find_cointegrated_pairs_windows(prices, high_corr_pairs=None, significance=0.05, window_size=720, min_pass_fraction=0.5):
    
    """
    Check pairs of symbols for cointegration across multiple time windows.
    
    1. If high_corr_pairs is None or empty, the function will check *all* possible 
       pair combinations from the columns of 'prices'.
    2. Otherwise, it will only check the candidate pairs in high_corr_pairs (which 
       might have come from a high-correlation filter).
    3. The full timeframe is split into windows (via your split_price_series_into_windows 
       helper) of size 'window_size'.
    4. For each pair, we apply the coint_test_modified in each window and track 
       how many windows pass the significance threshold (p-value < significance).
    5. Pairs that pass in at least 'min_pass_fraction' of valid windows are retained.

    Parameters:
        prices (pd.DataFrame): DataFrame of prices, each column is a symbol.
        high_corr_pairs (list or None): List of (sym1, sym2, correlation), or None/empty to test all pairs.
        significance (float): Significance level for cointegration test. Default 0.05.
        window_size (int): Number of data points (rows) per window. e.g. 720 for 30 days of hourly data.
        min_pass_fraction (float): Min fraction of windows where p-value < significance.

    Returns:
        tuple: (cointegrated_pairs, window_results)
            - cointegrated_pairs (list): 
                [ (sym1, sym2, pass_fraction, avg_pvalue, correlation), ... ]
            - window_results (dict):
                { (sym1, sym2): [pvalue_window1, pvalue_window2, ...], ... }
    """
    # 1) If 'high_corr_pairs' is None or empty, build a list of *all possible* pair combos from prices.columns
    if not high_corr_pairs:  # covers None or empty list
        symbols = prices.columns
        high_corr_pairs = []
        for i in range(len(symbols)):
            for j in range(i+1, len(symbols)):
                # We'll set correlation to None (or 0) since we don't know it
                high_corr_pairs.append((symbols[i], symbols[j], None))

    # 2) Split the full data into windows
    windows = split_price_series_into_windows(prices, window_size)
    num_windows = len(windows)
    
    cointegrated_pairs = []
    window_results = {}
    
    # 3) Loop through each pair and test cointegration in each window
    for sym1, sym2, corr_val in high_corr_pairs:
        pvalues = []
        
        for window_df in windows:
            try:
                S1 = window_df[sym1]
                S2 = window_df[sym2]
                # Modified cointegration test that returns (pvalue, <other info>)
                pvalue, _ = coint_test_modified(S1, S2)
            except Exception:
                pvalue = np.nan
            pvalues.append(pvalue)
        
        pvalues_array = np.array(pvalues)
        valid = np.isfinite(pvalues_array)  # windows with valid p-values
        if valid.sum() == 0:
            pass_fraction = 0.0
            avg_pvalue = np.nan
        else:
            pass_count = np.sum(pvalues_array[valid] < significance)
            pass_fraction = pass_count / valid.sum()
            avg_pvalue = np.nanmean(pvalues_array[valid])
        
        # Store all window p-values for reference
        window_results[(sym1, sym2)] = pvalues
        
        # 4) Keep pairs that pass in enough windows
        if pass_fraction >= min_pass_fraction:
            # If correlation was None, we can just store 0 or None here
            cval = corr_val if corr_val is not None else 0.0
            cointegrated_pairs.append((sym1, sym2, pass_fraction, avg_pvalue, cval))
    
    # 5) Sort results: pass_fraction desc, then avg_pvalue asc
    cointegrated_pairs.sort(key=lambda x: (-x[2], x[3]))
    
    # Optional: Print results
    if cointegrated_pairs:
        print("\nCointegrated pairs (across windows):")
        for pair in cointegrated_pairs:
            print(f"{pair[0]} & {pair[1]}: pass fraction={pair[2]:.2f}, avg p-value={pair[3]:.4f}, correlation={pair[4]}")
    else:
        print("\nNo cointegrated pairs found across the windows.")
    
    return cointegrated_pairs, window_results

# Example usage:
# Assuming `prices` is your DataFrame with 1 year of hourly data and 
# `high_corr_pairs` is a list of candidate pairs (sym1, sym2, correlation)
# For instance:
# high_corr_pairs = [('BTC/USDT', 'ETH/USDT', 0.9), ('XRP/USDT', 'LTC/USDT', 0.85), ... ]
# cointegrated_pairs, window_results = find_cointegrated_pairs(prices, high_corr_pairs, significance=0.05, window_size=720, min_pass_fraction=0.5)


def plot_spread_in_windows(sym1, sym2, windows, window_results, significance=0.05):
    """
    Plots the spread for each rolling/chunked window for a given pair (sym1, sym2),
    labeling each subplot with the coint test p-value for that window (obtained
    from the window_results dictionary).

    Arguments:
        sym1 (str): Symbol name for the first series
        sym2 (str): Symbol name for the second series
        windows (list of pd.DataFrame): List of windowed data (e.g., from
            split_price_series_into_windows). Each window_df should contain
            columns [sym1, sym2].
        window_results (dict): Dictionary returned by find_cointegrated_pairs_windows,
            typically under window_results[(sym1, sym2)] = [pval_win1, pval_win2, ...].
            The length of this list should match the number of windows.
        significance (float): Significance level. Used to highlight if p-value < significance.

    Returns:
        None. Displays a figure with subplots (one per window).
    """

    n_windows = len(windows)
    if n_windows == 0:
        print("No windows to plot.")
        return

    # Fetch the list of p-values from window_results for this specific pair
    pvalues = window_results.get((sym1, sym2), [])
    if len(pvalues) != n_windows:
        print(f"Warning: Number of p-values ({len(pvalues)}) does not match number of windows ({n_windows}).")
        print("Plotting will proceed, but alignment may be incorrect.")
    
    # Create subplots (one per window)
    fig, axes = plt.subplots(n_windows, 1, figsize=(10, 3 * n_windows), sharex=False)

    # If there's only 1 window, axes is not an array, so wrap it
    if n_windows == 1:
        axes = [axes]

    for i, window_df in enumerate(windows):
        ax = axes[i]
        # Extract the series for the current window
        S1 = window_df[sym1].dropna()
        S2 = window_df[sym2].dropna()

        # Align them just in case (inner join on the index)
        S1, S2 = S1.align(S2, join='inner')

        # Quick OLS to get alpha, beta for this window
        # OLS: S1 = alpha + beta * S2
        X = sm.add_constant(S2)
        model = sm.OLS(S1, X, missing='drop').fit()
        alpha = model.params['const']
        beta = model.params[S2.name]

        # Spread: S1 - alpha - beta*S2
        spread = S1 - alpha - beta * S2

        # Plot the spread
        ax.plot(spread.index, spread, label=f"Spread (Window {i+1})")

        # Horizontal line at mean
        mean_spread = spread.mean()
        ax.axhline(mean_spread, color='r', linestyle='--', label='Mean')

        # Retrieve p-value for this window if it exists
        if i < len(pvalues):
            pval = pvalues[i]
        else:
            pval = np.nan

        below_signif = (pval < significance)
        if np.isfinite(pval):
            coint_label = f"p={pval:.4g} (<{significance})" if below_signif else f"p={pval:.4g}"
        else:
            coint_label = "p=NaN"

        ax.set_title(f"Window {i+1}: {sym1} & {sym2} | coint {coint_label}")
        ax.legend(loc='best')

    plt.tight_layout()
    plt.show()




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


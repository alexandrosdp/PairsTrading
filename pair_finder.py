import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller

import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox

from cointegration_testing import coint_test_modified


# --------------------------
# 2. Pre-Screening & Cointegration Testing
# --------------------------

#TODO: Add Sector Categorization 
#TODO: Add community detection


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



def find_cointegrated_pairs(prices,high_corr_pairs, significance=0.05):
    """
    Check all pairs of assets for cointegration using the Engle-Granger two-step method.
    
    Parameters:
        prices (pd.DataFrame): DataFrame containing closing prices of assets with each column representing a symbol.
        significance (float, optional): Significance level for the cointegration test. Default is 0.05.
    
    Returns:
        tuple: A tuple containing:
            - pairs (list): List of tuples (symbol1, symbol2, p-value) for pairs where the p-value is below the significance level.
            - pvalue_matrix (np.ndarray): Matrix of p-values for all tested pairs.
    """
    n = prices.shape[1]
    keys = prices.columns
    pvalue_matrix = np.ones((n, n))

    #Create a dataframe to store the residuals for each cointegrated pair
    residuals_df = pd.DataFrame()

    cointegrated_pairs = []

    for sym1, sym2, corr_val in high_corr_pairs:
        S1 = prices[sym1]
        S2 = prices[sym2]
        pvalue, res_adf = coint_test_modified(S1, S2)

        #Add p value to matrix
        i = keys.get_loc(sym1)
        j = keys.get_loc(sym2)
        pvalue_matrix[i, j] = pvalue
        pvalue_matrix[j, i] = pvalue
        
        # If the p-value is less than the significance level, consider the pair cointegrated.
        if pvalue < significance:

            cointegrated_pairs.append((sym1, sym2, pvalue, corr_val))

            #Store the residuals for each cointegrated pair in the dataframe
            residuals_df[sym1 + '_' + sym2] = res_adf
            
            
    if cointegrated_pairs:
        print("\nCointegrated pairs (from pre-filtered high-correlation pairs):")
        for pair in cointegrated_pairs:
            print(f"{pair[0]} & {pair[1]}: p-value = {pair[2]:.4f}, correlation = {pair[3]:.4f}")
    else:
        print("\nNo cointegrated pairs found among the high-correlation pairs.")

    return cointegrated_pairs, pvalue_matrix, residuals_df

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


# --------------------------
# 3. Strategy Functions
# --------------------------

def compute_spread(S1, S2):
    """
    Compute the spread between two asset price series by regressing S1 on S2.
    
    Parameters:
        S1 (pd.Series): Time series data for the first asset.
        S2 (pd.Series): Time series data for the second asset.
    
    Returns:
        tuple: A tuple containing:
            - spread (pd.Series): The residual series (spread) obtained from the regression.
            - beta (float): The hedge ratio (slope coefficient) from the regression.
    """
    S2_const = sm.add_constant(S2)
    model = sm.OLS(S1, S2_const).fit()
    beta = model.params[1]
    spread = S1 - beta * S2
    return spread, beta



def backtest_pair(spread, entry_threshold=1.0, exit_threshold=0.0):
    """
    Generate trading signals based on the z-score of the spread and simulate positions over time.
    
    Parameters:
        spread (pd.Series): The spread series between two asset prices.
        entry_threshold (float, optional): The z-score level at which to enter a position. Default is 1.0.
        exit_threshold (float, optional): The z-score level at which to exit a position. Default is 0.0.
    
    Returns:
        tuple: A tuple containing:
            - zscore (pd.Series): The z-score of the spread.
            - positions (pd.Series): Series of trading positions over time 
              (1 for long spread, -1 for short spread, 0 for no position).
    """
    # Compute z-score of the spread
    spread_mean = spread.mean()
    spread_std = spread.std()
    zscore = (spread - spread_mean) / spread_std
    
    positions = []
    position = 0  # 1 for long spread, -1 for short spread, 0 for no position.
    for z in zscore:
        # Entry conditions: enter short if z > entry_threshold, long if z < -entry_threshold.
        if position == 0:
            if z > entry_threshold:
                position = -1
            elif z < -entry_threshold:
                position = 1
        # Exit conditions: exit when the z-score reverts close to 0.
        elif position == 1 and z >= -exit_threshold:
            position = 0
        elif position == -1 and z <= exit_threshold:
            position = 0
        positions.append(position)
    positions = pd.Series(positions, index=spread.index)
    return zscore, positions

def simulate_strategy(S1, S2, positions, beta):
    """
    Simulate the performance of a pairs trading strategy based on the computed positions.
    
    Parameters:
        S1 (pd.Series): Time series data for the first asset.
        S2 (pd.Series): Time series data for the second asset.
        positions (pd.Series): Series of trading positions generated from the backtest (aligned with spread timestamps).
        beta (float): Hedge ratio computed from the cointegrating regression.
    
    Returns:
        tuple: A tuple containing:
            - pnl (pd.Series): Daily profit and loss (PnL) of the strategy.
            - cum_pnl (pd.Series): Cumulative PnL over time.
    """
    # Recompute spread using the hedge ratio
    spread, _ = compute_spread(S1, S2)
    # Daily change in spread (price difference)
    spread_diff = spread.diff()
    # Calculate PnL: shift positions to avoid lookahead bias
    pnl = -positions.shift(1) * spread_diff
    pnl = pnl.fillna(0)
    cum_pnl = pnl.cumsum()
    return pnl, cum_pnl

# --------------------------
# 4. Main Execution: Data Gathering, Pair Screening, and Backtesting
# --------------------------
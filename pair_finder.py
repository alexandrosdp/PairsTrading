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

def analyze_adf_residuals_from_adfuller(y, maxlag=None, autolag='AIC', regression='c', lb_lags=10, plot=True):
    """
    Analyze the residuals produced by the ADF regression using adfuller, ensuring that the 
    same parameters (maxlag, autolag, regression) are used as in the cointegration test (coint).
    
    The function runs the ADF test on the series `y` using:
        ts.adfuller(y, maxlag=maxlag, autolag=autolag, regression=regression, regresults=True)
    and then extracts the regression results (which include the residuals). It then
    analyzes the residuals for autocorrelation using an ACF plot, the Durbin-Watson statistic, 
    and the Ljung-Box test.
    
    Parameters:
        y (pd.Series): The input time series.
        maxlag (int, optional): Maximum number of lags to include. Default is None.
        autolag (str, optional): Method for selecting the lag length (e.g., 'AIC'). Default is 'AIC'.
        regression (str, optional): Type of regression/trend to include (e.g., 'c' for constant). Default is 'c'.
        lb_lags (int, optional): Number of lags to use in the Ljung-Box test. Default is 10.
        plot (bool, optional): Whether to display an ACF plot of the residuals. Default is True.
    
    Returns:
        model: The regression results object from adfuller.
        residuals (pd.Series): Residuals of the test regression.
        dw_stat (float): The Durbin-Watson statistic for the residuals.
        lb_test (pd.DataFrame): Results of the Ljung-Box test.
    """
    # Run adfuller with regresults=True so that the underlying regression results are returned.
    adf_result = adfuller(y, maxlag=maxlag, autolag=autolag, regression=regression, regresults=True)
    # The adfuller result tuple is:
    # (test_statistic, p-value, usedlag, nobs, critical_values, icbest, regresults)
    reg_results = adf_result[6]  # This is the regression results object.
    residuals = reg_results.resid
    
    # Plot the Autocorrelation Function (ACF) of the residuals if requested.
    if plot:
        plt.figure(figsize=(10, 4))
        plot_acf(residuals, lags=20)
        plt.title("ACF of ADF Regression Residuals")
        plt.show()
    
    # Compute the Durbin-Watson statistic to quantify autocorrelation in the residuals.
    dw_stat = sm.stats.stattools.durbin_watson(residuals)
    
    # Perform the Ljung-Box test on the residuals.
    lb_test = acorr_ljungbox(residuals, lags=[lb_lags], return_df=True)
    
    return reg_results, residuals, dw_stat, lb_test




def find_cointegrated_pairs(prices, significance=0.05):
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
    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            S1 = prices[keys[i]]
            S2 = prices[keys[j]]
            score, pvalue, _ = coint(S1, S2)
            pvalue_matrix[i, j] = pvalue
            if pvalue < significance:
                pairs.append((keys[i], keys[j], pvalue))

    return pairs, pvalue_matrix


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
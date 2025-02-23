import pandas as pd
import numpy as np
import statsmodels.api as sm
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
    pnl = positions.shift(1) * spread_diff #The pnl at time t is the position at time t-1 times the change in spread from t-1 to t
    pnl = pnl.fillna(0)
    cum_pnl = pnl.cumsum()
    return pnl, cum_pnl


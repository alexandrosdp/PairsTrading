import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
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

    #Note: 
    # Although the full regression model is S1 = α + βS2 + ϵ, this function computes the spread without including the intercept α. In many cointegration or pairs trading setups, the focus is on the relative movement captured by β.

    S2_const = sm.add_constant(S2) #Adds a constant to the independent variable
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

    #Count long and short positions

    longs = 0
    shorts = 0

    for z in zscore:
        # Entry conditions: enter short if z > entry_threshold, long if z < -entry_threshold.
        if position == 0:
            if z > entry_threshold:
                position = -1
                shorts += 1
            elif z < -entry_threshold:
                position = 1
                longs += 1
        # Exit conditions: exit when the z-score reverts close to 0.
        elif position == 1 and z >= -exit_threshold:
            position = 0
        elif position == -1 and z <= exit_threshold:
            position = 0
        positions.append(position)
    positions = pd.Series(positions, index=spread.index)
    
    return zscore, positions, longs, shorts

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

def plot_trading_simulation(S1, S2, sym1,sym2,zscore, positions, cum_pnl): 
        
        """
        Plot the trading simulation results including stock prices, z-score, trading positions, and cumulative PnL.

        Parameters:
            S1 (pd.Series): Time series data for the first asset.
            S2 (pd.Series): Time series data for the second asset.
            zscore (pd.Series): The z-score of the spread.
            positions (pd.Series): Series of trading positions generated from the backtest.
            cum_pnl (pd.Series): Cumulative profit and loss (PnL) of the strategy.
        """

    
        # Visualization
        plt.figure(figsize=(12, 15))

        plt.subplot(5, 1, 1)

        #Plot s1 and s2 with separate y-axis
        
        ax1 = plt.gca()
        ax2 = ax1.twinx()
        ax1.plot(S1, label=sym1, color='blue')
        ax2.plot(S2, label=sym2, color='red')
        ax1.set_ylabel(sym1, color='blue')
        ax2.set_ylabel(sym2, color='red')
        plt.title(f"Stock Prices: {sym1} and {sym2}")
        plt.legend()


    
        plt.subplot(5, 1, 2)
        plt.plot(zscore, label='Z-Score')
        plt.axhline(0, color='grey', linestyle='--', label='Mean')
        plt.axhline(1.0, color='red', linestyle='--', label='Upper threshold')
        plt.axhline(-1.0, color='green', linestyle='--', label='Lower threshold')
        plt.title("Z-Score of Spread")
        plt.legend()
        
        
        plt.subplot(5, 1, 3)
        plt.plot(positions, label='Positions', drawstyle='steps-mid')
        plt.title("Trading Positions")
        plt.legend()

        # plt.tight_layout()
        # plt.show()

    
        plt.subplot(5, 1, 4)
        plt.plot(cum_pnl, label='Cumulative PnL')
        plt.title("Strategy Performance (Cumulative PnL)")
        plt.legend()
    
        plt.tight_layout()
        plt.show()


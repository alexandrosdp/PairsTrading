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

def compute_spread_series(S1, S2, window_size=None):
    """
    Compute the dynamic spread between two asset price series by performing rolling regressions
    to re-estimate the hedge ratio using sliding windows.
    
    The spread at time t is computed as:
        spread_t = S1_t - beta_t * S2_t,
    where beta_t is the hedge ratio estimated over the previous window_size observations.
    
    Parameters:
        S1 (pd.Series): Time series data for the first asset.
        S2 (pd.Series): Time series data for the second asset.
        window_size (int, optional): Number of observations to use in each rolling regression.
                                     If None, the entire series is used (resulting in a static hedge ratio).
    
    Returns:
        tuple: A tuple containing:
            - spread (pd.Series): The dynamic spread series.
            - beta_series (pd.Series): A series of hedge ratios (β) computed for each time t. For t values
              before a full window is available, the value will be NaN.
    """
    
    # If no window size is specified, use the entire series (static beta)
    if window_size is None:
        window_size = len(S1)
    
    # Prepare empty series for beta and spread, with the same index as S1
    beta_series = pd.Series(index=S1.index, dtype=float)
    spread_series = pd.Series(index=S1.index, dtype=float)
    
    # Loop over the time series, starting from the first time index where a full window is available.
    for t in range(window_size - 1, len(S1)):
        # Get the rolling window data ending at time t
        S1_window = S1.iloc[t - window_size + 1 : t + 1]
        S2_window = S2.iloc[t - window_size + 1 : t + 1]
        
        # Add a constant to S2 (as in your original regression) 
        X = sm.add_constant(S2_window)
        model = sm.OLS(S1_window, X).fit()
        beta_t = model.params[1]
        
        # Record the beta for the current time t
        beta_series.iloc[t] = beta_t
        # Compute the spread at time t using the dynamically estimated beta
        spread_series.iloc[t] = S1.iloc[t] - beta_t * S2.iloc[t]
    
    return spread_series, beta_series



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

def simulate_strategy_monetary_sl(S1, S2, positions, beta, account_balance=1000, risk_per_trade=100,
                                  entry_threshold_short=1.0, stop_loss_threshold_short=2.0,
                                  entry_threshold_long=-1.0, stop_loss_threshold_long=-2.0,
                                  window_for_std=20):
    """
    Simulate the monetary profit and loss (PnL) for a pairs trading strategy with a stop-loss rule based on
    a fixed threshold plus one standard deviation.

    For a short spread trade:
      - Entry is triggered when the spread reaches the upper threshold (e.g., z-score = 1).
      - At entry, the spread is assumed to be: 
             entry_spread = mean_spread_window + entry_threshold_short * std_spread_window.
      - The stop-loss is set at:
             stop_loss_price = mean_spread_window + stop_loss_threshold_short * std_spread_window.
      - The trade size is computed so that a move from entry_spread to stop_loss_price results in a loss equal to risk_per_trade.
    
    For a long spread trade (signal = +1):
      - Entry is triggered when the spread reaches the lower threshold (e.g., z-score = -1):
             entry_spread = mean_spread_window + entry_threshold_long * std_spread_window.
      - The stop-loss is set at:
             stop_loss_price = mean_spread_window + stop_loss_threshold_long * std_spread_window.
      - Trade size is computed similarly.

    Once a trade is active, the monetary PnL is computed period by period using the trade size and the
    change in the spread from the entry level. If the spread reaches the stop-loss level, the trade is closed and
    the loss is capped at -risk_per_trade.

    Parameters:
        S1 (pd.Series): Price series for asset 1.
        S2 (pd.Series): Price series for asset 2.
        positions (pd.Series): Trading signals (1 for long spread, -1 for short spread, 0 for no position), aligned with the price series.
        beta (float): Hedge ratio computed from the cointegrating regression.
        account_balance (float, optional): Total account balance (for reference). Default is 1000.
        risk_per_trade (float, optional): The maximum monetary loss allowed per trade (e.g., 100€). Default is 100.
        entry_threshold_short (float, optional): For a short trade, the z-score threshold at entry. Default is 1.0.
        stop_loss_threshold_short (float, optional): For a short trade, the z-score for the stop loss. Default is 2.0.
        entry_threshold_long (float, optional): For a long trade, the z-score threshold at entry. Default is -1.0.
        stop_loss_threshold_long (float, optional): For a long trade, the z-score for the stop loss. Default is -2.0.
        window_for_std (int, optional): Number of periods to compute the spread's rolling standard deviation at trade entry. Default is 20.
    
    Returns:
        tuple: A tuple containing:
            - pnl_series (pd.Series): Period-by-period monetary PnL.
            - cum_pnl (pd.Series): Cumulative monetary PnL over time.
    """

    # First, compute the spread using the provided beta.
    # (Note: This spread is computed over the full series.)
    spread_full = S1 - beta * S2
    
    pnl_series = pd.Series(0.0, index=S1.index)
    trade_state = None  # To store details of the active trade.
    
    for t in range(len(S1)):
        current_signal = positions.iloc[t]
        
        # No trade currently active: check if a trade is initiated.
        if trade_state is None:
            if current_signal != 0:
                # Record trade entry details.
                entry_idx = t
                entry_S1 = S1.iloc[t]
                entry_S2 = S2.iloc[t]
                
                # Compute the entry spread using the current data point.
                entry_spread = spread_full.iloc[t]
                
                # Determine the window to compute rolling statistics.
                window_start = max(0, t - window_for_std + 1)
                spread_window = spread_full.iloc[window_start:t+1]
                mean_window = spread_window.mean()
                std_window = spread_window.std()
                # To avoid division by zero
                if std_window == 0:
                    std_window = 1e-8
                
                # Based on the trade direction, set entry threshold and stop loss level in price units.
                if current_signal == -1:  # short trade: entry when spread is high.
                    # Entry price corresponds to: mean + entry_threshold_short * std.
                    desired_entry = mean_window + entry_threshold_short * std_window
                    # Stop loss is at: mean + stop_loss_threshold_short * std.
                    stop_loss_price = mean_window + stop_loss_threshold_short * std_window
                elif current_signal == 1:  # long trade: entry when spread is low.
                    desired_entry = mean_window + entry_threshold_long * std_window
                    stop_loss_price = mean_window + stop_loss_threshold_long * std_window
                
                # For simplicity, we assume the trade is executed at the observed spread.
                # (In practice, you might enforce that the entry_spread is close to desired_entry.)
                # Calculate trade size such that a move from desired_entry to stop_loss equals risk_per_trade.
                move_required = abs(stop_loss_price - desired_entry)
                trade_size = risk_per_trade / move_required if move_required != 0 else 0
                
                trade_state = {
                    'entry_index': t,
                    'entry_spread': entry_spread,
                    'trade_size': trade_size,
                    'direction': current_signal,  # +1 for long, -1 for short.
                    'stop_loss_price': stop_loss_price,
                    'desired_entry': desired_entry
                }
        else:
            # Trade is active.
            current_spread = spread_full.iloc[t]
            direction = trade_state['direction']
            trade_size = trade_state['trade_size']
            entry_spread = trade_state['entry_spread']
            stop_loss_price = trade_state['stop_loss_price']
            
            # Calculate current profit:
            # For a short trade (direction = -1), profit is: trade_size * (entry_spread - current_spread).
            # For a long trade (direction = 1), profit is: trade_size * (current_spread - entry_spread).
            if direction == -1:
                current_trade_pnl = trade_size * (entry_spread - current_spread)
                # Check if stop loss has been hit: for short trade, stop loss is triggered if current_spread >= stop_loss_price.
                stop_triggered = current_spread >= stop_loss_price
            else:  # direction == 1
                current_trade_pnl = trade_size * (current_spread - entry_spread)
                # For a long trade, stop loss is triggered if current_spread <= stop_loss_price.
                stop_triggered = current_spread <= stop_loss_price
            
            # Record incremental pnl for this period.
            # To avoid lookahead bias, assume the pnl realized in period t is the change from previous period.
            # Here we simply assign the current trade pnl difference.
            if 'prev_trade_pnl' not in trade_state:
                trade_state['prev_trade_pnl'] = 0.0
            pnl_increment = current_trade_pnl - trade_state['prev_trade_pnl']
            pnl_series.iloc[t] = pnl_increment
            trade_state['prev_trade_pnl'] = current_trade_pnl
            
            # If the stop loss is triggered, adjust pnl so that the loss equals -risk_per_trade.
            if stop_triggered:
                # Determine adjustment required.
                loss_excess = current_trade_pnl + risk_per_trade  # Note: current_trade_pnl should be negative at stop.
                pnl_series.iloc[t] -= loss_excess  # Adjust the incremental pnl so that total loss equals -risk_per_trade.
                # Close the trade.
                trade_state = None
            
            # Alternatively, if the position signal goes to 0, close the trade.
            if positions.iloc[t] == 0:
                trade_state = None

    cum_pnl = pnl_series.cumsum()
    return pnl_series, cum_pnl

# Example usage:
# Assume S1 and S2 are pandas Series of hourly prices for two cryptocurrencies,
# positions is the trading signals series generated by your backtest_pair function,
# and beta is computed by compute_spread.
#
# pnl_series, cum_pnl = simulate_strategy_monetary_sl(S1, S2, positions, beta, account_balance=1000, risk_per_trade=100)
# print("Monetary PnL:")
# print(pnl_series.tail())
# cum_pnl.plot(title="Cumulative Monetary PnL")




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


        # Identify trade entry points: where the position goes from 0 to a nonzero value. THIS WILL NOT COUNT TRADES THAT ARE OPENED ON THE FIRST DAY FROM TIME 0
        trade_entries = positions[(positions != 0) & (positions.shift(1) == 0)]
        # Separate long and short entries
        long_entries = trade_entries[trade_entries == 1]
        short_entries = trade_entries[trade_entries == -1]

        print(f"Long Entries: {len(long_entries)}, Short Entries: {len(short_entries)}")
        
            
        # Visualization
        plt.figure(figsize=(15, 20))

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
        plt.axhline(1.0, color='green', linestyle='--', label='Upper threshold')
        plt.axhline(-1.0, color='green', linestyle='--', label='Lower threshold')
        plt.axhline(3.0, color='red', linestyle='--', label='Upper SL')
        plt.axhline(-3.0, color='red', linestyle='--', label='Lower SL')


        plt.title("Z-Score of Spread")
        plt.legend()


        # Plot markers for trade entries:
        # For long entries (positions = +1), use green upward triangles.
        plt.scatter(long_entries.index, zscore.loc[long_entries.index], marker='^', 
                    color='green', s=100, label='Long Entry')
        # For short entries (positions = -1), use red downward triangles.
        plt.scatter(short_entries.index, zscore.loc[short_entries.index], marker='v', 
                    color='red', s=100, label='Short Entry')
        
        
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


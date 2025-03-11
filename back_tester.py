import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt


#-------------------------------------------------------------------------------
#                     Functions To Backtest Strategy
#-------------------------------------------------------------------------------



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
    alpha_series = pd.Series(index=S1.index, dtype=float)
    spread_series = pd.Series(index=S1.index, dtype=float)
    
    # Loop over the time series, starting from the first time index where a full window is available.
    for t in range(window_size, len(S1)): #Start from the first time index where a full window is available
        # Get the rolling window data ending at time t
        S1_window = S1.iloc[t - window_size: t]  # S1 data for the window
        S2_window = S2.iloc[t - window_size: t]  # S2 data for the window
        
        # Perform OLS regression to estimate the hedge ratio beta for the window
        X = sm.add_constant(S2_window)
        model = sm.OLS(S1_window, X).fit()

        #Record the intercept (alpha) and hedge ratio (beta) for the current window
        #Include alpha since Engle–Granger cointegration typically uses the intercept in the regression formula for the spread
        alpha_t = model.params.iloc[0]
        beta_t = model.params.iloc[1]
        
        # Record the beta for the current time t
        beta_series.iloc[t] = beta_t
        alpha_series.iloc[t] = alpha_t

        # Compute the spread at time t using the dynamically estimated beta
        spread_series.iloc[t] = S1.iloc[t] - alpha_t - beta_t * S2.iloc[t]

    return spread_series, beta_series, alpha_series

def compute_rolling_zscore(spread_series, window_size):
    
    """
    Compute the rolling z-score of a spread series using a moving window.
    
    The z-score at time t is calculated as:
        zscore_t = (spread_series_t - rolling_mean_t) / rolling_std_t,
    where rolling_mean_t and rolling_std_t are computed over the previous
    window_size observations.
    
    Parameters:
        spread_series (pd.Series): The spread series (e.g., output from compute_spread_series).
        window_size (int): The number of observations over which to compute the rolling mean
                           and standard deviation.
    
    Returns:
        tuple: A tuple containing:
            - zscore (pd.Series): The rolling z-score series.
            - rolling_mean (pd.Series): The rolling mean of the spread.
            - rolling_std (pd.Series): The rolling standard deviation of the spread.
    """

    #You can test the logic of this function using this dummy code:
    #-------------------------------------------------------------
    # prices = pd.Series([100, 105, 110, 120, 130], 
    #                index=pd.date_range("2024-03-01", periods=5))

    # # 3-day trailing average, not including the current day
    # rolling_avg = prices.rolling(window=3, closed="left",min_periods=3).mean()

    # df = pd.DataFrame({'price': prices, 'trailing_avg': rolling_avg})
    # print(df)
    #-------------------------------------------------------------


    # Compute the rolling mean and standard deviation
    rolling_mean = spread_series.rolling(window=window_size, closed = "left", min_periods=window_size).mean() #closed='left' excludes the current row from the window, avoiding using the current data point at time t in the calculation. min_periods=window_size ensures that the first window has at least window_size observations.
    rolling_std = spread_series.rolling(window=window_size,  closed = "left", min_periods=window_size).std()
    
    # Compute the rolling z-score
    zscore = (spread_series - rolling_mean) / rolling_std #The z-score at time t is calculated as: zscore_t = (spread_series_t - rolling_mean_t) / rolling_std_t
    
    #Ensure index of zscore is a datetime index
    zscore.index = pd.to_datetime(zscore.index)

    return zscore, rolling_mean, rolling_std

# Example usage:
# Assuming 'spread_series' is a pandas Series obtained from compute_spread_series(S1, S2, window_size)
# and you want to use a window of, say, 720 observations (e.g., roughly one month for hourly data):
# zscore_series, roll_mean, roll_std = compute_rolling_zscore(spread_series, window_size=720)

def backtest_pair_rolling(spread_series,S1,S2, zscore, entry_threshold=1.0, exit_threshold=0.1, stop_loss_threshold=2.0):
    """
    Generate trading signals based on the rolling z-score of the spread series using a moving window.
    
    The rolling z-score is computed using the provided window_size. The z-score at time t is:
        zscore_t = (spread_series_t - rolling_mean_t) / rolling_std_t,
    where rolling_mean_t and rolling_std_t are computed over the previous window_size observations.
    
    The trading rules are:
      - When no position is open (position == 0):
          * If zscore >= entry_threshold, enter a short spread (position = -1).
          * If zscore <= -entry_threshold, enter a long spread (position = 1).
      - When a position is open, exit (set position to 0) when the z-score reverts near 0:
          * For a long position (position = 1), exit when zscore >= -exit_threshold.
          * For a short position (position = -1), exit when zscore <= exit_threshold.
    
    Parameters:
        spread_series (pd.Series): The spread series between two asset prices.
        window_size (int): The number of observations to use in computing the rolling z-score.
        entry_threshold (float, optional): The z-score threshold for entering a trade. Default is 1.0.
        exit_threshold (float, optional): The z-score level for exiting a trade. Default is 0.0.
        stop_loss_threshold (float, optional): The z-score level for a stop-loss exit. Default is 2.0.
    
    Returns:
        tuple: A tuple containing:
            - positions (pd.Series): A series of trading signals over time 
              (1 for long spread, -1 for short spread, 0 for no position).
            -Win_indexs (list): A list of the indexs of winning trades
            -Loss_indexs (list): A list of the indexs of losing

    """
    
    positions = []
    position = 0  # 1 for long spread, -1 for short spread, 0 for no position.

    # win_count = 0 #Count of winning trades
    win_indexs = [] #Index of winning trades

    # loss_count = 0 #Count of losing trades
    loss_indexs = [] #Index of losing trades

    # When we get stopped out, we set stop_out=True and remain out until
    # |z| <= exit_threshold again
    stop_out = False

    #Create a list to track the price changes in each trafing period (from the start of the trade to the end)
    # Lists to track price changes for each leg.
    price_changes_S1 = [] #Percentage change in price of S1
    price_changes_S2 = [] #Percentage change in price of S2
    
    # Initialize entry prices.
    entry_price_S1 = None
    entry_price_S2 = None
    
    for t, z in enumerate(zscore):

        # If we don't have a valid z-score (e.g., before the window is full), remain flat.
        if pd.isna(z):
            #print("INVALID Z-SCORE DETECTED")
            positions.append(0)
            price_changes_S1.append(0)
            price_changes_S2.append(0)
            continue
        
        #If we are not currentlty in a position
        if position == 0:
                
                #If we are stopped out, we remain flat until |z| <= exit_threshold
                if stop_out:

                    if abs(z) <= exit_threshold:
        
                         # Once z <= exit_threshold, we allow new trades again
                        stop_out = False

                    # remain flat
                    positions.append(0)
                    price_changes_S1.append(0)
                    price_changes_S2.append(0)
                    continue
                    
                #If we are not stopped out, check for entry
                else:
                    # No position -> check entry
                    if z >= entry_threshold:
                        position = -1  # Short spread
                        
                        entry_price_S1 = S1.iloc[t]
                        entry_price_S2 = S2.iloc[t]

                    elif z <= -entry_threshold:
                        position = +1  # Long spread

                        entry_price_S1 = S1.iloc[t]
                        entry_price_S2 = S2.iloc[t]


        #If we are currently in a position
        else:

            if position == +1:  # Long spread
                # Normal exit condition => if z crosses above -exit_threshold
                if z >= -exit_threshold:
                    position = 0
                    win_indexs.append(zscore.index[t]) #Appends actual datetime index of the winning trade

                    # Record the exit prices for each leg
                    exit_price_S1 = S1.iloc[t]
                    exit_price_S2 = S2.iloc[t]
                    
                    #Record the price changes for each leg
                    pc_S1_percentage = (exit_price_S1 - entry_price_S1) / entry_price_S1
                    pc_S2_percentage = (entry_price_S2 - exit_price_S2) / entry_price_S2

                    price_changes_S1.append(pc_S1_percentage)
                    price_changes_S2.append(pc_S2_percentage)
                    entry_price_S1 = None
                    entry_price_S2 = None

                # Stop-loss => if z <= -stop_loss_threshold
                elif z <= -stop_loss_threshold:
                    position = 0
                    loss_indexs.append(zscore.index[t])

                    # Record the exit prices for each leg
                    exit_price_S1 = S1.iloc[t]
                    exit_price_S2 = S2.iloc[t]

                    #Record the price changes for each leg
                    pc_S1_percentage = (exit_price_S1 - entry_price_S1) / entry_price_S1
                    pc_S2_percentage = (entry_price_S2 - exit_price_S2) / entry_price_S2

                    price_changes_S1.append(pc_S1_percentage)
                    price_changes_S2.append(pc_S2_percentage)
                    
                    stop_out = True
                    entry_price_S1 = None
                    entry_price_S2 = None
                
                else:
                    price_changes_S1.append(0)
                    price_changes_S2.append(0)


            elif position == -1:  # Short spread
                # Normal exit => if z crosses below exit_threshold
                if z <= exit_threshold:
                    position = 0
                    win_indexs.append(zscore.index[t])
                    
                    # Record the exit prices for each leg
                    exit_price_S1 = S1.iloc[t]
                    exit_price_S2 = S2.iloc[t]

                    # For short spread: profit S1 = entry - exit, profit S2 = exit - entry.
                    pc_S1_percentage = (entry_price_S1 - exit_price_S1)/ entry_price_S1
                    pc_S2_percentage = (exit_price_S2 - entry_price_S2) / entry_price_S2

                    price_changes_S1.append(pc_S1_percentage)
                    price_changes_S2.append(pc_S2_percentage)
                    
                    entry_price_S1 = None
                    entry_price_S2 = None

                # Stop-loss => if z >= stop_loss_threshold
                elif z >= stop_loss_threshold:
                    position = 0
                    loss_indexs.append(zscore.index[t])

                    # Record the exit prices for each leg
                    exit_price_S1 = S1.iloc[t]
                    exit_price_S2 = S2.iloc[t]

                    # For short spread: profit S1 = entry - exit, profit S2 = exit - entry.
                    pc_S1_percentage = (entry_price_S1 - exit_price_S1)/ entry_price_S1
                    pc_S2_percentage = (exit_price_S2 - entry_price_S2) / entry_price_S2

                    price_changes_S1.append(pc_S1_percentage)
                    price_changes_S2.append(pc_S2_percentage)

                    stop_out = True

                    entry_price_S1 = None
                    entry_price_S2 = None


                else:
                    price_changes_S1.append(0)
                    price_changes_S2.append(0)
                    
        

        positions.append(position)

            
    positions = pd.Series(positions, index=spread_series.index)

    #Compute number of wins and losses
    num_wins = len(win_indexs)
    num_losses = len(loss_indexs)

    print(f"Total trades closed: {num_wins+num_losses} (Wins={num_wins}, Losses={num_losses})")
    print(f"Win rate: {num_wins/(num_wins+num_losses):.2f}")


    return positions, win_indexs, loss_indexs, price_changes_S1, price_changes_S2

# Example usage:
# Suppose 'spread_series' is a pandas Series representing the spread between two assets,
# and you wish to compute a rolling z-score over a window of 720 observations (e.g., roughly one month for hourly data).
# zscore_series, positions_series = backtest_pair_rolling(spread_series, window_size=720, entry_threshold=1.0, exit_threshold=0.0)



def simulate_true_strategy_rolling(S1, S2, positions, beta_series):
    """
    Simulate the true performance of a pairs trading strategy using a dynamic (rolling) hedge ratio.
    
    In this strategy:
      - A long spread (position = +1) means:
          * Long 1 unit of S1
          * Short beta_series[t] units of S2 at time t
      - A short spread (position = -1) means:
          * Short 1 unit of S1
          * Long beta_series[t] units of S2 at time t
    
    The profit or loss from a trade between time t-1 and t is computed as:
        For a long spread:
            pnl = (S1[t] - S1[t-1]) - beta_series[t-1] * (S2[t] - S2[t-1])
        For a short spread:
            pnl = -(S1[t] - S1[t-1]) + beta_series[t-1] * (S2[t] - S2[t-1])
    
    We use the previous period's position (i.e. positions.shift(1)) and the previous period's
    beta (i.e. beta_series.shift(1)) to avoid lookahead bias.
    
    Parameters:
        S1 (pd.Series): Time series of asset 1 prices.
        S2 (pd.Series): Time series of asset 2 prices.
        positions (pd.Series): Trading signals (1 for long spread, -1 for short spread, 0 for no position).
        beta_series (pd.Series): A time series of hedge ratios computed using a rolling regression.
    
    Returns:
        tuple: A tuple containing:
            - pnl (pd.Series): The period-by-period profit and loss of the strategy.
            - cum_pnl (pd.Series): The cumulative profit and loss over time.
    """
    # Calculate the changes in asset prices.
    delta_S1 = S1.diff()
    delta_S2 = S2.diff()
    
    # Shift positions and beta_series by one period to avoid lookahead bias.
    shifted_positions = positions.shift(1)
    shifted_beta = beta_series.shift(1)
    
    # Compute the period-by-period PnL using the dynamic beta.
    # For a long spread (position = 1), pnl = (ΔS1) - beta * (ΔS2).
    # For a short spread (position = -1), pnl = -(ΔS1) + beta * (ΔS2).
    pnl = shifted_positions * (delta_S1 - shifted_beta * delta_S2)
    pnl = pnl.fillna(0)
    
    # Compute cumulative PnL.
    cum_pnl = pnl.cumsum()
    
    return pnl, cum_pnl


# def simulate_strategy_pnl(
#     S1, 
#     S2, 
#     positions, 
#     beta_series=None, 
#     initial_capital=1000.0
# ):
#     """
#     Compute daily/cumulative PnL for a pairs strategy that uses a hedge ratio (beta).

#     If beta_series is None, we assume beta=1 at all times.
#     Otherwise, beta_series[t] is the ratio for day t: 
#         e.g. if position=+1, we buy shares in S1, 
#         short beta_series[t]*shares_of_S2.

#     Args:
#         S1 (pd.Series): Price series for S1
#         S2 (pd.Series): Price series for S2
#         positions (pd.Series): 0=flat, +1=long spread, -1=short spread
#         beta_series (pd.Series or float): The hedge ratio(s). If None => 1.0
#         initial_capital (float): e.g. 1000 EUR

#     Returns:
#         daily_pnl (pd.Series)
#         cum_pnl  (pd.Series)
#         cum_pnl_pct (pd.Series)
#     """

#     #The standard pairs logic: If β=2, it means “S1 moves 2 times as much as S2.” So to offset that effect, you hold 2 times as much notional in S2 as you do in S1.

#     # If no beta_series, use 1.0
#     if beta_series is None:
#         # Create a series of 1.0 with same index
#         beta_series = pd.Series(1.0, index=S1.index)

#     # Convert to arrays for speed if you prefer
#     daily_pnl = []

#     shares_S1_list = []
#     shares_S2_list = []

#     short_profit_list = []

#     # We'll iterate from i=1..end to compute daily PnL from i-1 -> i
#     for i in range(1, len(S1)):
#         prev_pos = positions.iloc[i-1]
#         if prev_pos == 0:
#             # no position => daily PnL=0
#             daily_pnl.append(0.0)
#             continue
        
#         # Price changes
#         dS1 = S1.iloc[i] - S1.iloc[i-1]
#         dS2 = S2.iloc[i] - S2.iloc[i-1]

#         # The ratio for day i-1
#         beta = beta_series.iloc[i-1]

#         # number of shares for each leg
#         # total capital is initial_capital. We won't re-invest daily
#         # Let's define a function that for a +1 spread,
#         # we put half capital in S1, half in S2, but scale S2 by beta.
#         # One approach:
#         #   Notional_S1 = initial_capital / (1+beta)
#         #   Notional_S2 = beta * Notional_S1 = (beta/(1+beta))*initial_capital
#         # Then shares_S1 = Notional_S1 / price_of_S1
#         #     shares_S2 = Notional_S2 / price_of_S2
#         # if pos=+1 => we are long S1, short S2
#         # if pos=-1 => we are short S1, long S2

#         # compute these for day i-1
#         #This determines how we allocate our capital between the two assets
#         Notional_S1 = initial_capital / (1.0 + beta)
#         Notional_S2 = beta * Notional_S1

#         # share counts, based on the prices from i-1
#         # We are using the prices from i-1 to determine the number of shares we would have bought/sold
#         shares_S1 = Notional_S1 / S1.iloc[i-1]
#         shares_S2 = Notional_S2 / S2.iloc[i-1]

#         shares_S1_list.append(shares_S1)
#         shares_S2_list.append(shares_S2)
        
#         # now the PnL depends on whether pos=+1 or -1
#         if prev_pos == +1:
#             # +1 => long S1 => daily PnL from S1 is shares_S1 * dS1
#             #        short S2 => daily PnL from S2 is shares_S2 * (-dS2)

#             # long_profit = shares_S1 * dS1
#             # short_profit = shares_S2 * (-dS2)
#             # day_pnl = long_profit + short_profit
            
#             day_pnl = shares_S1 * dS1 + shares_S2 * (-dS2)

            
#         else:
#             # prev_pos == -1 => short S1 => daily PnL from S1 => shares_S1 * (-dS1)
#             #                => long S2 => shares_S2 * dS2
#             # short_profit = shares_S1 * (-dS1)
#             # long_profit = shares_S2 * dS2
#             # day_pnl = short_profit + long_profit

#             day_pnl = shares_S1 * (-dS1) + shares_S2 * dS2
#            # day_pnl = shares_S1 * (-dS1) + shares_S2 * dS2

#         daily_pnl.append(day_pnl)

#     # daily_pnl is one element shorter than S1, so let's align indexing
#     daily_pnl_series = pd.Series([0.0] + daily_pnl, index=S1.index)

#     cum_pnl_series = daily_pnl_series.cumsum()
#     cum_pnl_pct_series = (cum_pnl_series / initial_capital) * 100.0

#     return daily_pnl_series, cum_pnl_series, cum_pnl_pct_series, shares_S1_list, shares_S2_list, 

def simulate_strategy_trade_pnl(S1, S2, positions, beta_series=None, initial_capital=1000.0, tx_cost=None):
    """
    Compute the profit (or loss) for each trade by measuring the price change from the time the trade is opened
    until it is closed, and adjust for transaction costs if provided.
    
    For a long spread (positions = +1):
      - You go long S1 and short S2.
      - At entry, allocate capital as:
            Notional_S1 = initial_capital / (1 + beta_entry)
            Notional_S2 = beta_entry * Notional_S1.
      - Shares are:
            shares_S1 = Notional_S1 / S1_entry,
            shares_S2 = Notional_S2 / S2_entry.
      - Gross profit is:
            profit = shares_S1 * (S1_exit - S1_entry) + shares_S2 * (S2_entry - S2_exit).
    
    For a short spread (positions = -1):
      - You go short S1 and long S2.
      - Gross profit is:
            profit = shares_S1 * (S1_entry - S1_exit) + shares_S2 * (S2_exit - S2_entry).
    
    Transaction fees are applied to both the entry and exit of each leg.
      - For example, if tx_cost = 0.001 (0.10%), then each trade (buy or sell) on each asset is charged 0.10% of the transaction value.
    
    Args:
        S1 (pd.Series): Price series for asset S1.
        S2 (pd.Series): Price series for asset S2.
        positions (pd.Series): Trading signals (0 = flat, +1 = long spread, -1 = short spread).
        beta_series (pd.Series or float, optional): Hedge ratio(s). If None, beta is assumed to be 1.0.
        initial_capital (float, optional): The capital allocated per trade (default 1000.0).
        tx_cost (float, optional): Transaction cost per trade as a fraction (e.g., 0.001 for 0.10%).
                                   If None, no transaction costs are applied.
    
    Returns:
        tuple: A tuple containing:
            - trade_profits (list): A list of net profit values for each closed trade.
            - cumulative_profit (pd.Series): Cumulative profit over time (indexed by trade exit times).
            - entry_indices (list): A list of indices (timestamps) when trades were opened.
            - exit_indices (list): A list of indices (timestamps) when trades were closed.
    """
    
    # If no beta_series is provided, assume beta=1.0.
    if beta_series is None:
        beta_series = pd.Series(1.0, index=S1.index)
    elif not isinstance(beta_series, pd.Series):
        beta_series = pd.Series(beta_series, index=S1.index)
    
    trade_profits = []
    entry_indices = []
    exit_indices = []

    long_spread = False
    short_spread = False

    long_spread_loss_count = 0
    short_spread_loss_count = 0
    
    in_trade = False  # Flag indicating whether a trade is active.
    trade_direction = 0  # +1 for long spread, -1 for short spread.
    entry_index = None
    entry_price_S1 = None
    entry_price_S2 = None
    beta_entry = None  # Hedge ratio at entry.
    
    # Loop over the positions series.
    for t in range(len(positions)):
        current_pos = positions.iloc[t]
        if not in_trade:
            # Look for a trade entry: when the position changes from 0 to nonzero.
            if current_pos != 0:
                in_trade = True
                trade_direction = current_pos

                if trade_direction == 1:
                    long_spread = True
                elif trade_direction == -1:
                    short_spread = True


                entry_index = positions.index[t]
                entry_price_S1 = S1.iloc[t]
                entry_price_S2 = S2.iloc[t]
                beta_entry = beta_series.iloc[t]
                entry_indices.append(entry_index)
        else:
            # A trade is active; check for trade exit (when the position returns to 0).
            if current_pos == 0:
                exit_index = positions.index[t]
                exit_price_S1 = S1.iloc[t]
                exit_price_S2 = S2.iloc[t]
                exit_indices.append(exit_index)
                
                # Compute notional allocation based on initial capital and beta_entry.
                Notional_S1 = initial_capital / (1.0 + beta_entry)
                Notional_S2 = beta_entry * Notional_S1
                
                # Compute share counts at entry.
                shares_S1 = Notional_S1 / entry_price_S1
                shares_S2 = Notional_S2 / entry_price_S2
                
                # Compute gross profit based on trade direction.
                if trade_direction == 1:
                    # Long spread: long S1, short S2.
                    gross_profit_S1 = shares_S1 * (exit_price_S1 - entry_price_S1)
                    gross_profit_S2 = shares_S2 * (entry_price_S2 - exit_price_S2)
                    gross_profit = gross_profit_S1 + gross_profit_S2



                elif trade_direction == -1:
                    # Short spread: short S1, long S2.
                    gross_profit_S1 = shares_S1 * (entry_price_S1 - exit_price_S1)
                    gross_profit_S2 = shares_S2 * (exit_price_S2 - entry_price_S2)
                    gross_profit = gross_profit_S1 + gross_profit_S2

                    if(gross_profit/initial_capital < 0):

                        short_spread_loss_count += 1

                else:
                    gross_profit = 0.0
                
                # If transaction costs are provided, calculate fees for each leg at entry and exit.
                if tx_cost is not None:
                    fee_S1_entry = tx_cost * (shares_S1 * entry_price_S1)
                    fee_S1_exit = tx_cost * (shares_S1 * exit_price_S1)
                    fee_S2_entry = tx_cost * (shares_S2 * entry_price_S2)
                    fee_S2_exit = tx_cost * (shares_S2 * exit_price_S2)
                    total_fees = fee_S1_entry + fee_S1_exit + fee_S2_entry + fee_S2_exit
                else:
                    total_fees = 0.0
                
                # Net trade profit is gross profit minus total transaction fees.
                net_trade_profit = gross_profit - total_fees
                
                #If the trade is a loss, increment the loss count
                if net_trade_profit/initial_capital < 0:

                    if long_spread:
                        long_spread_loss_count += 1
                    elif short_spread:
                        short_spread_loss_count += 1
                

                
                trade_profits.append(net_trade_profit)
                
                # Reset trade state.
                in_trade = False
                trade_direction = 0
                entry_index = None
                entry_price_S1 = None
                entry_price_S2 = None
                beta_entry = None

                # Reset spread direction flags.
                long_spread = False
                short_spread = False
                
                
    # Compute cumulative profit from the list of trade profits.
    cumulative_profit = np.cumsum(trade_profits)
    cumulative_profit_series = pd.Series(cumulative_profit, index=exit_indices)
    
    return trade_profits, cumulative_profit_series, entry_indices, exit_indices, long_spread_loss_count, short_spread_loss_count

# Example usage:
# daily_pnl, cum_pnl, entry_indices, exit_indices = simulate_strategy_trade_pnl(S1, S2, positions, beta_series, initial_capital=1000.0, tx_cost=0.001)



def plot_trading_simulation(
    S1, 
    S2, 
    sym1, 
    sym2, 
    zscore, 
    positions, 
    entry_threshold,
    stop_loss_threshold,
    cum_pnl,
    win_indexs=None,     # list of indices where trades ended in a "win"
    loss_indexs=None,    # list of indices where trades ended in a "loss"
    window_start=None,
    window_end=None
): 
    """
    Plot the trading simulation results including:
      - Stock prices (S1, S2)
      - Z-score
      - Trading positions
      - Cumulative PnL
    
    Also highlights periods for each trade in light green (if it ended in a win)
    or light red (if it ended in a loss), and draws black vertical lines at the
    start and end of each trade to "border" those highlights.

    For a long spread (positions=+1):
        - S1 is long (green '^' marker)
        - S2 is short (red 'v' marker)
    For a short spread (positions=-1):
        - S1 is short (red 'v' marker)
        - S2 is long (green '^' marker)

    Args:
        S1 (pd.Series): Price series for the first asset.
        S2 (pd.Series): Price series for the second asset.
        sym1 (str): Label for the first asset in plots.
        sym2 (str): Label for the second asset in plots.
        zscore (pd.Series): The rolling z-score of the spread.
        positions (pd.Series): 0=flat, +1=long spread, -1=short spread.
        cum_pnl (pd.Series): Cumulative PnL of the strategy.
        win_indexs (list): Indices where trades ended in a win.
        loss_indexs (list): Indices where trades ended in a loss.
        window_start, window_end: Optional start/end for slicing the data.
    """

    # Default empty lists if none provided
    if win_indexs is None:
        win_indexs = []
    if loss_indexs is None:
        loss_indexs = []


    # Slice data if window bounds are provided
    if window_start is not None or window_end is not None:
        S1 = S1.loc[window_start:window_end]
        S2 = S2.loc[window_start:window_end]
        zscore = zscore.loc[window_start:window_end]
        positions = positions.loc[window_start:window_end]
        cum_pnl = cum_pnl.loc[window_start:window_end]
    
    win_indexs = [idx for idx in win_indexs if idx in positions.index]
    loss_indexs = [idx for idx in loss_indexs if idx in positions.index]

    # Identify trade entry points: position changes from 0 to ±1
    trade_entries = positions[(positions != 0) & (positions.shift(1) == 0)]
    long_entries = trade_entries[trade_entries == 1]
    short_entries = trade_entries[trade_entries == -1]

    print(f"Long Entries In Window: {len(long_entries)}, Short Entries In Window: {len(short_entries)}")
    print(f"Wins In Window: {len(win_indexs)}, Losses In Window: {len(loss_indexs)}")

    # Determine intervals for each trade: from trade open to trade close
    trades = []
    current_pos = 0
    trade_start = None

    # We'll parse the positions series to find (start_idx, end_idx, outcome)
    for i in range(len(positions)):
        idx = positions.index[i]
        pos = positions.iloc[i]
        prev_pos = positions.iloc[i-1] if i > 0 else 0

        # If we just opened a trade
        if pos != 0 and prev_pos == 0:
            trade_start = idx
            current_pos = pos

        # If we just closed a trade
        if pos == 0 and prev_pos != 0:
            trade_end = idx
            # Determine outcome
            if trade_end in win_indexs:
                outcome = "win"
            elif trade_end in loss_indexs:
                outcome = "loss"
            else:
                outcome = "unknown"  # e.g. normal exit or partial data

            trades.append((trade_start, trade_end, outcome))
            trade_start = None
            current_pos = 0

    # Plot figure
    plt.figure(figsize=(15, 20))

    # Subplot 1: S1 & S2 with trade intervals & markers
    plt.subplot(5, 1, 1)
    ax1 = plt.gca()
    ax2 = ax1.twinx()

    ax1.plot(S1, label=sym1, color='blue')
    ax2.plot(S2, label=sym2, color='red')

    ax1.set_ylabel(sym1, color='blue')
    ax2.set_ylabel(sym2, color='red')
    plt.title(f"Stock Prices: {sym1} and {sym2}")

    # Highlight the entire time region from trade_start to trade_end
    # in lightgreen for wins, lightcoral for losses.
    # Additionally, draw black vertical lines at the start and end.
    for (start_idx, end_idx, outcome) in trades:
        if start_idx is None or end_idx is None:
            continue
        if outcome == "win":
            ax1.axvspan(start_idx, end_idx, facecolor='lightgreen', alpha=0.2)
        elif outcome == "loss":
            ax1.axvspan(start_idx, end_idx, facecolor='lightcoral', alpha=0.2)
        else:
            # If you want to highlight "unknown" trades differently, do so here
            pass

        # Add black vertical lines at trade start & end
        ax1.axvline(start_idx, color='black', linestyle='-', linewidth=1)
        ax1.axvline(end_idx, color='black', linestyle='-', linewidth=1)

    # Markers for trade entries on S1 (ax1)
    ax1.scatter(long_entries.index, S1.loc[long_entries.index], 
                marker='^', color='green', s=100, label='S1 Long Entry (Long Spread)')
    ax1.scatter(short_entries.index, S1.loc[short_entries.index],
                marker='v', color='red', s=100, label='S1 Short Entry (Short Spread)')

    # Markers for trade entries on S2 (ax2)
    ax2.scatter(long_entries.index, S2.loc[long_entries.index],
                marker='v', color='red', s=100, label='S2 Short Entry (Long Spread)')
    ax2.scatter(short_entries.index, S2.loc[short_entries.index],
                marker='^', color='green', s=100, label='S2 Long Entry (Short Spread)')


    # Subplot 2: Z-score
    plt.subplot(5, 1, 2)
    plt.plot(zscore, label='Z-Score', color='purple', marker='o')
    plt.axhline(0, color='grey', linestyle='--', label='Mean')
    plt.axhline(entry_threshold, color='green', linestyle='--', label='±1.0 Entry threshold')
    plt.axhline(-entry_threshold, color='green', linestyle='--')
    plt.axhline(stop_loss_threshold, color='red', linestyle='--', label='±stop_loss_threshold Stop-loss')
    plt.axhline(-stop_loss_threshold, color='red', linestyle='--')

    plt.title("Z-Score of Spread")
    plt.scatter(long_entries.index, zscore.loc[long_entries.index], marker='^', 
                color='green', s=100, label='Long Entry')
    plt.scatter(short_entries.index, zscore.loc[short_entries.index], marker='v', 
                color='red', s=100, label='Short Entry')

    # # Subplot 3: Trading Positions
    # plt.subplot(5, 1, 3)
    # plt.plot(positions, label='Positions', drawstyle='steps-mid')
    # plt.title("Trading Positions")
    # plt.legend()

    # Subplot 4: Cumulative PnL
    plt.subplot(5, 1, 3)
    plt.plot(cum_pnl, label='Cumulative PnL')
    plt.title("Strategy Performance (Cumulative PnL)")
    plt.legend()

    plt.tight_layout()
    plt.show()


#-------------------------------------------------------------------------------
#                      Old Functions (Kept for Reference)
#-------------------------------------------------------------------------------


# def compute_spread(S1, S2):
#     """
#     Compute the spread between two asset price series by regressing S1 on S2.
    
#     Parameters:
#         S1 (pd.Series): Time series data for the first asset.
#         S2 (pd.Series): Time series data for the second asset.
    
#     Returns:
#         tuple: A tuple containing:
#             - spread (pd.Series): The residual series (spread) obtained from the regression.
#             - beta (float): The hedge ratio (slope coefficient) from the regression.
#     """

#     #Note: 
#     #Full regression model is S1 = α + βS2 + ϵ,

#     S2_const = sm.add_constant(S2) #Adds a constant to the independent variable
#     model = sm.OLS(S1, S2_const).fit()
#     beta = model.params[1]
#     alpha = model.params[0]
#     spread = S1 - alpha - beta * S2
#     return spread, beta


# def simulate_strategy(S1, S2, positions, beta):
#     """
#     Simulate the performance of a pairs trading strategy based on the computed positions.
    
#     Parameters:
#         S1 (pd.Series): Time series data for the first asset.
#         S2 (pd.Series): Time series data for the second asset.
#         positions (pd.Series): Series of trading positions generated from the backtest (aligned with spread timestamps).
#         beta (float): Hedge ratio computed from the cointegrating regression.
    
#     Returns:
#         tuple: A tuple containing:
#             - pnl (pd.Series): Daily profit and loss (PnL) of the strategy.
#             - cum_pnl (pd.Series): Cumulative PnL over time.
#     """
#     # Recompute spread using the hedge ratio
#     spread, _ = compute_spread(S1, S2)
#     # Daily change in spread (price difference)
#     spread_diff = spread.diff()
#     # Calculate PnL: shift positions to avoid lookahead bias
#     pnl = positions.shift(1) * spread_diff #The pnl at time t is the position at time t-1 times the change in spread from t-1 to t
#     pnl = pnl.fillna(0)
#     cum_pnl = pnl.cumsum()
#     return pnl, cum_pnl


# def backtest_pair(spread, entry_threshold=1.0, exit_threshold=0.0):
#     """
#     Generate trading signals based on the z-score of the spread and simulate positions over time.
    
#     Parameters:
#         spread (pd.Series): The spread series between two asset prices.
#         entry_threshold (float, optional): The z-score level at which to enter a position. Default is 1.0.
#         exit_threshold (float, optional): The z-score level at which to exit a position. Default is 0.0.
    
#     Returns:
#         tuple: A tuple containing:
#             - zscore (pd.Series): The z-score of the spread.
#             - positions (pd.Series): Series of trading positions over time 
#               (1 for long spread, -1 for short spread, 0 for no position).
#     """
#     # Compute z-score of the spread
#     spread_mean = spread.mean()
#     spread_std = spread.std()
#     zscore = (spread - spread_mean) / spread_std
    
#     positions = []
#     position = 0  # 1 for long spread, -1 for short spread, 0 for no position.

#     #Count long and short positions

#     for z in zscore:
#         # Entry conditions: enter short if z > entry_threshold, long if z < -entry_threshold.
#         if position == 0:
#             if z >= entry_threshold:
#                 position = -1
#             elif z <= -entry_threshold:
#                 position = 1
#         # Exit conditions: exit when the z-score reverts close to 0.
#         elif position == 1 and z >= -exit_threshold:
#             position = 0
#         elif position == -1 and z <= exit_threshold:
#             position = 0
#         positions.append(position)
#     positions = pd.Series(positions, index=spread.index)
    
#     return zscore, positions

# def simulate_true_strategy(S1, S2, positions, beta):
#     """
#     Simulate the true performance of a pairs trading strategy by calculating the profit and loss
#     from the individual asset positions when a trade is active.

#     In this strategy:
#       - A long spread (position = +1) means:
#           * Long 1 unit of S1
#           * Short beta units of S2
#       - A short spread (position = -1) means:w
#           * Short 1 unit of S1
#           * Long beta units of S2

#     The profit or loss from a trade between time t-1 and t is computed as:
#         For a long spread:
#             pnl = (S1[t] - S1[t-1]) - beta * (S2[t] - S2[t-1])
#         For a short spread:
#             pnl = -(S1[t] - S1[t-1]) + beta * (S2[t] - S2[t-1])
    
#     We use the previous period's position (i.e. positions.shift(1)) to avoid lookahead bias.

#     Parameters:
#         S1 (pd.Series): Time series of asset 1 prices.
#         S2 (pd.Series): Time series of asset 2 prices.
#         positions (pd.Series): Trading signals (1 for long spread, -1 for short spread, 0 for no position).
#         beta (float): Hedge ratio estimated from the cointegrating regression.

#     Returns:
#         tuple: A tuple containing:
#             - pnl (pd.Series): The period-by-period profit and loss of the strategy.
#             - cum_pnl (pd.Series): The cumulative profit and loss over time.
#     """
#     # Calculate the changes in asset prices
#     delta_S1 = S1.diff()
#     delta_S2 = S2.diff()
    
#     # Shift positions by one period to ensure that today's position is applied to tomorrow's returns
#     shifted_positions = positions.shift(1) #You generate a trading signal based on information up to time t−1 and then use that signal to trade during the period from t−1 to t.
    
#     # Compute profit and loss:
#     # For a long spread (position = 1): pnl = (ΔS1) - beta * (ΔS2)
#     # For a short spread (position = -1): pnl = -(ΔS1) + beta * (ΔS2)
#     # This formula works for both cases when multiplied by the signal (shifted_positions).
#     pnl = shifted_positions * (delta_S1 - beta * delta_S2)
#     pnl = pnl.fillna(0)
    
#     # Compute cumulative profit and loss
#     cum_pnl = pnl.cumsum()
    
#     return pnl, cum_pnl

# Example usage:
# Assume S1 and S2 are the price series of two cryptocurrencies,
# positions is the trading signal series generated by backtest_pair,
# and beta is the hedge ratio from compute_spread.
#
# pnl, cum_pnl = simulate_true_strategy(S1, S2, positions, beta)
# print(pnl.tail())
# cum_pnl.plot(title="Cumulative PnL")


# def simulate_strategy_monetary_sl(S1, S2, positions, beta, account_balance=1000, risk_per_trade=100,
#                                   entry_threshold_short=1.0, stop_loss_threshold_short=2.0,
#                                   entry_threshold_long=-1.0, stop_loss_threshold_long=-2.0,
#                                   window_for_std=20):
#     """
#     Simulate the monetary profit and loss (PnL) for a pairs trading strategy with a stop-loss rule based on
#     a fixed threshold plus one standard deviation.

#     For a short spread trade:
#       - Entry is triggered when the spread reaches the upper threshold (e.g., z-score = 1).
#       - At entry, the spread is assumed to be: 
#              entry_spread = mean_spread_window + entry_threshold_short * std_spread_window.
#       - The stop-loss is set at:
#              stop_loss_price = mean_spread_window + stop_loss_threshold_short * std_spread_window.
#       - The trade size is computed so that a move from entry_spread to stop_loss_price results in a loss equal to risk_per_trade.
    
#     For a long spread trade (signal = +1):
#       - Entry is triggered when the spread reaches the lower threshold (e.g., z-score = -1):
#              entry_spread = mean_spread_window + entry_threshold_long * std_spread_window.
#       - The stop-loss is set at:
#              stop_loss_price = mean_spread_window + stop_loss_threshold_long * std_spread_window.
#       - Trade size is computed similarly.

#     Once a trade is active, the monetary PnL is computed period by period using the trade size and the
#     change in the spread from the entry level. If the spread reaches the stop-loss level, the trade is closed and
#     the loss is capped at -risk_per_trade.

#     Parameters:
#         S1 (pd.Series): Price series for asset 1.
#         S2 (pd.Series): Price series for asset 2.
#         positions (pd.Series): Trading signals (1 for long spread, -1 for short spread, 0 for no position), aligned with the price series.
#         beta (float): Hedge ratio computed from the cointegrating regression.
#         account_balance (float, optional): Total account balance (for reference). Default is 1000.
#         risk_per_trade (float, optional): The maximum monetary loss allowed per trade (e.g., 100€). Default is 100.
#         entry_threshold_short (float, optional): For a short trade, the z-score threshold at entry. Default is 1.0.
#         stop_loss_threshold_short (float, optional): For a short trade, the z-score for the stop loss. Default is 2.0.
#         entry_threshold_long (float, optional): For a long trade, the z-score threshold at entry. Default is -1.0.
#         stop_loss_threshold_long (float, optional): For a long trade, the z-score for the stop loss. Default is -2.0.
#         window_for_std (int, optional): Number of periods to compute the spread's rolling standard deviation at trade entry. Default is 20.
    
#     Returns:
#         tuple: A tuple containing:
#             - pnl_series (pd.Series): Period-by-period monetary PnL.
#             - cum_pnl (pd.Series): Cumulative monetary PnL over time.
#     """

#     # First, compute the spread using the provided beta.
#     # (Note: This spread is computed over the full series.)
#     spread_full = S1 - beta * S2
    
#     pnl_series = pd.Series(0.0, index=S1.index)
#     trade_state = None  # To store details of the active trade.
    
#     for t in range(len(S1)):
#         current_signal = positions.iloc[t]
        
#         # No trade currently active: check if a trade is initiated.
#         if trade_state is None:
#             if current_signal != 0:
#                 # Record trade entry details.
#                 entry_idx = t
#                 entry_S1 = S1.iloc[t]
#                 entry_S2 = S2.iloc[t]
                
#                 # Compute the entry spread using the current data point.
#                 entry_spread = spread_full.iloc[t]
                
#                 # Determine the window to compute rolling statistics.
#                 window_start = max(0, t - window_for_std + 1)
#                 spread_window = spread_full.iloc[window_start:t+1]
#                 mean_window = spread_window.mean()
#                 std_window = spread_window.std()
#                 # To avoid division by zero
#                 if std_window == 0:
#                     std_window = 1e-8
                
#                 # Based on the trade direction, set entry threshold and stop loss level in price units.
#                 if current_signal == -1:  # short trade: entry when spread is high.
#                     # Entry price corresponds to: mean + entry_threshold_short * std.
#                     desired_entry = mean_window + entry_threshold_short * std_window
#                     # Stop loss is at: mean + stop_loss_threshold_short * std.
#                     stop_loss_price = mean_window + stop_loss_threshold_short * std_window
#                 elif current_signal == 1:  # long trade: entry when spread is low.
#                     desired_entry = mean_window + entry_threshold_long * std_window
#                     stop_loss_price = mean_window + stop_loss_threshold_long * std_window
                
#                 # For simplicity, we assume the trade is executed at the observed spread.
#                 # (In practice, you might enforce that the entry_spread is close to desired_entry.)
#                 # Calculate trade size such that a move from desired_entry to stop_loss equals risk_per_trade.
#                 move_required = abs(stop_loss_price - desired_entry)
#                 trade_size = risk_per_trade / move_required if move_required != 0 else 0
                
#                 trade_state = {
#                     'entry_index': t,
#                     'entry_spread': entry_spread,
#                     'trade_size': trade_size,
#                     'direction': current_signal,  # +1 for long, -1 for short.
#                     'stop_loss_price': stop_loss_price,
#                     'desired_entry': desired_entry
#                 }
#         else:
#             # Trade is active.
#             current_spread = spread_full.iloc[t]
#             direction = trade_state['direction']
#             trade_size = trade_state['trade_size']
#             entry_spread = trade_state['entry_spread']
#             stop_loss_price = trade_state['stop_loss_price']
            
#             # Calculate current profit:
#             # For a short trade (direction = -1), profit is: trade_size * (entry_spread - current_spread).
#             # For a long trade (direction = 1), profit is: trade_size * (current_spread - entry_spread).
#             if direction == -1:
#                 current_trade_pnl = trade_size * (entry_spread - current_spread)
#                 # Check if stop loss has been hit: for short trade, stop loss is triggered if current_spread >= stop_loss_price.
#                 stop_triggered = current_spread >= stop_loss_price
#             else:  # direction == 1
#                 current_trade_pnl = trade_size * (current_spread - entry_spread)
#                 # For a long trade, stop loss is triggered if current_spread <= stop_loss_price.
#                 stop_triggered = current_spread <= stop_loss_price
            
#             # Record incremental pnl for this period.
#             # To avoid lookahead bias, assume the pnl realized in period t is the change from previous period.
#             # Here we simply assign the current trade pnl difference.
#             if 'prev_trade_pnl' not in trade_state:
#                 trade_state['prev_trade_pnl'] = 0.0
#             pnl_increment = current_trade_pnl - trade_state['prev_trade_pnl']
#             pnl_series.iloc[t] = pnl_increment
#             trade_state['prev_trade_pnl'] = current_trade_pnl
            
#             # If the stop loss is triggered, adjust pnl so that the loss equals -risk_per_trade.
#             if stop_triggered:
#                 # Determine adjustment required.
#                 loss_excess = current_trade_pnl + risk_per_trade  # Note: current_trade_pnl should be negative at stop.
#                 pnl_series.iloc[t] -= loss_excess  # Adjust the incremental pnl so that total loss equals -risk_per_trade.
#                 # Close the trade.
#                 trade_state = None
            
#             # Alternatively, if the position signal goes to 0, close the trade.
#             if positions.iloc[t] == 0:
#                 trade_state = None

#     cum_pnl = pnl_series.cumsum()
#     return pnl_series, cum_pnl


# Example usage:
# Assume S1 and S2 are pandas Series of hourly prices for two cryptocurrencies,
# positions is the trading signals series generated by your backtest_pair function,
# and beta is computed by compute_spread.
#
# pnl_series, cum_pnl = simulate_strategy_monetary_sl(S1, S2, positions, beta, account_balance=1000, risk_per_trade=100)
# print("Monetary PnL:")
# print(pnl_series.tail())
# cum_pnl.plot(title="Cumulative Monetary PnL")





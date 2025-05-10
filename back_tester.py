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

        #Including intercept
        # Compute the spread at time t using the dynamically estimated beta
        spread_series.iloc[t] = S1.iloc[t] - alpha_t - beta_t * S2.iloc[t]

        #Excluding intercept
        #spread_series.iloc[t] = S1.iloc[t] - beta_t * S2.iloc[t]



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


def backtest_pair_rolling(
    S1, S2, zscore, entry_threshold=1.0, exit_threshold=0.0, stop_loss_threshold=2.0
):
    """
    A bar-based backtest that does NOT do linear interpolation,
    
    Trading rules (discrete bar):
      - We check the z-score at each bar's close:
          * If position == 0 (flat) and zscore >= +entry_threshold, open short spread at this bar's price (position = -1).
          * If position == 0 (flat) and zscore <= -entry_threshold, open long spread at this bar's price (position = +1).
      - If position == +1 (long spread) and zscore >= -exit_threshold => exit (win).
        * Or if zscore <= -stop_loss_threshold => exit (loss).
      - If position == -1 (short spread) and zscore <= +exit_threshold => exit (win).
        * Or if zscore >= +stop_loss_threshold => exit (loss).

    Returns:
        positions (pd.Series): The discrete position (+1, -1, or 0) at each bar.
        trade_entries (list): A list of entry dicts: {'time','S1','S2','z','position'}.
        trade_exits (list): A list of exit dicts:  {'time','S1','S2','z','exit_type'}.

    Note: No sub-bar interpolation or same-bar entry/exit logic.
          We simply act on each bar's z-score once we have it.
    """

    positions = []
    position  = 0   # +1 (long spread), -1 (short spread), or 0 (flat)
    stopped_out_position = 0 #Store the that you were in when you were stopped out
    stop_out  = False

    trade_entries = []
    trade_exits   = []

    entry_price_S1 = None
    entry_price_S2 = None
    entry_z        = None

    z_index = zscore.index

    for t, current_index in enumerate(z_index):
        current_z  = zscore.loc[current_index]
        current_S1 = S1.loc[current_index]
        current_S2 = S2.loc[current_index]

        # Initialization: first bar
        if t == 0:
            positions.append(0)
            continue

        # If z-score is NaN, remain in the current position
        if pd.isna(current_z):
            positions.append(position)
            continue

        # CASE 1: No open position
        if position == 0:
            if stop_out:
                
                #If you were in a long position, you can re-enter if the z-score is equal to or crosses the zero line from below
                if stopped_out_position == 1:

                    if current_z >= exit_threshold:
                        stop_out = False
                        stopped_out_position = 0    
                    positions.append(0)

                #If you were in a short position, you can re-enter if the z-score is equal to or crosses the zero line from above
                elif stopped_out_position == -1:
                    if current_z <= -exit_threshold:
                        stop_out = False
                        stopped_out_position = 0
                    positions.append(0)

                # # If you want to prevent re-entry until z is near zero:
                # if abs(current_z) <= 0.1:
                #     stop_out = False
                # # remain flat
                # positions.append(0)

            else:
                # If zscore >= +entry_threshold => short spread
                if current_z >= entry_threshold:
                    position      = -1
                    entry_price_S1 = current_S1
                    entry_price_S2 = current_S2
                    entry_z        = current_z
                    trade_entries.append({
                        'time': current_index,
                        'S1':   entry_price_S1,
                        'S2':   entry_price_S2,
                        'z':    entry_z,
                        'position': position
                    })
                    positions.append(position)

                # If zscore <= -entry_threshold => long spread
                elif current_z <= -entry_threshold:
                    position       = 1
                    entry_price_S1 = current_S1
                    entry_price_S2 = current_S2
                    entry_z        = current_z
                    trade_entries.append({
                        'time': current_index,
                        'S1':   entry_price_S1,
                        'S2':   entry_price_S2,
                        'z':    entry_z,
                        'position': position
                    })
                    positions.append(position)
                else:
                    # no entry
                    positions.append(0)

        # CASE 2: We already have a position
        else:          
            if position == 1:  # LONG SPREAD
                # Normal exit => if zscore >= -exit_threshold => 'win'
                if current_z >= -exit_threshold:
                    trade_exits.append({
                        'time': current_index,
                        'S1':   current_S1,
                        'S2':   current_S2,
                        'z':    current_z,
                        'exit_type': 'win'
                    })
                    position      = 0
                    entry_price_S1 = None
                    entry_price_S2 = None
                    stop_out      = False
                # Stop-loss => if zscore <= -stop_loss_threshold => 'loss'
                elif current_z <= -stop_loss_threshold:
                    trade_exits.append({
                        'time': current_index,
                        'S1':   current_S1,
                        'S2':   current_S2,
                        'z':    current_z,
                        'exit_type': 'loss'
                    })
                    stopped_out_position = 1
                    position       = 0
                    entry_price_S1 = None
                    entry_price_S2 = None
                    stop_out       = True
                positions.append(position)

            elif position == -1:  # SHORT SPREAD
                # Normal exit => if zscore <= exit_threshold => 'win'
                if current_z <= exit_threshold:
                    trade_exits.append({
                        'time': current_index,
                        'S1':   current_S1,
                        'S2':   current_S2,
                        'z':    current_z,
                        'exit_type': 'win'
                    })
                    position       = 0
                    entry_price_S1 = None
                    entry_price_S2 = None
                    stop_out       = False
                # Stop-loss => if zscore >= stop_loss_threshold => 'loss'
                elif current_z >= stop_loss_threshold:
                    trade_exits.append({
                        'time': current_index,
                        'S1':   current_S1,
                        'S2':   current_S2,
                        'z':    current_z,
                        'exit_type': 'loss'
                    })
                    stopped_out_position = -1
                    position       = 0
                    entry_price_S1 = None
                    entry_price_S2 = None
                    stop_out       = True
                positions.append(position)


    #Check if we are still in a position at the end of the backtest
    if position != 0:

        trade_exits.append({
            'time': current_index,
            'S1':   current_S1,
            'S2':   current_S2,
            'z':    current_z,
            'exit_type': 'forced_exit' #We are forced to exit the position as we are at the end of the backtest
        })


    # Convert positions list to Series
    positions = pd.Series(positions, index=zscore.index)

    # Print basic stats
    total_closed = len(trade_exits)
    wins  = sum(1 for e in trade_exits if e['exit_type'] == 'win')
    losses= sum(1 for e in trade_exits if e['exit_type'] == 'loss')

    # print(f"Total trades closed: {total_closed} (Wins={wins}, Losses={losses})")
    # if total_closed > 0:
    #     print(f"Win rate: {wins / total_closed:.2f}")

    return positions, trade_entries, trade_exits



def backtest_pair_rolling_order_book(
    spread_series, 
    S1_mid_price, 
    S2_mid_price,
    S1_ask,
    S1_ask_amount,
    S1_bid,
    S1_bid_amount,
    S2_ask,
    S2_ask_amount,
    S2_bid,
    S2_bid_amount,
    zscore, entry_threshold=1.0,
    exit_threshold=0.0, 
    stop_loss_threshold=2.0
):
    """
    A bar-based backtest that does NOT do linear interpolation,
    
    Trading rules (discrete bar):
      - We check the z-score at each bar's close:
          * If position == 0 (flat) and zscore >= +entry_threshold, open short spread at this bar's price (position = -1).
          * If position == 0 (flat) and zscore <= -entry_threshold, open long spread at this bar's price (position = +1).
      - If position == +1 (long spread) and zscore >= -exit_threshold => exit (win).
        * Or if zscore <= -stop_loss_threshold => exit (loss).
      - If position == -1 (short spread) and zscore <= +exit_threshold => exit (win).
        * Or if zscore >= +stop_loss_threshold => exit (loss).

    Returns:
        positions (pd.Series): The discrete position (+1, -1, or 0) at each bar.
        trade_entries (list): A list of entry dicts: {'time','S1','S2','z','position'}.
        trade_exits (list): A list of exit dicts:  {'time','S1','S2','z','exit_type'}.

    Note: No sub-bar interpolation or same-bar entry/exit logic.
          We simply act on each bar's z-score once we have it.
    """

    positions = []
    position  = 0   # +1 (long spread), -1 (short spread), or 0 (flat)
    stopped_out_position = 0 #Store the that you were in when you were stopped out
    stop_out  = False

    trade_entries = []
    trade_exits   = []

    entry_price_S1 = None
    entry_price_S2 = None
    entry_z        = None

    z_index = zscore.index

    for t, current_index in enumerate(z_index):

        #Get Current Z-Score At Index
        #-------------------
        current_z  = zscore.loc[current_index]

        #Get Current Mid Prices At Index
        #-------------------
        current_S1_mid_price = S1_mid_price.loc[current_index]
        current_S2_mid_price = S2_mid_price.loc[current_index]

        #Get Current Ask/Bid Prices And Amounts At Index For S1
        #-------------------
        current_S1_ask = S1_ask.loc[current_index]
        current_S1_ask_amount = S1_ask_amount.loc[current_index]
        current_S1_bid = S1_bid.loc[current_index]
        current_S1_bid_amount = S1_bid_amount.loc[current_index]

        #Get Current Ask/Bid Prices And Amounts At Index For S2
        #-------------------
        current_S2_ask = S2_ask.loc[current_index]
        current_S2_ask_amount = S2_ask_amount.loc[current_index]
        current_S2_bid = S2_bid.loc[current_index]
        current_S2_bid_amount = S2_bid_amount.loc[current_index]
        
        # Initialization: first bar
        if t == 0:
            positions.append(0)
            continue

        # If z-score is NaN, remain in the current position
        if pd.isna(current_z):
            positions.append(position)
            continue

        # CASE 1: No open position
        if position == 0:
            if stop_out:
                
                #If you were in a long position, you can re-enter if the z-score is equal to or crosses the zero line from below
                if stopped_out_position == 1:

                    if current_z >= exit_threshold:
                        stop_out = False
                        stopped_out_position = 0    
                    positions.append(0)

                #If you were in a short position, you can re-enter if the z-score is equal to or crosses the zero line from above
                elif stopped_out_position == -1:
                    if current_z <= -exit_threshold:
                        stop_out = False
                        stopped_out_position = 0
                    positions.append(0)

                # # If you want to prevent re-entry until z is near zero:
                # if abs(current_z) <= 0.1:
                #     stop_out = False
                # # remain flat
                # positions.append(0)

            else:
                # If zscore >= +entry_threshold => short spread: Short S1, Long S2
                if current_z >= entry_threshold:
                    position      = -1
                    entry_price_S1 = current_S1_bid #Short S1 at best bid price
                    entry_price_S2 = current_S2_ask #Long S2 at best ask price
                    amount_S1 = current_S1_bid_amount
                    amount_S2 = current_S2_ask_amount
                    entry_z        = current_z
                    trade_entries.append({
                        'time': current_index,
                        'S1':   entry_price_S1,
                        'S2':   entry_price_S2,
                        'S1 Amount': amount_S1,
                        'S2 Amount': amount_S2,
                        'z':    entry_z,
                        'position': position
                    })
                    positions.append(position)

                # If zscore <= -entry_threshold => long spread: Long S1, Short S2
                elif current_z <= -entry_threshold:
                    position       = 1
                    entry_price_S1 = current_S1_ask #Long S1 at best ask price
                    entry_price_S2 = current_S2_bid #Short S2 at best bid price
                    amount_S1 = current_S1_ask_amount
                    amount_S2 = current_S2_bid_amount
                    entry_z        = current_z
                    trade_entries.append({
                        'time': current_index,
                        'S1':   entry_price_S1,
                        'S2':   entry_price_S2,
                        'S1 Amount': amount_S1,
                        'S2 Amount': amount_S2,
                        'z':    entry_z,
                        'position': position
                    })
                    positions.append(position)
                else:
                    # no entry
                    positions.append(0)

        # CASE 2: We already have a position
        else:
            if position == 1:  # LONG SPREAD: Long S1, Short S2 (To close the position, we need to sell S1 and buy S2)
                # Normal exit => if zscore >= -exit_threshold => 'win'
                if current_z >= -exit_threshold:
                    trade_exits.append({
                        'time': current_index,
                        'S1':   current_S1_bid, #Sell S1 at best bid price
                        'S2':   current_S2_ask, #Buy S2 at best ask price
                        'S1 Amount': current_S1_bid_amount,
                        'S2 Amount': current_S2_ask_amount,
                        'z':    current_z,
                        'exit_type': 'win'
                    })
                    position      = 0
                    entry_price_S1 = None
                    entry_price_S2 = None
                    stop_out      = False
                # Stop-loss => if zscore <= -stop_loss_threshold => 'loss'
                elif current_z <= -stop_loss_threshold:
                    trade_exits.append({
                        'time': current_index,
                        'S1':   current_S1_bid, #Sell S1 at best bid price
                        'S2':   current_S2_ask, #Buy S2 at best ask price
                        'S1 Amount': current_S1_bid_amount,
                        'S2 Amount': current_S2_ask_amount,
                        'z':    current_z,
                        'exit_type': 'loss'
                    })
                    stopped_out_position = 1
                    position       = 0
                    entry_price_S1 = None
                    entry_price_S2 = None
                    stop_out       = True
                positions.append(position)

            elif position == -1:  # SHORT SPREAD: Short S1, Long S2 (To close the position, we need to buy S1 and sell S2)
                # Normal exit => if zscore <= exit_threshold => 'win'
                if current_z <= exit_threshold:
                    trade_exits.append({
                        'time': current_index,
                        'S1':   current_S1_ask, #Buy S1 at best ask price
                        'S2':   current_S2_bid, #Sell S2 at best bid price
                        'S1 Amount': current_S1_ask_amount,
                        'S2 Amount': current_S2_bid_amount,
                        'z':    current_z,
                        'exit_type': 'win'
                    })
                    position       = 0
                    entry_price_S1 = None
                    entry_price_S2 = None
                    stop_out       = False
                # Stop-loss => if zscore >= stop_loss_threshold => 'loss'
                elif current_z >= stop_loss_threshold:
                    trade_exits.append({
                      'time': current_index,
                        'S1':   current_S1_ask, #Buy S1 at best ask price
                        'S2':   current_S2_bid, #Sell S2 at best bid price
                        'S1 Amount': current_S1_ask_amount,
                        'S2 Amount': current_S2_bid_amount,
                        'z':    current_z,
                        'exit_type': 'loss'
                    })
                    stopped_out_position = -1
                    position       = 0
                    entry_price_S1 = None
                    entry_price_S2 = None
                    stop_out       = True
                positions.append(position)

    # Convert positions list to Series
    positions = pd.Series(positions, index=spread_series.index)

    # Print basic stats
    total_closed = len(trade_exits)
    wins  = sum(1 for e in trade_exits if e['exit_type'] == 'win')
    losses= sum(1 for e in trade_exits if e['exit_type'] == 'loss')

    print(f"Total trades closed: {total_closed} (Wins={wins}, Losses={losses})")
    if total_closed > 0:
        print(f"Win rate: {wins / total_closed:.2f}")

    return positions, trade_entries, trade_exits


def compute_sharpe_ratio(initial_capital,trade_profits, risk_free_rate=0.0):

    """
    Calculate the Sharpe ratio of a trading strategy.
    The Sharpe ratio is a measure of risk-adjusted return.
    It is calculated as the mean return of the strategy minus the risk-free rate,
    divided by the standard deviation of the returns.
    Parameters:
    initial_capital (float): The initial capital of the portfolio.
    trade_profits (list): A list of trade profits.
    risk_free_rate (float): The risk-free rate of return.
    Returns:
    sharpe_ratio (float): The Sharpe ratio of the strategy.
    """

    trade_profits = np.array(trade_profits)

    trade_returns = trade_profits/ initial_capital

    mean_return = np.mean(trade_returns)
    std_return = np.std(trade_returns)
    sharpe_ratio = (mean_return - risk_free_rate) / std_return

    return sharpe_ratio


def compute_max_drawdown(initial_capital,cumulative_profit_series):

    """
    Calculate the maximum drawdown of a portfolio.
    The maximum drawdown is defined as the maximum observed loss from a peak to a trough
    in the portfolio value.
    Parameters:
    initial_capital (float): The initial capital of the portfolio.
    cumulative_profit_series (pd.Series): A series of cumulative profits over time.
    Returns:
    max_drawdown (float): The maximum drawdown value.
    max_drawdown_percentage (float): The maximum drawdown percentage.
    """
 
    # Calculate the portfolio value over time
    portfolio_values = initial_capital + cumulative_profit_series

    # Calculate the drawdown
    rolling_max = portfolio_values.cummax()
    drawdown = (portfolio_values - rolling_max) / rolling_max
    drawdown = drawdown.dropna()
    # Calculate the maximum drawdown
    max_drawdown = drawdown.min() #We use min bevause drawdown is negative!
    # Calculate the maximum drawdown percentage
    max_drawdown_percentage = max_drawdown * 100

    max_drawdown_percentage = abs(max_drawdown_percentage) #Convert to positive value

    return max_drawdown_percentage


def compute_mean_absolute_percent_delta_betas(beta_series, entry_times, exit_times):

    """
    Calculate the mean absolute percent delta betas for each trade.
    This function computes the absolute percent difference between the beta at the trade entry
    and the betas in the trade range (between entry and exit times).

    Parameters:
    beta_series (pd.Series): A series of betas (hedge ratios)
    entry_times (list): A list of entry times for trades
    exit_times (list): A list of exit times for trades

    Returns:
    mean_absolute_percent_delta_betas (list): A list of mean absolute percent delta betas for each trade
    """


    #Get actual Betas used for each trade entry
    beta_entries = [beta_series[entry] for entry in entry_times]

    trade_ranges = [(entry, exit) for entry, exit in zip(entry_times, exit_times)]

    #Get beta series in the trade ranges

    beta_series_trade_ranges = [beta_series[entry:exit].iloc[1:] for entry, exit in trade_ranges] #iloc[1:] to exclude the first beta value which is the same as the entry beta

    #Calculate the percent absolute difference between the beta at the trade entry and the betas in the trade range
    absolute_percent_delta_betas = [np.abs(((entry_beta - beta_range)/entry_beta)*100) for entry_beta, beta_range in zip(beta_entries, beta_series_trade_ranges)]

    #Calculate the mean delta beta for each trade
    mean_absolute_percent_delta_betas = [np.mean(delta) for delta in absolute_percent_delta_betas]


    return mean_absolute_percent_delta_betas




def compute_key_metrics(sym1, sym2, S1,S2,initial_capital,trade_profits,cumulative_profit_series, entry_times, exit_times, beta_series):
    
    #Trade losses and wins
    trade_losses = [profit for profit in trade_profits if profit < 0]
    trade_wins = [profit for profit in trade_profits if profit > 0]
    
    #Compute non-stop loss win rate (assuming there is no stop loss)
    non_stop_loss_win_rate = (len(trade_wins) / (len(trade_wins) + len(trade_losses))) * 100

    #Compute trade durations
    trade_durations = [(exit - entry).total_seconds() / 60 for entry, exit in zip(entry_times, exit_times)]

    # Percentage changes in S1 and S2 each trade
    trade_percentage_changes = []
    for entry_time, exit_time in zip(entry_times, exit_times):
        # Calculate percentage change for S1 and S2
        price_change_S1 = (S1[exit_time] - S1[entry_time]) / S1[entry_time] * 100
        price_change_S2 = (S2[exit_time] - S2[entry_time]) / S2[entry_time] * 100
        trade_percentage_changes.append((price_change_S1, price_change_S2))

    S1_trade_returns = [percentage_changes[0] for percentage_changes in trade_percentage_changes]
    S2_trade_returns = [percentage_changes[1] for percentage_changes in trade_percentage_changes]

    #Compute mean absolute percent delta betas
    mean_absolute_percent_delta_betas = compute_mean_absolute_percent_delta_betas(beta_series, entry_times, exit_times)

    #Compute average absolute percent delta betas
    average_absolute_percent_delta_beta = np.mean(mean_absolute_percent_delta_betas)

    #Compute median absolute percent delta betas
    median_absolute_percent_delta_beta = np.median(mean_absolute_percent_delta_betas)

    #Compute average trade duration
    mean_trade_duration = np.mean(trade_durations)

    #Compute average S1 and S2 trade returns
    average_S1_trade_returns = np.mean(S1_trade_returns)
    average_S2_trade_returns = np.mean(S2_trade_returns)
    
    #Compute entry betas
    entry_betas = beta_series[entry_times]

    #Compute average entry beta
    avg_entry_beta = np.mean(entry_betas)

    #Compute beta series returns
    beta_series_returns = beta_series.pct_change().dropna()

    #Compute standard of beta series returns
    std_beta_series = beta_series_returns.std() * 100 #Convert to percentage since beta_series_returns is in decimal format
    print(f"Std of beta series returns: {std_beta_series:.4f}") 

    #Compute Median Absolute Deviation of the beta series returns 
    median = np.median(beta_series_returns)
    mad = np.median(np.abs(beta_series_returns - median)) * 100 #Convert to percentage since beta_series_returns is in decimal format

    sharpe_ratio = compute_sharpe_ratio(initial_capital,trade_profits)

    max_drawdown = compute_max_drawdown(initial_capital,cumulative_profit_series)

    key_metrics_df = pd.DataFrame({
    'Pair': f"{sym1} ~ {sym2}",
    'Total return (%)': cumulative_profit_series[-1]/initial_capital * 100,
    'Sharpe ratio': sharpe_ratio,
    'Max drawdown (%)': max_drawdown,
    'Number of trades': len(trade_profits),
    'Non-stop loss win rate (%)': non_stop_loss_win_rate,
    f'Mean trade duration/Reversion speed from threshold (mins)': mean_trade_duration,
    'Average entry beta': avg_entry_beta,
    'Mean Absolute Percent Delta Beta (%)': average_absolute_percent_delta_beta,
    'Median Absolute Percent Delta Beta (%)': median_absolute_percent_delta_beta,
    'Beta series returns std (%)': std_beta_series,
    'Beta series returns MAD (%)': mad,
    #'Spread series z-score std (%)': z_score_spread_std,
    'Average S1 trade returns (%)': average_S1_trade_returns,
    'Average S2 trade returns (%)': average_S2_trade_returns,
    },index=[0])

    return key_metrics_df


#Final backtest function

def backtest(prices_df, **params):

    #Params:
    #----------------------------------------------
    initial_capital = params.get("initial_capital")
    window_size = params.get("window_size")
    entry_threshold = params.get("entry_threshold")
    exit_threshold = params.get("exit_threshold")
    stop_loss_threshold = params.get("stop_loss_threshold")
    tx_cost= params.get("tx_cost")

    #Backtest functions:
    #----------------------------------------------
    sym1, sym2 = prices_df.columns

    S1 = prices_df[sym1]
    S2 = prices_df[sym2]

    print("-------------------------------------------------------")
    print("INITIATING BACKTEST FOR PAIR: ", sym1, "~", sym2, "🚀")
    print("-------------------------------------------------------")

    # Compute the spread series and beta_series 
    spread_series, beta_series, alpha_series = compute_spread_series(S1, S2, window_size)
    #print(f"Hedge ratio (beta) for {sym1} ~ {sym2}: {beta:.4f}")

    # Compute rolling z-score using the provided helper function.
    zscore_series, rolling_mean, rolling_std = compute_rolling_zscore(spread_series, window_size)

    # Gather trade entries and exits
    positions, trade_entries, trade_exits = backtest_pair_rolling(S1,S2,zscore_series, entry_threshold, exit_threshold, stop_loss_threshold)

    #Compute trade profits
    trade_profits, net_trade_profits_S1, net_trade_profits_S2,cumulative_profit_series, entry_times, exit_times = simulate_strategy_trade_pnl(trade_entries, trade_exits, initial_capital, beta_series, tx_cost)

    #Compute key metrics
    #----------------------------------------------
    key_metrics_df = compute_key_metrics(sym1, sym2,S1,S2,initial_capital,trade_profits,cumulative_profit_series, entry_times, exit_times, beta_series)

    print("-------------------------------------------------------")
    print("BACKTEST COMPLETED FOR: ", sym1, "~", sym2, "✅")
    print("-------------------------------------------------------")


    return key_metrics_df





#OLD BACK TEST FUNCTION WITH INTERPOLATION
#-------------------------------------------------------------


# def backtest_pair_rolling(spread_series, S1, S2, zscore, entry_threshold=1.0, exit_threshold=0.0, stop_loss_threshold=2.0):
#     """
#     Generate trading signals based on the rolling z-score of the spread series using a moving window,
#     and record the interpolated entry and exit prices (and z-scores) for more accurate profit calculations.
    
#     This version uses linear interpolation to estimate the exact prices at which the z-score 
#     reaches the entry threshold and then immediately checks if an exit (normal or stop-loss) 
#     is triggered in the same time step.
    
#     Trading rules:
#       - When no position is open (position == 0):
#           * If the z-score crosses above entry_threshold, enter a short spread (position = -1).
#           * If the z-score crosses below -entry_threshold, enter a long spread (position = +1).
#       - When in a position:
#           * For a long spread (position = +1), exit when z-score crosses upward past -exit_threshold.
#           * For a short spread (position = -1), exit when z-score crosses downward past exit_threshold.
#       - Stop-loss conditions:
#           * For a long spread, exit if z-score goes below -stop_loss_threshold.
#           * For a short spread, exit if z-score goes above stop_loss_threshold.
    
#     Returns:
#         positions (pd.Series): Trading signals over time.
#         trade_entries (list): List of dictionaries for trade entries.
#         trade_exits (list): List of dictionaries for trade exits.
#         price_changes_S1 (list): List of percentage price changes for S1 from entry to exit.
#         price_changes_S2 (list): List of percentage price changes for S2 from entry to exit.
#     """


#     positions = []
#     position = 0  # 1 for long spread, -1 for short spread, 0 for flat.
#     trade_entries = []
#     trade_exits = []
#     price_changes_S1 = []
#     price_changes_S2 = []
#     stop_out = False

#     # Variables to hold the interpolated entry values.
#     entry_price_S1 = None
#     entry_price_S2 = None
#     entry_z = None  # z-score at entry

#     # For interpolation, keep track of previous values.
#     prev_z = None
#     prev_S1 = None
#     prev_S2 = None
#     prev_index = None

#     for t, current_index in enumerate(zscore.index):
#         current_z = zscore.loc[current_index]
#         current_S1 = S1.loc[current_index]
#         current_S2 = S2.loc[current_index]

#         # Initialization: first time step.
#         if t == 0:
#             positions.append(0)
#             prev_z = current_z
#             prev_S1 = current_S1
#             prev_S2 = current_S2
#             prev_index = current_index
#             continue

#         # If current z is not valid, remain flat.
#         if pd.isna(current_z):
#             positions.append(0)
#             price_changes_S1.append(0)
#             price_changes_S2.append(0)
#             prev_z = current_z
#             prev_S1 = current_S1
#             prev_S2 = current_S2
#             prev_index = current_index
#             continue

#         # Not in a trade.
#         if position == 0:
#             if stop_out:
#                 if abs(current_z) <= 0.1: #Allow re-entry if z-score is within -0.10 and 0.10 (close enough to zero)
#                     stop_out = False
#                 positions.append(0)
#                 price_changes_S1.append(0)
#                 price_changes_S2.append(0)
#             else:
#                 # Check for entry conditions with interpolation.
#                 # Long spread entry: when z crosses below -entry_threshold.
#                 if (prev_z is not None) and (prev_z > -entry_threshold) and (current_z <= -entry_threshold):
#                     lam_entry = (-entry_threshold - prev_z) / (current_z - prev_z)
#                     interpolated_entry_S1 = prev_S1 + lam_entry * (current_S1 - prev_S1)
#                     interpolated_entry_S2 = prev_S2 + lam_entry * (current_S2 - prev_S2)
#                     entry_price_S1 = interpolated_entry_S1
#                     entry_price_S2 = interpolated_entry_S2
#                     entry_z = -entry_threshold
#                     position = 1
#                     trade_entries.append({
#                         'time': current_index,
#                         'S1': entry_price_S1,
#                         'S2': entry_price_S2,
#                         'z': entry_z,
#                         'position': 1
#                     })
#                     # Immediately check for exit conditions within same iteration.
#                     # For a long spread, normal exit if current_z >= -exit_threshold.
#                     if current_z >= -exit_threshold:
#                         lam_exit = (-exit_threshold - (-entry_threshold)) / (current_z - (-entry_threshold))
#                         interpolated_exit_S1 = entry_price_S1 + lam_exit * (current_S1 - entry_price_S1)
#                         interpolated_exit_S2 = entry_price_S2 + lam_exit * (current_S2 - entry_price_S2)
#                         exit_z = -exit_threshold
#                         trade_exits.append({
#                             'time': current_index,
#                             'S1': interpolated_exit_S1,
#                             'S2': interpolated_exit_S2,
#                             'z': exit_z,
#                             'exit_type': 'win'
#                         })
#                         change_S1 = (interpolated_exit_S1 - entry_price_S1) / entry_price_S1 * 100
#                         change_S2 = (entry_price_S2 - interpolated_exit_S2) / entry_price_S2 * 100
#                         price_changes_S1.append(change_S1)
#                         price_changes_S2.append(change_S2)
#                         position = 0  # trade closes immediately
#                     # Or, stop-loss exit if current_z <= -stop_loss_threshold.
#                     elif current_z <= -stop_loss_threshold:
#                         lam_exit = (-stop_loss_threshold - (-entry_threshold)) / (current_z - (-entry_threshold))
#                         interpolated_exit_S1 = entry_price_S1 + lam_exit * (current_S1 - entry_price_S1)
#                         interpolated_exit_S2 = entry_price_S2 + lam_exit * (current_S2 - entry_price_S2)
#                         exit_z = -stop_loss_threshold
#                         trade_exits.append({
#                             'time': current_index,
#                             'S1': interpolated_exit_S1,
#                             'S2': interpolated_exit_S2,
#                             'z': exit_z,
#                             'exit_type': 'loss'
#                         })
#                         change_S1 = (interpolated_exit_S1 - entry_price_S1) / entry_price_S1 * 100
#                         change_S2 = (entry_price_S2 - interpolated_exit_S2) / entry_price_S2 * 100
#                         price_changes_S1.append(change_S1)
#                         price_changes_S2.append(change_S2)
#                         stop_out = True
#                         position = 0
#                     positions.append(position)
#                     # If no immediate exit condition is met, position remains active.
#                     if position != 0:
#                         price_changes_S1.append(0)
#                         price_changes_S2.append(0)
#                 # Short spread entry: when z crosses above entry_threshold.
#                 elif (prev_z is not None) and (prev_z < entry_threshold) and (current_z >= entry_threshold):
#                     lam_entry = (entry_threshold - prev_z) / (current_z - prev_z)
#                     interpolated_entry_S1 = prev_S1 + lam_entry * (current_S1 - prev_S1)
#                     interpolated_entry_S2 = prev_S2 + lam_entry * (current_S2 - prev_S2)
#                     entry_price_S1 = interpolated_entry_S1
#                     entry_price_S2 = interpolated_entry_S2
#                     entry_z = entry_threshold
#                     position = -1
#                     trade_entries.append({
#                         'time': current_index,
#                         'S1': entry_price_S1,
#                         'S2': entry_price_S2,
#                         'z': entry_z,
#                         'position': -1
#                     })
#                     # Immediately check for exit conditions.
#                     # For a short spread, normal exit if current_z <= exit_threshold.
#                     if current_z <= exit_threshold:
#                         lam_exit = (exit_threshold - entry_threshold) / (current_z - entry_threshold)
#                         interpolated_exit_S1 = entry_price_S1 + lam_exit * (current_S1 - entry_price_S1)
#                         interpolated_exit_S2 = entry_price_S2 + lam_exit * (current_S2 - entry_price_S2)
#                         exit_z = exit_threshold
#                         trade_exits.append({
#                             'time': current_index,
#                             'S1': interpolated_exit_S1,
#                             'S2': interpolated_exit_S2,
#                             'z': exit_z,
#                             'exit_type': 'win'
#                         })
#                         change_S1 = (entry_price_S1 - interpolated_exit_S1) / entry_price_S1 * 100
#                         change_S2 = (interpolated_exit_S2 - entry_price_S2) / entry_price_S2 * 100
#                         price_changes_S1.append(change_S1)
#                         price_changes_S2.append(change_S2)
#                         position = 0
#                     # Or, stop-loss exit if current_z >= stop_loss_threshold.
#                     elif current_z >= stop_loss_threshold:
#                         lam_exit = (stop_loss_threshold - entry_threshold) / (current_z - entry_threshold)
#                         interpolated_exit_S1 = entry_price_S1 + lam_exit * (current_S1 - entry_price_S1)
#                         interpolated_exit_S2 = entry_price_S2 + lam_exit * (current_S2 - entry_price_S2)
#                         exit_z = stop_loss_threshold
#                         trade_exits.append({
#                             'time': current_index,
#                             'S1': interpolated_exit_S1,
#                             'S2': interpolated_exit_S2,
#                             'z': exit_z,
#                             'exit_type': 'loss'
#                         })
#                         change_S1 = (entry_price_S1 - interpolated_exit_S1) / entry_price_S1 * 100
#                         change_S2 = (interpolated_exit_S2 - entry_price_S2) / entry_price_S2 * 100
#                         price_changes_S1.append(change_S1)
#                         price_changes_S2.append(change_S2)
#                         stop_out = True
#                         position = 0
#                     positions.append(position)
#                     if position != 0:
#                         price_changes_S1.append(0)
#                         price_changes_S2.append(0)
#                 else:
#                     positions.append(0)
#                     price_changes_S1.append(0)
#                     price_changes_S2.append(0)
#         else:
#             # A trade is active from a previous iteration.
#             if position == 1:  # Long spread trade.
#                 if (prev_z is not None) and (prev_z < -exit_threshold) and (current_z >= -exit_threshold): #If the z-score crosses upward past -exit_threshold.
#                     lam = (-exit_threshold - prev_z) / (current_z - prev_z)
#                     interpolated_S1 = prev_S1 + lam * (current_S1 - prev_S1)
#                     interpolated_S2 = prev_S2 + lam * (current_S2 - prev_S2)
#                     exit_price_S1 = interpolated_S1
#                     exit_price_S2 = interpolated_S2
#                     exit_z = -exit_threshold
#                     trade_exits.append({
#                         'time': current_index,
#                         'S1': exit_price_S1,
#                         'S2': exit_price_S2,
#                         'z': exit_z,
#                         'exit_type': 'win'
#                     })
#                     change_S1 = (exit_price_S1 - entry_price_S1) / entry_price_S1 * 100
#                     change_S2 = (entry_price_S2 - exit_price_S2) / entry_price_S2 * 100
#                     price_changes_S1.append(change_S1)
#                     price_changes_S2.append(change_S2)
#                     position = 0
#                     entry_price_S1 = None
#                     entry_price_S2 = None
#                 elif (prev_z is not None) and (prev_z > -stop_loss_threshold) and (current_z <= -stop_loss_threshold):
#                     lam = (-stop_loss_threshold - prev_z) / (current_z - prev_z)
#                     interpolated_S1 = prev_S1 + lam * (current_S1 - prev_S1)
#                     interpolated_S2 = prev_S2 + lam * (current_S2 - prev_S2)
#                     exit_price_S1 = interpolated_S1
#                     exit_price_S2 = interpolated_S2
#                     exit_z = -stop_loss_threshold
#                     trade_exits.append({
#                         'time': current_index,
#                         'S1': exit_price_S1,
#                         'S2': exit_price_S2,
#                         'z': exit_z,
#                         'exit_type': 'loss'
#                     })
#                     change_S1 = (exit_price_S1 - entry_price_S1) / entry_price_S1 * 100
#                     change_S2 = (entry_price_S2 - exit_price_S2) / entry_price_S2 * 100
#                     price_changes_S1.append(change_S1)
#                     price_changes_S2.append(change_S2)
#                     stop_out = True
#                     position = 0
#                     entry_price_S1 = None
#                     entry_price_S2 = None
#                 else:
#                     price_changes_S1.append(0)
#                     price_changes_S2.append(0)
#                 positions.append(position)
#             elif position == -1:  # Short spread trade.
#                 if (prev_z is not None) and (prev_z > exit_threshold) and (current_z <= exit_threshold): #If the z-score crosses downward past exit_threshold.
#                     lam = (exit_threshold - prev_z) / (current_z - prev_z)
#                     interpolated_S1 = prev_S1 + lam * (current_S1 - prev_S1)
#                     interpolated_S2 = prev_S2 + lam * (current_S2 - prev_S2)
#                     exit_price_S1 = interpolated_S1
#                     exit_price_S2 = interpolated_S2
#                     exit_z = exit_threshold
#                     trade_exits.append({
#                         'time': current_index,
#                         'S1': exit_price_S1,
#                         'S2': exit_price_S2,
#                         'z': exit_z,
#                         'exit_type': 'win'
#                     })
#                     change_S1 = (entry_price_S1 - exit_price_S1) / entry_price_S1 * 100
#                     change_S2 = (exit_price_S2 - entry_price_S2) / entry_price_S2 * 100
#                     price_changes_S1.append(change_S1)
#                     price_changes_S2.append(change_S2)
#                     position = 0
#                     entry_price_S1 = None
#                     entry_price_S2 = None
#                 elif (prev_z is not None) and (prev_z < stop_loss_threshold) and (current_z >= stop_loss_threshold):
#                     lam = (stop_loss_threshold - prev_z) / (current_z - prev_z)
#                     interpolated_S1 = prev_S1 + lam * (current_S1 - prev_S1)
#                     interpolated_S2 = prev_S2 + lam * (current_S2 - prev_S2)
#                     exit_price_S1 = interpolated_S1
#                     exit_price_S2 = interpolated_S2
#                     exit_z = stop_loss_threshold
#                     trade_exits.append({
#                         'time': current_index,
#                         'S1': exit_price_S1,
#                         'S2': exit_price_S2,
#                         'z': exit_z,
#                         'exit_type': 'loss'
#                     })
#                     change_S1 = (entry_price_S1 - exit_price_S1) / entry_price_S1 * 100
#                     change_S2 = (exit_price_S2 - entry_price_S2) / entry_price_S2 * 100
#                     price_changes_S1.append(change_S1)
#                     price_changes_S2.append(change_S2)
#                     stop_out = True
#                     position = 0
#                     entry_price_S1 = None
#                     entry_price_S2 = None
#                 else:
#                     price_changes_S1.append(0)
#                     price_changes_S2.append(0)
#                 positions.append(position)

#         prev_z = current_z
#         prev_S1 = current_S1
#         prev_S2 = current_S2
#         prev_index = current_index

#     positions = pd.Series(positions, index=spread_series.index)

#     print(f"Total trades closed: {len(trade_exits)} (Wins={sum(1 for e in trade_exits if e['exit_type']=='win')}, Losses={sum(1 for e in trade_exits if e['exit_type']=='loss')})")
#     total_trades = len(trade_exits)
#     if total_trades > 0:
#         win_rate = sum(1 for e in trade_exits if e['exit_type']=='win') / total_trades
#         print(f"Win rate: {win_rate:.2f}")

#     return positions, trade_entries, trade_exits, price_changes_S1, price_changes_S2



# Example usage:
# Suppose 'spread_series' is a pandas Series representing the spread between two assets,
# and you wish to compute a rolling z-score over a window of 720 observations (e.g., roughly one month for hourly data).
# zscore_series, positions_series = backtest_pair_rolling(spread_series, window_size=720, entry_threshold=1.0, exit_threshold=0.0)



# def simulate_true_strategy_rolling(S1, S2, positions, beta_series):
#     """
#     Simulate the true performance of a pairs trading strategy using a dynamic (rolling) hedge ratio.
    
#     In this strategy:
#       - A long spread (position = +1) means:
#           * Long 1 unit of S1
#           * Short beta_series[t] units of S2 at time t
#       - A short spread (position = -1) means:
#           * Short 1 unit of S1
#           * Long beta_series[t] units of S2 at time t
    
#     The profit or loss from a trade between time t-1 and t is computed as:
#         For a long spread:
#             pnl = (S1[t] - S1[t-1]) - beta_series[t-1] * (S2[t] - S2[t-1])
#         For a short spread:
#             pnl = -(S1[t] - S1[t-1]) + beta_series[t-1] * (S2[t] - S2[t-1])
    
#     We use the previous period's position (i.e. positions.shift(1)) and the previous period's
#     beta (i.e. beta_series.shift(1)) to avoid lookahead bias.
    
#     Parameters:
#         S1 (pd.Series): Time series of asset 1 prices.
#         S2 (pd.Series): Time series of asset 2 prices.
#         positions (pd.Series): Trading signals (1 for long spread, -1 for short spread, 0 for no position).
#         beta_series (pd.Series): A time series of hedge ratios computed using a rolling regression.
    
#     Returns:
#         tuple: A tuple containing:
#             - pnl (pd.Series): The period-by-period profit and loss of the strategy.
#             - cum_pnl (pd.Series): The cumulative profit and loss over time.
#     """
#     # Calculate the changes in asset prices.
#     delta_S1 = S1.diff()
#     delta_S2 = S2.diff()
    
#     # Shift positions and beta_series by one period to avoid lookahead bias.
#     shifted_positions = positions.shift(1)
#     shifted_beta = beta_series.shift(1)
    
#     # Compute the period-by-period PnL using the dynamic beta.
#     # For a long spread (position = 1), pnl = (ΔS1) - beta * (ΔS2).
#     # For a short spread (position = -1), pnl = -(ΔS1) + beta * (ΔS2).
#     pnl = shifted_positions * (delta_S1 - shifted_beta * delta_S2)
#     pnl = pnl.fillna(0)
    
#     # Compute cumulative PnL.
#     cum_pnl = pnl.cumsum()
    
#     return pnl, cum_pnl



# def simulate_strategy_trade_pnl(S1, S2, positions,initial_capital, beta_series=None, tx_cost=None):
#     """
#     Compute the profit (or loss) for each trade by measuring the price change from the time the trade is opened
#     until it is closed, and adjust for transaction costs if provided.
    
#     For a long spread (positions = +1):
#       - You go long S1 and short S2.
#       - At entry, allocate capital as:
#             Notional_S1 = initial_capital / (1 + beta_entry)
#             Notional_S2 = beta_entry * Notional_S1.
#       - Shares are:
#             shares_S1 = Notional_S1 / S1_entry,
#             shares_S2 = Notional_S2 / S2_entry.
#       - Gross profit is:
#             profit = shares_S1 * (S1_exit - S1_entry) + shares_S2 * (S2_entry - S2_exit).
    
#     For a short spread (positions = -1):
#       - You go short S1 and long S2.
#       - Gross profit is:
#             profit = shares_S1 * (S1_entry - S1_exit) + shares_S2 * (S2_exit - S2_entry).
    
#     Transaction fees are applied to both the entry and exit of each leg.
#       - For example, if tx_cost = 0.001 (0.10%), then each trade (buy or sell) on each asset is charged 0.10% of the transaction value.
    
#     Args:
#         S1 (pd.Series): Price series for asset S1.
#         S2 (pd.Series): Price series for asset S2.
#         positions (pd.Series): Trading signals (0 = flat, +1 = long spread, -1 = short spread).
#         beta_series (pd.Series or float, optional): Hedge ratio(s). If None, beta is assumed to be 1.0.
#         initial_capital (float, optional): The capital allocated per trade (default 1000.0).
#         tx_cost (float, optional): Transaction cost per trade as a fraction (e.g., 0.001 for 0.10%).
#                                    If None, no transaction costs are applied.
    
#     Returns:
#         tuple: A tuple containing:
#             - trade_profits (list): A list of net profit values for each closed trade.
#             - cumulative_profit (pd.Series): Cumulative profit over time (indexed by trade exit times).
#             - entry_indices (list): A list of indices (timestamps) when trades were opened.
#             - exit_indices (list): A list of indices (timestamps) when trades were closed.
#     """
    
#     # If no beta_series is provided, assume beta=1.0.
#     if beta_series is None:
#         beta_series = pd.Series(1.0, index=S1.index)
#     elif not isinstance(beta_series, pd.Series):
#         beta_series = pd.Series(beta_series, index=S1.index)
    
#     trade_profits = []
#     entry_indices = []
#     exit_indices = []

#     long_spread = False
#     short_spread = False

#     long_spread_loss_count = 0
#     short_spread_loss_count = 0

#     number_of_dual_leg_profits = 0
    
#     in_trade = False  # Flag indicating whether a trade is active.
#     trade_direction = 0  # +1 for long spread, -1 for short spread.
#     entry_index = None
#     entry_price_S1 = None
#     entry_price_S2 = None
#     beta_entry = None  # Hedge ratio at entry.

    
#     # Loop over the positions series.
#     for t in range(len(positions)):
#         current_pos = positions.iloc[t]
#         if not in_trade:
#             # Look for a trade entry: when the position changes from 0 to nonzero.
#             if current_pos != 0:
#                 in_trade = True
#                 trade_direction = current_pos

#                 if trade_direction == 1:
#                     long_spread = True
#                 elif trade_direction == -1:
#                     short_spread = True


#                 entry_index = positions.index[t]
#                 entry_price_S1 = S1.iloc[t]
#                 entry_price_S2 = S2.iloc[t]
#                 beta_entry = beta_series.iloc[t]
#                 entry_indices.append(entry_index)
#         else:
#             # A trade is active; check for trade exit (when the position returns to 0).
#             if current_pos == 0:
#                 exit_index = positions.index[t]
#                 exit_price_S1 = S1.iloc[t]
#                 exit_price_S2 = S2.iloc[t]
#                 exit_indices.append(exit_index)
                
#                 # Compute notional allocation based on initial capital and beta_entry.
#                 Notional_S1 = initial_capital / (1.0 + beta_entry)
#                 Notional_S2 = beta_entry * Notional_S1

                
#                 # Compute share counts at entry.
#                 shares_S1 = Notional_S1 / entry_price_S1
#                 shares_S2 = Notional_S2 / entry_price_S2

                
#                 # Compute gross profit based on trade direction.
#                 if trade_direction == 1:
#                     # Long spread: long S1, short S2.
#                     gross_profit_S1 = shares_S1 * (exit_price_S1 - entry_price_S1)
#                     gross_profit_S2 = shares_S2 * (entry_price_S2 - exit_price_S2)
#                     gross_profit = gross_profit_S1 + gross_profit_S2

#                     if (gross_profit_S1 > 0 and gross_profit_S2 > 0):

#                         number_of_dual_leg_profits += 1
#                         print(f"Dual Leg profit: {gross_profit_S1}, {gross_profit_S2}")

#                     else:

#                         print(f"Dual Leg loss: {gross_profit_S1}, {gross_profit_S2}")


#                 elif trade_direction == -1:
#                     # Short spread: short S1, long S2.
#                     gross_profit_S1 = shares_S1 * (entry_price_S1 - exit_price_S1)
#                     gross_profit_S2 = shares_S2 * (exit_price_S2 - entry_price_S2)
#                     gross_profit = gross_profit_S1 + gross_profit_S2

#                     if (gross_profit/initial_capital).item() < 0:

#                         short_spread_loss_count += 1

#                     if(gross_profit_S1 > 0 and gross_profit_S2 > 0):

#                         number_of_dual_leg_profits += 1
#                         print(f"Dual Leg profit: {gross_profit_S1}, {gross_profit_S2}")
#                     else:

#                         print(f"Dual Leg loss: {gross_profit_S1}, {gross_profit_S2}")

#                 else:
#                     gross_profit = 0.0
                
#                 # If transaction costs are provided, calculate fees for each leg at entry and exit.
#                 if tx_cost is not None:
#                     fee_S1_entry = tx_cost * (shares_S1 * entry_price_S1)
#                     fee_S1_exit = tx_cost * (shares_S1 * exit_price_S1)
#                     fee_S2_entry = tx_cost * (shares_S2 * entry_price_S2)
#                     fee_S2_exit = tx_cost * (shares_S2 * exit_price_S2)
#                     total_fees = fee_S1_entry + fee_S1_exit + fee_S2_entry + fee_S2_exit
#                 else:
#                     total_fees = 0.0
                
                
#                 #print(f"Trade Profit: {gross_profit}, Total Fees: {total_fees}")

#                 # Net trade profit is gross profit minus total transaction fees.
#                 net_trade_profit = gross_profit - total_fees
                
#                 #If the trade is a loss, increment the loss count
#                 if net_trade_profit/initial_capital < 0:

#                     if long_spread:
#                         long_spread_loss_count += 1
#                     elif short_spread:
#                         short_spread_loss_count += 1
                

                
#                 trade_profits.append(net_trade_profit)
                
#                 # Reset trade state.
#                 in_trade = False
#                 trade_direction = 0
#                 entry_index = None
#                 entry_price_S1 = None
#                 entry_price_S2 = None
#                 beta_entry = None

#                 # Reset spread direction flags.
#                 long_spread = False
#                 short_spread = False
                
                
#     # Compute cumulative profit from the list of trade profits.
#     cumulative_profit = np.cumsum(trade_profits)
#     cumulative_profit_series = pd.Series(cumulative_profit, index=exit_indices)
    
#     return trade_profits, cumulative_profit_series, entry_indices, exit_indices, long_spread_loss_count, short_spread_loss_count, number_of_dual_leg_profits

def simulate_strategy_trade_pnl(trade_entries, trade_exits, initial_capital, beta_series, tx_cost):
    """
    Compute the profit (or loss) for each trade by measuring the price change from the interpolated
    entry to the interpolated exit, using the paired trade_entries and trade_exits objects.
    
    For a long spread (trade entry 'position' = +1):
      - You go long S1 and short S2.
      - At entry, allocate capital as:
            Notional_S1 = initial_capital / (1 + beta_entry)
            Notional_S2 = beta_entry * Notional_S1.
      - Shares are calculated using the interpolated entry prices:
            shares_S1 = Notional_S1 / entry_price_S1,
            shares_S2 = Notional_S2 / entry_price_S2.
      - Gross profit is:
            profit = shares_S1 * (exit_price_S1 - entry_price_S1) + shares_S2 * (entry_price_S2 - exit_price_S2).
    
    For a short spread (trade entry 'position' = -1):
      - You go short S1 and long S2.
      - Gross profit is:
            profit = shares_S1 * (entry_price_S1 - exit_price_S1) + shares_S2 * (exit_price_S2 - entry_price_S2).
    
    Transaction fees are applied to both the entry and exit of each leg:
      - For example, if tx_cost = 0.001 (0.10%), then each trade (buy or sell) on each asset is charged 0.10% of the transaction value.
    
    Args:
        S1 (pd.Series): Price series for asset S1.
        S2 (pd.Series): Price series for asset S2.
        trade_entries (list): List of dictionaries for trade entries with keys:
                              'time', 'S1', 'S2', 'z', 'position'.
        trade_exits (list): List of dictionaries for trade exits with keys:
                            'time', 'S1', 'S2', 'z', 'exit_type'.
        initial_capital (float): The capital allocated per trade.
        beta_series (pd.Series or float, optional): Hedge ratio(s). If None, beta is assumed to be 1.0.
        tx_cost (float, optional): Transaction cost per trade as a fraction (e.g., 0.001 for 0.10%).
                                   If None, no transaction costs are applied.
    
    Returns:
        tuple: A tuple containing:
            - trade_profits (list): A list of net profit values for each closed trade.
            - cumulative_profit_series (pd.Series): Cumulative profit over time (indexed by trade exit times).
            - entry_times (list): A list of trade entry timestamps.
            - exit_times (list): A list of trade exit timestamps.
    """


    # # If beta_series is not provided, use 1.0.
    # if beta_series is None:
    #     beta_series = pd.Series(1.0, index=S1.index)
    # elif not isinstance(beta_series, pd.Series):
    #     beta_series = pd.Series(beta_series, index=S1.index)
    
    trade_profits = []
    net_trade_profits_S1 = []
    net_trade_profits_S2 = []
    entry_times = []
    exit_times = []
    
    # For tracking loss counts and dual-leg profits (if needed).
    long_spread_loss_count = 0
    short_spread_loss_count = 0
    number_of_dual_leg_profits = 0

    trade_count = 0

    # Loop over the paired trades.
    # It is assumed that trade_entries and trade_exits are already properly paired in order.
    for entry, exit in zip(trade_entries, trade_exits):
        entry_time = pd.to_datetime(entry['time'])
        exit_time = pd.to_datetime(exit['time'])

        # Determine beta at the entry time using the provided beta_series.
        beta_entry = beta_series.loc[entry_time] if entry_time in beta_series.index else print("Error: Beta not found at entry time.")

        #Take absolute value of beta_entry to avoid negative notional allocation or a blow up of the notional value as B approaches -1
        beta_abs = abs(beta_entry)

        # Notional allocation.
        Notional_S1 = initial_capital / (1.0 + beta_abs)
        Notional_S2 = beta_abs * Notional_S1

        # Compute share counts based on the interpolated entry prices.
        shares_S1 = Notional_S1 / entry['S1']
        shares_S2 = Notional_S2 / entry['S2']

        # Calculate gross profit depending on trade direction.
        if entry['position'] == 1:  # Long spread: long S1, short S2.

            if beta_entry > 0:
                gross_profit_S1 = shares_S1 * (exit['S1'] - entry['S1'])
                gross_profit_S2 = shares_S2 * (entry['S2'] - exit['S2'])
                gross_profit = gross_profit_S1 + gross_profit_S2
            else:
                #For negative beta a long spread means long S1 and long S2
                gross_profit_S1 = shares_S1 * (exit['S1'] - entry['S1'])
                gross_profit_S2 = shares_S2 * (exit['S2'] - entry['S2'])
                gross_profit = gross_profit_S1 + gross_profit_S2

            # For diagnostic purposes, you can check dual leg performance.
            if (gross_profit_S1 > 0 and gross_profit_S2 > 0):
                number_of_dual_leg_profits += 1

        elif entry['position'] == -1:  # Short spread: short S1, long S2.

            if beta_entry > 0:
                gross_profit_S1 = shares_S1 * (entry['S1'] - exit['S1'])
                gross_profit_S2 = shares_S2 * (exit['S2'] - entry['S2'])
                gross_profit = gross_profit_S1 + gross_profit_S2
            else:
                #For negative beta a short spread means short S1 and short S2
                gross_profit_S1 = shares_S1 * (entry['S1'] - exit['S1'])
                gross_profit_S2 = shares_S2 * (entry['S2'] - exit['S2'])
                gross_profit = gross_profit_S1 + gross_profit_S2

            if (gross_profit_S1 > 0 and gross_profit_S2 > 0):
                number_of_dual_leg_profits += 1
        else:
            gross_profit = 0.0

        # Compute transaction fees if provided.
        if tx_cost is not None:
            fee_S1_entry = tx_cost * (shares_S1 * entry['S1'])
            fee_S1_exit  = tx_cost * (shares_S1 * exit['S1'])
            fee_S2_entry = tx_cost * (shares_S2 * entry['S2'])
            fee_S2_exit  = tx_cost * (shares_S2 * exit['S2'])
            total_fees = fee_S1_entry + fee_S1_exit + fee_S2_entry + fee_S2_exit
        else:
            total_fees = 0.0

        #Calculate net trade profit for individual legs


        net_trade_profit_S1 = gross_profit_S1 - (fee_S1_entry + fee_S1_exit)
        net_trade_profit_S2 = gross_profit_S2 - (fee_S2_entry + fee_S2_exit)
        net_trade_profits_S1.append(net_trade_profit_S1)
        net_trade_profits_S2.append(net_trade_profit_S2)

        net_trade_profit = gross_profit - total_fees

        # print(f"Trade Num: {trade_count}")
        # print("-----------------------------------------------")
        # print(f"Trade type: {entry['position']}, Entry time: {entry_time}, Exit time: {exit_time}") #If entry['position'] == 1, long spread; if -1, short spread.
        # print(f"Net Trade profit (includes fees): {net_trade_profit}")
        # print(f"Beta at entry (Not the absolute value): {beta_entry}")
        # print(f"Notional S1: {Notional_S1}, Notional S2: {Notional_S2}, Shares S1 : {shares_S1}, Shares S2: {shares_S2}")
        # print(f"Percentage change S1: {((exit['S1'] - entry['S1']) / entry['S1']) * 100:.2f}%, Percentage change S2: {((exit['S2'] - entry['S2']) /entry['S2']) * 100:.2f}%")
        trade_profits.append(net_trade_profit)
        entry_times.append(entry_time)
        exit_times.append(exit_time)

        # #For compounding
        # #----------------------------------------------------------------------------------------------------------------
        # print("Trade profit: -->", net_trade_profit)
        # initial_capital = initial_capital + net_trade_profit #Update the initial capital with the net trade profit for this trade
        # print("New initial capital: -->", initial_capital)

        # Count losses.
        if net_trade_profit < 0:
            if entry['position'] == 1:
                long_spread_loss_count += 1
            elif entry['position'] == -1:
                short_spread_loss_count += 1

        # Count the number of trades.        
        trade_count += 1
        

    # Compute cumulative profit series.
    cumulative_profit = np.cumsum(trade_profits)
    cumulative_profit_series = pd.Series(cumulative_profit, index=exit_times)
    
    #initial_capital = 10_000.0 #Reset initial capital

    # # Print diagnostics.
    # print(f"Total trades: {len(trade_profits)}")
    # print(f"Number of profitable trades (proft > 0): {sum(1 for profit in trade_profits if profit > 0)}")
    # print(f"Number of non-profitable trades (proft < 0): {sum(1 for profit in trade_profits if profit < 0)}")
    # print(f"Total return €: {cumulative_profit[-1]:.2f}")
    # print(f"Total return %: {(cumulative_profit[-1] / initial_capital) * 100:.2f}%")
    # print(f"Long spread losses: {long_spread_loss_count}, Short spread losses: {short_spread_loss_count}")
    # print(f"Number of Dual-leg profitable trades: {number_of_dual_leg_profits}")
    # print(f"Dual leg trade profit rate: {number_of_dual_leg_profits / len(trade_profits) * 100:.2f}%")



    return trade_profits, net_trade_profits_S1, net_trade_profits_S2,cumulative_profit_series, entry_times, exit_times


# Example usage:
# daily_pnl, cum_pnl, entry_indices, exit_indices = simulate_strategy_trade_pnl(S1, S2, positions, beta_series, initial_capital=1000.0, tx_cost=0.001)



def plot_trading_simulation(
    S1, 
    S2, 
    sym1, 
    sym2, 
    zscore, 
    positions, 
    trade_profits,
    entry_threshold,
    stop_loss_threshold,
    trade_entries=None,  # list of dicts: { 'time', 'S1', 'S2', 'z', 'position' }
    trade_exits=None,    # list of dicts: { 'time', 'S1', 'S2', 'z', 'exit_type' }
    window_start=None,
    window_end=None
): 
    """
    Plot the trading simulation results including:
      - Underlying stock prices (S1, S2)
      - Rolling z-score of the spread
      - Trading positions
      - Cumulative PnL
      
    The plot highlights trade intervals:
      - Light green for trades that ended as a "win"
      - Light red for trades that ended as a "loss"
    and draws vertical black lines at the start and end of each trade.
    
    Additionally, the function plots scatter points at the exact (interpolated) entry and exit points 
    for both the price series and the z-score.
    
    Args:
        S1 (pd.Series): Price series for asset S1.
        S2 (pd.Series): Price series for asset S2.
        sym1 (str): Label for asset S1.
        sym2 (str): Label for asset S2.
        zscore (pd.Series): Rolling z-score series of the spread.
        positions (pd.Series): Trading signals (0=flat, +1=long spread, -1=short spread).
        entry_threshold (float): The entry threshold used in the back-test.
        stop_loss_threshold (float): The stop-loss threshold used in the back-test.
        cum_pnl (pd.Series): Cumulative profit curve.
        trade_entries (list): List of dictionaries for trade entries with keys: 'time', 'S1', 'S2', 'z', 'position'.
        trade_exits (list): List of dictionaries for trade exits with keys: 'time', 'S1', 'S2', 'z', 'exit_type'.
        window_start, window_end: Optional window slicing for the data. Can be strings or Timestamps.
    """

    # Convert window bounds to Timestamps if given as strings.
    # if window_start is not None:
    #     window_start = pd.to_datetime(window_start)
    # if window_end is not None:
    #     window_end = pd.to_datetime(window_end)

    # Slice data series to the window.
    if window_start is not None or window_end is not None:
        S1 = S1.loc[window_start:window_end]
        S2 = S2.loc[window_start:window_end]
        zscore = zscore.loc[window_start:window_end]
        positions = positions.loc[window_start:window_end]
        #cum_pnl = cum_pnl.loc[window_start:window_end]

    # Pair trade entries and exits into a unified trade list.
    trades = []

    if trade_entries is not None and trade_exits is not None:
        # First, sort both lists by time.
        trade_entries = sorted(trade_entries, key=lambda e: pd.to_datetime(e['time']))
        trade_exits = sorted(trade_exits, key=lambda e: pd.to_datetime(e['time']))
        # Pair them using zip.
        paired_trades = list(zip(trade_entries, trade_exits)) #Remeber trade entries and exits are dictionaries!📌
        # Filter to keep only trades that fully occur within the window.
        filtered_trades = []
        for entry, exit in paired_trades:
            entry_time = pd.to_datetime(entry['time'])
            exit_time = pd.to_datetime(exit['time'])
            if (window_start is None or entry_time >= window_start) and (window_end is None or exit_time <= window_end):
                filtered_trades.append((entry, exit))
        trades = [(entry['time'], exit['time'], exit['exit_type']) for entry, exit in filtered_trades]
        # Also update trade_entries and trade_exits to the filtered ones:
        trade_entries = [entry for entry, exit in filtered_trades]
        trade_exits = [exit for entry, exit in filtered_trades]
    # else:
    #     # Fallback: derive trades from positions (less preferred)
    #     current_pos = 0
    #     trade_start = None
    #     for i in range(len(positions)):
    #         idx = positions.index[i]
    #         pos = positions.iloc[i]
    #         prev_pos = positions.iloc[i-1] if i > 0 else 0
    #         if pos != 0 and prev_pos == 0:
    #             trade_start = idx
    #             current_pos = pos
    #         if pos == 0 and prev_pos != 0:
    #             trade_end = idx
    #             outcome = "unknown"
    #             trades.append((trade_start, trade_end, outcome))
    #             trade_start = None
    #             current_pos = 0

    # print(f"Number of trade entries: {len(trade_entries) if trade_entries is not None else 'N/A'}")
    # print(f"Number of trade exits: {len(trade_exits) if trade_exits is not None else 'N/A'}")
    # print(f"Total paired trades: {len(trades)}")
    
    plt.figure(figsize=(15, 20))

    # Subplot 1: Underlying Price Series with Trade Interval Highlights.
    plt.subplot(5, 1, 1)
    ax1 = plt.gca()
    ax2 = ax1.twinx()

    ax1.plot(S1, label=sym1, color='blue')
    ax2.plot(S2, label=sym2, color='red')
    ax1.set_ylabel(sym1, color='blue')
    ax2.set_ylabel(sym2, color='red')
    plt.title(f"Underlying Prices: {sym1} and {sym2}")

    #print("Trades in window:")
    # Highlight each trade interval.
    i = 0
    for (start_time, end_time, outcome) in trades:
        i += 1
        print(f"Trade {i} : {start_time} to {end_time} ({outcome})")
        if start_time is None or end_time is None:
            continue
        if outcome == "win":
            ax1.axvspan(start_time, end_time, facecolor='lightgreen', alpha=0.3)
        elif outcome == "loss":
            ax1.axvspan(start_time, end_time, facecolor='lightcoral', alpha=0.3)
        else:
            ax1.axvspan(start_time, end_time, facecolor='grey', alpha=0.3)
        ax1.axvline(start_time, color='black', linestyle='-', linewidth=1)
        ax1.axvline(end_time, color='black', linestyle='-', linewidth=1)

    # Plot scatter points for the exact (interpolated) entry prices.
    if trade_entries is not None:
        
        #These bools are used simply to ensure that the labels are only added once to the plot 1️⃣

        plotted_S1_long = plotted_S1_short = plotted_S2_long = plotted_S2_short = False

        for entry in trade_entries:
            t = entry['time']
            price_S1 = entry['S1']
            price_S2 = entry['S2']
            pos = entry['position']
            if pos == 1:  # Long spread: long S1, short S2.
                #print(f"Long Spread Entry: {t}, {price_S1}, {price_S2}")
                if not plotted_S1_long:
                    ax1.scatter(t, price_S1, marker='^', color='green', s=100, label='S1 Long Entry (Interp)')
                    plotted_S1_long = True
                else:
                    ax1.scatter(t, price_S1, marker='^', color='green', s=100)
                if not plotted_S2_long:
                    ax2.scatter(t, price_S2, marker='v', color='red', s=100, label='S2 Short Entry (Interp)')
                    plotted_S2_long = True
                else:
                    ax2.scatter(t, price_S2, marker='v', color='red', s=100)
            elif pos == -1:  # Short spread: short S1, long S2.
                #print(f"Short Spread Entry: {t}, {price_S1}, {price_S2}")
                if not plotted_S1_short:
                    ax1.scatter(t, price_S1, marker='v', color='red', s=100, label='S1 Short Entry (Interp)')
                    plotted_S1_short = True
                else:
                    ax1.scatter(t, price_S1, marker='v', color='red', s=100)
                if not plotted_S2_short:
                    ax2.scatter(t, price_S2, marker='^', color='green', s=100, label='S2 Long Entry (Interp)')
                    plotted_S2_short = True
                else:
                    ax2.scatter(t, price_S2, marker='^', color='green', s=100)
    # else:
    #     # Fallback if trade_entries is not provided.
    #     long_entries = positions[(positions != 0) & (positions.shift(1) == 0) & (positions == 1)].index.tolist()
    #     short_entries = positions[(positions != 0) & (positions.shift(1) == 0) & (positions == -1)].index.tolist()
    #     ax1.scatter(long_entries, S1.loc[long_entries], marker='^', color='green', s=100, label='S1 Long Entry')
    #     ax1.scatter(short_entries, S1.loc[short_entries], marker='v', color='red', s=100, label='S1 Short Entry')
    #     ax2.scatter(long_entries, S2.loc[long_entries], marker='v', color='red', s=100, label='S2 Short Entry')
    #     ax2.scatter(short_entries, S2.loc[short_entries], marker='^', color='green', s=100, label='S2 Long Entry')

    # Subplot 2: Z-Score Plot with Interpolated Entry/Exit Markers.
    plt.subplot(5, 1, 2)
    plt.plot(zscore, label='Z-Score', color='purple', marker='o', markersize=3)
    plt.axhline(0, color='grey', linestyle='--', label='Mean')
    plt.axhline(entry_threshold, color='green', linestyle='--', label='Entry Threshold')
    plt.axhline(-entry_threshold, color='green', linestyle='--')


    if(stop_loss_threshold < 10):
        plt.axhline(stop_loss_threshold, color='red', linestyle='--', label='Stop-Loss Threshold')
        plt.axhline(-stop_loss_threshold, color='red', linestyle='--')
    else:
        print("Stop loss threshold is too high to be plotted")

    plt.title("Rolling Z-Score of Spread")
    # Plot interpolated entry markers for z-score.
    if trade_entries is not None:

        #These bools are used simply to ensure that the labels are only added once to the plot 1️⃣
        plotted_z_entry_long = plotted_z_entry_short = False
        
        for entry in trade_entries:
            t = entry['time']
            z_val = entry['z']
            if entry['position'] == 1:
                if not plotted_z_entry_long:
                    plt.scatter(t, z_val, marker='^', color='green', s=100, label='Long Entry Z (Interp)')
                    plotted_z_entry_long = True
                else:
                    plt.scatter(t, z_val, marker='^', color='green', s=100)
            elif entry['position'] == -1:
                if not plotted_z_entry_short:
                    plt.scatter(t, z_val, marker='v', color='red', s=100, label='Short Entry Z (Interp)')
                    plotted_z_entry_short = True
                else:
                    plt.scatter(t, z_val, marker='v', color='red', s=100)
    # Plot interpolated exit markers for z-score.
    if trade_exits is not None:

        #These bools are used simply to ensure that the labels are only added once to the plot 1️⃣
        plotted_z_exit_win = plotted_z_exit_loss = False

        for exit in trade_exits:
            t = exit['time']
            z_val = exit['z']
            if exit['exit_type'] == 'win':
                if not plotted_z_exit_win:
                    plt.scatter(t, z_val, marker='o', color='blue', s=100, label='Exit Z (Win, Interp)')
                    plotted_z_exit_win = True
                else:
                    plt.scatter(t, z_val, marker='o', color='blue', s=100)
            elif exit['exit_type'] == 'loss':
                if not plotted_z_exit_loss:
                    plt.scatter(t, z_val, marker='o', color='orange', s=100, label='Exit Z (Loss, Interp)')
                    plotted_z_exit_loss = True
                else:
                    plt.scatter(t, z_val, marker='o', color='orange', s=100)

    plt.legend()

    # # Subplot 3: Cumulative Profit Curve.
    # plt.subplot(5, 1, 3)
    # plt.plot(cum_pnl, label="Cumulative Profit", color='green')
    # plt.title("Cumulative Profit Curve")
    # plt.legend()

    # Subplot 4: Trading Positions.
    plt.subplot(5, 1, 4)
    plt.plot(positions, label="Trading Positions", drawstyle='steps-mid')
    plt.title("Trading Positions")
    plt.legend()

    # (Optional) Subplot 5: Trade Returns Histogram.
    plt.subplot(5, 1, 5)
    plt.title("Trade Returns Histogram (Optional)")
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





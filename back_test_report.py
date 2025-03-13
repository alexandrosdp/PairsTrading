

from data_pre_processing import *
from pair_finder import *
from back_tester import *
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os
from matplotlib.backends.backend_pdf import PdfPages


def generate_back_test_report(prices,**params):

    #XVS/USDT_2024_30m & QI/USDT_2024_30m


    crypto_1 = prices.columns[0]
    crypto_2 = prices.columns[1]

    cointegrated_pairs = [(crypto_1,
    crypto_2,
    0.731393,
    0.8501886494196308,
    0.22)]

    #Params:
    window_size = params.get("window_size")
    entry_threshold = params.get("entry_threshold")
    exit_threshold = params.get("exit_threshold")
    stop_loss_threshold = params.get("stop_loss_threshold")
    timeframe_str = params.get("timeframe_str", "Timeframe: 1 Year (30min Data)")
    year_str = params.get("year_str", "Year: 2024")
    min_pass_fraction = params.get("min_pass_fraction")
    significance = params.get("significance")

    high_corr_pairs = []

    #Results from cointegration tests
    cointegrated_pairs, window_results, pair_corr, pass_fraction, avg_pvalue = find_cointegrated_pairs_windows(prices, high_corr_pairs, significance, window_size, min_pass_fraction)

    sym1, sym2 = prices.columns
    print(f"\nTesting strategy on pair: {sym1} and {sym2}...")
    S1 = prices[sym1]
    S2 = prices[sym2]
    
    # Compute the spread series and beta_series 
    spread_series, beta_series, alpha_series = compute_spread_series(S1, S2, window_size)
    #print(f"Hedge ratio (beta) for {sym1} ~ {sym2}: {beta:.4f}")
    
    # Compute rolling z-score using the provided helper function.
    zscore_series, rolling_mean, rolling_std = compute_rolling_zscore(spread_series, window_size)
    
    # Generate trading signals (positions) based on the spread's z-score
    positions_series,  win_indexs, loss_indexs, price_changes_S1, price_changes_S2 = backtest_pair_rolling(spread_series,S1,S2,zscore_series, entry_threshold, exit_threshold, stop_loss_threshold)
    
    initial_capital = 10_000.0
    #tx_cost= 0.00031 #0.031% transaction cost
    tx_cost= 0.00025 #0.025% transaction cost

    trade_profits, cumulative_profit_series, entry_indices, exit_indices, long_spread_loss_count, short_spread_loss_count, number_of_dual_leg_profits = simulate_strategy_trade_pnl(S1, S2, positions_series, initial_capital, beta_series,tx_cost)

    #------------------------------------
    #Average Absolute Percentage Reversion
    #------------------------------------


    #Get prices at the start and end of the first trade
    S1_price_start = S1[entry_indices]
    S2_price_start = S2[entry_indices]

    S1_price_end = S1.loc[exit_indices]
    S2_price_end = S2.loc[exit_indices]


    #Calculate the price changes by converting the series to numpy arrays
    S1_price_start = S1_price_start.to_numpy()
    S2_price_start = S2_price_start.to_numpy()

    S1_price_end = S1_price_end.to_numpy()
    S2_price_end = S2_price_end.to_numpy()

    print("BEFORE ALIGNMENT")

    #Make sure the arrays have the same shape
    print(S1_price_start.shape)
    print(S2_price_start.shape)


    print(S1_price_end.shape)
    print(S2_price_end.shape)


    if S1_price_start.shape > S1_price_end.shape:
        S1_price_start = S1_price_start[:len(S1_price_end)]

    if S2_price_start.shape > S2_price_end.shape:
        S2_price_start = S2_price_start[:len(S2_price_end)]
    
    print("AFTER ALIGNMENT")

    #Make sure the arrays have the same shape
    print(S1_price_start.shape)
    print(S2_price_start.shape)

    print(S1_price_end.shape)
    print(S2_price_end.shape)

    #Get the price changes
    S1_price_change = S1_price_end - S1_price_start
    S2_price_change = S2_price_end - S2_price_start

    #Get percentage price changes
    S1_price_change_percent = S1_price_change/S1_price_start * 100
    S2_price_change_percent = S2_price_change/S2_price_start * 100

    #Get absolute percentage price changes
    S1_price_change_percent_abs = np.abs(S1_price_change_percent)
    S2_price_change_percent_abs = np.abs(S2_price_change_percent)


    avg_abs_s1_price_change_percent = np.mean(S1_price_change_percent_abs)
    avg_abs_s2_price_change_percent = np.mean(S2_price_change_percent_abs)

    Average_Absolute_Percentage_Reversion = (avg_abs_s1_price_change_percent + avg_abs_s2_price_change_percent)/2

    #--------------------------------------------------------------------
    # Other Performance Metrics
    #--------------------------------------------------------------------
    total_trades = len(win_indexs) + len(loss_indexs)
    win_rate = len(win_indexs) / total_trades if total_trades > 0 else 0
    total_return_pct = (cumulative_profit_series.iloc[-1] / initial_capital) * 100 if total_trades > 0 else 0
    
    # Compute maximum drawdown:
    running_max = cumulative_profit_series.cummax()
    drawdown = running_max - cumulative_profit_series
    max_drawdown = drawdown.max()

    # Prepare parameters used in the back-test as a dictionary.
    parameters_used = {
        "Window Size": window_size,
        "Entry Threshold": entry_threshold,
        "Exit Threshold": exit_threshold,
        "Stop Loss Threshold": stop_loss_threshold,
        "Initial Capital": initial_capital,
        "Transaction Cost": tx_cost
    }
    parameters_df = pd.DataFrame(list(parameters_used.items()), columns=["", ""])

    # Prepare a summary DataFrame of metrics.
    summary_metrics = {
        "Total Trades": total_trades,
        "Win Rate": f"{win_rate:.2f}",
        "Total Return (%)": f"{total_return_pct:.2f}",
        "Max Drawdown": f"{max_drawdown:.2f}",
        "Avg Abs % Reversion": f"{Average_Absolute_Percentage_Reversion:.2f}",
        "Pass Fraction": f"{pass_fraction:.2f}",
        "Avg p-value": f"{avg_pvalue:.2f}",
        "Correlation": f"{pair_corr:.5f}"
    }
    #summary_df = pd.DataFrame([summary_metrics])

    summary_df = pd.DataFrame(list(summary_metrics.items()), columns=["Metric", "Value"])
    
    pdf_filename = f"backtest_report_{sym1}_{sym2}.pdf"

    pdf_dir = os.path.dirname(pdf_filename)
    # Create the directory if it doesn't exist.
    if pdf_dir and not os.path.exists(pdf_dir):
        os.makedirs(pdf_dir, exist_ok=True)


    with PdfPages(pdf_filename) as pdf:
        # Page 1: Title and Summary Metrics Table.
        plt.figure(figsize=(8.27, 11.69))  # A4 portrait size in inches.
        plt.axis('off')
        plt.title("Pairs Trading Back Test Report", fontsize=20,weight = 'bold',pad=50)

        # Add subheading with the pair information.
        pair_heading = f"Pair: {sym1} vs {sym2}"
        plt.text(0.5, 0.90, pair_heading, horizontalalignment='center', fontsize=16, transform=plt.gcf().transFigure)
        
        # Add subtitles for timeframe and year below the pair heading.
        plt.text(0.5, 0.86, timeframe_str, horizontalalignment='center', fontsize=14, transform=plt.gcf().transFigure)
        plt.text(0.5, 0.84, year_str, horizontalalignment='center', fontsize=14, transform=plt.gcf().transFigure)

        # Parameters Table (placed at the top half of the page)

        plt.text(0.5, 0.78, "Back-Test Parameters", horizontalalignment='center', fontsize=14, transform=plt.gcf().transFigure)

        param_table = plt.table(cellText=parameters_df.values,
                                loc='upper center',
                                cellLoc='left',
                                bbox=[0.15, 0.65, 0.70, 0.20])
        param_table.auto_set_font_size(False)
        param_table.set_fontsize(12)


        # Summary Metrics Table (placed below the parameters table)
        plt.text(0.5, 0.55, "Summary Metrics", horizontalalignment='center', fontsize=14, transform=plt.gcf().transFigure)

        summary_table = plt.table(cellText=summary_df.values,
                                  loc='center',
                                  cellLoc='left',
                                  bbox=[0.15, 0.35, 0.70, 0.20])
        summary_table.auto_set_font_size(False)
        summary_table.set_fontsize(12)
        
        pdf.savefig()
        plt.close()

        # Page 2: Price Series and Cumulative Profit.
        plt.figure(figsize=(11, 8.5))
        plt.subplot(2, 1, 1)
        plt.plot(S1, label=sym1, color='blue')
        plt.plot(S2, label=sym2, color='red')
        plt.title("Underlying Price Time Series")
        plt.legend()
        plt.subplot(2, 1, 2)
        plt.plot(cumulative_profit_series, label="Cumulative Profit", color='green')
        plt.title("Cumulative Profit Curve")
        plt.legend()
        pdf.savefig()
        plt.close()

        # Page 3: Z-Score Distribution.
        plt.figure(figsize=(11, 8.5))
        plt.hist(zscore_series.dropna(), bins=50, color='skyblue', edgecolor='black')
        plt.title("Distribution of Rolling Z-Score")
        plt.xlabel("Z-Score")
        plt.ylabel("Frequency")
        pdf.savefig()
        plt.close()

        # Page 4: Trade Returns Distribution.
        plt.figure(figsize=(11, 8.5))
        trade_returns_pct = (np.array(trade_profits) / initial_capital) * 100
        plt.hist(trade_returns_pct, bins=50, color='lightgreen', edgecolor='black')
        plt.title("Distribution of Trade Returns (%)")
        plt.xlabel("Return (%)")
        plt.ylabel("Frequency")
        pdf.savefig()
        plt.close()

        # Page 5: Zscore Spread Time Series and Trading Positions.
        plt.figure(figsize=(11, 8.5))
        plt.subplot(2, 1, 1)
        plt.plot(zscore_series, label="Z-Score Spread")
        plt.title("Spread Series")
        plt.legend()
        plt.subplot(2, 1, 2)
        plt.plot(positions_series, drawstyle='steps-mid', label="Positions")
        plt.title("Trading Positions")
        plt.legend()
        pdf.savefig()
        plt.close()

    
    print(f"PDF report generated  ✅: {pdf_filename}")



if __name__ == '__main__':

    prices = pd.read_csv("binance_data/Staked_ETH_Bybit/merged_closing_prices.csv", index_col=0, parse_dates=True)

    params = {
    "window_size": 336,
    "entry_threshold": 4.0,
    "exit_threshold": 0.1,
    "stop_loss_threshold": 25,
    "timeframe_str": "Timeframe: 1 Year (30min Data)",
    "year_str": "Year: 2024",
    "min_pass_fraction": 0.5,
    "significance": 0.05
    }


    generate_back_test_report(prices, **params)
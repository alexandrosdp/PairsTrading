

from data_pre_processing import *
from pair_finder import *
from back_tester import *
import itertools
import pandas as pd
import numpy as np
from experiments import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os
from matplotlib.backends.backend_pdf import PdfPages



def calculate_avg_abs_pct_reversion(trade_entries, trade_exits):
    """
    Calculate the average absolute percentage reversion from trade entry to trade exit
    for both asset price series (S1 and S2), using the interpolated trade data.
    
    Args:
        trade_entries (list): List of dictionaries for trade entries.
            Each dict should have keys: 'time', 'S1', 'S2', 'z', 'position'
        trade_exits (list): List of dictionaries for trade exits.
            Each dict should have keys: 'time', 'S1', 'S2', 'z', 'exit_type'
    
    Returns:
        overall_avg (float): The average of the average absolute percentage changes for S1 and S2.
    """
    S1_pct_changes = []
    S2_pct_changes = []
    
    # Loop over paired trade entries and exits.
    for entry, exit in zip(trade_entries, trade_exits):
        entry_S1 = entry['S1']
        entry_S2 = entry['S2']
        exit_S1 = exit['S1']
        exit_S2 = exit['S2']
        
        # Calculate absolute percentage change for each asset.
        pct_change_S1 = np.abs((exit_S1 - entry_S1) / entry_S1 * 100)
        pct_change_S2 = np.abs((exit_S2 - entry_S2) / entry_S2 * 100)
        
        S1_pct_changes.append(pct_change_S1)
        S2_pct_changes.append(pct_change_S2)
    
    # Compute the average absolute percentage change for S1 and S2.
    avg_abs_S1 = np.mean(S1_pct_changes) if S1_pct_changes else 0
    avg_abs_S2 = np.mean(S2_pct_changes) if S2_pct_changes else 0
    
    # Overall average is the average of the two.
    overall_avg = (avg_abs_S1 + avg_abs_S2) / 2
    
    print(f"Average absolute S1 price change percent: {avg_abs_S1:.2f}%")
    print(f"Average absolute S2 price change percent: {avg_abs_S2:.2f}%")
    print(f"Average of average absolute price changes: {overall_avg:.2f}%")
    
    return overall_avg




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
    initial_capital = params.get("initial_capital")
    tx_cost= params.get("tx_cost")
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
    print(f"\nTesting strategy on pair: {sym1} and {sym2} ...")
    S1 = prices[sym1]
    S2 = prices[sym2]

    # Compute the spread series and beta_series 
    spread_series, beta_series, alpha_series = compute_spread_series(S1, S2, window_size)
    #print(f"Hedge ratio (beta) for {sym1} ~ {sym2}: {beta:.4f}")

    # Compute rolling z-score using the provided helper function.
    zscore_series, rolling_mean, rolling_std = compute_rolling_zscore(spread_series, window_size)

    # Generate trading signals (positions) based on the spread's z-score
    positions, trade_entries, trade_exits = backtest_pair_rolling(spread_series,S1,S2,zscore_series, entry_threshold, exit_threshold, stop_loss_threshold)


    # # Identify trade entry points: position changes from 0 to ±1
    # trade_entries = positions_series[(positions_series != 0) & (positions_series.shift(1) == 0)]
    # long_entries = trade_entries[trade_entries == 1]
    # short_entries = trade_entries[trade_entries == -1]

    trade_profits, net_trade_profits_S1, net_trade_profits_S2,cumulative_profit_series, entry_times, exit_times = simulate_strategy_trade_pnl(trade_entries, trade_exits, initial_capital, beta_series, tx_cost)
    #--------------------------------------------------------------------
    # Performance Metrics
    #--------------------------------------------------------------------

    
    total_trades = len(trade_profits)
    total_closed = len(trade_exits)
    wins  = sum(1 for e in trade_exits if e['exit_type'] == 'win')
    win_rate = (wins / total_closed) 

    #losses= sum(1 for e in trade_exits if e['exit_type'] == 'loss')
    
    total_return_pct = (cumulative_profit_series.iloc[-1] / initial_capital) * 100 

    overall_avg_reversion = calculate_avg_abs_pct_reversion(trade_entries, trade_exits)
    
    # Compute maximum drawdown:
    running_max = cumulative_profit_series.cummax()
    drawdown = running_max - cumulative_profit_series
    max_drawdown = drawdown.max()

    #Experiments

    print(f"Running threshold experiment ...")
    entry_stop_map = {
    "1": (1, 2),
    "2": (2, 3),  
    "3": (3, 4),
    "4": (4, 5),
    "5": (5, 6)
    }

    percentage_returns_list_thresholds = threshold_experiment(prices,entry_stop_map,initial_capital)

    print(f"Running transaction cost experiment ...")

    #Create transaction costs list from Binance (Maker fees)
    tx_costs = {
    "Regular User": 0.001000,
    "VIP 1": 0.000900,
    "VIP 2": 0.000800,
    "VIP 3": 0.000400,
   # "VIP 4": 0.000400,
    "VIP 5": 0.000250,
    "VIP 6": 0.000200,
    "VIP 7": 0.000190,
    "VIP 8": 0.000160,
    "VIP 9": 0.000110
    }

    percentage_returns_list_transaction_costs = transaction_cost_experiment(prices,tx_costs,initial_capital)


    # Prepare parameters used in the back-test as a dictionary.
    parameters_used = {
        "Window Size": window_size,
        "Entry Threshold": entry_threshold,
        "Exit Threshold": exit_threshold,
        "Stop Loss Threshold": stop_loss_threshold,
        "Initial Capital": initial_capital,
        "Transaction Cost (%)": tx_cost * 100
    }

    parameters_df = pd.DataFrame(list(parameters_used.items()), columns=["", ""])

    # Prepare a summary DataFrame of metrics.
    summary_metrics = {
        "Total Trades": total_trades,
        "Win Rate": f"{win_rate:.2f}",
        "Total Return (%)": f"{total_return_pct:.2f}",
        "Max Drawdown": f"{max_drawdown:.2f}",
        "Avg Abs % Reversion": f"{overall_avg_reversion:.2f}",
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
        plt.xlabel("Time")
        plt.ylabel("Profit (€)")
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

        #Plot experiment results

        plt.figure(figsize=(11, 8.5))

        # Prepare data
        x_labels = [f"{key * 100:.3f}" for key in percentage_returns_list_transaction_costs.keys()]
        heights = list(percentage_returns_list_transaction_costs.values())

        # Create bar plot
        bars = plt.bar(x_labels, heights, color='orange')
        plt.xlabel('Transaction costs (%)')
        plt.ylabel('Returns (%)')
        plt.title(f'Percentage returns for {sym1} and {sym2} pair for different transaction costs')
        
        # Add labels on top of each bar
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, height,
                    f'{height:.2f}%', ha='center', va='bottom', fontsize=9)

        pdf.savefig()
        plt.close()

        plt.figure(figsize=(11, 8.5))
        # Prepare data
        x_labels = [f"{key}" for key in percentage_returns_list_thresholds.keys()]
        heights = list(percentage_returns_list_thresholds.values())
        
        # Create bar plot
        bars = plt.bar(x_labels, heights, color='green')
        plt.xlabel('Thresholds')
        plt.ylabel('Returns (%)')
        plt.title(f'Percentage returns for {sym1} and {sym2} pair for different thresholds')

        # Add labels on top of each bar
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, height,
                    f'{height:.2f}%', ha='center', va='bottom', fontsize=9)

        pdf.savefig()
        plt.close()



        # # Page 5: Zscore Spread Time Series and Trading Positions.
        # plt.figure(figsize=(11, 8.5))
        # plt.plot(zscore_series, label='Z-Score', color='purple', marker='o')
        # plt.axhline(0, color='grey', linestyle='--', label='Mean')
        # plt.axhline(entry_threshold, color='green', linestyle='--', label='±1.0 Entry threshold')
        # plt.axhline(-entry_threshold, color='green', linestyle='--')
        # plt.axhline(stop_loss_threshold, color='red', linestyle='--', label='±stop_loss_threshold Stop-loss')
        # plt.axhline(-stop_loss_threshold, color='red', linestyle='--')

        # plt.title("Z-Score Of Spread With Trade Entries")
        # plt.scatter(long_entries.index, zscore_series.loc[long_entries.index], marker='^', 
        #             color='green', s=100, label='Long Entry')
        # plt.scatter(short_entries.index, zscore_series.loc[short_entries.index], marker='v', 
        #             color='red', s=100, label='Short Entry')
        # plt.legend()
        # pdf.savefig()
        # plt.close()

    
    print(f"PDF report generated  ✅: {pdf_filename}")



if __name__ == '__main__':

                                                    #CLOSING PRICES
    #---------------------------------------------------------------------------------------------------------------------
    #ETH/WBETH
    #prices = pd.read_csv('binance_data/ETH_and_WBETH/2024/1m/merged_closing_prices.csv', index_col=0, parse_dates=True)

    #BTC/WBTC
    #prices = pd.read_csv("binance_data/Wrapped BTC/2024/1m/merged_closing_prices.csv", index_col=0, parse_dates=True)
    
                                                    #ORDER BOOK
    #---------------------------------------------------------------------------------------------------------------------

    #BTC/WBTC (ORDER BOOK)
    #prices = pd.read_csv('order_book_data/merged_data/1min/btc_wbtc_combined_1m.csv', index_col=0, parse_dates=True)
    #prices = prices[['btc_mid_price', 'wbtc_mid_price']] #ONLY TAKE MID PRICES

    # #ETH/WETH (ORDER BOOK)
    # prices = pd.read_csv('order_book_data/merged_data/1min/eth_wbeth_combined_1m.csv', index_col=0, parse_dates=True)
    # prices = prices[['eth_mid_price','wbeth_mid_price']]



    #SOL and BNSOL
    #prices = pd.read_csv('binance_data/SOL_and_BNSOL/2025/5m/merged_closing_prices.csv', index_col=0, parse_dates=True)

    #prices = pd.read_csv('binance_data/top_100_tickers/2024/1m/merged_closing_prices.csv', index_col=0, parse_dates=True)

    #prices = prices[['XRP/USDT_2024_1m', 'ADA/USDT_2024_1m']]


    #Final dataset:
    prices = pd.read_csv('tardis_data/final_in_sample_dataset/final_in_sample_dataset_5min_2024.csv', index_col=0, parse_dates=True)

    #Filter by XRP and ADA
    #prices = prices[['XRPUSDT_2024_5m', 'ADAUSDT_2024_5m']]

    #Filter by MANAUSDT_2024_5m ~ SANDUSDT_2024_5m
    prices = prices[['MANAUSDT_2024_5m', 'SANDUSDT_2024_5m']]

    #Filter by AXSUSDT_2024_5m ~ MANAUSDT_2024_5m
    #prices = prices[['AXSUSDT_2024_5m', 'MANAUSDT_2024_5m']]

    params = {
    "initial_capital": 10_000.0,
    "tx_cost": 0.00,
    "window_size": 288,
    "entry_threshold": 3,
    "exit_threshold": 0,
    "stop_loss_threshold": 4,
    "timeframe_str": "Timeframe: January - June (1min Data)",
    "year_str": "Year: 2024",
    "min_pass_fraction": 0,
    "significance": 0.05
    }


    generate_back_test_report(prices, **params)

from data_pre_processing import *
from pair_finder import *
from back_tester import *
import itertools



def threshold_experiment(prices,entry_stop_map,initial_capital):


    percentage_returns_list = {} #Store the return for each threshold

    #Params:
    #--------------------------------
    window_size = 1440
    exit_threshold= 0

    tx_cost = 0.00025

    #Params:
    #Back test
    sym1, sym2 = prices.columns
    print(f"\nTesting strategy on pair: {sym1} and {sym2} ...")
    S1 = prices[sym1]
    S2 = prices[sym2]


    # Compute the spread series and beta_series 
    spread_series, beta_series, alpha_series = compute_spread_series(S1, S2, window_size)
    #print(f"Hedge ratio (beta) for {sym1} ~ {sym2}: {beta:.4f}")

    # Compute rolling z-score using the provided helper function.
    zscore_series, rolling_mean, rolling_std = compute_rolling_zscore(spread_series, window_size)


    for entry_threshold, stop_loss_threshold in entry_stop_map.values():

        print(f"\nTesting strategy with entry threshold: {entry_threshold} and stop loss threshold: {stop_loss_threshold} ...")

        # Generate trading signals (positions) based on the spread's z-score
        positions, trade_entries, trade_exits = backtest_pair_rolling(spread_series,S1,S2,zscore_series, entry_threshold, exit_threshold, stop_loss_threshold)

        trade_profits, cumulative_profit_series, entry_times, exit_times = simulate_strategy_trade_pnl(trade_entries, trade_exits, initial_capital, beta_series, tx_cost)

        percentage_returns_list[(entry_threshold, stop_loss_threshold)] = (cumulative_profit_series[-1]/initial_capital) * 100


    return percentage_returns_list


def transaction_cost_experiment(prices,tx_costs_map,initial_capital):



    percentage_returns_list = {} #Store the return for each transaction cost

    #Params:
    #--------------------------------
    window_size = 1440
    entry_threshold= 2
    exit_threshold= 0
    stop_loss_threshold= 3

    #Params:
    #Back test
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


    for tx_cost in list(tx_costs_map.values()):

        print(f"\nTesting strategy with transaction cost: {tx_cost} ...")

        trade_profits, cumulative_profit_series, entry_times, exit_times = simulate_strategy_trade_pnl(trade_entries, trade_exits, initial_capital, beta_series, tx_cost)

        percentage_returns_list[tx_cost] = (cumulative_profit_series[-1]/initial_capital) * 100


    return percentage_returns_list    


if __name__ == '__main__':

    initial_capital = 10_000.0
    file_path_sol_bnsol = 'binance_data/SOL_and_BNSOL/2025/1m/merged_closing_prices.csv'
    file_path_eth_wbeth = 'binance_data/ETH_and_WBETH/2024/1m/merged_closing_prices.csv'  
    file_path_eth_wbtc =  'binance_data/ETH_and_WBTC/2024/1m/merged_closing_prices.csv'


                                    #Transaction cost experiment
    #---------------------------------------------------------------------------------------------

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

    prices = pd.read_csv(file_path_sol_bnsol, index_col = 0, parse_dates=True)

    percentage_returns_list_sol_bnsol = transaction_cost_experiment(prices,tx_costs,initial_capital)

    print('Percentage returns for SOL and BNSOL pair for different transaction costs:')
    print(percentage_returns_list_sol_bnsol)

    #Create bar plot for the percentage returns
    plt.bar([f"{key * 100:.2f}%" for key in percentage_returns_list_sol_bnsol.keys()], list(percentage_returns_list_sol_bnsol.values()))
    plt.xlabel('Transaction costs (%)')
    plt.ylabel('Returns (%)')
    plt.title('Percentage returns for SOL and BNSOL pair for different transaction costs')

    plt.show()

                                    #Thresholds cost experiment
    #---------------------------------------------------------------------------------------------

    # entry_stop_map = {
    # "1": (1, 2),
    # "2": (2, 3),  
    # "3": (3, 4),
    # "4": (4, 5),
    # "5": (5, 6)
    # }

    # percentage_returns_list_sol_bnsol = threshold_experiment(file_path_sol_bnsol,entry_stop_map,initial_capital)  

    # print('Percentage returns for SOL and BNSOL pair for different thresholds:')
    # print(percentage_returns_list_sol_bnsol)



    # #Create bar plot for the percentage returns
    # plt.bar([str(key) for key in percentage_returns_list_sol_bnsol.keys()], list(percentage_returns_list_sol_bnsol.values()))
    # plt.xlabel('Thresholds')
    # plt.ylabel('Returns (%)')
    # plt.title('Percentage returns for SOL and BNSOL pair for different thresholds')

    # plt.show()


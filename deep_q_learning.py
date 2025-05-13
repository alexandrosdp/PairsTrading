

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from back_tester import * 
import matplotlib.pyplot as plt
import copy


def simulate_strategy(spread_trading_window,prices_trading_window,beta_series_trading_window,initial_capital,tx_cost, entry, stop):
    """
    Placeholder for the user's existing simulation function.
    It should take:
      - spread_trading_window: pd.Dataframe of z-score spreads for the trading window (length T)
      - prices_trading_window: pd.Dataframe of raw prices for the two assets over the trading windfow (same length)
      - entry: entry threshold (in σ-units)
      - stop: stop-loss threshold (in σ-units)
    And return:
      - reward: An event based reward for the trading window
    """

    
    sym1, sym2 = prices_trading_window.columns
    S1 = prices_trading_window[sym1]
    S2 = prices_trading_window[sym2]

    
    positions, trade_entries, trade_exits = backtest_pair_rolling(S1=S1,
                                                                  S2=S2,
                                                                  zscore=spread_trading_window, 
                                                                  entry_threshold = entry, 
                                                                  exit_threshold = 0,  #The exit threshold is always the mean of the zscore
                                                                  stop_loss_threshold = stop)
    
    if not trade_exits: #If there are no trades, return no reward no profit and No entry (None)
        return 0.0,0.0,None

    #Calculate trade profit for first trade in this trading window
    trade_profits, _,_,_,_,_ = simulate_strategy_trade_pnl([trade_entries[0]], [trade_exits[0]], initial_capital, beta_series_trading_window, tx_cost) #We wrap the trade entries and exits in lists to match the expected input format of the simulate_strategy_trade_pnl function.

    trade_entry = trade_entries[0] #The first trade entry

    #print("TRADE ENTRIES: ",[trade_entries[0]], "TRADE EXITS: ",[trade_exits[0]], "TRADE PROFITS: ",trade_profits)

    #Only focus on the first trade for now!
    first_trade = trade_exits[0]['exit_type']

    if first_trade == 'win':
        return 1000.0,trade_profits,trade_entry
    elif first_trade == 'loss':
        return -1000.0,trade_profits,trade_entry
    elif first_trade == 'forced_exit':
        return -500.0,trade_profits,trade_entry
    


    # if (len(trade_exits) != 0): #If there are any trades

    #     for trade_exit in trade_exits:

    #         if(trade_exit['exit_type'] == 'win'):
    #             reward = 1000
    #         elif(trade_exit['exit_type'] == 'loss'):
    #             reward = -1000
    #         elif(trade_exit['exit_type'] == 'forced_exit'):
    #             reward = -500
    # else:
    #     reward = 0



    # a catch-all for any unexpected exit_type
    return 0.0,trade_profits,trade_entry



class DQN(nn.Module):
    """
    Simple feed-forward neural network for Q-value approximation.
    Input: flattened formation-window of z-score spreads (shape: [F])
    Output: one Q-value per action (shape: [n_actions])
    """
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 64):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            #nn.Dropout(p=dropout_p),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            #nn.Dropout(p=dropout_p),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    """
    Fixed-size circular buffer for experience replay.
    Stores tuples of (state, action, reward, next_state, done).
    """
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity) # stores the most recent experiences. A deque is a data structure that allows for fast appends and pops from both ends. When it gets full, we remove the first element (oldest episode) and append a new episode at the end, hence why we chose a deque

    def push(self, state, action, reward, next_state, done): #Add a new experience to the buffer. If the buffer is full, the oldest experience will be removed.
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int): # Sample a random batch (of size batch_size) of experiences from the buffer.
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch) # Unzip the batch into separate lists for each component of the experience tuple.
        return ( #Converting lists to appropriate formats
            np.vstack(states), #Stack the states vertically to create a 2D array.
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.vstack(next_states),
            np.array(dones, dtype=np.uint8)
        )

    def __len__(self):
        return len(self.buffer) #Returns the current number of experiences in the buffer.


class PairsTradingEnv:
    """
    Environment for the pairs trading DQN.
    Each episode = one trading window (T days).
    State = last F days of z-score spread.
    Action = pick one (entry, stop-loss) threshold pair.
    Reward = profit from simulate_strategy.
    """
    def __init__(self,
                 spreads: pd.Series, #Full z-score spread time series
                 prices: pd.DataFrame, #Full raw price time series
                 beta_series: pd.Series, #Full beta time series
                 initial_capital: float, # Initial capital for the strategy
                 tx_cost: float, # Transaction cost (not used in this version)
                 entry_stop_pairs: list, # List of (entry, stop-loss) pairs
                 formation_window: int, # Length of formation window (F)
                 trading_window: int): # Length of trading window (T)
        
        self.spreads = spreads
        self.prices = prices
        self.beta_series = beta_series # Placeholder for beta series
        self.initial_capital = initial_capital # Initial capital for the strategy
        self.tx_cost = tx_cost
        self.entry_stop_pairs = entry_stop_pairs
        self.F = formation_window
        self.T = trading_window

        # number of non-overlapping episodes we can run
        self.max_episodes = (len(spreads) - self.F) // self.T #For example, if F=30 and T=15, then the maximum number of episodes is (total_days - 30) // 15.
        self.current_episode = 0 #Tracks the current episode index.

    def reset(self):
        """
        Reset to the start of the next episode.
        Returns the initial state (F-length spread history).
        """
        if self.current_episode >= self.max_episodes:
            self.current_episode = 0 #If all episodes have been exhausted, the current_episode counter is reset to 0.
        start = self.current_episode * self.T #Index of the start of the current episode.
        state = self.spreads.iloc[start : start + self.F].to_numpy().astype(np.float32) #Slide the window forward by T days to get the formation window. The trader wants to trade immediately afte their previous trading window ended, so this allows for that (see MS whiteboard)
        return state

    def step(self, action: int):
        """
        Execute ONE episode using the chosen action.
        Returns: next_state, reward, done, info
        """

        start = self.current_episode * self.T # start of the current episode
        # full window includes formation + trading

        spread_trading_window = self.spreads.iloc[start + self.F : start + self.F + self.T] #The spread trading window
        prices_trading_window = self.prices.iloc[start + self.F : start + self.F + self.T] #The price trading window
        beta_series_trading_window = self.beta_series.iloc[start + self.F : start + self.F + self.T] #The beta series trading window

        entry, stop = self.entry_stop_pairs[action] #The selected entry and stop-loss thresholds based on the action taken.
        
        # simulate_strategy should return profit over the T-day trading window
        reward,profits,trade_entry = simulate_strategy(spread_trading_window,prices_trading_window,beta_series_trading_window,self.initial_capital,self.tx_cost, entry, stop)

        # build next state by shifting formation window forward by T days
        next_start = (self.current_episode + 1) * self.T
        next_state = self.spreads[next_start:next_start + self.F].to_numpy().astype(np.float32) # The next state is the spread window for the next episode.

        # done flag when we've exhausted all episodes
        done = (self.current_episode + 1 >= self.max_episodes) #The episode is done when the current episode index exceeds the maximum number of episodes (done gets set to True).
        self.current_episode += 1 # Increment the current episode index.

        return next_state, float(reward),profits,trade_entry, done, {} #An empty dictionary is returned as the info parameter, which can be used to pass additional information if needed.


def train_dqn(spreads_train: pd.Series,
              prices_train: pd.DataFrame,
              beta_series_train: pd.Series,
              spreads_val: pd.Series,
              prices_val: pd.DataFrame,
              beta_series_val: pd.Series,
              initial_capital: float,
              tx_cost: float,
              entry_stop_pairs: list,
              F: int,
              T: int,
              num_epochs: int = 10,
              batch_size: int = 32,
              gamma: float = 0.99,
              lr: float = 1e-3,
              epsilon_start: float = 1.0,
              epsilon_end: float = 0.01,
              epsilon_decay: float = 0.995,
              replay_capacity: int = 1000,
              target_update_freq: int = 5, # After how many epochs to copy the online network’s weights into the target network.
              hidden_dim: int = 64): # Number of neurons in each hidden layer of your DQN.
    """
    Main training loop for DQN.
    - Builds environment, networks, replay buffer, and optimizer.
    - Runs episodes in epochs to train the online network.
    - Periodically syncs target network.
    """

    # Setup device, envoronment, and action space
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #Use GPU if available, otherwise use CPU.
    env = PairsTradingEnv(spreads_train,prices_train,beta_series_train,initial_capital,tx_cost, entry_stop_pairs, F, T) #Initialize the environment with the provided spreads, prices, entry-stop pairs, formation window, and trading window.
    n_actions = len(entry_stop_pairs) # Number of discrete actions (entry-stop pairs).


    # Create online and target networks
    online_net = DQN(F, n_actions, hidden_dim).to(device) #The online network is the one that will be trained. The to device method moves the model to the specified device (GPU or CPU).
    target_net = DQN(F, n_actions, hidden_dim).to(device) #The target network is used to stabilize training. It is a copy of the online network.
    target_net.load_state_dict(online_net.state_dict()) # Initialize target network with the same weights as the online network.

    # Create optimzer and replay buffer
    optimizer = optim.Adam(online_net.parameters(), lr=lr) #Adam optimizer is used for training the online network. The learning rate and weight decay are specified.
    replay_buffer = ReplayBuffer(replay_capacity)
    epsilon = epsilon_start

    epoch_loss_history = []  # collect avg batch loss per epochh
    reward_history = [] # collect average reward per epoch


    # Initialize the best validation reward, weights and patience
    best_val_reward = -float('inf')
    best_weights    = copy.deepcopy(online_net.state_dict())
    patience        = 0
    patience_limit  = 20

    validation_reward_history = [] # collect validation reward per epoch

    # Training loop: Each epoch consists of running through all available trading-window episodes exactly once (in order), collecting rewards and experiences.
    for epoch in range(1, num_epochs + 1):

        state = env.reset() # Reset the environment to get the initial state.
        epoch_rewards = []
        epoch_batch_losses = []  # collect minibatch losses this epoch


        # Loop through all episodes in this epoch
        while True:
            # ε-greedy action selection

            #Explore with probability ε by picking a uniform random action.
            if random.random() < epsilon:
                action = random.randrange(n_actions)
            else:
                # Exploit by selecting the action with the highest Q-value from the online network.
                with torch.no_grad():
                    s_v = torch.from_numpy(state).unsqueeze(0).to(device) 
                    q_vals = online_net(s_v)
                    action = q_vals.argmax(dim=1).item()  

            next_state, reward,profits,trade_entry, done, _ = env.step(action)
            replay_buffer.push(state, action, reward, next_state, done)
            epoch_rewards.append(reward)
            state = next_state

            # Perform learning step if enough data
            if len(replay_buffer) >= batch_size:
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size) # Sample a random batch of experiences from the replay buffer.
            
                #Convert to PyTorch tensors and move to device
                states_v = torch.from_numpy(states).to(device) 
                next_states_v = torch.from_numpy(next_states).to(device) 
                actions_v = torch.tensor(actions, dtype=torch.int64, device=device)
                rewards_v = torch.tensor(rewards, dtype=torch.float32, device=device)
                dones_v = torch.tensor(dones, dtype=torch.bool, device=device)

                # Q(s,a)
                q_values = online_net(states_v)
                state_action_values = q_values.gather(1, actions_v.unsqueeze(1)).squeeze(1) #extracts the Q-values corresponding to the actions taken (actions_v) in the sampled experiences. This ensures that only the Q-values for the actions actually taken are used in the loss calculation.

                # max_a' Q_target(s', a')
                with torch.no_grad(): # No gradient tracking needed for target network
                    next_q_values = target_net(next_states_v)
                    max_next_q_values = next_q_values.max(1)[0] #computes the maximum Q-value across all possible actions for each next state. This represents the best possible future reward.
                    expected_values = rewards_v + gamma * max_next_q_values * (~dones_v) #~dones_v: Ensures that no future rewards are added if the episode has ended (done = True)

                # compute loss and update network
                loss = nn.MSELoss()(state_action_values, expected_values) # Mean Squared Error loss between the predicted Q-values and the expected Q-values.
                epoch_batch_losses.append(loss.item())
                optimizer.zero_grad()  # Clears the gradients of the online network to ensure that gradients from the previous step do not accumulate
                loss.backward() #Computes the gradients of the loss with respect to the online network's parameters using backpropagation
                optimizer.step() # Updates the online network's parameters using the computed gradients.

                # ─── Soft‐update target network ───────────────────────────
                #nstead of a hard copy every K epochs, we nudge θ⁻ toward θ each step
                #In this case we are retaining 99% of the previous target weight and adding 1% of the online network’s weight every step. This “soft” update smooths the transition, rather than doing a full 100% copy all at once.
                τ = 0.01
                for tgt_param, src_param in zip(target_net.parameters(),
                                                online_net.parameters()): #airs them in the same order: the first weight in the target net with the first weight in the online net, the first bias with the first bias, and so on
                    tgt_param.data.mul_(1.0 - τ) #Multiply target network parameters by (1 - τ) 
                    tgt_param.data.add_(τ * src_param.data) #Add the scaled online network parameters to the target network parameters. 


            if done: #If we have exhausted all trading window episodes, break out of the loop.
                break

        # Decay exploration rate
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        
         # # Periodically update target network
        # if epoch % target_update_freq == 0:
        #     target_net.load_state_dict(online_net.state_dict())

         # record average batch loss this epoch
        avg_epoch_loss = np.mean(epoch_batch_losses) if epoch_batch_losses else 0.0
        epoch_loss_history.append(avg_epoch_loss)

        avg_reward = np.mean(epoch_rewards) #Record the average reward for this epoch, which is mean of the rewards collected over the episodes in this epoch.
        reward_history.append(avg_reward) #Append the average reward for this epoch to the reward history for later analysis.
        print(f"Epoch {epoch:02d} | AvgReward: {avg_reward:.2f} | Epsilon: {epsilon:.3f}")


        # Start checking validation performance after 100 epochs

        #Put the online network in evaluation mode
        # ——————————————— Validation pass ————————————————————
        _,_,_,_,val_metrics = evaluate_dqn(
            online_net,
            spreads_val, prices_val, beta_series_val,
            initial_capital, tx_cost,
            entry_stop_pairs, F, T
        )

        avg_val_reward = val_metrics['avg_reward']
        validation_reward_history.append(avg_val_reward) #Append the average reward for this epoch to the validation reward history for later analysis.

        # print(f"Epoch {epoch:03d} | ValReward: {avg_val_reward:.2f}  "
        #     f"(best={best_val_reward:.2f}, patience={patience})")

        # if avg_val_reward > best_val_reward:
        #     best_val_reward = avg_val_reward
        #     best_weights    = copy.deepcopy(online_net.state_dict())
        #     patience        = 0
        # else:
        #     patience += 1
        #     if patience >= patience_limit:
        #         print("⏹ Early stopping triggered")
        #         break

    
    # # Reload best‐epoch weights and test
    # online_net.load_state_dict(best_weights)

    return online_net, replay_buffer,epoch_loss_history, reward_history,validation_reward_history #Return the trained online network, replay buffer, loss history, and reward history.




def evaluate_dqn(
    online_net,           # your trained DQN
    spreads: pd.Series,   # test‐set z‐scores
    prices: pd.DataFrame, # test‐set prices
    beta_series: pd.Series, # test‐set betas
    initial_capital: float, # Initial capital for the strategy
    tx_cost: float, # Transaction cost (not used in this version)
    entry_stop_pairs: list,
    F: int, T: int,
):
    """
    Runs the greedy policy (epsilon=0) over all test episodes,
    returns reward list and simple classification metrics.
    """
    device = next(online_net.parameters()).device
    #env = PairsTradingEnv(spreads, prices, entry_stop_pairs, F, T)
    env = PairsTradingEnv(spreads, prices,beta_series,initial_capital,tx_cost, entry_stop_pairs, F, T) #Initialize the environment with the provided spreads, prices, entry-stop pairs, formation window, and trading window.


    test_rewards = []
    actions = []
    trade_profits = []
    win, loss, forced, none = 0, 0, 0, 0
    episodes = [] # List to store episode metadata

    state = env.reset() # Reset the environment to get the initial state.
    done = False # Initialize the done flag to False.
    # Force greedy policy
    while not done:
        
        # Get the current episode index and calculate the start and end indices for the trading window (To be used for plotting)
        ep_idx     = env.current_episode #Starts at 0
        start_pos  = ep_idx * T 
        form_end   = start_pos + F
        trade_start = spreads.index[form_end]
        trade_end   = spreads.index[form_end + T - 1] #We subtract 1 because the python range function is exclusive of the end index (remember if you specify 5 at the end, it will get the first 5 elements, but the last element will  actually be at index 4, not 5).

        # Greedy action
        with torch.no_grad():
            s_v = torch.from_numpy(state).unsqueeze(0).to(device)
            action = online_net(s_v).argmax(dim=1).item() # Select the action with the highest Q-value from the online network.
        entry, stop = entry_stop_pairs[action] #Remeber action is an index into the entry_stop_pairs list, which contains tuples of (entry, stop-loss) pairs.

      

        next_state,reward,profits,trade_entry, done, _ = env.step(action)

        # Record episode metadata
        episodes.append({
            'trade_start': trade_start,
            'trade_end':   trade_end,
            'trade_entry_metadata': trade_entry,
            'entry':       entry,
            'stop':        stop
        })

        #Collect results
        actions.append((entry,stop)) # Append the action taken to the actions list.
        trade_profits.append(profits) # Append the profits from the trade to the trade profits list.
        test_rewards.append(reward)
        if   reward == 1000.0:  win    += 1
        elif reward == -1000.0: loss   += 1
        elif reward == -500.0:  forced += 1
        else:                   none   += 1

        # Update the state to the next state.
        state = next_state 


    #Compute summary metrics
    total = len(test_rewards)
    metrics = {
        "avg_reward":    np.mean(test_rewards),
        "win_rate":      win/total,
        "loss_rate":     loss/total,
        "forced_rate":   forced/total,
        "no_trade_rate": none/total,
    }

    return test_rewards,trade_profits,actions,episodes,metrics



def plot_episodes(prices: pd.DataFrame,
                  zscores: pd.Series,
                  episodes: list,
                  F: int,
                  T: int,
                  episode_idx: int = None):
    """
    Plot either all episodes or a single episode zoom.

    Args:
        prices: DataFrame indexed by datetime, columns=[S1, S2]
        zscores: Series indexed by datetime of the z-score spread
        episodes: List of dicts with keys:
            'trade_start', 'trade_end', 'entry', 'stop'
        F: Formation window length
        T: Trading window length
        episode_idx: If int, zoom on that episode; if None, plot all
    """
    if episode_idx is None:
        # Full-series plot
        fig, (ax_price, ax_z) = plt.subplots(2, 1, sharex=True, figsize=(14, 8))
        ax_price.plot(prices.index, prices.iloc[:, 0], label=prices.columns[0])
        ax_price.plot(prices.index, prices.iloc[:, 1], label=prices.columns[1])
        ax_price.set_ylabel("Price"); ax_price.legend(loc="upper left")
        ax_price.set_title("Asset Prices with Trading Windows")
        ax_z.plot(zscores.index, zscores.values, color='black')
        ax_z.set_ylabel("Z-score Spread"); ax_z.set_title("Z-score with Thresholds")
        for ep in episodes:
            t0, t1 = ep['trade_start'], ep['trade_end']
            entry, stop = ep['entry'], ep['stop']
            ax_price.axvspan(t0, t1, color='orange', alpha=0.1)
            ax_z.axvspan(t0, t1, color='orange', alpha=0.1)
            ax_z.hlines([ entry, -entry], t0, t1, colors='green', linestyles='--')
            ax_z.hlines([ stop,  -stop ], t0, t1, colors='red',   linestyles='--')
        ax_z.set_xlabel("Time")
        plt.tight_layout(); plt.show()
    else:
        # Single-episode zoom
        ep = episodes[episode_idx]

        print(f"Episode {ep}:")

        # compute formation start by going back F bars from trade_start
        pos_t0 = zscores.index.get_loc(ep['trade_start']) # get the numerical index position of trade_start (remember the index of the zscore series is datetime, and in the next line we want to do a slice (pos_t0 - F), which requires a numerical index)
        form_start = zscores.index[pos_t0 - F] # formation start
        trade_start, trade_end = ep['trade_start'], ep['trade_end']
        entry, stop = ep['entry'], ep['stop']
        prices_win = prices.loc[form_start:trade_end] #window is from formation start to trade end
        zscores_win = zscores.loc[form_start:trade_end] #window is from formation start to trade end

        #Plot prices over the formation and trading window for this episode
        fig, (ax_price, ax_z) = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
        ax_price.plot(prices_win.index, prices_win.iloc[:,0], label=prices_win.columns[0])
        ax_price.plot(prices_win.index, prices_win.iloc[:,1], label=prices_win.columns[1])
        ax_price.axvline(trade_start, color='blue', linestyle='--', label='Trade Start')
        ax_price.axvline(trade_end,   color='red',  linestyle='--', label='Trade End')
        ax_price.set_ylabel("Price"); ax_price.legend(loc='upper left')
        ax_price.set_title(f"Episode {episode_idx}: Formation+Trading")

        #Plot the zscore over the formation and trading window for this episode
        ax_z.plot(zscores_win.index, zscores_win.values, color='black')
        ax_z.axvspan(trade_start, trade_end, color='orange', alpha=0.1)
        ax_z.hlines([ entry, -entry], trade_start, trade_end, colors='green', linestyles='--', label='Entry ±σ')
        ax_z.hlines([ stop,  -stop ], trade_start, trade_end, colors='red',   linestyles='--', label='Stop ±σ')
        ax_z.hlines(0, trade_start, trade_end, colors='black', linestyles='--', label='Mean')
        ax_z.axvline(trade_start, color='blue', linestyle='--')
        ax_z.axvline(trade_end,   color='red',  linestyle='--')
        ax_z.set_ylabel("Z-score"); ax_z.legend(loc='upper left')
        ax_z.set_xlabel("Time")
        plt.tight_layout(); plt.show()







# # Example usage:
# if __name__ == "__main__":


#     #Data
#     prices = pd.read_csv('tardis_data/final_in_sample_dataset/final_in_sample_dataset_5min_2024.csv', index_col=0, parse_dates=True)
#     prices = prices[['MANAUSDT_2024_5m', 'SANDUSDT_2024_5m']]

#     #Only use the first month of prices for now
#     start_date = pd.to_datetime('2024-01-01 00:00:00')
#     end_date = pd.to_datetime('2024-01-31 23:55:00') 

#     prices = prices.loc[start_date:end_date] # 2880 rows = 1 month of 5-minute data

#     #Params for spread calculation
#     window_size = 288 # It seems like as this increases, the percent absolute delta beta error decreases!

#     # Load your precomputed z-score spreads and raw prices as NumPy arrays
#     sym1, sym2 = prices.columns
#     S1 = prices[sym1]
#     S2 = prices[sym2]

#     # Compute the spread series and beta_series 
#     spread_series, beta_series, alpha_series = compute_spread_series(S1, S2, window_size)
#     #print(f"Hedge ratio (beta) for {sym1} ~ {sym2}: {beta:.4f}")

#     # Compute rolling z-score using the provided helper function.
#     zscore_series, rolling_mean, rolling_std = compute_rolling_zscore(spread_series, window_size)
    
#     # Define your discrete threshold pairs: [(entry1, stop1), (entry2, stop2), ...]
#     entry_stop_pairs = [(0.5, 2.5), (1.0, 3.0), (1.5, 4.0), (2.0, 4.5), (2.5, 5.0), (3.0, 5.5)]
#     # Training parameters
#     F, T = 200, 100
#     online_net, replay_buffer,loss_history, reward_history = train_dqn(zscore_series, prices, entry_stop_pairs, F, T, num_epochs=20)

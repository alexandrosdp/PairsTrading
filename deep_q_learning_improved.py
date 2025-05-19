



import numpy as np
import pandas as pd
import scipy
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from back_tester import * 
import matplotlib.pyplot as plt
import copy


def simulate_strategy(
    spread_trading_window: pd.Series,
    prices_trading_window: pd.DataFrame,
    beta_series_trading_window: pd.Series,
    initial_capital: float,
    tx_cost: float,
    entry_thr: float,
    stop_thr: float,
):
    """
    Run one backtest over exactly this divergence→reversion cycle (no fixed T).

    Returns:
      reward (float),  profit (float),  entry_meta (dict),  exit_meta (dict)
    Reward is shaped purely on the event:
      - miss      = never hit entry_thr   → reward = -1
      - win       = hit entry_thr & mean-revert   → reward = +1
      - stop_loss = hit entry_thr then stop_loss → reward = -1
    """
    # 1) peel off S1/S2 series
    sym1, sym2 = prices_trading_window.columns
    S1 = prices_trading_window[sym1]
    S2 = prices_trading_window[sym2]

    # 2) backtest, get all entry/exit events in this window
    positions, trade_entries, trade_exits = backtest_pair_rolling(
        S1=S1,
        S2=S2,
        zscore=spread_trading_window,
        entry_threshold=entry_thr,
        exit_threshold=0.0,     # mean-reversion
        stop_loss_threshold=stop_thr,
        agent_trader=True
    )

    # 3) if we never even crossed the chosen entry band → negative reward
    if not trade_entries or not trade_exits:
        return -1.0, 0.0, None, None

    # 4) focus on the *first* trade
    entry_meta = trade_entries[0]
    exit_meta  = trade_exits[0]

    # 5) compute the P&L of that one trade (for logging only)
    #    simulate_strategy_trade_pnl expects lists of entries/exits:
    pnl_list, _, _, _, _, _ = simulate_strategy_trade_pnl(
        [entry_meta], [exit_meta],
        initial_capital, beta_series_trading_window, tx_cost
    )
    profit = pnl_list[0]

    # 6) shape the reward purely on the exit_type
    etype = exit_meta['exit_type']
    if etype == 'win':
        reward = +1.0
    elif etype == 'loss':
        reward = -1.0
    elif etype == 'forced_exit':
        reward = 0 # Forced exits are only possible in the last cycle of the training set, so we can ignore them for now.

    return reward, profit, entry_meta, exit_meta



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
                 ): # Length of trading window (T)
        
        self.spreads = spreads
        self.prices = prices
        self.beta_series = beta_series # Placeholder for beta series
        self.initial_capital = initial_capital # Initial capital for the strategy
        self.tx_cost = tx_cost
        self.entry_stop_pairs = entry_stop_pairs
        
        #Create cycles from the z-score spread time series. The min_threshold is the minimum z-score value to consider a cycle, and tol is the tolerance for near-zero values.
        self.create_cycles(min_threshold = 1, tol = 0.10) 

        # now we have self.spread_cycles, self.price_cycles, self.beta_cycles
        # build a list of feature‐vectors summarizing each cycle:

        self.state_features = [
            np.array([
            cycle.mean(), cycle.std(), len(cycle),
            cycle.min(), cycle.max(),
            scipy.stats.skew(cycle),
            ], dtype=np.float32)
            for cycle in self.spread_cycles
        ]

        self.cycle_idx = 0


    def create_cycles(self, min_threshold: float, tol: float = 0.10):
        """
        Split the series into (divergence → revert) cycles at or above min_threshold.
        Each cycle begins when |z| first exceeds min_threshold and ends when z crosses 0.
        """
        self.spread_cycles = []
        self.price_cycles  = []
        self.beta_cycles   = []

        z = self.spreads  # pd.Series of z-scores
        p = self.prices   # pd.DataFrame aligned with z
        b = self.beta_series  # pd.Series aligned with z

        # 1) Trim initial “off‐zero” drift:
        first_near_zero = (z.abs() <= tol).to_numpy().argmax()

        mask = (z.abs() <= tol).to_numpy()
        if not mask.any(): 
            first_near_zero = 0
        else:
            first_near_zero = mask.argmax()


        z = z.iloc[first_near_zero:]
        p = p.iloc[first_near_zero:]
        b = b.iloc[first_near_zero:]

        in_cycle = False
        cycle_start = None
        entry_sign = 0

        for i, val in enumerate(z):
            if not in_cycle:
                # wait for a threshold cross
                if val >= min_threshold:
                    in_cycle = True
                    entry_sign = +1
                    cycle_start = i
                elif val <= -min_threshold:
                    in_cycle = True
                    entry_sign = -1
                    cycle_start = i
            else:
                # we’re in a cycle—look for crossing back through zero
                if (entry_sign == 1 and val <= 0) or (entry_sign == -1 and val >= 0):
                    cycle_end = i
                    # SLICE out the cycle *including* start and end bar
                    span = slice(cycle_start, cycle_end + 1)

                    self.spread_cycles.append(z.iloc[span])
                    self.price_cycles .append(p.iloc[span])
                    self.beta_cycles  .append(b.iloc[span])

                    in_cycle = False
                    cycle_start = None
                    entry_sign = 0
        # done

        # assign back so env can use them
        self.spreads = z
        self.prices  = p
        self.beta_series = b


    def reset(self):
        """Start at the first cycle’s summary‐stats."""
        self.cycle_idx = 0
        return self.state_features[0]

    def step(self, action: int):

        """
        One RL step ≡ one real divergence→reversion (or stop) cycle.
        State = summary‐features of cycle[k]
        Trading window = cycle[k+1]
        """
        # 1) Which cycle we’re trading?
        k = self.cycle_idx

        # 2) Our look‐back was cycle[k]: so the state was state_features[k]
        #    Now we simulate on the *next* cycle:
        spread_cycle = self.spread_cycles[k+1]
        price_cycle  = self.price_cycles[k+1]
        beta_cycle   = self.beta_cycles[k+1]

        entry_thr, stop_thr = self.entry_stop_pairs[action]

        # 3) Run existing backtest over exactly THIS cycle:
        reward, profit, entry_meta, exit_meta = simulate_strategy(
            spread_cycle, price_cycle, beta_cycle,
            self.initial_capital, self.tx_cost,
            entry_thr, stop_thr
        )

        # 4) Advance to the next cycle’s features
        self.cycle_idx += 1
        done = (self.cycle_idx >= len(self.state_features) - 1)

        if not done:
            next_state = self.state_features[self.cycle_idx]
        else:
            next_state = np.zeros_like(self.state_features[0])

        info = {
        "entry_meta": entry_meta,
        "exit_meta": exit_meta,
        "profit": profit
        }

        return next_state, float(reward), done, info
    

def train_dqn(spreads_train: pd.Series,
              prices_train: pd.DataFrame,
              beta_series_train: pd.Series,
              spreads_val: pd.Series,
              prices_val: pd.DataFrame,
              beta_series_val: pd.Series,
              initial_capital: float,
              tx_cost: float,
              entry_stop_pairs: list,
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

    env = PairsTradingEnv(
        spreads_train, prices_train, beta_series_train,
        initial_capital, tx_cost,
        entry_stop_pairs)
    
    print("Number of cycles:", len(env.spread_cycles))
    print("CYCLES:")
    print("-------")
    print(env.spread_cycles)
    
    # print("Number of cycles:", len(env.spread_cycles))

    # print("First cycle:")
    # print("------------")
    # print(env.spread_cycles[0])

    # print("Last cycle:")
    # print("------------")
    # print(env.spread_cycles[-1])

    n_actions = len(entry_stop_pairs) # Number of discrete actions (entry-stop pairs).

    input_dim = len(env.state_features[0]) # The input dimension is the length of the state feature vector, which is the same for all cycles.
    # Create online and target networks
    online_net = DQN(input_dim, n_actions, hidden_dim).to(device) #The online network is the one that will be trained. The to device method moves the model to the specified device (GPU or CPU).
    target_net = DQN(input_dim, n_actions, hidden_dim).to(device) #The target network is used to stabilize training. It is a copy of the online network.
    target_net.load_state_dict(online_net.state_dict()) # Initialize target network with the same weights as the online network.

    # Create optimzer and replay buffer
    optimizer = optim.Adam(online_net.parameters(), lr=lr) #Adam optimizer is used for training the online network. The learning rate and weight decay are specified.
    replay_buffer = ReplayBuffer(replay_capacity)
    epsilon = epsilon_start

    epoch_loss_history = []  # collect avg batch loss per epochh
    reward_history = [] # collect average reward per epoch


    # # Initialize the best validation reward, weights and patience
    # best_val_reward = -float('inf')
    # best_weights    = copy.deepcopy(online_net.state_dict())
    # patience        = 0
    # patience_limit  = 20

    validation_reward_history = [] # collect validation reward per epoch
    win_rate_history = [] # collect average win rate per epoch
    loss_rate_history = [] # collect average loss rate per epoch
    forced_rate_history = [] # collect average forced exit rate per epoch
    none_rate_history = [] # collect average no trade rate per epoch
    

    # Training loop: Each epoch consists of running through all available trading-window episodes exactly once (in order), collecting rewards and experiences.
    for epoch in range(1, num_epochs + 1):

        state = env.reset() # Reset the environment to get the initial state.
        epoch_rewards = []
        epoch_batch_losses = []  # collect minibatch losses this epoch
        win_count = loss_count = forced_count = none_count = 0

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

            next_state, reward, done, info = env.step(action) 

                
            entry_meta = info['entry_meta']
            exit_meta  = info['exit_meta']
            profit     = info['profit']

            exit_type = exit_meta['exit_type'] if exit_meta is not None else 'none'


            if   exit_type == 'win':          win_count    += 1
            elif exit_type == 'loss':    loss_count   += 1
            elif exit_type == 'forced_exit':  forced_count += 1
            elif exit_type == 'none':         none_count   += 1


            replay_buffer.push(state, action, reward, next_state, done)
            epoch_rewards.append(reward)
            state = next_state

            # ----—-------- DQN update ----------——  

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
                td_loss = nn.MSELoss()(state_action_values, expected_values) # Mean Squared Error loss between the predicted Q-values and the expected Q-values.
                epoch_batch_losses.append(td_loss.item())
                optimizer.zero_grad()  # Clears the gradients of the online network to ensure that gradients from the previous step do not accumulate
                td_loss.backward() #Computes the gradients of the loss with respect to the online network's parameters using backpropagation
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

        #Win rates
        total = win_count + loss_count + forced_count + none_count
        win_rate = win_count / total if total > 0 else 0
        loss_rate = loss_count / total if total > 0 else 0
        forced_rate = forced_count / total if total > 0 else 0
        none_rate = none_count / total if total > 0 else 0

        win_rate_history.append(win_rate) #Append the win rate for this epoch to the win rate history for later analysis.
        loss_rate_history.append(loss_rate) #Append the loss rate for this epoch to the loss rate history for later analysis.
        forced_rate_history.append(forced_rate) #Append the forced exit rate for this epoch to the forced exit rate history for later analysis.
        none_rate_history.append(none_rate) #Append the no trade rate for this epoch to the no trade rate history for later analysis.

        # Start checking validation performance after 100 epochs

        #Put the online network in evaluation mode
        # ——————————————— Validation pass ————————————————————
        _,_,_,_,val_metrics = evaluate_dqn(
            online_net,
            spreads_val, prices_val, beta_series_val,
            initial_capital, tx_cost,
            entry_stop_pairs
        )

        avg_val_reward = val_metrics['avg_reward']
        validation_reward_history.append(avg_val_reward) #Append the average reward for this epoch to the validation reward history for later analysis.
    
    training_metrics = {
        "win_rates": win_rate_history,
        "loss_rates": loss_rate_history,
        "forced_rates": forced_rate_history,
        "no_trade_rates": none_rate_history
    }

    # # Reload best‐epoch weights and test
    # online_net.load_state_dict(best_weights)

    return online_net, replay_buffer,epoch_loss_history, reward_history,validation_reward_history,training_metrics #Return the trained online network, replay buffer, loss history, and reward history.




def evaluate_dqn(
    online_net,            # your trained DQN
    spreads: pd.Series,    # test‐set z‐scores
    prices: pd.DataFrame,  # test‐set prices
    beta_series: pd.Series,# test‐set betas
    initial_capital: float,
    tx_cost: float,
    entry_stop_pairs: list,
):
    """
    Runs the greedy (ε=0) policy on the test‐set cycles,
    returns (test_rewards, trade_profits, actions, episodes, metrics).
    """
    device = next(online_net.parameters()).device

    # 1) Initialize env with no fixed T—just F and your cycles logic
    env = PairsTradingEnv(
        spreads, prices, beta_series,
        initial_capital, tx_cost,
        entry_stop_pairs
    )

    test_rewards   = []
    trade_profits  = []
    actions        = []
    episodes       = []  # metadata per cycle
    win = loss = forced = none = 0


    # 2) Start at cycle #0
    state = env.reset()
    done  = False

    # 3) Step through *each* cycle until we exhaust them
    while not done:
        k = env.cycle_idx  # which cycle am I about to trade?

        # Greedy action
        with torch.no_grad():
            sv     = torch.from_numpy(state).unsqueeze(0).to(device)
            action = online_net(sv).argmax(dim=1).item()
        entry_thr, stop_thr = entry_stop_pairs[action]

        # 4) Take exactly one cycle → get back reward, profit, entry & exit metadata
        next_state, reward, done, info = env.step(action)

        entry_meta = info['entry_meta']
        exit_meta  = info['exit_meta']
        profit     = info['profit']

        exit_type = exit_meta['exit_type'] if exit_meta is not None else 'none'



        # 5) Record what just happened
        episodes.append({
            'cycle_idx':        k,
            'entry_meta':       entry_meta,
            'exit_meta':        exit_meta,
            'entry_threshold':  entry_thr,
            'stop_threshold':   stop_thr
        })
        actions.append((entry_thr, stop_thr))
        test_rewards.append(reward)
        trade_profits.append(profit)

        if   exit_type == 'win':          win    += 1
        elif exit_type == 'loss':    loss   += 1
        elif exit_type == 'forced_exit':  forced += 1
        elif exit_type == 'none':         none   += 1

        # 6) advance
        state = next_state

    # 7) Compute simple metrics
    total = len(test_rewards)

    test_metrics = {
        "avg_reward":    np.mean(test_rewards),
        "win_rate":      win/total,
        "loss_rate":     loss/total,
        "forced_rate":   forced/total,  # if you ever use that code path
        "no_trade_rate": none/total,
    }

    return test_rewards, trade_profits, actions, episodes, test_metrics


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

from binance.client import Client
import pandas as pd

# Set your Binance API keys (leave empty if not using authenticated requests)
API_KEY = ""  # Optional
API_SECRET = ""

# Initialize Binance client
client = Client(API_KEY, API_SECRET)

# Define asset and time frame
SYMBOL = "BTCUSDT"  # Must match your dataset
INTERVAL = "1h"  # Binance API now requires interval as a string (e.g., '1m', '1h', '1d')

# Define the time range
DAYS = 7
start_str = (pd.Timestamp.utcnow() - pd.Timedelta(days=DAYS)).strftime('%Y-%m-%d %H:%M:%S')

# Fetch data from Binance
klines = client.get_historical_klines(SYMBOL, INTERVAL, start_str)

# Convert to DataFrame
binance_df = pd.DataFrame(klines, columns=[
    "Open Time", "Open", "High", "Low", "Close", "Volume",
    "Close Time", "Quote Asset Volume", "Number of Trades",
    "Taker Buy Base Volume", "Taker Buy Quote Volume", "Ignore"
])

# Convert timestamps to readable dates
binance_df["Open Time"] = pd.to_datetime(binance_df["Open Time"], unit="ms")
binance_df["Close Time"] = pd.to_datetime(binance_df["Close Time"], unit="ms")

# Drop unnecessary column
binance_df.drop(columns=["Ignore"], inplace=True)

# Save for comparison
binance_df.to_csv("binance_api_data.csv", index=False)
print("✅ Binance API data saved to binance_api_data.csv")

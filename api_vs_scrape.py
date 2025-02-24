import pandas as pd

# Load your merged scraped dataset
scraped_df = pd.read_csv("merged_data/1h/BTCUSDT_1h_merged.csv")

# Convert timestamps to datetime format
scraped_df["Open Time"] = pd.to_datetime(scraped_df["Open Time"])

# Load Binance API dataset
binance_df = pd.read_csv("binance_api_data.csv")

# Convert timestamps to datetime format
binance_df["Open Time"] = pd.to_datetime(binance_df["Open Time"])

# Merge the two datasets for comparison
comparison_df = scraped_df.merge(binance_df, on="Open Time", suffixes=("_scraped", "_binance"))

# Identify mismatches
mismatches = comparison_df[
    (comparison_df["Open_scraped"] != comparison_df["Open_binance"]) |
    (comparison_df["High_scraped"] != comparison_df["High_binance"]) |
    (comparison_df["Low_scraped"] != comparison_df["Low_binance"]) |
    (comparison_df["Close_scraped"] != comparison_df["Close_binance"])
]

# Save mismatches for review
if not mismatches.empty:
    mismatches.to_csv("data_mismatches.csv", index=False)
    print("⚠️ Mismatches found! Saved to data_mismatches.csv.")
else:
    print("✅ No mismatches found. Your data matches Binance API data!")

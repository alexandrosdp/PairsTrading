import requests
import os
import zipfile
import pandas as pd
from datetime import datetime, timedelta

# Define parameters
CATEGORY = "smart-contract-platform"  # Specify the desired category
TOP_N = 10  # Number of top coins to fetch from the category (THE SCRIPT WILL ACTULLY FETCH N * 2 TOP COINS IN CASE SOME ARE MISSING ON BINANCE)
INTERVAL = "1h"  # Change to "1m", "1d", etc.
YEAR = 2024  # Specify the year to fetch

# Directories
CSV_DIR = "binance_csvs"
CATEGORY_DIR = f"binance_data/{CATEGORY}/{INTERVAL}/"

# Binance Kline Data Column Names
COLUMNS = [
    "Open Time", "Open", "High", "Low", "Close", "Volume",
    "Close Time", "Quote Asset Volume", "Number of Trades",
    "Taker Buy Base Volume", "Taker Buy Quote Volume", "Ignore"
]

# Function to fetch available categories from CoinGecko
# Function to fetch available categories from CoinGecko
def fetch_categories():
    url = "https://api.coingecko.com/api/v3/coins/categories/list"
    response = requests.get(url)

    if response.status_code != 200:
        print(f"❌ Failed to fetch categories from CoinGecko. HTTP Status: {response.status_code}")
        return []

    try:
        data = response.json()

        if not isinstance(data, list):  # Ensure response is a list
            print(f"⚠️ Unexpected API response format from CoinGecko: {data}")
            return []

        # Extract category IDs and names
        categories = []
        for cat in data:
            category_id = cat.get("category_id")  # Ensure we fetch 'category_id'
            category_name = cat.get("name", "Unknown Name")
            
            if category_id:
                categories.append({"id": category_id, "name": category_name})  # Store both ID and name
            else:
                print(f"⚠️ Skipping category with missing ID: {cat}")  # Debugging print

        return categories
    except Exception as e:
        print(f"⚠️ Error parsing JSON response from CoinGecko: {e}")
        return []

# Function to fetch trading pairs from Binance
def fetch_binance_trading_pairs():
    url = "https://api.binance.com/api/v3/exchangeInfo"
    response = requests.get(url)
    if response.status_code != 200:
        print("❌ Failed to fetch trading pairs from Binance.")
        return set()

    data = response.json()
    trading_pairs = {symbol['symbol'] for symbol in data['symbols']}
    return trading_pairs

# Function to get the top N coins in a category from CoinGecko
def get_top_coins(category, top_n):
    categories = fetch_categories()
    
    if not categories:
        print("❌ No categories found from CoinGecko. Check API response.")
        return []
    
    try:
        category_ids = [cat.get('id', 'UNKNOWN_ID') for cat in categories if isinstance(cat, dict)]
        # print("CATEGORY IDS:")
        # print(category_ids)
        if category not in category_ids:
            print(f"❌ Category '{category}' not found in CoinGecko.")
            print("Available categories are:")
            for cat in categories:
                print(f" - {cat.get('id', 'UNKNOWN_ID')}: {cat.get('name', 'Unknown Name')}")
            return []

        url = "https://api.coingecko.com/api/v3/coins/markets"
        params = {
            "vs_currency": "usd",
            "category": category,
            "order": "market_cap_desc",
            "per_page": top_n * 2,  # Fetch more in case some are missing on Binance
            "page": 1
        }

        response = requests.get(url, params=params)
        if response.status_code != 200:
            print(f"❌ Failed to fetch top coins from CoinGecko for category '{category}'. HTTP Status: {response.status_code}")
            return []

        data = response.json()
        
        if not isinstance(data, list):  # Ensure response is a list
            print(f"⚠️ Unexpected API response format from CoinGecko: {data}")
            return []

        # Extract coin names and symbols
        coin_list = [(coin.get("name", "Unknown"), coin.get("symbol", "").upper() + "USDT") for coin in data]

        print(f"\n📌 Initial {top_n} Coins in Category '{category}':")
        for i, (name, symbol) in enumerate(coin_list[:top_n], 1):
            print(f"   {i}. {name} ({symbol})")

        return coin_list  # Return tuples of (name, symbol)

    except Exception as e:
        print(f"⚠️ Error processing CoinGecko data: {e}")
        return []

# Function to filter available coins on Binance
def get_available_coins():

    binance_pairs = fetch_binance_trading_pairs()
    coin_list = get_top_coins(CATEGORY, TOP_N * 2)  # Fetch more than TOP_N

    available_coins = []
    for name, symbol in coin_list:
        if symbol in binance_pairs:
            available_coins.append((name, symbol))
        if len(available_coins) == TOP_N:
            break  # Stop once we have enough available coins

    print("\n✅ Final selected coins available on Binance:")
    for i, (name, symbol) in enumerate(available_coins, 1):
        print(f"   {i}. {name} ({symbol})")

    return [symbol for _, symbol in available_coins]  # Return only Binance trading pairs

# Generate filenames for all days in the specified year
def generate_filenames(symbol):
    start_date = datetime(YEAR, 1, 1)
    end_date = datetime(YEAR, 12, 31)
    delta = timedelta(days=1)

    filenames = []
    while start_date <= end_date:
        filenames.append(f"{symbol}-{INTERVAL}-{start_date.strftime('%Y-%m-%d')}.zip")
        start_date += delta

    return filenames

# Function to download, extract, and remove zip files
def download_and_extract(symbol, file_list):
    if not os.path.exists(CSV_DIR):
        os.makedirs(CSV_DIR)

    for file_name in file_list:
        file_url = f"https://data.binance.vision/data/spot/daily/klines/{symbol}/{INTERVAL}/{file_name}"
        zip_path = os.path.join(CSV_DIR, file_name)

        # Skip if already processed
        if os.path.exists(zip_path.replace(".zip", ".csv")):
            print(f"✅ {file_name} already processed, skipping.")
            continue

        print(f"⬇️ Downloading {file_url}...")
        response = requests.get(file_url, stream=True)
        if response.status_code == 200:
            with open(zip_path, "wb") as file:
                for chunk in response.iter_content(chunk_size=1024):
                    file.write(chunk)

            print(f"📂 Extracting {file_name}...")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(CSV_DIR)

            os.remove(zip_path)  # Delete the zip file after extraction
            print(f"🗑️ Deleted {file_name}, only CSV saved.")
        else:
            print(f"❌ Failed to download: {file_url}")

# Function to merge all CSVs into a single cleaned CSV file
def merge_csvs(symbol):
    csv_files = [os.path.join(CSV_DIR, f) for f in os.listdir(CSV_DIR) if f.startswith(symbol) and f.endswith(".csv")]
    print(f"📂 Found {len(csv_files)} CSV files for {symbol}.")

    if not csv_files:
        print(f"⚠️ No CSV files found for {symbol}. Skipping merge.")
        return

    if not os.path.exists(CATEGORY_DIR):
        os.makedirs(CATEGORY_DIR)

    print(f"📊 Merging CSV files for {symbol}...")
    df_list = []
    
    for f in csv_files:
        try:
            df = pd.read_csv(f, header=None)
            df.columns = COLUMNS  # Assign proper column names
            df.dropna(how="all", inplace=True)  # Remove empty rows

            # Convert timestamps
            for col in ["Open Time", "Close Time"]:
                if df[col].max() > 1e13:  # If timestamps are too large, they are likely in µs
                    df[col] = df[col] // 1000  # Convert µs to ms

                df[col] = pd.to_datetime(df[col], unit="ms")  # Convert to human-readable datetime

            df_list.append(df)
        except Exception as e:
            print(f"⚠️ Error reading {f}: {e}")

    if not df_list:
        print(f"⚠️ No valid data found for {symbol}. Skipping merging step.")
        return

    # Concatenate all DataFrames and sort by Open Time
    merged_df = pd.concat(df_list, ignore_index=True)
    merged_df = merged_df.sort_values(by="Open Time")

    output_path = os.path.join(CATEGORY_DIR, f"{symbol}_merged.csv")
    merged_df.to_csv(output_path, index=False)
    print(f"✅ Merged CSV saved to {output_path}")

# Main execution
if __name__ == "__main__":
    
    selected_coins = get_available_coins()  # Get Binance-compatible coins
    print(selected_coins)
    for symbol in selected_coins:
        file_list = generate_filenames(symbol)
        download_and_extract(symbol, file_list)
        merge_csvs(symbol)

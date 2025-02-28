import requests
import os
import zipfile
import pandas as pd
from datetime import datetime, timedelta
import statsmodels.api as sm

# ----------------- CONFIGURATION -----------------
SECTOR_COIN_LIST = {
    # Example structure
    "From_Paper": ["BTCEUR", "BTCGBP"]
}

INTERVAL = "1m"  # Binance interval ("1m", "1h", "1d", etc.)
YEAR = 2023      # Year for data

# If None (or empty list), fetch entire year. Otherwise, specify a list of months (1..12), e.g. [1, 2, 7]
SELECTED_MONTHS = [10,11] # Fetch only October and November

# Directory to store data
CSV_DIR = "binance_csvs"
BASE_DIR = "binance_data"

# Binance Kline Data Column Names
COLUMNS = [
    "Open Time", "Open", "High", "Low", "Close", "Volume",
    "Close Time", "Quote Asset Volume", "Number of Trades",
    "Taker Buy Base Volume", "Taker Buy Quote Volume", "Ignore"
]


# ----------------- HELPER FUNCTIONS -----------------
def check_binance_availability():
    """
    Checks which coins in SECTOR_COIN_LIST are available on Binance.
    Filters out coins that are not listed.
    """
    print("🔎 Checking Binance for available coins...")
    url = "https://api.binance.com/api/v3/exchangeInfo"
    response = requests.get(url)
    if response.status_code != 200:
        print("❌ Failed to fetch trading pairs from Binance.")
        return {}

    binance_symbols = {s['symbol'] for s in response.json()['symbols']}
    available_coins = {}

    for sector, coins in SECTOR_COIN_LIST.items():
        valid_coins = [coin for coin in coins if coin in binance_symbols]
        if valid_coins:
            available_coins[sector] = valid_coins
        else:
            print(f"⚠️ No available pairs found for sector: {sector}")

    print("\n✅ Available Binance Coins by Sector:")
    for sector, coins in available_coins.items():
        print(f"📌 {sector}: {', '.join(coins)}")

    return available_coins


def generate_filenames(symbol):
    """
    Generates the filenames for each day of the specified YEAR/INTERVAL,
    optionally restricted to SELECTED_MONTHS if it's not None.
    """
    filenames = []

    if SELECTED_MONTHS:
        # Build day ranges only for the selected months
        for month in SELECTED_MONTHS:
            start_date = datetime(YEAR, month, 1)

            # Increment day by day until we leave that month (or the year ends)
            current_month = month
            day_delta = timedelta(days=1)

            while start_date.month == current_month and start_date.year == YEAR:
                file_name = f"{symbol}-{INTERVAL}-{start_date.strftime('%Y-%m-%d')}.zip"
                filenames.append(file_name)
                start_date += day_delta

    else:
        # Default: entire year
        start_date = datetime(YEAR, 1, 1)
        end_date = datetime(YEAR, 12, 31)
        delta = timedelta(days=1)
        while start_date <= end_date:
            file_name = f"{symbol}-{INTERVAL}-{start_date.strftime('%Y-%m-%d')}.zip"
            filenames.append(file_name)
            start_date += delta

    return filenames


def download_and_extract(symbol, sector, file_list):
    """
    Downloads and extracts Binance Kline data for a given trading pair.
    """
    sector_dir = f"{BASE_DIR}/{sector}/{YEAR}/{INTERVAL}/"
    os.makedirs(sector_dir, exist_ok=True)

    for file_name in file_list:
        file_url = f"https://data.binance.vision/data/spot/daily/klines/{symbol}/{INTERVAL}/{file_name}"
        zip_path = os.path.join(sector_dir, file_name)

        # If the CSV from that date is already present, skip re-downloading
        csv_candidate = zip_path.replace(".zip", ".csv")
        if os.path.exists(csv_candidate):
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
                zip_ref.extractall(sector_dir)

            os.remove(zip_path)  # Delete the zip file after extraction
            print(f"🗑️ Deleted {file_name}, only CSV saved.")
        else:
            print(f"❌ Failed to download: {file_url}")


def merge_csvs(symbol, sector):
    """
    Merges all CSV files for a symbol into a single file,
    then deletes the individual daily CSVs after merging.
    """
    sector_dir = f"{BASE_DIR}/{sector}/{YEAR}/{INTERVAL}/"
    csv_files = [
        os.path.join(sector_dir, f)
        for f in os.listdir(sector_dir)
        if f.startswith(symbol) and f.endswith(".csv")
    ]
    
    print(f"📂 Found {len(csv_files)} CSV files for {symbol}.")

    if not csv_files:
        print(f"⚠️ No CSV files found for {symbol}. Skipping merge.")
        return

    print(f"📊 Merging CSV files for {symbol}...")
    df_list = []

    for f in csv_files:
        try:
            df = pd.read_csv(f, header=None)
            df.columns = COLUMNS
            df.dropna(how="all", inplace=True)

            for col in ["Open Time", "Close Time"]:
                if df[col].max() > 1e13:
                    df[col] = df[col] // 1000
                df[col] = pd.to_datetime(df[col], unit="ms")

            df_list.append(df)
        except Exception as e:
            print(f"⚠️ Error reading {f}: {e}")

    if not df_list:
        print(f"⚠️ No valid data found for {symbol}. Skipping merging step.")
        return

    merged_df = pd.concat(df_list, ignore_index=True)
    merged_df = merged_df.sort_values(by="Open Time")
    
    output_file = os.path.join(sector_dir, f"{symbol}_{YEAR}_{INTERVAL}.csv")
    merged_df.to_csv(output_file, index=False)
    print(f"✅ Merged CSV saved: {output_file}")

    # --- Delete individual CSVs after merging ---
    for f in csv_files:
        # Skip the newly created merged file
        if f == output_file:
            continue
        try:
            os.remove(f)
            # print(f"🗑️ Deleted {f}")
        except OSError as e:
            print(f"⚠️ Error deleting {f}: {e}")


# ----------------- MAIN SCRIPT -----------------
if __name__ == "__main__":
    # Example: SELECTED_MONTHS = [1, 2, 12]  # fetch January, February, December of the YEAR
    # If SELECTED_MONTHS is None or an empty list, the entire year is fetched

    available_coins = check_binance_availability()

    for sector, coins in available_coins.items():
        for coin in coins:
            print(f"\n🚀 Processing {coin} in {sector} sector...")

            # Now generate filenames for the entire YEAR or only SELECTED_MONTHS
            filenames = generate_filenames(coin)
            download_and_extract(coin, sector, filenames)
            merge_csvs(coin, sector)

    print("\n🎉 Data collection complete!")

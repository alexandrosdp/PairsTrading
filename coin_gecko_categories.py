import requests

def fetch_categories():
    url = "https://api.coingecko.com/api/v3/coins/categories/list"
    response = requests.get(url)
    if response.status_code != 200:
        print("❌ Failed to fetch categories from CoinGecko.")
        return []

    return response.json()

# Fetch and display all available categories
categories = fetch_categories()

if categories:
    print("\n📌 **Available Categories on CoinGecko:**")
    for category in categories:
        print(f" - {category['id']}: {category['name']}")
else:
    print("⚠️ No categories found.")

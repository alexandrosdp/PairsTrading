import requests

def fetch_categories():
    url = "https://api.coingecko.com/api/v3/coins/categories/list"
    response = requests.get(url)
    
    if response.status_code != 200:
        print("❌ Failed to fetch categories from CoinGecko.")
        return []
    
    data = response.json()
    
    # Print the raw data to inspect its structure
    print("\n📌 **Raw API Response from CoinGecko:**")
    print(data)  # Print full response
    
    return data

# Fetch and display all available categories
categories = fetch_categories()

if categories:
    print("\n📌 **Available Categories on CoinGecko:**")
    for category in categories:
        print(category)  # Print each category item to see its structure
else:
    print("⚠️ No categories found.")

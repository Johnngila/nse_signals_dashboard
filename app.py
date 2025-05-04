import requests
from bs4 import BeautifulSoup
import pandas as pd

def fetch_nse_data():
    url = "https://live.mystocks.co.ke/"
    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")

    table = soup.find("table", {"id": "stock-table"})
    rows = table.find_all("tr")[1:]  # skip header

    data = []
    for row in rows:
        cols = row.find_all("td")
        if len(cols) < 8:
            continue
        symbol = cols[0].text.strip()
        price = cols[1].text.strip()
        change = cols[2].text.strip()
        volume = cols[5].text.strip()
        data.append({
            "Symbol": symbol,
            "Price": price,
            "Change": change,
            "Volume": volume
        })

    df = pd.DataFrame(data)
    return df

# Load historical data from CSV
stock_data = pd.read_csv('NSE_data_all_stocks.csv')

# Test the function
if __name__ == "__main__":
    try:
        df = fetch_nse_data()
        print("Successfully fetched stock data:")
        print(df.head())
        print("\nHistorical stock data:")
        print(stock_data.head())
    except Exception as e:
        print(f"Error fetching stock data: {e}")
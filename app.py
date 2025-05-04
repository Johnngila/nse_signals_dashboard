import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import sys

# Add the utils directory to the path so we can import modules from it
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

# Import the more robust data fetcher
from data_fetcher import fetch_stock_data

# Create data directory if it doesn't exist
os.makedirs(os.path.join(os.path.dirname(__file__), 'data'), exist_ok=True)

def fetch_nse_data():
    """
    A simplified wrapper around the more robust fetch_stock_data function
    from utils/data_fetcher.py
    """
    try:
        # Try to use the robust implementation first
        return fetch_stock_data(period="1d", source="nse")
    except Exception as e:
        print(f"Error using data_fetcher.py implementation: {e}")
        
        # Fall back to the simple implementation
        return fetch_nse_data_simple()

def fetch_nse_data_simple():
    """The original simple implementation as a fallback"""
    url = "https://live.mystocks.co.ke/"
    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")

    # Try to find the table with id "stock-table" first
    table = soup.find("table", {"id": "stock-table"})
    
    # If not found, try other common table selectors
    if table is None:
        # Try to find the first table on the page
        table = soup.find("table")
        
        # If still not found, try to find tables with common class names
        if table is None:
            table = soup.find("table", {"class": ["stock-table", "market-data", "quotes-table"]})
            
    # If no table was found at all, raise an error
    if table is None:
        raise Exception("Could not find stock data table on the webpage")
        
    rows = table.find_all("tr")[1:]  # skip header

    data = []
    # Load the stock codes file to help with mapping
    stock_codes_df = pd.read_csv('NSE_data_all_stocks.csv')
    stock_codes = stock_codes_df['Code'].tolist()
    
    for row in rows:
        try:
            cols = row.find_all("td")
            if len(cols) < 6:  # Changed from 8 to 6 to be more lenient
                continue
                
            # Use safer indexing with default values if columns don't exist
            symbol = cols[0].text.strip() if len(cols) > 0 else "Unknown"
            
            # Try to get price from column 1, but fallback to other columns if needed
            price = "N/A"
            for i in range(1, min(4, len(cols))):
                price_text = cols[i].text.strip()
                if price_text and any(c.isdigit() for c in price_text):
                    price = price_text.replace(',', '')  # Remove commas from price
                    break
                    
            # Similarly handle change and volume with fallbacks
            change = cols[2].text.strip() if len(cols) > 2 else "0"
            volume = cols[5].text.strip() if len(cols) > 5 else "0"
            
            # Clean up the symbol to match our Code format
            clean_symbol = symbol.strip().upper()
            
            # Try to match with our known codes
            matched_symbol = None
            for code in stock_codes:
                if clean_symbol == code or clean_symbol.startswith(code):
                    matched_symbol = code
                    break
            
            # Use matched symbol if found, otherwise use original
            final_symbol = matched_symbol if matched_symbol else clean_symbol
            
            # Only add rows that have at least a symbol and valid price
            if final_symbol != "Unknown" and price != "N/A":
                data.append({
                    "Symbol": final_symbol,
                    "Price": price,
                    "Change": change,
                    "Volume": volume
                })
        except Exception as e:
            print(f"Error processing row: {e}")
            continue

    df = pd.DataFrame(data)
    
    # Save the fetched prices to a CSV file for the streamlit app to use
    try:
        df.to_csv('data/current_prices.csv', index=False)
        print(f"Saved {len(df)} current prices to data/current_prices.csv")
    except Exception as e:
        print(f"Error saving prices to CSV: {e}")
        
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
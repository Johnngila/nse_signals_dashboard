import os
import sys
import pandas as pd
from datetime import datetime

# Add the utils directory to the path so we can import modules from it
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

# Import the data fetcher
from data_fetcher import fetch_stock_data

def download_nse_pricelist():
    """
    Download the latest NSE pricelist for all stocks and save as a CSV
    """
    # Load the list of all stocks from the CSV file
    all_stocks_df = pd.read_csv('NSE_data_all_stocks.csv')
    all_stocks = all_stocks_df['Code'].tolist()
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    print(f"Fetching latest price data for {len(all_stocks)} NSE stocks...")
    
    # Fetch data for all stocks
    try:
        # Use the fetch_stock_data function to get the latest prices
        price_data = fetch_stock_data(tickers=all_stocks, period="1d", source="nse")
        
        if price_data is not None and not price_data.empty:
            # Get the latest date
            latest_date = price_data.index[-1]
            date_str = latest_date.strftime('%Y-%m-%d')
            
            # Convert from wide to long format for better CSV structure
            price_list = []
            
            for stock in price_data.columns:
                if not pd.isna(price_data[stock].iloc[-1]):
                    price_list.append({
                        'Date': date_str,
                        'Symbol': stock,
                        'Price': price_data[stock].iloc[-1]
                    })
            
            # Create a DataFrame from the price list
            pricelist_df = pd.DataFrame(price_list)
            
            # Add stock information from the all_stocks_df
            pricelist_df = pricelist_df.merge(
                all_stocks_df, 
                left_on='Symbol', 
                right_on='Code',
                how='left'
            )
            
            # Keep only the necessary columns
            pricelist_df = pricelist_df[['Date', 'Symbol', 'Name', 'Sector', 'Price']]
            
            # Save to CSV
            output_file = f'data/NSE_pricelist_{date_str.replace("-", "")}.csv'
            pricelist_df.to_csv(output_file, index=False)
            
            print(f"Successfully downloaded NSE pricelist with {len(pricelist_df)} stocks")
            print(f"Saved to: {output_file}")
            print("\nSample of the pricelist:")
            print(pricelist_df.head())
            
            return output_file
        else:
            print("Error: No price data was returned")
            return None
    
    except Exception as e:
        print(f"Error downloading NSE pricelist: {e}")
        return None

if __name__ == "__main__":
    download_nse_pricelist()
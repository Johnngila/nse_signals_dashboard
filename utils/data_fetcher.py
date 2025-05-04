import pandas as pd
import numpy as np
import datetime as dt
import yfinance as yf
import requests
from bs4 import BeautifulSoup
import time
import os
import json
from datetime import datetime, timedelta

def fetch_stock_data(tickers=None, period="1mo", source="nse"):
    """
    Fetch stock data for the given tickers
    
    Parameters:
    -----------
    tickers : list or None
        List of stock tickers to fetch. If None, default NSE stocks are used.
    period : str
        Time period to fetch data for (e.g., '1d', '1mo', '1y')
    source : str
        Data source to use ('yahoo', 'nse', 'alphavantage')
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing stock price data
    """
    # Default NSE Kenya stocks if none provided
    if tickers is None:
        tickers = [
            'SCOM',  # Safaricom
            'EQTY',  # Equity Group
            'KCB',   # KCB Group
            'EABL',  # East African Breweries
            'BAT'    # British American Tobacco
        ]
    
    print(f"Fetching data for {tickers} from source: {source}, period: {period}")
    
    # Handle different data sources
    if source.lower() == 'yahoo':
        try:
            # Add .NR extension for Yahoo Finance NSE tickers if not already present
            yahoo_tickers = [f"{ticker}.NR" if not ticker.endswith('.NR') else ticker for ticker in tickers]
            
            print(f"Using Yahoo Finance with tickers: {yahoo_tickers}")
            
            # Try to fetch data from Yahoo Finance
            data = yf.download(yahoo_tickers, period=period, group_by='ticker')
            
            # If data is multi-level, get the closing prices
            if isinstance(data.columns, pd.MultiIndex):
                close_data = data.loc[:, (slice(None), 'Close')]
                # Flatten the column names
                close_data.columns = [x[0].replace('.NR', '') for x in close_data.columns]
            else:
                close_data = data['Close']
            
            print(f"Yahoo Finance data shape: {close_data.shape}")
            if close_data.empty:
                print("Yahoo Finance returned empty data, trying NSE fallback")
                return fetch_nse_data(tickers, period)
                
            return close_data
        
        except Exception as e:
            print(f"Error fetching data from Yahoo Finance: {e}")
            # Try NSE source as backup
            return fetch_nse_data(tickers, period)
            
    elif source.lower() == 'nse':
        return fetch_nse_data(tickers, period)
        
    elif source.lower() == 'alphavantage':
        # Alpha Vantage data fetching logic would go here
        # For now, try NSE source
        print("Alpha Vantage API not implemented, trying NSE data")
        return fetch_nse_data(tickers, period)
    
    elif source.lower() == 'mock':
        print("Using mock data source")
        return generate_mock_data(tickers, days=30)
    
    else:
        raise ValueError(f"Unsupported data source: {source}")

def fetch_nse_data(tickers=None, period="1mo"):
    """
    Fetch stock data directly from NSE Kenya website and cached sources
    
    Parameters:
    -----------
    tickers : list or None
        List of stock tickers to fetch. If None, default NSE stocks are used.
    period : str
        Time period to fetch data for (e.g., '1d', '1mo', '1y')
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing stock price data
    """
    if tickers is None:
        tickers = [
            'SCOM',  # Safaricom
            'EQTY',  # Equity Group
            'KCB',   # KCB Group
            'EABL',  # East African Breweries
            'BAT'    # British American Tobacco
        ]
    
    print(f"fetch_nse_data called for tickers: {tickers}, period: {period}")
    
    # Create data directory if it doesn't exist
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    # Define cache file path
    cache_file = os.path.join(data_dir, 'nse_data_cache.json')
    print(f"Cache file path: {cache_file}")
    
    # Try to get fresh data from NSE website
    try:
        # Get current market data
        print("Attempting to fetch current prices from NSE...")
        current_data = fetch_nse_current_prices(tickers)
        
        if not current_data:
            print("WARNING: No current data found from NSE website")
        else:
            print(f"Retrieved current prices for: {list(current_data.keys())}")
        
        # Load historical data from cache if available
        historical_data = {}
        if os.path.exists(cache_file):
            print("Loading data from cache file...")
            with open(cache_file, 'r') as f:
                cache = json.load(f)
                historical_data = cache.get('historical_data', {})
                last_update = datetime.fromisoformat(cache.get('last_update', '2000-01-01'))
                
                print(f"Cache last updated: {last_update}")
                print(f"Cached tickers: {list(historical_data.keys())}")
                
                # Check if cache is recent (less than 24 hours old)
                if (datetime.now() - last_update).total_seconds() > 86400:  # 24 hours in seconds
                    # If cache is old, supplement with NSE historical data
                    print("Cache is older than 24 hours, fetching historical data...")
                    historical_data = fetch_nse_historical_data(tickers, historical_data)
        else:
            # If no cache exists, get historical data
            print("No cache file found, fetching historical data...")
            historical_data = fetch_nse_historical_data(tickers)
        
        # Merge current with historical data
        all_data = merge_current_with_historical(current_data, historical_data)
        
        # Debug output
        for ticker in all_data:
            print(f"Ticker {ticker} has {len(all_data[ticker])} data points")
        
        # If we have no data points at all, return empty DataFrame instead of mock data
        all_empty = all(len(all_data.get(ticker, {})) == 0 for ticker in tickers)
        if all_empty:
            print("WARNING: No data available from NSE or cache.")
            return pd.DataFrame()  # Return empty DataFrame instead of mock data
        
        # Save updated data to cache
        cache = {
            'historical_data': all_data,
            'last_update': datetime.now().isoformat()
        }
        with open(cache_file, 'w') as f:
            json.dump(cache, f)
            
        # Convert to DataFrame and filter by period
        result_df = create_dataframe_from_merged_data(all_data, period)
        print(f"Final result dataframe shape: {result_df.shape}")
        
        return result_df
        
    except Exception as e:
        print(f"Error fetching NSE data: {e}")
        import traceback
        traceback.print_exc()
        
        # If there's a cache file, try to use it
        if os.path.exists(cache_file):
            try:
                print("Attempting to load from cache after error...")
                with open(cache_file, 'r') as f:
                    cache = json.load(f)
                    all_data = cache.get('historical_data', {})
                    result_df = create_dataframe_from_merged_data(all_data, period)
                    print(f"Retrieved data from cache with shape: {result_df.shape}")
                    return result_df
            except Exception as cache_error:
                print(f"Error loading cache: {cache_error}")
        
        # Return empty DataFrame instead of mock data
        print("All attempts failed, no data available")
        return pd.DataFrame()

def fetch_nse_current_prices(tickers):
    """Fetch current stock prices from MyStocks Kenya website"""
    current_data = {}
    
    try:
        # MyStocks Kenya URL
        url = "https://live.mystocks.co.ke/"
        
        # Send request with headers to mimic a browser
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'max-age=0'
        }
        
        print(f"Attempting to fetch stock prices from MyStocks Kenya...")
        response = requests.get(url, headers=headers, timeout=20)
        response.raise_for_status()
        
        # Parse HTML
        soup = BeautifulSoup(response.text, 'lxml')
        
        # Save the HTML for debugging
        debug_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'mystocks_debug.txt')
        with open(debug_file, 'w', encoding='utf-8') as f:
            f.write(response.text)
        print(f"Saved MyStocks HTML for debugging to {debug_file}")
        
        # The MyStocks website has a section with corporate actions showing stock symbols
        # We'll extract stocks from these sections and try to match with our tickers
        today = datetime.now().strftime('%Y-%m-%d')
        
        # Look for stock symbols in the corporate actions section
        stock_elements = soup.select(".corporate_actions a")
        
        for element in stock_elements:
            try:
                # Extract ticker from the element
                ticker = element.text.strip()
                
                # Check if this ticker is in our list (case insensitive)
                if ticker.upper() in [t.upper() for t in tickers]:
                    # Now we need to find the price for this ticker
                    # This is more challenging as it requires navigating to the stock's detail page
                    # For now, we'll try to extract price information from the main page
                    
                    # Check if there's a parent element with price information
                    parent_element = element.parent.parent
                    if parent_element:
                        price_text = parent_element.text.strip()
                        # Try to extract price using pattern matching
                        # Look for patterns like "KES X.XX" or similar
                        if "KES" in price_text:
                            # Extract the price - this is a simplified approach
                            try:
                                # Split by KES and take the next part
                                parts = price_text.split("KES")
                                if len(parts) > 1:
                                    # Take the part after KES and clean it
                                    price_part = parts[1].strip()
                                    # Extract numeric value (assuming it's at the beginning)
                                    price_value = ""
                                    for char in price_part:
                                        if char.isdigit() or char == ".":
                                            price_value += char
                                        else:
                                            break
                                    
                                    if price_value:
                                        price = float(price_value)
                                        
                                        # Store the price
                                        if ticker not in current_data:
                                            current_data[ticker] = {}
                                        
                                        current_data[ticker][today] = price
                                        print(f"Found price for {ticker}: {price}")
                            except Exception as e:
                                print(f"Error extracting price for {ticker}: {e}")
            
            except Exception as e:
                print(f"Error processing stock element: {e}")
                continue
        
        # If we didn't get many tickers, try a different approach
        if len(current_data) < len(tickers) * 0.5:  # If we got less than half the tickers
            print("Few tickers found with first method, trying alternative approach...")
            
            # Try to find all potential ticker symbols on the page
            potential_tickers = soup.select("a[href^='/stock/']")
            
            for element in potential_tickers:
                try:
                    # Extract ticker from href
                    href = element.get('href', '')
                    if href.startswith('/stock/'):
                        ticker = href.split('/stock/')[1].strip()
                        
                        # Check if this ticker is in our list
                        if ticker.upper() in [t.upper() for t in tickers] and ticker not in current_data:
                            # Found a ticker, now try to fetch its price
                            # We'll need to make another request to the stock's detail page
                            stock_url = f"https://live.mystocks.co.ke/stock/{ticker}"
                            
                            try:
                                stock_response = requests.get(stock_url, headers=headers, timeout=10)
                                stock_response.raise_for_status()
                                
                                stock_soup = BeautifulSoup(stock_response.text, 'lxml')
                                
                                # Look for the price in the stock detail page
                                # The exact structure might vary, so we'll look for common patterns
                                price_elements = stock_soup.select(".price, .stock-price, .current-price")
                                
                                if price_elements:
                                    price_text = price_elements[0].text.strip()
                                    # Clean up the price text
                                    price_text = price_text.replace('KES', '').replace(',', '').strip()
                                    
                                    if price_text and price_text not in ["-", "N/A"]:
                                        price = float(price_text)
                                        
                                        # Store the price
                                        if ticker not in current_data:
                                            current_data[ticker] = {}
                                        
                                        current_data[ticker][today] = price
                                        print(f"Found price for {ticker} (method 2): {price}")
                            except Exception as e:
                                print(f"Error fetching detail page for {ticker}: {e}")
                except Exception as e:
                    print(f"Error processing ticker link: {e}")
                    continue
        
        # If we still don't have enough data, try a third approach using the entire page text
        if len(current_data) < len(tickers) * 0.7:  # If we still haven't found most tickers
            print("Still missing tickers, trying text search approach...")
            
            page_text = soup.text
            
            for ticker in tickers:
                # Skip tickers we already found
                if ticker in current_data:
                    continue
                
                # Search for the ticker in the page text
                ticker_pos = page_text.find(ticker)
                if ticker_pos != -1:
                    # Found the ticker, now try to find a nearby price
                    # Look for "KES" near the ticker (within 100 characters)
                    surrounding_text = page_text[max(0, ticker_pos - 50):min(len(page_text), ticker_pos + 150)]
                    
                    if "KES" in surrounding_text:
                        try:
                            # Extract price using the same pattern matching as before
                            parts = surrounding_text.split("KES")
                            if len(parts) > 1:
                                price_part = parts[1].strip()
                                price_value = ""
                                for char in price_part:
                                    if char.isdigit() or char == ".":
                                        price_value += char
                                    else:
                                        break
                                
                                if price_value:
                                    price = float(price_value)
                                    
                                    # Store the price
                                    if ticker not in current_data:
                                        current_data[ticker] = {}
                                    
                                    current_data[ticker][today] = price
                                    print(f"Found price for {ticker} (method 3): {price}")
                        except Exception as e:
                            print(f"Error extracting price for {ticker} from text: {e}")
        
        # If we still don't have enough data, try a fourth approach using a secondary page
        if len(current_data) < len(tickers):
            print("Still missing tickers, trying market data page...")
            
            # Try the quotes page which might have more structured data
            quotes_url = "https://live.mystocks.co.ke/quotes"
            
            try:
                quotes_response = requests.get(quotes_url, headers=headers, timeout=15)
                quotes_response.raise_for_status()
                
                quotes_soup = BeautifulSoup(quotes_response.text, 'lxml')
                
                # Look for a table with stock quotes
                quote_tables = quotes_soup.select("table")
                
                for table in quote_tables:
                    rows = table.select("tr")
                    
                    for row in rows:
                        try:
                            cells = row.select("td")
                            if len(cells) >= 2:  # Need at least ticker and price
                                ticker_cell = cells[0].text.strip()
                                
                                # Check if this is one of our tickers
                                ticker = ticker_cell.split()[0] if ticker_cell else ""  # Take first word as ticker
                                
                                if ticker and ticker.upper() in [t.upper() for t in tickers] and ticker not in current_data:
                                    # Look for price in subsequent cells
                                    for cell in cells[1:]:
                                        cell_text = cell.text.strip()
                                        
                                        # Try to parse as a number
                                        if cell_text:
                                            try:
                                                # Clean up and convert to float
                                                price_text = cell_text.replace('KES', '').replace(',', '').strip()
                                                
                                                # Validate that this looks like a price
                                                if price_text and price_text not in ["-", "N/A"] and any(c.isdigit() for c in price_text):
                                                    price = float(price_text)
                                                    
                                                    # Check if price is reasonable (not too high or low)
                                                    if 0.1 <= price <= 10000:  # Reasonable range for KES
                                                        # Store the price
                                                        if ticker not in current_data:
                                                            current_data[ticker] = {}
                                                        
                                                        current_data[ticker][today] = price
                                                        print(f"Found price for {ticker} (method 4): {price}")
                                                        break  # Found price, move to next row
                                            except Exception as e:
                                                # Not a valid price, try next cell
                                                continue
                        except Exception as e:
                            print(f"Error processing quote row: {e}")
                            continue
            except Exception as e:
                print(f"Error fetching quotes page: {e}")
        
        # If we still don't have data for all tickers, try Yahoo Finance as a fallback
        missing_tickers = [t for t in tickers if t not in current_data]
        if missing_tickers:
            print(f"Missing prices for {len(missing_tickers)} tickers, trying Yahoo Finance fallback...")
            
            for ticker in missing_tickers:
                try:
                    yahoo_ticker = f"{ticker}.NR"
                    stock = yf.Ticker(yahoo_ticker)
                    info = stock.info
                    
                    # Get the current price
                    if 'currentPrice' in info:
                        price = info['currentPrice']
                        
                        if ticker not in current_data:
                            current_data[ticker] = {}
                        
                        current_data[ticker][today] = price
                        print(f"Found price for {ticker} from Yahoo Finance: {price}")
                except Exception as e:
                    print(f"Error fetching Yahoo Finance data for {ticker}: {e}")
                    continue
        
        return current_data
        
    except Exception as e:
        print(f"Error fetching MyStocks Kenya prices: {e}")
        import traceback
        traceback.print_exc()
        return {}

def fetch_nse_data_alternate(tickers):
    """Alternate method to fetch NSE data using a different source"""
    current_data = {}
    
    try:
        # Alternative data source: businesstoday.co.ke might have NSE data
        url = "https://businesstoday.co.ke/category/markets/"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'lxml')
        
        # Look for NSE price data in the content
        article_content = soup.select('.entry-content p')
        
        today = datetime.now().strftime('%Y-%m-%d')
        
        for ticker in tickers:
            # This is just an example approach - actual website structure may vary
            for paragraph in article_content:
                text = paragraph.text
                if ticker in text and 'KES' in text:
                    # Try to extract price
                    try:
                        # This regex pattern is simplistic and would need to be adjusted
                        price_text = text.split(ticker)[1].split('KES')[1].strip()
                        price = float(price_text.replace(',', ''))
                        
                        if ticker not in current_data:
                            current_data[ticker] = {}
                        current_data[ticker][today] = price
                    except:
                        continue
        
        return current_data
        
    except Exception as e:
        print(f"Error fetching alternate NSE data: {e}")
        return {}

def fetch_nse_historical_data(tickers, existing_data=None):
    """Fetch historical data for NSE stocks"""
    if existing_data is None:
        existing_data = {}
    
    # Create a deep copy to avoid modifying the input
    historical_data = {ticker: existing_data.get(ticker, {}).copy() for ticker in tickers}
    
    try:
        # For each ticker, try to fetch historical data
        for ticker in tickers:
            # NSE doesn't have a direct API for historical data
            # We can try to use Yahoo Finance with .NR suffix as a source for historical data
            yahoo_ticker = f"{ticker}.NR"
            
            try:
                # Get data from Yahoo
                stock_data = yf.download(yahoo_ticker, period="1y", progress=False)
                
                # If we got data, process it
                if not stock_data.empty:
                    for date, row in stock_data.iterrows():
                        date_str = date.strftime('%Y-%m-%d')
                        close_price = row['Close']
                        
                        if ticker not in historical_data:
                            historical_data[ticker] = {}
                        historical_data[ticker][date_str] = close_price
            except Exception as e:
                print(f"Error fetching Yahoo data for {ticker}: {e}")
                continue
                
        return historical_data
        
    except Exception as e:
        print(f"Error fetching historical NSE data: {e}")
        return existing_data  # Return what we already have

def merge_current_with_historical(current_data, historical_data):
    """Merge current day's data with historical data"""
    merged_data = {}
    
    # Start with all historical data
    for ticker in historical_data:
        merged_data[ticker] = historical_data[ticker].copy()
    
    # Add/update with current data
    for ticker in current_data:
        if ticker not in merged_data:
            merged_data[ticker] = {}
        
        # Update with current data
        for date, price in current_data[ticker].items():
            merged_data[ticker][date] = price
    
    return merged_data

def create_dataframe_from_merged_data(merged_data, period="1mo"):
    """Convert merged data dictionary to a DataFrame and filter by period"""
    # Create a list of all dates from all tickers
    all_dates = set()
    for ticker in merged_data:
        all_dates.update(merged_data[ticker].keys())
    
    # Sort dates
    all_dates = sorted(all_dates)
    
    # Create DataFrame with dates as index
    df = pd.DataFrame(index=pd.to_datetime(all_dates))
    
    # Add each ticker as a column
    for ticker in merged_data:
        ticker_data = merged_data[ticker]
        series_data = {pd.to_datetime(date): price for date, price in ticker_data.items()}
        df[ticker] = pd.Series(series_data)
    
    # Filter by period
    if period == "1d":
        df = df.last('1D')
    elif period == "1w":
        df = df.last('7D')
    elif period == "1mo":
        df = df.last('30D')
    elif period == "3mo":
        df = df.last('90D')
    elif period == "6mo":
        df = df.last('180D')
    elif period == "1y":
        df = df.last('365D')
    
    # Sort index and forward fill any missing values
    df = df.sort_index().ffill()
    
    return df

def generate_mock_data(tickers, days=30):
    """Generate mock stock data for demonstration purposes"""
    end_date = dt.datetime.now()
    start_date = end_date - dt.timedelta(days=days)
    
    date_range = pd.date_range(start=start_date, end=end_date, freq='B')
    
    data = {}
    for ticker in tickers:
        # Start with a random base price between 50 and 500
        base_price = np.random.uniform(50, 500)
        
        # Generate random price movements
        daily_returns = np.random.normal(0.0005, 0.015, size=len(date_range))
        prices = base_price * (1 + np.cumsum(daily_returns))
        
        data[ticker] = prices
        
    # Create DataFrame with the generated data
    mock_df = pd.DataFrame(data, index=date_range)
    return mock_df

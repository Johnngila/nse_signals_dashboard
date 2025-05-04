import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
import numpy as np
from utils.signal_engine import generate_signals

# Set page config
st.set_page_config(page_title="NSE Signals Dashboard", layout="wide")

# Title and description
st.title("NSE Stock Market Dashboard")
st.markdown("A dashboard for NSE stock market data and signals")

# Load stock data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('NSE_data_all_stocks.csv')
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

# Load sector data
@st.cache_data
def load_sector_data():
    try:
        df = pd.read_csv('NSE_data_stock_market_sectors.csv')
        return df
    except Exception as e:
        st.error(f"Error loading sector data: {e}")
        return pd.DataFrame()

# Load and merge price data
@st.cache_data
def load_and_merge_price_data():
    """
    Load and merge price data from all available sources, prioritizing the most recent data.
    This function tries to load data from multiple sources and combines them intelligently.
    """
    # Initialize an empty dataframe for the final result
    final_price_data = pd.DataFrame()
    
    # Track data sources
    data_sources = []
    
    # 1. Try to load from Prices.csv (main format with current and previous prices)
    try:
        prices_df = pd.read_csv('Prices.csv')
        if not prices_df.empty:
            # Get the actual column names (dynamic approach instead of hardcoding)
            cols = prices_df.columns.tolist()
            
            # Expect at least these columns: Symbol, Name, current price, current volume, previous price, previous volume
            if len(cols) >= 6 and 'Symbol' in cols and 'Name' in cols:
                # Create a structured dataframe
                price_df = pd.DataFrame()
                price_df['Symbol'] = prices_df['Symbol']
                price_df['Name'] = prices_df['Name']
                
                # Get price columns (excluding Symbol and Name)
                data_cols = [col for col in cols if col not in ['Symbol', 'Name']]
                
                # Determine which columns are which based on naming patterns
                current_price_col = data_cols[0]  # First data column (e.g. "2-May")
                current_volume_col = data_cols[1]  # Second data column (e.g. "Volume - May 2")
                prev_price_col = data_cols[2]  # Third data column (e.g. "24-Apr")
                prev_volume_col = data_cols[3]  # Fourth data column (e.g. "Volume - April 24")
                
                print(f"Using dynamic columns: {current_price_col}, {current_volume_col}, {prev_price_col}, {prev_volume_col}")
                
                # Current price data
                price_df['Price'] = prices_df[current_price_col]
                price_df['Volume'] = prices_df[current_volume_col]
                
                # Previous price data
                price_df['PreviousPrice'] = prices_df[prev_price_col]
                price_df['PreviousVolume'] = prices_df[prev_volume_col]
                
                # Store original column names for reference
                price_df.attrs['current_price_col'] = current_price_col
                price_df.attrs['current_volume_col'] = current_volume_col
                price_df.attrs['prev_price_col'] = prev_price_col
                price_df.attrs['prev_volume_col'] = prev_volume_col
                
                # Calculate price change and percentage
                price_df['PriceChange'] = price_df['Price'] - price_df['PreviousPrice']
                # Avoid division by zero
                price_df['PriceChangePercent'] = np.where(
                    price_df['PreviousPrice'] != 0,
                    (price_df['PriceChange'] / price_df['PreviousPrice']) * 100,
                    0  # Set to 0 when previous price is 0
                )
                
                # Clean up numeric columns
                numeric_cols = ['Price', 'Volume', 'PreviousPrice', 'PreviousVolume', 
                              'PriceChange', 'PriceChangePercent']
                for col in numeric_cols:
                    if col in price_df.columns:
                        try:
                            # Handle different formats including commas, spaces, and other non-numeric characters
                            if price_df[col].dtype == object:
                                price_df[col] = price_df[col].astype(str).str.replace(',', '').str.replace(' ', '')
                                price_df[col] = pd.to_numeric(price_df[col], errors='coerce')
                            elif not np.issubdtype(price_df[col].dtype, np.number):
                                price_df[col] = pd.to_numeric(price_df[col], errors='coerce')
                        except Exception as e:
                            print(f"Error converting {col}: {e}")
                            # If conversion fails, try a different approach
                            try:
                                price_df[col] = price_df[col].replace('[\£,\$,\€,\,]', '', regex=True)
                                price_df[col] = pd.to_numeric(price_df[col], errors='coerce')
                            except Exception as e2:
                                print(f"Second attempt at converting {col} failed: {e2}")
                
                final_price_data = price_df
                data_sources.append('Prices.csv')
                print(f"Loaded price data from Prices.csv: {len(final_price_data)} rows")
                
                # Print sample data to debug
                print(f"Sample data: {final_price_data.head(2).to_dict()}")
            else:
                print(f"Unexpected column structure in Prices.csv. Found columns: {cols}")
    except Exception as e:
        print(f"Error loading Prices.csv: {e}")
    
    # 2. Try to load from data/prices.csv (if Prices.csv failed or to supplement data)
    if final_price_data.empty or len(final_price_data) < 10:  # If we have very little data so far
        try:
            prices_csv_df = pd.read_csv('data/prices.csv')
            
            if not prices_csv_df.empty:
                print(f"Loaded data from data/prices.csv with {len(prices_csv_df)} rows")
                
                # If we have no data yet, use this as primary source
                if final_price_data.empty:
                    final_price_data = prices_csv_df
                    data_sources.append('data/prices.csv (primary)')
                else:
                    # Otherwise, merge in any missing symbols
                    current_symbols = set(final_price_data['Symbol'])
                    new_symbols = [s for s in prices_csv_df['Symbol'] if s not in current_symbols]
                    
                    if new_symbols:
                        missing_rows = prices_csv_df[prices_csv_df['Symbol'].isin(new_symbols)]
                        final_price_data = pd.concat([final_price_data, missing_rows], ignore_index=True)
                        data_sources.append('data/prices.csv (supplemental)')
                        print(f"Added {len(new_symbols)} missing symbols from data/prices.csv")
        except Exception as e:
            print(f"Error loading data/prices.csv: {e}")
    
    # 3. Try to load from current_prices.csv
    try:
        current_df = pd.read_csv('data/current_prices.csv')
        
        if not current_df.empty:
            # If we already have data
            if not final_price_data.empty:
                # Merge missing stocks
                missing_symbols = [s for s in current_df['Symbol'] if s not in final_price_data['Symbol'].values]
                
                if missing_symbols:
                    missing_df = current_df[current_df['Symbol'].isin(missing_symbols)]
                    final_price_data = pd.concat([final_price_data, missing_df], ignore_index=True)
                    print(f"Added {len(missing_symbols)} missing stocks from current_prices.csv")
                
                # Also update any stocks that might have missing Price values
                for idx, row in final_price_data.iterrows():
                    if pd.isna(row.get('Price')) and row['Symbol'] in current_df['Symbol'].values:
                        current_price = current_df[current_df['Symbol'] == row['Symbol']]['Price'].iloc[0]
                        final_price_data.at[idx, 'Price'] = current_price
                        print(f"Updated missing price for {row['Symbol']} from current_prices.csv")
            else:
                # Use current_prices.csv as the primary data source
                final_price_data = current_df
                print(f"Loaded price data from current_prices.csv: {len(final_price_data)} rows")
            
            data_sources.append('current_prices.csv')
    except Exception as e:
        print(f"Error loading current_prices.csv: {e}")
    
    # 4. If still no data, try to fetch new data
    if final_price_data.empty:
        try:
            from app import fetch_nse_data
            df = fetch_nse_data()
            
            if df is not None and not df.empty:
                final_price_data = df
                print(f"Fetched new price data: {len(final_price_data)} rows")
                
                # Save the new data for future use
                try:
                    os.makedirs('data', exist_ok=True)
                    df.to_csv('data/current_prices.csv', index=False)
                    print("Saved new price data to data/current_prices.csv")
                except Exception as save_error:
                    print(f"Error saving new price data: {save_error}")
                
                data_sources.append('fresh fetch')
            else:
                print("Fetched data is empty")
        except Exception as e:
            print(f"Error fetching new data: {e}")
    
    # Final data quality check and cleanup
    if not final_price_data.empty:
        # Make sure we have the essential columns
        essential_cols = ['Symbol', 'Price']
        for col in essential_cols:
            if col not in final_price_data.columns:
                print(f"WARNING: Missing essential column: {col}")
                if col == 'Symbol' and 'Code' in final_price_data.columns:
                    final_price_data['Symbol'] = final_price_data['Code']
                    print("Used 'Code' column as 'Symbol'")
        
        # Convert Price to numeric if needed and ensure no NaN values
        numeric_cols = ['Price', 'Volume', 'PreviousPrice', 'PreviousVolume', 
                      'PriceChange', 'PriceChangePercent']
        for col in numeric_cols:
            if col in final_price_data.columns:
                try:
                    if not pd.api.types.is_numeric_dtype(final_price_data[col]):
                        final_price_data[col] = pd.to_numeric(final_price_data[col], errors='coerce')
                    # Fill NaN values with appropriate defaults
                    if col in ['Price', 'PreviousPrice']:
                        final_price_data[col] = final_price_data[col].fillna(0)
                    elif col in ['Volume', 'PreviousVolume']:
                        final_price_data[col] = final_price_data[col].fillna(0).astype(int)
                    elif col in ['PriceChange', 'PriceChangePercent']:
                        final_price_data[col] = final_price_data[col].fillna(0)
                except Exception as e:
                    print(f"Error finalizing {col}: {e}")
        
        print(f"Final price data has {len(final_price_data)} rows from sources: {', '.join(data_sources)}")
    else:
        print("WARNING: Could not load any price data from any source")
    
    return final_price_data

@st.cache_data
def load_price_data():
    """
    Load price data from available sources.
    This is now a wrapper around the more comprehensive load_and_merge_price_data function.
    Kept for backward compatibility.
    """
    return load_and_merge_price_data()

# Load signals data
@st.cache_data
def load_signals_data(ticker, signal_type="Moving Average Crossover", **params):
    """
    Load or generate trading signals for the specified ticker and signal type
    """
    try:
        # Generate signals using the signal_engine module
        signals_df = generate_signals(
            signal_type=signal_type,
            tickers=[ticker],
            **params
        )
        
        # If we got valid signals, return them
        if not signals_df.empty:
            return signals_df
        else:
            st.warning(f"No signal data could be generated for {ticker}. Check if price history is available.")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Error generating signals: {e}")
        return pd.DataFrame()

# Function to extract current and previous dates from price data sources
def extract_dates_from_price_data():
    """Extract the current and previous dates from the price data sources"""
    # Set default dates based on current context (May 5, 2025)
    dates = {
        'current_date': '2025-05-02',  # Default to current market date (May 2, 2025)
        'previous_date': '2025-04-24'  # Default previous market date (April 24, 2025)
    }
    
    try:
        # Try to extract dates from Prices.csv (most reliable source)
        try:
            prices_df = pd.read_csv('Prices.csv')
            if not prices_df.empty:
                # Get the column headers which contain the dates
                date_cols = [col for col in prices_df.columns if col not in ['Symbol', 'Name']]
                
                if len(date_cols) >= 4:  # We expect at least 4 columns (2 dates, 2 volumes)
                    # First date column (e.g., "2-May")
                    current_date_col = date_cols[0]
                    
                    # Previous date is embedded in the 4th column header (e.g., "Volume - April 24")
                    previous_date_col = date_cols[3]
                    
                    # Current date processing
                    from datetime import datetime
                    current_year = datetime.now().year
                    
                    if '-' in current_date_col:
                        # Format: "2-May"
                        day, month = current_date_col.split('-')
                        try:
                            month_num = datetime.strptime(month, '%b').month
                            dates['current_date'] = f"{current_year}-{month_num:02d}-{int(day):02d}"
                        except:
                            print(f"Could not parse current date: {current_date_col}")
                    
                    # Previous date processing
                    if 'Volume - ' in previous_date_col:
                        prev_date_str = previous_date_col.replace('Volume - ', '')
                        try:
                            # Try to parse date formats like "April 24"
                            prev_date = datetime.strptime(prev_date_str, '%B %d')
                            dates['previous_date'] = f"{current_year}-{prev_date.month:02d}-{prev_date.day:02d}"
                        except:
                            print(f"Could not parse previous date: {prev_date_str}")
                    
                    print(f"Extracted dates from Prices.csv: Current={dates['current_date']}, Previous={dates['previous_date']}")
        except Exception as e:
            print(f"Error extracting dates from Prices.csv: {e}")
        
        # Return the extracted dates
        return dates
    except Exception as e:
        print(f"Error in extract_dates_from_price_data: {e}")
        return dates

# Function to create signal charts
def create_signal_chart(signals_df, stock_code, analysis_type, params):
    signal_col = f"{stock_code}_Signal"
    price_col = f"{stock_code}_Price"
    
    if signal_col not in signals_df.columns or price_col not in signals_df.columns:
        st.warning(f"No signal data available for {stock_code}")
        return
    
    # Create a plotly figure for signal visualization
    fig = make_subplots(rows=2, cols=1, 
                         shared_xaxes=True, 
                         vertical_spacing=0.1,
                         row_heights=[0.7, 0.3])
    
    # Add price line to top subplot
    fig.add_trace(
        go.Scatter(
            x=signals_df.index,
            y=signals_df[price_col],
            name="Price",
            line=dict(color='royalblue', width=1.5)
        ),
        row=1, col=1
    )
    
    # Add buy signals
    buy_signals = signals_df[signals_df[signal_col] == 1]
    fig.add_trace(
        go.Scatter(
            x=buy_signals.index,
            y=buy_signals[price_col],
            mode='markers',
            marker=dict(symbol='triangle-up', size=12, color='green'),
            name='Buy Signal'
        ),
        row=1, col=1
    )
    
    # Add sell signals
    sell_signals = signals_df[signals_df[signal_col] == -1]
    fig.add_trace(
        go.Scatter(
            x=sell_signals.index,
            y=sell_signals[price_col],
            mode='markers',
            marker=dict(symbol='triangle-down', size=12, color='red'),
            name='Sell Signal'
        ),
        row=1, col=1
    )
    
    # Add technical indicators depending on the analysis type
    if analysis_type == "Moving Average Crossover":
        # Add short moving average
        short_window = params.get("short_window", 20)
        long_window = params.get("long_window", 50)
        
        signals_df[f'{stock_code}_SMA_{short_window}'] = signals_df[price_col].rolling(window=short_window).mean()
        signals_df[f'{stock_code}_SMA_{long_window}'] = signals_df[price_col].rolling(window=long_window).mean()
        
        fig.add_trace(
            go.Scatter(
                x=signals_df.index,
                y=signals_df[f'{stock_code}_SMA_{short_window}'],
                name=f'SMA {short_window}',
                line=dict(color='orange', width=1.5)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=signals_df.index,
                y=signals_df[f'{stock_code}_SMA_{long_window}'],
                name=f'SMA {long_window}',
                line=dict(color='magenta', width=1.5)
            ),
            row=1, col=1
        )
        
    elif analysis_type == "RSI":
        # If RSI is already in the signals dataframe
        rsi_col = f"{stock_code}_RSI"
        if rsi_col in signals_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=signals_df.index,
                    y=signals_df[rsi_col],
                    name='RSI',
                    line=dict(color='purple', width=1.5)
                ),
                row=2, col=1
            )
            
            # Add overbought and oversold lines
            overbought = params.get("overbought", 70)
            oversold = params.get("oversold", 30)
            
            fig.add_shape(
                type="line", line=dict(dash='dash', color='red', width=1),
                y0=overbought, y1=overbought, x0=signals_df.index[0], x1=signals_df.index[-1],
                row=2, col=1
            )
            
            fig.add_shape(
                type="line", line=dict(dash='dash', color='green', width=1),
                y0=oversold, y1=oversold, x0=signals_df.index[0], x1=signals_df.index[-1],
                row=2, col=1
            )
            
            # Set y-axis range for RSI
            fig.update_yaxes(range=[0, 100], row=2, col=1)
            
    elif analysis_type == "MACD":
        # If MACD is already in the signals dataframe
        macd_col = f"{stock_code}_MACD"
        signal_line_col = f"{stock_code}_MACD_Signal"
        histogram_col = f"{stock_code}_MACD_Hist"
        
        if macd_col in signals_df.columns and signal_line_col in signals_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=signals_df.index,
                    y=signals_df[macd_col],
                    name='MACD',
                    line=dict(color='blue', width=1.5)
                ),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=signals_df.index,
                    y=signals_df[signal_line_col],
                    name='Signal Line',
                    line=dict(color='red', width=1.5)
                ),
                row=2, col=1
            )
            
            if histogram_col in signals_df.columns:
                # Color histogram bars based on value
                colors = ['green' if val > 0 else 'red' for val in signals_df[histogram_col]]
                
                fig.add_trace(
                    go.Bar(
                        x=signals_df.index,
                        y=signals_df[histogram_col],
                        name='Histogram',
                        marker_color=colors
                    ),
                    row=2, col=1
                )
    
    elif analysis_type == "Bollinger Bands":
        # If Bollinger Bands are already in the signals dataframe
        bb_upper_col = f"{stock_code}_BB_Upper"
        bb_middle_col = f"{stock_code}_BB_Middle"
        bb_lower_col = f"{stock_code}_BB_Lower"
        
        if bb_upper_col in signals_df.columns and bb_middle_col in signals_df.columns and bb_lower_col in signals_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=signals_df.index,
                    y=signals_df[bb_upper_col],
                    name='Upper Band',
                    line=dict(color='rgba(250, 0, 0, 0.5)', width=1)
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=signals_df.index,
                    y=signals_df[bb_middle_col],
                    name='Middle Band',
                    line=dict(color='rgba(0, 0, 250, 0.5)', width=1)
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=signals_df.index,
                    y=signals_df[bb_lower_col],
                    name='Lower Band',
                    line=dict(color='rgba(250, 0, 0, 0.5)', width=1),
                    fill='tonexty',
                    fillcolor='rgba(200, 200, 200, 0.3)'
                ),
                row=1, col=1
            )
    
    # Update layout
    fig.update_layout(
        height=600,
        title_text=f"{stock_code} - {analysis_type} Analysis",
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    # Add volume histogram if needed for future enhancement
    
    # Customize axes
    fig.update_yaxes(title_text="Price (KES)", row=1, col=1)
    fig.update_yaxes(title_text="Indicator", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    
    # Display chart
    st.plotly_chart(fig, use_container_width=True)

# Function to display analysis explanation
def display_analysis_explanation(analysis_type, params):
    st.subheader("About This Analysis")
    
    if analysis_type == "Moving Average Crossover":
        st.write(f"""
        **Moving Average Crossover** is a trend-following strategy that uses two moving averages:
        
        - Short-term SMA: {params.get('short_window', 20)} days
        - Long-term SMA: {params.get('long_window', 50)} days
        
        **Buy Signal**: When the short-term SMA crosses above the long-term SMA
        
        **Sell Signal**: When the short-term SMA crosses below the long-term SMA
        
        This strategy works best in trending markets and may generate false signals in sideways markets.
        """)
    
    elif analysis_type == "RSI":
        st.write(f"""
        **Relative Strength Index (RSI)** is a momentum oscillator that measures the speed and change of price movements.
        
        - RSI Window: {params.get('window', 14)} days
        - Overbought Level: {params.get('overbought', 70)}
        - Oversold Level: {params.get('oversold', 30)}
        
        **Buy Signal**: When RSI crosses above the oversold level from below
        
        **Sell Signal**: When RSI crosses below the overbought level from above
        
        RSI works best in ranging markets and may provide early signals in trending markets.
        """)
    
    elif analysis_type == "MACD":
        st.write(f"""
        **Moving Average Convergence Divergence (MACD)** is a trend-following momentum indicator.
        
        - Fast EMA: {params.get('fast_period', 12)} days
        - Slow EMA: {params.get('slow_period', 26)} days
        - Signal Line EMA: {params.get('signal_period', 9)} days
        
        **Buy Signal**: When MACD line crosses above the signal line
        
        **Sell Signal**: When MACD line crosses below the signal line
        
        MACD is versatile and can identify both trend direction and momentum.
        """)
    
    elif analysis_type == "Bollinger Bands":
        st.write(f"""
        **Bollinger Bands** are volatility bands placed above and below a moving average.
        
        - Moving Average Window: {params.get('window', 20)} days
        - Standard Deviation: {params.get('num_std', 2.0)}
        
        **Buy Signal**: When price touches or crosses below the lower band and then starts moving upward
        
        **Sell Signal**: When price touches or crosses above the upper band and then starts moving downward
        
        Bollinger Bands are useful for identifying overbought and oversold conditions, especially in ranging markets.
        """)
    
    elif analysis_type == "Mean Reversion":
        st.write(f"""
        **Mean Reversion** strategy is based on the concept that prices tend to revert to their mean over time.
        
        - Lookback Period: {params.get('lookback_period', 20)} days
        - Standard Deviation Threshold: {params.get('std_dev_threshold', 2.0)}
        
        **Buy Signal**: When price deviates significantly below the mean
        
        **Sell Signal**: When price returns to or exceeds the mean
        
        This strategy works well in range-bound or cyclical markets but may underperform in strongly trending markets.
        """)
    
    else:
        st.write(f"Analysis type: {analysis_type}")
        st.write("Parameters:", params)

# Function to generate stock recommendations based on selected analysis method
def generate_recommendations(stock_data, price_data, analysis_type="Moving Average Crossover"):
    """
    Generate buy, sell, and hold recommendations based on the selected analysis method
    
    Parameters:
    -----------
    stock_data : DataFrame
        Stock information data
    price_data : DataFrame
        Current price data
    analysis_type : str
        The selected analysis method
    
    Returns:
    --------
    dict
        Dictionary containing buy, sell, and hold recommendations
    """
    recommendations = {
        'buy': [],
        'sell': [],
        'hold': []
    }
    
    # Limit the number of stocks to analyze (for performance)
    max_stocks_to_analyze = 20
    
    # Get list of all stocks
    all_stocks = sorted(stock_data['Code'].unique()) if 'Code' in stock_data.columns else []
    
    # Create a copy of stock data so we don't modify the original
    stocks_to_analyze = all_stocks[:max_stocks_to_analyze]
    
    # Define parameters based on analysis type
    params = {}
    if analysis_type == "Moving Average Crossover":
        params = {
            "short_window": 20,
            "long_window": 50,
            "confirmation_window": 3,
            "trend_filter": True
        }
    elif analysis_type == "RSI":
        params = {
            "window": 14,
            "overbought": 70,
            "oversold": 30,
            "confirmation_days": 2
        }
    elif analysis_type == "MACD":
        params = {
            "fast_period": 12,
            "slow_period": 26,
            "signal_period": 9,
            "zero_line_filter": True
        }
    elif analysis_type == "Bollinger Bands":
        params = {
            "window": 20,
            "num_std": 2.0,
            "mean_reversion": True
        }
    elif analysis_type == "Mean Reversion":
        params = {
            "lookback_period": 20,
            "std_dev_threshold": 2.0,
            "exit_threshold": 0.5,
            "max_holding": 10
        }
    elif analysis_type == "Linear Regression":
        params = {
            "window": 20,
            "forecast_period": 5,
            "confidence": 0.95,
            "entry_threshold": 0.02
        }
    elif analysis_type == "Relative Value":
        params = {
            "lookback": 30,
            "threshold": 1.5
        }
    elif analysis_type == "Statistical Arbitrage":
        params = {
            "lookback": 20,
            "entry_z": 2.0,
            "exit_z": 0.5,
            "max_pairs": 3
        }
    elif analysis_type == "Volume Price Trend":
        params = {
            "window": 14,
            "threshold": 1.0
        }
    elif analysis_type == "GARCH Volatility":
        params = {
            "window": 100,
            "forecast_periods": 5,
            "vol_threshold": 0.4
        }
    
    try:
        # Create a progress bar for processing
        progress_placeholder = st.empty()
        progress_bar = progress_placeholder.progress(0)
        
        # Analyze each stock
        for i, stock_code in enumerate(stocks_to_analyze):
            # Update progress
            progress_percent = int((i + 1) / len(stocks_to_analyze) * 100)
            progress_bar.progress(progress_percent)
            
            try:
                # Get stock information
                stock_info = stock_data[stock_data['Code'] == stock_code]
                if stock_info.empty:
                    continue
                
                # Get price information
                price_info = None
                if not price_data.empty and 'Symbol' in price_data.columns and stock_code in price_data['Symbol'].values:
                    price_info = price_data[price_data['Symbol'] == stock_code]
                    if price_info.empty or 'Price' not in price_info.columns:
                        continue
                else:
                    continue
                
                # Generate signals for this stock
                signals_df = generate_signals(
                    signal_type=analysis_type,
                    tickers=[stock_code],
                    **params
                )
                
                if signals_df.empty:
                    continue
                
                # Check if we have a signal column for this stock
                signal_col = f"{stock_code}_Signal"
                price_col = f"{stock_code}_Price"
                
                if signal_col not in signals_df.columns or price_col not in signals_df.columns:
                    continue
                
                # Get the latest signal
                latest_signal = signals_df[signal_col].iloc[-1]
                
                # Create stock recommendation object
                stock_rec = {
                    'symbol': stock_code,
                    'name': stock_info['Name'].iloc[0],
                    'price': price_info['Price'].iloc[0],
                    'sector': stock_info['Sector'].iloc[0] if 'Sector' in stock_info.columns else 'Unknown'
                }
                
                # Add to appropriate recommendation list
                if latest_signal == 1:  # Buy signal
                    recommendations['buy'].append(stock_rec)
                elif latest_signal == -1:  # Sell signal
                    recommendations['sell'].append(stock_rec)
                else:  # Hold or no signal
                    recommendations['hold'].append(stock_rec)
                    
            except Exception as e:
                # Skip on error
                print(f"Error analyzing {stock_code}: {e}")
                continue
                
        # Clear progress bar when done
        progress_placeholder.empty()
        
    except Exception as e:
        st.error(f"Error generating recommendations: {e}")
    
    return recommendations

# Initialize session states
def init_session_state():
    if 'view' not in st.session_state:
        st.session_state.view = "stock"  # Default view: stock or sector
    if 'selected_stock' not in st.session_state:
        st.session_state.selected_stock = None
    if 'selected_sector' not in st.session_state:
        st.session_state.selected_sector = None
    if 'selected_analysis' not in st.session_state:
        st.session_state.selected_analysis = "Moving Average Crossover"
    # Initialize favorite stocks (with some defaults that can be changed)
    if 'favorite_stocks' not in st.session_state:
        st.session_state.favorite_stocks = []  # Start with an empty list
    # Add a counter for cycling through favorite stocks in the auto-display feature
    if 'favorite_index' not in st.session_state:
        st.session_state.favorite_index = 0
    # Add a timestamp to control the cycling interval
    if 'last_update_time' not in st.session_state:
        st.session_state.last_update_time = pd.Timestamp.now()

# Function to create volume analysis chart
def create_volume_analysis_chart(stock_code, price_data, window=5):
    """
    Create a comprehensive volume analysis chart for a stock
    
    Parameters:
    -----------
    stock_code : str
        The stock symbol/code to analyze
    price_data : DataFrame
        The price data containing volume information
    window : int
        Window for moving averages
        
    Returns:
    --------
    fig : plotly.graph_objects.Figure or None
        The plotly figure with volume analysis, or None if data unavailable
    """
    # Check if we have the necessary data
    if not price_data.empty and 'Symbol' in price_data.columns and stock_code in price_data['Symbol'].values:
        price_row = price_data[price_data['Symbol'] == stock_code]
        
        if price_row.empty or 'Volume' not in price_row.columns or 'PreviousVolume' not in price_row.columns:
            return None
        
        # Get current and previous volume
        current_vol = price_row['Volume'].iloc[0]
        prev_vol = price_row['PreviousVolume'].iloc[0]
        
        if pd.isna(current_vol) or pd.isna(prev_vol):
            return None
        
        # Extract dates
        date_info = extract_dates_from_price_data()
        current_date_str = date_info['current_date']
        previous_date_str = date_info['previous_date']
        
        # Format dates for display
        current_date_display = format_date_for_display(current_date_str)
        previous_date_display = format_date_for_display(previous_date_str)
        
        # Calculate volume change
        vol_change = current_vol - prev_vol
        vol_change_percent = (vol_change / prev_vol) * 100 if prev_vol > 0 else 0
        
        # Get price data for context
        current_price = price_row['Price'].iloc[0] if 'Price' in price_row.columns else None
        prev_price = price_row['PreviousPrice'].iloc[0] if 'PreviousPrice' in price_row.columns else None
        price_change = price_row['PriceChange'].iloc[0] if 'PriceChange' in price_row.columns else None
        
        # Create a figure with two subplots: volume and price+volume correlation
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.6, 0.4],
            shared_xaxes=False,
            vertical_spacing=0.1,
            subplot_titles=("Volume Comparison", "Price-Volume Correlation")
        )
        
        # 1. Volume comparison bar chart
        date_labels = [previous_date_display, current_date_display]
        volumes = [prev_vol, current_vol]
        
        # Add volume bars
        fig.add_trace(
            go.Bar(
                x=date_labels,
                y=volumes,
                marker_color=['rgba(55, 83, 109, 0.7)', 'rgba(26, 118, 255, 0.7)'],
                text=[f"{int(vol):,}" for vol in volumes],
                textposition='auto',
                hoverinfo='text',
                hovertext=[
                    f"Date: {previous_date_display}<br>Volume: {int(prev_vol):,}",
                    f"Date: {current_date_display}<br>Volume: {int(current_vol):,}<br>Change: {vol_change_percent:.1f}%"
                ],
            ),
            row=1, col=1
        )
        
        # 2. Price-Volume correlation chart (scatter plot)
        if current_price is not None and prev_price is not None:
            # Create a scatter plot of price vs volume for the two data points
            fig.add_trace(
                go.Scatter(
                    x=[prev_price, current_price],
                    y=[prev_vol, current_vol],
                    mode='markers+lines+text',
                    marker=dict(
                        size=[12, 14],
                        color=['rgba(55, 83, 109, 0.9)', 'rgba(26, 118, 255, 0.9)']
                    ),
                    line=dict(
                        color='rgba(50, 50, 50, 0.5)',
                        dash='dot'
                    ),
                    text=[previous_date_display, current_date_display],
                    textposition="top center",
                    hoverinfo='text',
                    hovertext=[
                        f"Date: {previous_date_display}<br>Price: KES {prev_price:.2f}<br>Volume: {int(prev_vol):,}",
                        f"Date: {current_date_display}<br>Price: KES {current_price:.2f}<br>Volume: {int(current_vol):,}"
                    ],
                ),
                row=2, col=1
            )
        
        # Update layout
        fig.update_layout(
            title=f"Volume Analysis for {stock_code}",
            height=500,
            showlegend=False,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        # Update y-axes
        fig.update_yaxes(title_text="Volume (Shares)", tickformat=",", row=1, col=1)
        fig.update_yaxes(title_text="Volume (Shares)", tickformat=",", row=2, col=1)
        fig.update_xaxes(title_text="Date", row=1, col=1)
        fig.update_xaxes(title_text="Price (KES)", row=2, col=1)
        
        # Add volume change annotation
        direction = "↑" if vol_change > 0 else "↓" if vol_change < 0 else "→"
        color = "green" if vol_change > 0 else "red" if vol_change < 0 else "gray"
        
        fig.add_annotation(
            x=0.98, y=0.98,
            xref="paper", yref="paper",
            text=f"Volume Change: {direction} {abs(vol_change_percent):.1f}%",
            showarrow=False,
            font=dict(size=14, color=color),
            align="right",
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor=color,
            borderwidth=1,
            borderpad=4
        )
        
        # Add a correlation indication 
        if current_price is not None and prev_price is not None and price_change is not None:
            price_direction = price_change > 0
            volume_direction = vol_change > 0
            
            # Determine correlation message
            if price_direction == volume_direction:
                corr_message = "Price and volume moving in same direction"
                if price_direction:  # Both positive
                    corr_color = "green"
                    corr_interpretation = "Possible bullish confirmation"
                else:  # Both negative
                    corr_color = "red"
                    corr_interpretation = "Possible bearish confirmation"
            else:
                if price_direction:  # Price up, volume down
                    corr_color = "orange"
                    corr_message = "Price up but volume down"
                    corr_interpretation = "Possible weak rally"
                else:  # Price down, volume up
                    corr_color = "purple"
                    corr_message = "Price down but volume up"
                    corr_interpretation = "Possible selling pressure"
            
            fig.add_annotation(
                x=0.5, y=0.05,
                xref="paper", yref="paper",
                text=f"{corr_message}: {corr_interpretation}",
                showarrow=False,
                font=dict(size=12, color=corr_color),
                align="center",
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor=corr_color,
                borderwidth=1,
                borderpad=4
            )
        
        return fig
    
    return None

# Helper function to format dates
def format_date_for_display(date_str):
    """Format date string for display in the UI. 
    Converts dates like '2025-05-04' to 'May 4, 2025'"""
    try:
        from datetime import datetime
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        return date_obj.strftime('%b %d, %Y')
    except:
        return date_str  # Return original if parsing fails

# Function to display favorite stock analysis
def display_favorite_stock_analysis(stock_code, stock_data, price_data, analysis_type, params):
    # Get stock information
    stock_df = stock_data[stock_data['Code'] == stock_code]
    
    if stock_df.empty:
        st.warning(f"No data found for {stock_code}")
        return
    
    # Stock header and basic info - more compact
    st.write(f"**{stock_code}**: {stock_df['Name'].iloc[0][:15]}...")
    
    # Show price if available - more compact
    if not price_data.empty and 'Symbol' in price_data.columns and stock_code in price_data['Symbol'].values:
        price_row = price_data[price_data['Symbol'] == stock_code]
        if not price_row.empty and 'Price' in price_row.columns:
            price = price_row['Price'].iloc[0]
            
            # Get price change for delta display
            previous_price = price_row['PreviousPrice'].iloc[0] if 'PreviousPrice' in price_row.columns and not pd.isna(price_row['PreviousPrice'].iloc[0]) else None
            price_change_pct = price_row['PriceChangePercent'].iloc[0] if 'PriceChangePercent' in price_row.columns and not pd.isna(price_row['PriceChangePercent'].iloc[0]) else None
            
            # Display price with change percentage
            delta = f"{price_change_pct:.1f}%" if price_change_pct is not None else None
            st.metric("Price", f"KES {price}", delta=delta, delta_color="normal", label_visibility="visible")
            
            # Show volume if available - more compact
            if 'Volume' in price_row.columns and not pd.isna(price_row['Volume'].iloc[0]):
                volume = price_row['Volume'].iloc[0]
                st.caption(f"Volume: {int(volume):,} shares")
    
    # Generate signals for this stock
    with st.spinner(f"Analyzing..."):
        signals_df = load_signals_data(stock_code, analysis_type, **params)
        
        if not signals_df.empty:
            # Check if we have a signal column for this stock
            signal_col = f"{stock_code}_Signal"
            price_col = f"{stock_code}_Price"
            
            if signal_col in signals_df.columns and price_col in signals_df.columns:
                # Get the latest signal
                latest_signal = signals_df[signal_col].iloc[-1]
                
                # Display current signal - more compact
                signal_text = "HOLD"
                signal_color = "gray"
                
                if latest_signal == 1:
                    signal_text = "BUY"
                    signal_color = "green"
                elif latest_signal == -1:
                    signal_text = "SELL"
                    signal_color = "red"
                
                # Show signal prominently but more compact
                st.markdown(f"""
                <div style="background-color: {signal_color}; padding: 3px; border-radius: 3px; margin: 3px 0;">
                    <h4 style="color: white; text-align: center; margin: 0;">{signal_text}</h4>
                </div>
                """, unsafe_allow_html=True)
                
                # Create mini-chart - more compact
                fig = go.Figure()
                
                # Price line
                fig.add_trace(
                    go.Scatter(
                        x=signals_df.index[-20:],  # Last 20 days for more compact view
                        y=signals_df[price_col][-20:],
                        name="Price",
                        line=dict(color='royalblue', width=1.5)
                    )
                )
                
                # Add buy/sell markers
                buy_signals = signals_df[signals_df[signal_col] == 1][-20:]
                sell_signals = signals_df[signals_df[signal_col] == -1][-20:]
                
                fig.add_trace(
                    go.Scatter(
                        x=buy_signals.index,
                        y=buy_signals[price_col],
                        mode='markers',
                        marker=dict(symbol='triangle-up', size=8, color='green'),
                        name='Buy'
                    )
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=sell_signals.index,
                        y=sell_signals[price_col],
                        mode='markers',
                        marker=dict(symbol='triangle-down', size=8, color='red'),
                        name='Sell'
                    )
                )
                
                # Update layout for a very compact chart
                fig.update_layout(
                    height=150,
                    margin=dict(l=5, r=5, t=25, b=5),
                    title_text=f"{analysis_type[:12]}...",
                    xaxis_title=None,
                    yaxis_title=None,
                    showlegend=False,
                    font=dict(size=10),
                    plot_bgcolor='rgba(240,240,240,0.5)'
                )
                
                # Simplify x-axis
                fig.update_xaxes(showticklabels=False)
                
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                
                # Add volume indicator (compact version)
                if 'Volume' in price_row.columns and 'PreviousVolume' in price_row.columns:
                    current_vol = price_row['Volume'].iloc[0]
                    prev_vol = price_row['PreviousVolume'].iloc[0]
                    
                    if not pd.isna(current_vol) and not pd.isna(prev_vol):
                        vol_change = current_vol - prev_vol
                        vol_pct = (vol_change / prev_vol * 100) if prev_vol > 0 else 0
                        
                        vol_icon = "↑" if vol_change > 0 else "↓" if vol_change < 0 else "→"
                        vol_color = "green" if vol_change > 0 else "red" if vol_change < 0 else "gray"
                        
                        st.caption(f"Volume: {vol_icon} {abs(vol_pct):.1f}% ({int(current_vol):,} shares)")
        else:
            st.info(f"No signal data available for {stock_code}")

# Main function
def main():
    # Initialize session state
    init_session_state()
    
    # Load data
    stock_data = load_data()
    sector_data = load_sector_data()
    price_data = load_and_merge_price_data()
    
    # Extract dates from price data
    date_info = extract_dates_from_price_data()
    current_date_str = date_info['current_date']
    previous_date_str = date_info['previous_date']
    
    # Format dates for display
    current_date_display = format_date_for_display(current_date_str)
    previous_date_display = format_date_for_display(previous_date_str)
    
    if not stock_data.empty:
        # Get list of all stocks and sectors
        all_stocks = sorted(stock_data['Code'].unique()) if 'Code' in stock_data.columns else []
        all_sectors = sorted(stock_data['Sector'].unique().tolist()) if 'Sector' in stock_data.columns else []

        # Create a 3-column layout for the entire page: main content, favorites panel, and sidebar
        main_col, favorites_col = st.columns([4, 1])
        
        # FAVORITES PANEL - Right side auto-cycling stocks (moved above main content to ensure it renders first)
        with favorites_col:
            st.header("Favorites")
            
            # Check if we have any favorite stocks
            if not st.session_state.favorite_stocks:
                st.info("Add stocks to your favorites")
            else:
                # Create a container for the auto-cycling analysis
                auto_analysis_container = st.container(border=True)
                
                # Get current time and check if we need to update the display
                current_time = pd.Timestamp.now()
                time_diff = (current_time - st.session_state.last_update_time).total_seconds()
                
                # Update the display every 5 seconds
                if time_diff >= 5:
                    # Update the index to the next favorite stock
                    if len(st.session_state.favorite_stocks) > 0:
                        st.session_state.favorite_index = (st.session_state.favorite_index + 1) % len(st.session_state.favorite_stocks)
                        st.session_state.last_update_time = current_time
                
                # Get the current favorite stock to display
                if st.session_state.favorite_stocks:
                    current_fav_idx = min(st.session_state.favorite_index, len(st.session_state.favorite_stocks) - 1)
                    current_fav_stock = st.session_state.favorite_stocks[current_fav_idx]
                    
                    # Define analysis parameters
                    analysis_type = st.session_state.selected_analysis
                    params = {}
                    if analysis_type == "Moving Average Crossover":
                        params = {"short_window": 20, "long_window": 50, "confirmation_window": 3, "trend_filter": True}
                    elif analysis_type == "RSI":
                        params = {"window": 14, "overbought": 70, "oversold": 30, "confirmation_days": 2}
                    elif analysis_type == "MACD":
                        params = {"fast_period": 12, "slow_period": 26, "signal_period": 9, "zero_line_filter": True}
                    elif analysis_type == "Bollinger Bands":
                        params = {"window": 20, "num_std": 2.0, "mean_reversion": True}
                    elif analysis_type == "Mean Reversion":
                        params = {"lookback_period": 20, "std_dev_threshold": 2.0, "exit_threshold": 0.5}
                    elif analysis_type == "Linear Regression":
                        params = {"window": 20, "forecast_period": 5, "confidence": 0.95}
                    elif analysis_type == "Relative Value":
                        params = {"lookback": 30, "threshold": 1.5}
                    elif analysis_type == "Statistical Arbitrage":
                        params = {"lookback": 20, "entry_z": 2.0, "exit_z": 0.5}
                    elif analysis_type == "Volume Price Trend":
                        params = {"window": 14, "threshold": 1.0}
                    elif analysis_type == "GARCH Volatility":
                        params = {"window": 100, "forecast_periods": 5, "vol_threshold": 0.4}
                    
                    # Show the timer progress - more compact
                    time_remaining = max(0, 5 - time_diff)
                    progress_val = 1 - (time_remaining / 5)  # 0 to 1 progress value
                    progress_bar = st.progress(progress_val, text=f"Next: {time_remaining:.1f}s")
                    
                    # Display the navigation for favorites - more compact
                    fav_count = len(st.session_state.favorite_stocks)
                    st.caption(f"{current_fav_idx + 1}/{fav_count}")
                    
                    # Display the stock analysis in the auto-analysis container
                    with auto_analysis_container:
                        display_favorite_stock_analysis(current_fav_stock, stock_data, price_data, analysis_type, params)
                        
                        # Add button to view full analysis - smaller
                        if st.button(f"View Full", key=f"view_full_{current_fav_stock}"):
                            st.session_state.view = "stock"
                            st.session_state.selected_stock = current_fav_stock
                            st.rerun()

# Run the app
if __name__ == "__main__":
    main()

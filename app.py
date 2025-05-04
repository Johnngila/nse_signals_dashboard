# Streamlit NSE Dashboard Main App
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
from utils.data_fetcher import fetch_stock_data
from utils.signal_engine import generate_signals
import datetime as dt

# Set page config with updated theme - MUST BE CALLED FIRST
st.set_page_config(
    page_title="NSE Signals Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set Plotly theme for consistent styling
pio.templates.default = "plotly_white"

# Custom color theme with expanded palette
theme_colors = {
    "primary": "#1E88E5",      # Blue
    "secondary": "#26A69A",    # Teal
    "success": "#66BB6A",      # Green
    "danger": "#EF5350",       # Red
    "warning": "#FFCA28",      # Amber
    "info": "#29B6F6",         # Light Blue
    "light": "#E0E0E0",        # Light Grey
    "dark": "#212121",         # Dark Grey
    "background": "#FAFAFA",   # Almost White
    "purple": "#9C27B0",       # Purple
    "orange": "#FF9800",       # Orange
    "pink": "#EC407A",         # Pink
    "indigo": "#3F51B5",       # Indigo
}

# Enhanced CSS for better UI
st.markdown("""
<style>
    /* Main container styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Card styling */
    .stcard {
        background-color: white;
        border-radius: 0.5rem;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .stcard:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    
    /* Metric styling */
    .metric-row {
        display: flex;
        flex-wrap: wrap;
        gap: 1rem;
        margin-bottom: 1rem;
    }
    .metric-container {
        background-color: white;
        border-radius: 0.5rem;
        padding: 1rem 1.5rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        flex: 1;
        min-width: 120px;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #666;
        margin-bottom: 0.3rem;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: 600;
        color: #333;
    }
    
    /* Improved headers */
    h1, h2, h3 {
        color: #1E293B;
        font-weight: 700;
    }
    h2 {
        border-bottom: 1px solid #E0E0E0;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
    }
    h3 {
        margin-top: 1.5rem;
    }
    
    /* Button styling */
    .stButton button {
        border-radius: 0.5rem;
        font-weight: 500;
        transition: all 0.2s;
    }
    .stButton button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        white-space: nowrap;
        border-radius: 0.5rem 0.5rem 0 0;
        padding: 0 1rem;
        font-weight: 500;
    }
    
    /* Sidebar refinements */
    .sidebar .sidebar-content {
        background-color: #FAFAFA;
    }
    
    /* Section dividers */
    hr {
        margin: 2rem 0;
        border: none;
        height: 1px;
        background-color: #E0E0E0;
    }

    /* Footer styling */
    .footer {
        position: fixed;
        bottom: 0;
        right: 0;
        padding: 10px;
        background-color: rgba(255, 255, 255, 0.7);
        font-size: 0.8rem;
        border-top-left-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("NSE Signals")
st.sidebar.markdown("---")

# Data source selection
data_source = st.sidebar.selectbox("Data Source", ["NSE Kenya", "Yahoo Finance", "Mock Data"])
source_mapping = {"NSE Kenya": "nse", "Yahoo Finance": "yahoo", "Mock Data": "mock"}

# Time period selection
period = st.sidebar.selectbox("Time Period", ["1 Day", "1 Week", "1 Month", "3 Months", "6 Months", "1 Year"])
period_mapping = {"1 Day": "1d", "1 Week": "1w", "1 Month": "1mo", "3 Months": "3mo", "6 Months": "6mo", "1 Year": "1y"}

# Stock selection
default_stocks = ["SCOM", "EQTY", "KCB", "EABL", "BAT"]
all_nse_stocks = [
    "SCOM", "EQTY", "KCB", "EABL", "BAT", "COOP", "ABSA", "SCBK", "NCBA", "DTK",
    "JUB", "CTUM", "CARB", "TOTL", "KPLC", "KEGN", "KQ", "NBV", "NMG", "SASN"
]
selected_stocks = st.sidebar.multiselect("Select Stocks", all_nse_stocks, default=default_stocks)

# Navigation
page = st.sidebar.radio("Navigation", ["Dashboard", "Signals", "Analysis", "Trading Platforms", "Settings"])

# Function to load data
@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_data(source, period_code, tickers):
    # Show error if no stocks are selected
    if not tickers:
        st.error("Please select at least one stock to display")
        return None
    
    if source == "mock":
        # For mock data, determine number of days based on period
        days_map = {"1d": 1, "1w": 7, "1mo": 30, "3mo": 90, "6mo": 180, "1y": 365}
        days = days_map.get(period_code, 30)
        from utils.data_fetcher import generate_mock_data
        return generate_mock_data(tickers, days)
    else:
        try:
            # Try to fetch data from the specified source
            df = fetch_stock_data(tickers=tickers, period=period_code, source=source)
            
            # If the dataframe is empty, show clear error instead of using mock data
            if df is None or df.empty:
                st.error("Could not retrieve real price data from NSE. Please try again later or check your internet connection.")
                return None
            return df
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return None

# Main content
st.title("NSE Signals Dashboard")
st.caption(f"Data as of {dt.datetime.now().strftime('%Y-%m-%d %H:%M')}")

# Display content based on selected page
if page == "Dashboard":
    st.header("Market Overview")
    
    # Load the stock data
    with st.spinner("Loading market data..."):
        source_code = source_mapping.get(data_source, "nse")
        period_code = period_mapping.get(period, "1mo")
        
        # Fetch data
        df = load_data(source_code, period_code, selected_stocks)
    
    if df is not None and not df.empty:
        # Create a grid layout for key metrics
        st.subheader("Current Prices")
        
        # Create a card layout for current prices
        cols = st.columns(len(selected_stocks))
        for i, ticker in enumerate(selected_stocks):
            if ticker in df.columns:
                # Get latest price
                latest_price = df[ticker].iloc[-1]
                
                # Calculate change and percent change
                prev_price = df[ticker].iloc[-2] if len(df) > 1 else latest_price
                change = latest_price - prev_price
                pct_change = (change / prev_price) * 100 if prev_price > 0 else 0
                
                # Determine if the stock is up or down
                color = theme_colors["success"] if change >= 0 else theme_colors["danger"]
                arrow = "↑" if change >= 0 else "↓"
                
                # Display in a card format
                with cols[i]:
                    st.markdown(f"""
                    <div style="background-color: white; padding: 15px; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
                        <h3 style="margin: 0; color: #333;">{ticker}</h3>
                        <p style="font-size: 24px; font-weight: bold; margin: 10px 0;">KES {latest_price:.2f}</p>
                        <p style="color: {color}; margin: 0;">
                            {arrow} {abs(change):.2f} ({abs(pct_change):.2f}%)
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Add a price chart
        st.subheader("Price Movement")
        
        # Create tabs for different chart types
        chart_tabs = st.tabs(["Line Chart", "Candlestick", "Area Chart"])
        
        with chart_tabs[0]:
            # Line chart using Plotly
            fig = go.Figure()
            
            for ticker in selected_stocks:
                if ticker in df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=df.index,
                            y=df[ticker],
                            mode='lines',
                            name=ticker,
                            line=dict(width=2)
                        )
                    )
            
            # Update layout
            fig.update_layout(
                height=500,
                margin=dict(l=20, r=20, t=50, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                xaxis_title="Date",
                yaxis_title="Price (KES)",
                hovermode="x unified",
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with chart_tabs[1]:
            # For candlestick, we'd need OHLC data which we may not have
            # Let's create a simplified version using daily price changes
            if len(df) > 1:
                # Select a single stock to display in candlestick
                selected_ticker = st.selectbox("Select Stock for Candlestick View", selected_stocks)
                
                if selected_ticker in df.columns:
                    # Create OHLC-like data (simplified)
                    candle_data = pd.DataFrame()
                    candle_data['date'] = df.index
                    candle_data['ticker'] = selected_ticker
                    candle_data['open'] = df[selected_ticker].shift(1) 
                    candle_data['high'] = df[selected_ticker].rolling(2).max()
                    candle_data['low'] = df[selected_ticker].rolling(2).min()
                    candle_data['close'] = df[selected_ticker]
                    
                    # Clean up data
                    candle_data = candle_data.dropna()
                    
                    # Create candlestick chart
                    if not candle_data.empty:
                        fig = go.Figure(go.Candlestick(
                            x=candle_data['date'],
                            open=candle_data['open'],
                            high=candle_data['high'],
                            low=candle_data['low'],
                            close=candle_data['close'],
                            name=selected_ticker
                        ))
                        
                        fig.update_layout(
                            height=500,
                            margin=dict(l=20, r=20, t=50, b=20),
                            xaxis_title="Date",
                            yaxis_title="Price (KES)",
                            title=f"{selected_ticker} Price Movement"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Not enough data to display candlestick chart")
            else:
                st.warning("Candlestick chart requires at least 2 days of data")
        
        with chart_tabs[2]:
            # Area chart
            fig = go.Figure()
            
            for ticker in selected_stocks:
                if ticker in df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=df.index,
                            y=df[ticker],
                            mode='lines',
                            name=ticker,
                            fill='tozeroy',
                            opacity=0.6
                        )
                    )
            
            # Update layout
            fig.update_layout(
                height=500,
                margin=dict(l=20, r=20, t=50, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                xaxis_title="Date",
                yaxis_title="Price (KES)",
                hovermode="x unified",
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Add a volume chart or additional performance metrics
        st.subheader("Performance Metrics")
        
        # Prepare performance metrics
        metrics_data = {}
        for ticker in selected_stocks:
            if ticker in df.columns and len(df) > 0:
                current_price = df[ticker].iloc[-1]
                
                # Calculate metrics
                metrics_data[ticker] = {
                    "Current Price": f"KES {current_price:.2f}",
                    "1-Day Change": f"{((df[ticker].iloc[-1] / df[ticker].iloc[-2] - 1) * 100):.2f}%" if len(df) > 1 else "N/A",
                    "7-Day Change": f"{((df[ticker].iloc[-1] / df[ticker].iloc[-7] - 1) * 100):.2f}%" if len(df) > 7 else "N/A",
                    "30-Day Change": f"{((df[ticker].iloc[-1] / df[ticker].iloc[-30] - 1) * 100):.2f}%" if len(df) > 30 else "N/A",
                    "Volatility (30D)": f"{df[ticker].pct_change().rolling(30).std().iloc[-1] * 100:.2f}%" if len(df) > 30 else "N/A"
                }
        
        if metrics_data:
            # Convert to DataFrame for display
            metrics_df = pd.DataFrame(metrics_data).T
            st.dataframe(metrics_df, use_container_width=True)
        
        # Data source info
        st.caption(f"Data Source: {data_source} | Last Updated: {dt.datetime.now().strftime('%Y-%m-%d %H:%M')}")
        
    else:
        st.error("No data available for the selected stocks and time period. Please try different settings or check your internet connection.")
    
elif page == "Signals":
    st.header("Trading Signals")
    # Signals content will go here
    
elif page == "Analysis":
    st.header("Technical Analysis")
    # Analysis content will go here
    
elif page == "Trading Platforms":
    st.header("NSE Mobile & Online Trading Platforms")
    
    st.write("""
    The Nairobi Securities Exchange (NSE) offers various mobile and online trading platforms 
    through licensed stockbrokers. These platforms allow investors to buy and sell securities 
    directly from their phones or computers.
    """)
    
    # Create tabs for Online and Mobile trading
    tab1, tab2 = st.tabs(["Mobile Trading Apps", "Stockbrokers"])
    
    with tab1:
        st.subheader("Available Mobile Trading Apps")
        
        # Create a DataFrame to display broker apps info
        broker_apps = {
            "Broker": [
                "Dyer and Blair", "Kingdom Securities", "AIB-AXYS Africa Ltd", 
                "Sterling", "Faida Investment Bank", "NCBA", "Genghis", "EFG Hermes"
            ],
            "Android": [
                "✓", "✓", "✓", "✓", "✓", "✓", "✓", "✓"
            ],
            "iOS": [
                "✓", "❌", "✓", "✓", "✓", "✓", "✓", "✓"
            ]
        }
        
        broker_df = pd.DataFrame(broker_apps)
        st.dataframe(broker_df, use_container_width=True)
        
        st.info("""
        **How to start trading:**
        1. Select a licensed stockbroker
        2. Download their mobile app from Google Play or App Store
        3. Create an account and complete the KYC process
        4. Fund your account
        5. Start trading NSE listed securities
        """)
    
    with tab2:
        st.subheader("Licensed NSE Stockbrokers")
        
        brokers = [
            "Dyer and Blair", "Suntra Investment Bank", "Old Mutual", "SBG Securities",
            "Kingdom Securities", "AIB-AXYS Africa Ltd", "ABC Capital", "Sterling Capital",
            "Faida Investment Bank", "NCBA", "Genghis Capital", "Standard Investment Bank",
            "Kestrel Capital", "African Alliance", "KCB Capital"
        ]
        
        # Create a multicolumn layout for brokers
        cols = st.columns(3)
        for i, broker in enumerate(brokers):
            cols[i % 3].write(f"- {broker}")
        
        st.markdown("---")
        st.markdown("For more information, visit the [NSE Mobile and Online Trading page](https://www.nse.co.ke/mobile-and-online-trading/)")

elif page == "Settings":
    st.header("Settings")
    # Settings content will go here

# Footer with GitHub link
st.markdown(
    """
    <div class="footer">
        <a href="https://github.com/Johnngila/nse_signals_dashboard" target="_blank">GitHub Repository</a>
    </div>
    """,
    unsafe_allow_html=True
)

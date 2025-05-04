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
page = st.sidebar.radio("Navigation", ["Dashboard", "Signals", "Analysis", "Settings"])

# Function to load data
@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_data(source, period_code, tickers):
    if source == "mock":
        # For mock data, determine number of days based on period
        days_map = {"1d": 1, "1w": 7, "1mo": 30, "3mo": 90, "6mo": 180, "1y": 365}
        days = days_map.get(period_code, 30)
        from utils.data_fetcher import generate_mock_data
        return generate_mock_data(tickers, days)
    else:
        return fetch_stock_data(tickers=tickers, period=period_code, source=source)

# Main content
st.title("NSE Signals Dashboard")
st.caption(f"Data as of {dt.datetime.now().strftime('%Y-%m-%d %H:%M')}")

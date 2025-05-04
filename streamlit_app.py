import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

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

# Main function
def main():
    # Load data
    stock_data = load_data()
    sector_data = load_sector_data()
    
    # Sidebar
    st.sidebar.header("Filters")
    
    if not stock_data.empty:
        # Get list of stocks
        stocks = sorted(stock_data['Symbol'].unique()) if 'Symbol' in stock_data.columns else []
        
        # Stock selector
        selected_stock = st.sidebar.selectbox("Select Stock", stocks)
        
        # Display basic info
        st.header(f"Stock Data: {selected_stock}")
        
        # Filter data for selected stock
        if 'Symbol' in stock_data.columns:
            stock_df = stock_data[stock_data['Symbol'] == selected_stock]
            
            if not stock_df.empty:
                # Display dataframe
                st.subheader("Latest Data")
                st.dataframe(stock_df.head())
                
                # Plot if we have price and date columns
                if 'Price' in stock_df.columns and 'Date' in stock_df.columns:
                    st.subheader("Price History")
                    fig = px.line(stock_df, x='Date', y='Price', title=f"{selected_stock} Price History")
                    st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No stock data available. Please check your data file.")

    # Sector analysis section if sector data is available
    if not sector_data.empty:
        st.header("Sector Analysis")
        st.dataframe(sector_data.head())

if __name__ == "__main__":
    main()
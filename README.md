# NSE Signals Dashboard

A Streamlit-based dashboard for tracking Nairobi Securities Exchange (NSE) stocks and generating trading signals using various technical indicators.

## Features

- **Market Overview**: View current prices and trends for top NSE stocks
- **Trading Signals**: Generate signals based on multiple technical indicators
  - Moving Average Crossover
  - RSI (Relative Strength Index)
  - MACD (Moving Average Convergence Divergence)
  - Bollinger Bands
- **Technical Analysis**: Tools for analyzing stock performance (coming soon)
- **Customizable Settings**: Configure data sources and refresh rates

## Requirements

- Python 3.8+
- Required packages (install via `pip install -r requirements.txt`):
  - streamlit
  - pandas
  - numpy
  - matplotlib
  - yfinance (for data fetching from Yahoo Finance)

## Installation and Setup

1. Clone this repository:
   ```
   git clone <repository-url>
   cd nse-signals
   ```

2. Install required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

4. Open your browser and navigate to the URL displayed in the terminal (typically http://localhost:8501)

## Project Structure

```
├── app.py              # Main Streamlit application
├── requirements.txt    # Project dependencies
├── data/               # Directory for data storage
└── utils/
    ├── data_fetcher.py # Data retrieval functions
    └── signal_engine.py # Signal generation algorithms
```

## Usage

1. Navigate through the dashboard using the sidebar menu
2. View real-time (or mock) stock data from the NSE
3. Generate trading signals based on different technical indicators
4. Adjust settings according to your preferences

## Deployment

The app can be deployed to Streamlit Community Cloud for free hosting:
1. Push the code to a GitHub repository
2. Connect your repository to [Streamlit Community Cloud](https://streamlit.io/cloud)
3. Deploy directly from your GitHub repository

## Future Enhancements

- Historical data analysis
- More advanced technical indicators
- Backtesting functionality
- User authentication
- Portfolio tracking

## License

MIT

## Contact

Created by JNG Trading

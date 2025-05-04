import pandas as pd
import numpy as np
from utils.data_fetcher import fetch_stock_data
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from scipy import stats

def generate_signals(signal_type="Moving Average Crossover", tickers=None, **kwargs):
    """
    Generate trading signals based on technical indicators
    
    Parameters:
    -----------
    signal_type : str
        Type of signal to generate (Moving Average Crossover, RSI, MACD, Bollinger Bands, Custom)
    tickers : list or None
        List of stock tickers to analyze. If None, default NSE stocks are used.
    **kwargs : dict
        Additional parameters for signal generation
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing generated signals
    """
    try:
        # Fetch stock data
        data = fetch_stock_data(tickers=tickers)
        
        if data is None or data.empty:
            print("Error: No price data available for the selected tickers")
            return pd.DataFrame()
        
        # Check if we have enough data points for analysis
        min_required_points = 50  # Minimum data points needed for reliable signals
        if len(data) < min_required_points:
            print(f"Warning: Limited price history ({len(data)} points) may affect signal quality")
        
        # Generate signals based on the selected type
        try:
            if signal_type == "Moving Average Crossover":
                return generate_ma_crossover_signals(data, **kwargs)
            elif signal_type == "RSI":
                return generate_rsi_signals(data, **kwargs)
            elif signal_type == "MACD":
                return generate_macd_signals(data, **kwargs)
            elif signal_type == "Bollinger Bands":
                return generate_bollinger_signals(data, **kwargs)
            elif signal_type == "Custom":
                return generate_custom_signals(data, **kwargs)
            # New advanced models
            elif signal_type == "Mean Reversion":
                return generate_mean_reversion_signals(data, **kwargs)
            elif signal_type == "GARCH Volatility":
                return generate_garch_signals(data, **kwargs)
            elif signal_type == "Relative Value":
                return generate_relative_value_signals(data, **kwargs)
            elif signal_type == "Linear Regression":
                return generate_linear_regression_signals(data, **kwargs)
            elif signal_type == "Statistical Arbitrage":
                return generate_statistical_arbitrage_signals(data, **kwargs)
            elif signal_type == "Volume Price Trend":
                return generate_vpt_signals(data, **kwargs)
            else:
                print(f"Error: Unsupported signal type: {signal_type}")
                return pd.DataFrame()
        except Exception as e:
            print(f"Error generating {signal_type} signals: {e}")
            return pd.DataFrame()
    except Exception as e:
        print(f"Error in generate_signals: {e}")
        return pd.DataFrame()

# Existing signal generation functions...

def generate_mean_reversion_signals(data, lookback_period=20, std_dev_threshold=2.0, exit_threshold=0.5, max_holding=10):
    """
    Generate signals based on statistical mean reversion with z-score
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Stock price data
    lookback_period : int
        Period for calculating moving average and standard deviation
    std_dev_threshold : float
        Z-score threshold for entry signals
    exit_threshold : float
        Z-score threshold for exit signals
    max_holding : int
        Maximum holding period
    """
    signals = pd.DataFrame(index=data.index)
    
    # Process each stock
    for ticker in data.columns:
        # Calculate rolling mean and standard deviation
        rolling_mean = data[ticker].rolling(window=lookback_period).mean()
        rolling_std = data[ticker].rolling(window=lookback_period).std()
        
        # Calculate z-score
        z_score = (data[ticker] - rolling_mean) / rolling_std
        
        # Initialize signal and position columns
        signal_col = f"{ticker}_Signal"
        position_col = f"{ticker}_Position"
        signals[signal_col] = 0
        signals[position_col] = 0
        
        # Set initial positions based on z-score
        position = 0
        entry_date = None
        
        # Apply mean reversion strategy
        for i in range(lookback_period, len(signals)):
            if position == 0:  # No current position
                if z_score.iloc[i] <= -std_dev_threshold:
                    # Price significantly below mean (buy signal)
                    col_idx = signals.columns.get_loc(signal_col)
                    signals.iat[i, col_idx] = 1
                    position = 1
                    entry_date = signals.index[i]
                elif z_score.iloc[i] >= std_dev_threshold:
                    # Price significantly above mean (sell signal)
                    col_idx = signals.columns.get_loc(signal_col)
                    signals.iat[i, col_idx] = -1
                    position = -1
                    entry_date = signals.index[i]
            else:  # Already have a position
                days_held = (signals.index[i] - entry_date).days if entry_date else 0
                
                # Exit rules
                exit_signal = False
                
                # 1. Mean reversion exit - price moved back toward mean
                if (position == 1 and z_score.iloc[i] >= exit_threshold) or \
                   (position == -1 and z_score.iloc[i] <= -exit_threshold):
                    exit_signal = True
                    
                # 2. Time-based exit - held for too long
                if days_held >= max_holding:
                    exit_signal = True
                    
                # 3. Stop loss - moved further away (optional)
                if (position == 1 and z_score.iloc[i] <= -std_dev_threshold*1.5) or \
                   (position == -1 and z_score.iloc[i] >= std_dev_threshold*1.5):
                    exit_signal = True
                
                if exit_signal:
                    # Reset position
                    position = 0
                    entry_date = None
            
            # Store current position
            col_idx = signals.columns.get_loc(position_col)
            signals.iat[i, col_idx] = position
        
        # Add price and z-score columns for reference
        signals[f"{ticker}_Price"] = data[ticker]
        signals[f"{ticker}_Z_Score"] = z_score
        signals[f"{ticker}_Mean"] = rolling_mean
        signals[f"{ticker}_Upper"] = rolling_mean + (rolling_std * std_dev_threshold)
        signals[f"{ticker}_Lower"] = rolling_mean - (rolling_std * std_dev_threshold)
    
    # Drop NaN values
    signals = signals.dropna()
    
    return signals

def generate_linear_regression_signals(data, window=20, forecast_period=5, confidence=0.95, entry_threshold=0.02):
    """
    Generate signals based on linear regression prediction with confidence intervals
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Stock price data
    window : int
        Lookback period for regression
    forecast_period : int
        Number of periods to forecast ahead
    confidence : float
        Confidence level for prediction intervals (0-1)
    entry_threshold : float
        Minimum predicted return to generate signal
    """
    signals = pd.DataFrame(index=data.index)
    
    # Process each stock
    for ticker in data.columns:
        # Initialize signal and prediction columns
        signal_col = f"{ticker}_Signal"
        prediction_col = f"{ticker}_Prediction"
        lower_col = f"{ticker}_Lower_CI"
        upper_col = f"{ticker}_Upper_CI"
        
        signals[signal_col] = 0
        signals[prediction_col] = np.nan
        signals[lower_col] = np.nan
        signals[upper_col] = np.nan
        
        # Generate signals based on linear regression forecasts
        for i in range(window, len(data)):
            # Get historical data for this window
            y = data[ticker].iloc[i-window:i].values
            X = np.arange(window).reshape(-1, 1)
            
            try:
                # Fit linear regression
                model = LinearRegression()
                model.fit(X, y)
                
                # Calculate confidence intervals
                y_pred = model.predict(X)
                residuals = y - y_pred
                mse = np.mean(residuals**2)
                std_err = np.sqrt(mse)
                
                # Forecast next period
                future_X = np.array([[window]])
                forecast = model.predict(future_X)[0]
                
                # Calculate confidence intervals
                t_value = stats.t.ppf((1 + confidence) / 2, window - 2)
                pred_std_err = std_err * np.sqrt(1 + 1/window + (window - np.mean(X))**2 / np.sum((X - np.mean(X))**2))
                lower_ci = forecast - t_value * pred_std_err
                upper_ci = forecast + t_value * pred_std_err
                
                # Store prediction and confidence intervals
                signals.at[data.index[i], prediction_col] = forecast
                signals.at[data.index[i], lower_col] = lower_ci
                signals.at[data.index[i], upper_col] = upper_ci
                
                # Generate signals based on predicted return
                current_price = data[ticker].iloc[i]
                predicted_return = (forecast - current_price) / current_price
                
                if predicted_return > entry_threshold and (upper_ci - lower_ci) / current_price < entry_threshold * 2:
                    # Strong upward prediction with reasonable confidence
                    signals.at[data.index[i], signal_col] = 1
                elif predicted_return < -entry_threshold and (upper_ci - lower_ci) / current_price < entry_threshold * 2:
                    # Strong downward prediction with reasonable confidence
                    signals.at[data.index[i], signal_col] = -1
            except Exception as e:
                # Skip if regression fails, but log the error
                print(f"Error in linear regression for {ticker} at index {i}: {e}")
                continue
        
        # Add price for reference
        signals[f"{ticker}_Price"] = data[ticker]
    
    # Drop NaN values
    signals = signals.dropna()
    
    return signals

def generate_relative_value_signals(data, sector_mapping=None, lookback=30, threshold=1.5):
    """
    Generate signals based on relative value compared to sector or market peers
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Stock price data
    sector_mapping : dict or None
        Mapping of tickers to sectors
    lookback : int
        Lookback period for relative strength
    threshold : float
        Relative strength threshold for generating signals
    """
    signals = pd.DataFrame(index=data.index)
    
    # If no sector mapping provided, treat all stocks as one group
    if sector_mapping is None:
        sector_mapping = {ticker: 'Market' for ticker in data.columns}
    
    # Group stocks by sector
    sectors = {}
    for ticker in data.columns:
        sector = sector_mapping.get(ticker, 'Market')
        if sector not in sectors:
            sectors[sector] = []
        sectors[sector].append(ticker)
    
    # Process each stock
    for ticker in data.columns:
        # Get sector peers
        sector = sector_mapping.get(ticker, 'Market')
        peers = [peer for peer in sectors[sector] if peer != ticker]
        
        if not peers:
            continue  # Skip if no peers
        
        # Initialize signal and strength columns
        signal_col = f"{ticker}_Signal"
        rs_col = f"{ticker}_RS"  # Relative strength
        signals[signal_col] = 0
        signals[rs_col] = np.nan
        
        # Calculate returns
        returns = data.pct_change(periods=1)
        
        # Calculate peer average return series
        peer_returns = returns[peers].mean(axis=1)
        
        # Calculate relative strength (ratio of stock return to peer average return)
        # Using cumulative returns over lookback period
        for i in range(lookback, len(data)):
            stock_cum_return = (1 + returns[ticker].iloc[i-lookback:i]).prod() - 1
            peer_cum_return = (1 + peer_returns.iloc[i-lookback:i]).prod() - 1
            
            if peer_cum_return != 0:
                relative_strength = stock_cum_return / peer_cum_return if peer_cum_return != 0 else np.inf
            else:
                relative_strength = 1 if stock_cum_return == 0 else np.inf * np.sign(stock_cum_return)
            
            # Store relative strength
            signals.at[data.index[i], rs_col] = relative_strength
            
            # Generate signals based on relative strength threshold
            if relative_strength > threshold:
                # Outperforming peers by threshold (buy signal)
                signals.at[data.index[i], signal_col] = 1
            elif relative_strength < 1/threshold:
                # Underperforming peers by threshold (sell signal)
                signals.at[data.index[i], signal_col] = -1
        
        # Add price for reference
        signals[f"{ticker}_Price"] = data[ticker]
    
    # Drop NaN values
    signals = signals.dropna()
    
    return signals

def generate_statistical_arbitrage_signals(data, lookback=20, entry_z=2.0, exit_z=0.5, max_pairs=3):
    """
    Generate signals based on pairs trading using statistical arbitrage
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Stock price data
    lookback : int
        Lookback period for z-score calculation
    entry_z : float
        Z-score threshold for entry
    exit_z : float
        Z-score threshold for exit
    max_pairs : int
        Maximum number of pairs to consider per stock
    """
    signals = pd.DataFrame(index=data.index)
    
    # Need at least 2 stocks
    if len(data.columns) < 2:
        return signals
    
    # Find correlated pairs
    returns = data.pct_change().dropna()
    corr_matrix = returns.corr()
    
    # Process each stock
    for ticker in data.columns:
        # Find most correlated stocks
        corrs = corr_matrix[ticker].drop(ticker).sort_values(ascending=False)
        pairs = corrs.index[:max_pairs].tolist()
        
        # Initialize signal column
        signal_col = f"{ticker}_Signal"
        signals[signal_col] = 0
        
        # Analyze each pair
        for pair in pairs:
            # Get ratio of prices
            ratio = data[ticker] / data[pair]
            
            # Calculate z-score of ratio
            rolling_mean = ratio.rolling(window=lookback).mean()
            rolling_std = ratio.rolling(window=lookback).std()
            z_score = (ratio - rolling_mean) / rolling_std
            
            # Add z-score column for this pair
            z_col = f"{ticker}_{pair}_Z"
            signals[z_col] = z_score
            
            # Generate signals based on z-score
            for i in range(lookback, len(data)):
                if z_score.iloc[i] >= entry_z:
                    # Ratio is high - sell numerator (ticker), buy denominator (pair)
                    signals.at[data.index[i], signal_col] = -1
                elif z_score.iloc[i] <= -entry_z:
                    # Ratio is low - buy numerator (ticker), sell denominator (pair)
                    signals.at[data.index[i], signal_col] = 1
                elif abs(z_score.iloc[i]) <= exit_z:
                    # Ratio reverted to mean - exit position
                    signals.at[data.index[i], signal_col] = 0
        
        # Add price for reference
        signals[f"{ticker}_Price"] = data[ticker]
    
    # Drop NaN values
    signals = signals.dropna()
    
    return signals

def generate_vpt_signals(data, volume_data=None, window=14, threshold=1.0):
    """
    Generate signals based on Volume Price Trend (VPT) indicator
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Stock price data
    volume_data : pandas.DataFrame or None
        Volume data (if None, will try to derive from price data)
    window : int
        Lookback period for VPT calculation
    threshold : float
        Threshold for generating signals
    """
    signals = pd.DataFrame(index=data.index)
    
    # If no volume data provided, try to derive proxy (this won't be accurate)
    if volume_data is None:
        print("Warning: No volume data provided. Using price volatility as proxy.")
        # Use price volatility as a crude proxy for volume
        temp_returns = data.pct_change()
        volume_proxy = temp_returns.rolling(5).std() * 1000
        volume_data = volume_proxy
    
    # Process each stock
    for ticker in data.columns:
        if ticker not in volume_data.columns:
            continue  # Skip if no volume data
            
        # Calculate price change percentage
        price_change = data[ticker].pct_change()
        
        # Calculate VPT
        vpt = (volume_data[ticker] * price_change).cumsum()
        
        # Calculate VPT EMA
        vpt_ema = vpt.ewm(span=window).mean()
        
        # Initialize signal and indicator columns
        signal_col = f"{ticker}_Signal"
        signals[signal_col] = 0
        signals[f"{ticker}_VPT"] = vpt
        signals[f"{ticker}_VPT_EMA"] = vpt_ema
        
        # Calculate slope of VPT
        vpt_slope = vpt.diff(5) / vpt.shift(5)
        signals[f"{ticker}_VPT_Slope"] = vpt_slope
        
        # Generate signals based on VPT crossovers and slope
        for i in range(window + 5, len(data)):
            # VPT crosses above EMA with positive slope
            if (vpt.iloc[i] > vpt_ema.iloc[i]) and (vpt.iloc[i-1] <= vpt_ema.iloc[i-1]) and (vpt_slope.iloc[i] > threshold / 100):
                signals.at[data.index[i], signal_col] = 1
                
            # VPT crosses below EMA with negative slope
            elif (vpt.iloc[i] < vpt_ema.iloc[i]) and (vpt.iloc[i-1] >= vpt_ema.iloc[i-1]) and (vpt_slope.iloc[i] < -threshold / 100):
                signals.at[data.index[i], signal_col] = -1
        
        # Add price for reference
        signals[f"{ticker}_Price"] = data[ticker]
    
    # Drop NaN values
    signals = signals.dropna()
    
    return signals

def generate_garch_signals(data, window=100, forecast_periods=5, vol_threshold=0.4):
    """
    Generate signals based on GARCH volatility forecasting model
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Stock price data
    window : int
        Lookback window for GARCH model
    forecast_periods : int
        Number of periods to forecast volatility
    vol_threshold : float
        Volatility threshold (annualized) for generating signals
    """
    signals = pd.DataFrame(index=data.index)
    
    try:
        import arch
    except ImportError:
        print("Error: arch package not installed. Install with: pip install arch")
        return signals
    
    # Process each stock
    for ticker in data.columns:
        # Initialize signal and volatility columns
        signal_col = f"{ticker}_Signal"
        vol_col = f"{ticker}_GARCH_Vol"
        signals[signal_col] = 0
        signals[vol_col] = np.nan
        
        # Calculate returns
        returns = data[ticker].pct_change().dropna() * 100  # in percent
        
        # Generate signals using rolling window approach
        for i in range(window, len(returns)):
            try:
                # Get window of returns
                window_returns = returns.iloc[i-window:i]
                
                # Fit GARCH(1,1) model
                model = arch.arch_model(window_returns, vol='Garch', p=1, q=1)
                result = model.fit(disp='off', show_warning=False)
                
                # Forecast volatility
                forecast = result.forecast(horizon=forecast_periods)
                forecasted_vol = np.sqrt(forecast.variance.iloc[-1].mean())
                annualized_vol = forecasted_vol * np.sqrt(252) / 100  # annualize and convert back to decimals
                
                # Store forecasted volatility
                signals.at[data.index[i], vol_col] = annualized_vol
                
                # Generate signals based on volatility regimes
                if annualized_vol < vol_threshold * 0.5:
                    # Low volatility regime - typically good for long positions
                    signals.at[data.index[i], signal_col] = 1
                elif annualized_vol > vol_threshold:
                    # High volatility regime - typically good for short positions or staying out
                    signals.at[data.index[i], signal_col] = -1
            except Exception as e:
                # Skip if GARCH model fails to converge, but log the error
                print(f"Error in GARCH model for {ticker} at index {i}: {e}")
                continue
        
        # Add price for reference
        signals[f"{ticker}_Price"] = data[ticker]
    
    # Drop NaN values
    signals = signals.dropna()
    
    return signals

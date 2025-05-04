import pandas as pd
import numpy as np
from utils.data_fetcher import fetch_stock_data

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
    # Fetch stock data
    data = fetch_stock_data(tickers=tickers)
    
    if data is None or data.empty:
        return pd.DataFrame()
    
    # Generate signals based on the selected type
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
    else:
        raise ValueError(f"Unsupported signal type: {signal_type}")

def generate_ma_crossover_signals(data, short_window=20, long_window=50, confirmation_window=3, trend_filter=True):
    """
    Generate signals based on Moving Average Crossover with enhanced logic
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Stock price data
    short_window : int
        Short moving average window
    long_window : int
        Long moving average window
    confirmation_window : int
        Number of periods to confirm a signal
    trend_filter : bool
        Whether to apply trend filtering
    """
    signals = pd.DataFrame(index=data.index)
    
    # Process each stock
    for ticker in data.columns:
        # Calculate short and long moving averages
        short_ma = data[ticker].rolling(window=short_window, min_periods=1).mean()
        long_ma = data[ticker].rolling(window=long_window, min_periods=1).mean()
        
        # Calculate trend direction (using 50-day MA slope)
        if trend_filter:
            trend_ma = data[ticker].rolling(window=50, min_periods=1).mean()
            trend_direction = trend_ma.diff(periods=10).apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        else:
            trend_direction = pd.Series(1, index=data.index)  # No filtering
        
        # Initialize signal column and strength column
        signal_col = f"{ticker}_Signal"
        strength_col = f"{ticker}_Signal_Strength"
        signals[signal_col] = 0
        signals[strength_col] = 0
        
        # Calculate crossover condition
        crossover_up = (short_ma > long_ma) & (short_ma.shift(1) <= long_ma.shift(1))
        crossover_down = (short_ma < long_ma) & (short_ma.shift(1) >= long_ma.shift(1))
        
        # Calculate signal strength (percentage difference between MAs)
        signals[strength_col] = (short_ma - long_ma) / long_ma * 100
        
        # Apply custom signal generation logic
        for i in range(len(signals)):
            if i < confirmation_window:
                continue
                
            # Calculate confirmed signals with trend filter
            if crossover_up.iloc[i]:
                # Only generate buy signals in uptrend or if trend filter is disabled
                if trend_direction.iloc[i] >= 0 or not trend_filter:
                    signals.iloc[i, signals.columns.get_loc(signal_col)] = 1
            elif crossover_down.iloc[i]:
                # Only generate sell signals in downtrend or if trend filter is disabled
                if trend_direction.iloc[i] <= 0 or not trend_filter:
                    signals.iloc[i, signals.columns.get_loc(signal_col)] = -1
                    
            # Apply confirmation - signal must persist for confirmation_window periods
            if i >= confirmation_window:
                ma_diff_positive = short_ma.iloc[i-confirmation_window:i+1] > long_ma.iloc[i-confirmation_window:i+1]
                ma_diff_negative = short_ma.iloc[i-confirmation_window:i+1] < long_ma.iloc[i-confirmation_window:i+1]
                
                if not all(ma_diff_positive) and signals.iloc[i, signals.columns.get_loc(signal_col)] == 1:
                    signals.iloc[i, signals.columns.get_loc(signal_col)] = 0
                elif not all(ma_diff_negative) and signals.iloc[i, signals.columns.get_loc(signal_col)] == -1:
                    signals.iloc[i, signals.columns.get_loc(signal_col)] = 0
        
        # Add price and MA columns for reference
        signals[f"{ticker}_Price"] = data[ticker]
        signals[f"{ticker}_Short_MA"] = short_ma
        signals[f"{ticker}_Long_MA"] = long_ma
        signals[f"{ticker}_Trend"] = trend_direction if trend_filter else None
    
    # Drop NaN values
    signals = signals.dropna()
    
    return signals

def generate_rsi_signals(data, window=14, overbought=70, oversold=30, trend_window=50, confirmation_days=2):
    """
    Generate signals based on Relative Strength Index (RSI) with enhanced logic
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Stock price data
    window : int
        RSI calculation window
    overbought : int
        RSI level considered overbought
    oversold : int
        RSI level considered oversold
    trend_window : int
        Window for trend determination
    confirmation_days : int
        Number of days to confirm a signal
    """
    signals = pd.DataFrame(index=data.index)
    
    # Process each stock
    for ticker in data.columns:
        # Calculate RSI
        delta = data[ticker].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=window, min_periods=1).mean()
        avg_loss = loss.rolling(window=window, min_periods=1).mean()
        
        rs = avg_gain / avg_loss.replace(0, np.nan)  # Avoid division by zero
        rsi = 100 - (100 / (1 + rs))
        
        # Calculate trend direction using simple moving average
        trend_ma = data[ticker].rolling(window=trend_window, min_periods=1).mean()
        trend_direction = trend_ma.diff(periods=10).apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        
        # Initialize signal column and strength column
        signal_col = f"{ticker}_Signal"
        strength_col = f"{ticker}_Signal_Strength"
        signals[signal_col] = 0
        signals[strength_col] = 0
        
        # Calculate signal strength based on distance from thresholds
        signals[strength_col] = np.where(
            rsi < oversold, 
            (oversold - rsi) / oversold * 100,  # Buy strength
            np.where(
                rsi > overbought,
                (rsi - overbought) / (100 - overbought) * 100,  # Sell strength
                0  # No signal
            )
        )
        
        # RSI conditions with confirmation
        for i in range(confirmation_days, len(rsi)):
            # Buy signal: RSI has been below oversold for confirmation_days and is now rising
            if all(rsi.iloc[i-confirmation_days:i] < oversold) and rsi.iloc[i] > rsi.iloc[i-1]:
                # Only take buy signals if trend is up or neutral
                if trend_direction.iloc[i] >= 0:
                    signals.iloc[i, signals.columns.get_loc(signal_col)] = 1
                    
            # Sell signal: RSI has been above overbought for confirmation_days and is now falling
            elif all(rsi.iloc[i-confirmation_days:i] > overbought) and rsi.iloc[i] < rsi.iloc[i-1]:
                # Only take sell signals if trend is down or neutral
                if trend_direction.iloc[i] <= 0:
                    signals.iloc[i, signals.columns.get_loc(signal_col)] = -1
        
        # Add price, RSI, and trend columns for reference
        signals[f"{ticker}_Price"] = data[ticker]
        signals[f"{ticker}_RSI"] = rsi
        signals[f"{ticker}_Trend"] = trend_direction
    
    # Drop NaN values
    signals = signals.dropna()
    
    return signals

def generate_macd_signals(data, fast_period=12, slow_period=26, signal_period=9, zero_line_filter=True):
    """
    Generate signals based on Moving Average Convergence Divergence (MACD) with enhanced logic
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Stock price data
    fast_period : int
        Fast EMA period
    slow_period : int
        Slow EMA period
    signal_period : int
        Signal line period
    zero_line_filter : bool
        Whether to use MACD zero line filter
    """
    signals = pd.DataFrame(index=data.index)
    
    # Process each stock
    for ticker in data.columns:
        # Calculate MACD
        fast_ema = data[ticker].ewm(span=fast_period, adjust=False).mean()
        slow_ema = data[ticker].ewm(span=slow_period, adjust=False).mean()
        macd = fast_ema - slow_ema
        signal_line = macd.ewm(span=signal_period, adjust=False).mean()
        macd_histogram = macd - signal_line
        
        # Initialize signal and strength columns
        signal_col = f"{ticker}_Signal"
        strength_col = f"{ticker}_Signal_Strength"
        signals[signal_col] = 0
        signals[strength_col] = 0
        
        # Calculate crossover conditions
        crossover_up = (macd > signal_line) & (macd.shift(1) <= signal_line.shift(1))
        crossover_down = (macd < signal_line) & (macd.shift(1) >= signal_line.shift(1))
        
        # Calculate signal strength (percent of MACD to price)
        signals[strength_col] = (macd / data[ticker]) * 100
        
        # Apply signal generation with zero line filter if enabled
        for i in range(1, len(signals)):
            if crossover_up.iloc[i]:
                # If zero line filter is enabled, only generate buy when MACD is positive or turning positive
                if not zero_line_filter or macd.iloc[i] > 0 or (macd.iloc[i] > macd.iloc[i-1] and macd.iloc[i-1] < 0):
                    signals.iloc[i, signals.columns.get_loc(signal_col)] = 1
            elif crossover_down.iloc[i]:
                # If zero line filter is enabled, only generate sell when MACD is negative or turning negative
                if not zero_line_filter or macd.iloc[i] < 0 or (macd.iloc[i] < macd.iloc[i-1] and macd.iloc[i-1] > 0):
                    signals.iloc[i, signals.columns.get_loc(signal_col)] = -1
        
        # Add price, MACD, and signal line columns for reference
        signals[f"{ticker}_Price"] = data[ticker]
        signals[f"{ticker}_MACD"] = macd
        signals[f"{ticker}_Signal_Line"] = signal_line
        signals[f"{ticker}_Histogram"] = macd_histogram
    
    # Drop NaN values
    signals = signals.dropna()
    
    return signals

def generate_bollinger_signals(data, window=20, num_std=2, mean_reversion=False, trend_window=50):
    """
    Generate signals based on Bollinger Bands with enhanced logic
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Stock price data
    window : int
        Bollinger Bands calculation window
    num_std : float
        Number of standard deviations
    mean_reversion : bool
        Whether to use mean reversion strategy (otherwise, breakout strategy)
    trend_window : int
        Window for trend determination
    """
    signals = pd.DataFrame(index=data.index)
    
    # Process each stock
    for ticker in data.columns:
        # Calculate Bollinger Bands
        rolling_mean = data[ticker].rolling(window=window).mean()
        rolling_std = data[ticker].rolling(window=window).std()
        
        upper_band = rolling_mean + (rolling_std * num_std)
        lower_band = rolling_mean - (rolling_std * num_std)
        
        # Calculate trend for filtering signals
        trend_ma = data[ticker].rolling(window=trend_window, min_periods=1).mean()
        trend_direction = trend_ma.diff(periods=10).apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        
        # Calculate bandwidth and %B indicators
        bandwidth = (upper_band - lower_band) / rolling_mean * 100
        percent_b = (data[ticker] - lower_band) / (upper_band - lower_band)
        
        # Initialize signal column and strength column
        signal_col = f"{ticker}_Signal"
        strength_col = f"{ticker}_Signal_Strength"
        signals[signal_col] = 0
        signals[strength_col] = 0
        
        # Calculate signal strength based on position within bands
        signals[strength_col] = (percent_b - 0.5) * 200  # Scale to -100 to 100 range
        
        if mean_reversion:
            # Mean reversion strategy: Buy at lower band, sell at upper band
            # But only in sideways markets or in direction of the trend
            for i in range(1, len(signals)):
                # Buy signal when price touches or crosses below lower band
                if data[ticker].iloc[i] <= lower_band.iloc[i] and data[ticker].iloc[i-1] > lower_band.iloc[i-1]:
                    # In a downtrend, wait for confirmation of bounce with a rising day
                    if trend_direction.iloc[i] < 0:
                        if data[ticker].iloc[i] > data[ticker].iloc[i-1]:
                            signals.iloc[i, signals.columns.get_loc(signal_col)] = 1
                    else:
                        # In sideways or uptrend, take signal immediately
                        signals.iloc[i, signals.columns.get_loc(signal_col)] = 1
                        
                # Sell signal when price touches or crosses above upper band
                elif data[ticker].iloc[i] >= upper_band.iloc[i] and data[ticker].iloc[i-1] < upper_band.iloc[i-1]:
                    # In an uptrend, wait for confirmation of drop with a falling day
                    if trend_direction.iloc[i] > 0:
                        if data[ticker].iloc[i] < data[ticker].iloc[i-1]:
                            signals.iloc[i, signals.columns.get_loc(signal_col)] = -1
                    else:
                        # In sideways or downtrend, take signal immediately
                        signals.iloc[i, signals.columns.get_loc(signal_col)] = -1
        else:
            # Breakout strategy: Buy on upper band breakout, sell on lower band breakout
            for i in range(1, len(signals)):
                # Buy signal when price breaks above upper band in uptrend
                if data[ticker].iloc[i] > upper_band.iloc[i] and data[ticker].iloc[i-1] <= upper_band.iloc[i-1]:
                    if trend_direction.iloc[i] >= 0:  # Only in uptrend or sideways
                        signals.iloc[i, signals.columns.get_loc(signal_col)] = 1
                
                # Sell signal when price breaks below lower band in downtrend
                elif data[ticker].iloc[i] < lower_band.iloc[i] and data[ticker].iloc[i-1] >= lower_band.iloc[i-1]:
                    if trend_direction.iloc[i] <= 0:  # Only in downtrend or sideways
                        signals.iloc[i, signals.columns.get_loc(signal_col)] = -1
        
        # Add price and Bollinger Bands columns for reference
        signals[f"{ticker}_Price"] = data[ticker]
        signals[f"{ticker}_Upper_Band"] = upper_band
        signals[f"{ticker}_Lower_Band"] = lower_band
        signals[f"{ticker}_Middle_Band"] = rolling_mean
        signals[f"{ticker}_Bandwidth"] = bandwidth
        signals[f"{ticker}_Percent_B"] = percent_b
        signals[f"{ticker}_Trend"] = trend_direction
    
    # Drop NaN values
    signals = signals.dropna()
    
    return signals

def generate_custom_signals(data, ma_period=50, rsi_period=14, macd_fast=12, macd_slow=26, macd_signal=9, min_consensus=2):
    """
    Generate signals by combining multiple indicators (custom strategy)
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Stock price data
    ma_period : int
        Moving average period for trend determination
    rsi_period : int
        RSI calculation period
    macd_fast : int
        MACD fast period
    macd_slow : int
        MACD slow period
    macd_signal : int
        MACD signal line period
    min_consensus : int
        Minimum number of indicators that must agree for a signal
    """
    signals = pd.DataFrame(index=data.index)
    
    # Process each stock
    for ticker in data.columns:
        # Initialize signal columns
        signal_col = f"{ticker}_Signal"
        consensus_col = f"{ticker}_Consensus"
        signals[signal_col] = 0
        signals[consensus_col] = 0
        
        # 1. Trend determination using moving average
        ma = data[ticker].rolling(window=ma_period, min_periods=1).mean()
        trend = np.where(data[ticker] > ma, 1, -1)
        
        # 2. RSI calculation
        delta = data[ticker].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=rsi_period, min_periods=1).mean()
        avg_loss = loss.rolling(window=rsi_period, min_periods=1).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        
        # 3. MACD calculation
        fast_ema = data[ticker].ewm(span=macd_fast, adjust=False).mean()
        slow_ema = data[ticker].ewm(span=macd_slow, adjust=False).mean()
        macd = fast_ema - slow_ema
        signal_line = macd.ewm(span=macd_signal, adjust=False).mean()
        
        # 4. Price momentum (rate of change)
        momentum = data[ticker].pct_change(periods=10) * 100
        
        # Generate individual signals
        trend_signal = pd.Series(trend, index=data.index)
        rsi_signal = pd.Series(np.where(rsi < 30, 1, np.where(rsi > 70, -1, 0)), index=data.index)
        macd_signal_val = pd.Series(np.where(macd > signal_line, 1, np.where(macd < signal_line, -1, 0)), index=data.index)
        momentum_signal = pd.Series(np.where(momentum > 5, 1, np.where(momentum < -5, -1, 0)), index=data.index)
        
        # Calculate consensus - how many indicators agree with each other
        for i in range(len(signals)):
            buy_count = 0
            sell_count = 0
            
            # Count buy signals
            if trend_signal.iloc[i] == 1: buy_count += 1
            if rsi_signal.iloc[i] == 1: buy_count += 1
            if macd_signal_val.iloc[i] == 1: buy_count += 1
            if momentum_signal.iloc[i] == 1: buy_count += 1
            
            # Count sell signals
            if trend_signal.iloc[i] == -1: sell_count += 1
            if rsi_signal.iloc[i] == -1: sell_count += 1
            if macd_signal_val.iloc[i] == -1: sell_count += 1
            if momentum_signal.iloc[i] == -1: sell_count += 1
            
            # Store consensus level (positive for buy consensus, negative for sell consensus)
            signals.iloc[i, signals.columns.get_loc(consensus_col)] = buy_count - sell_count
            
            # Generate final signal based on consensus threshold
            if buy_count >= min_consensus:
                signals.iloc[i, signals.columns.get_loc(signal_col)] = 1
            elif sell_count >= min_consensus:
                signals.iloc[i, signals.columns.get_loc(signal_col)] = -1
        
        # Add all indicators for reference
        signals[f"{ticker}_Price"] = data[ticker]
        signals[f"{ticker}_MA"] = ma
        signals[f"{ticker}_RSI"] = rsi
        signals[f"{ticker}_MACD"] = macd
        signals[f"{ticker}_Signal_Line"] = signal_line
        signals[f"{ticker}_Momentum"] = momentum
    
    # Drop NaN values
    signals = signals.dropna()
    
    return signals

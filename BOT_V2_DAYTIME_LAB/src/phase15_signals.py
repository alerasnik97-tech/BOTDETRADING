
import pandas as pd
import numpy as np

def detect_post_news_continuation(df, news_df, params):
    """
    S1: Post-News Controlled Continuation
    params: {
        'block_mins': 30, 
        'range_mins': 15, 
        'lookback_mins': 120,
        'families': ['CPI', 'NFP', 'FOMC']
    }
    """
    df = df.copy()
    df['signal'] = 0
    df['range_high'] = np.nan
    df['range_low'] = np.nan
    
    # 1. Map news to df
    news_times = news_df['timestamp_ny'].values
    
    # Use vectorized approach for speed
    # For each news, define the window
    for nt in news_times:
        nt = pd.Timestamp(nt)
        block_end = nt + pd.Timedelta(minutes=params['block_mins'])
        range_end = block_end + pd.Timedelta(minutes=params['range_mins'])
        
        # Define range during [nt, block_end]
        mask_range = (df['timestamp_ny'] >= nt) & (df['timestamp_ny'] <= block_end)
        if not mask_range.any(): continue
        
        r_high = df.loc[mask_range, 'high_bid'].max()
        r_low = df.loc[mask_range, 'low_bid'].min()
        
        # Detect breakout after block_end
        # We signal on the first bar that closes outside the range between block_end and range_end
        mask_signal = (df['timestamp_ny'] > block_end) & (df['timestamp_ny'] <= range_end)
        if not mask_signal.any(): continue
        
        signals_subset = df.loc[mask_signal]
        entry_type = params.get('entry_type', 'close_outside')
        
        for idx, row in signals_subset.iterrows():
            is_long = (row['close_bid'] > r_high) if entry_type == 'close_outside' else (row['high_bid'] > r_high)
            is_short = (row['close_bid'] < r_low) if entry_type == 'close_outside' else (row['low_bid'] < r_low)
            
            if is_long:
                df.at[idx, 'signal'] = 1
                df.at[idx, 'range_high'] = r_high
                df.at[idx, 'range_low'] = r_low
                break 
            elif is_short:
                df.at[idx, 'signal'] = -1
                df.at[idx, 'range_high'] = r_high
                df.at[idx, 'range_low'] = r_low
                break
                
    return df

def detect_compression_breakout(df, params):
    """
    S2: Volatility Compression Breakout
    params: {
        'window_bars': 12, # 60m if M5
        'percentile': 20,
        'ema_filter': True
    }
    """
    df = df.copy()
    # Range = High - Low
    df['bar_range'] = df['high_bid'] - df['low_bid']
    df['rolling_range'] = df['high_bid'].rolling(params['window_bars']).max() - df['low_bid'].rolling(params['window_bars']).min()
    
    # Percentile threshold
    threshold = df['rolling_range'].expanding().quantile(params['percentile'] / 100.0)
    df['is_compressed'] = df['rolling_range'] < threshold
    
    # EMA Filter
    if params.get('ema_filter', False):
        df['ema_h1'] = df['close_bid'].ewm(span=50*12).mean() # Approx H1 EMA50 on M5
    
    df['signal'] = 0
    # Entry on breakout of the compression window
    # We need the range of the compression window
    # To avoid lookahead, we use shift(1)
    df['prev_high'] = df['high_bid'].rolling(params['window_bars']).max().shift(1)
    df['prev_low'] = df['low_bid'].rolling(params['window_bars']).min().shift(1)
    
    # Signal: current close > prev_high AND prev bar was compressed
    long_mask = (df['is_compressed'].shift(1)) & (df['close_bid'] > df['prev_high'])
    short_mask = (df['is_compressed'].shift(1)) & (df['close_bid'] < df['prev_low'])
    
    if params.get('ema_filter', False):
        long_mask &= (df['close_bid'] > df['ema_h1'])
        short_mask &= (df['close_bid'] < df['ema_h1'])
        
    df.loc[long_mask, 'signal'] = 1
    df.loc[short_mask, 'signal'] = -1
    
    return df

def detect_session_exhaustion(df, params):
    """
    S3: Session Exhaustion Fade
    params: {
        'atr_multiplier': 1.5,
        'lookback_atr': 14
    }
    """
    from phase15_helpers import calculate_atr
    df = df.copy()
    df['atr'] = calculate_atr(df, params['lookback_atr'])
    
    # Distance from Daily Open
    df['date'] = df['timestamp_ny'].dt.date
    df['day_open'] = df.groupby('date')['open_bid'].transform('first')
    df['dist_from_open'] = (df['close_bid'] - df['day_open']).abs()
    
    df['is_exhausted'] = df['dist_from_open'] > (params['atr_multiplier'] * df['atr'])
    
    # Fade signal: rejection candle after exhaustion
    # Simple rejection: candle color opposite to move
    df['signal'] = 0
    
    # Long fade (Price dropped too much, now green candle)
    long_mask = (df['is_exhausted']) & (df['close_bid'] < df['day_open']) & (df['close_bid'] > df['open_bid'])
    # Short fade (Price rose too much, now red candle)
    short_mask = (df['is_exhausted']) & (df['close_bid'] > df['day_open']) & (df['close_bid'] < df['open_bid'])
    
    df.loc[long_mask, 'signal'] = 1
    df.loc[short_mask, 'signal'] = -1
    
    return df

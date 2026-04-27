
import pandas as pd
import numpy as np

def detect_htf_sweep_ltf_confirm(df_ltf, htf_levels, params):
    """
    params: {
        "momentum_body_pct": 0.6,
        "max_bars_post_sweep": 6,
        "sl_buffer_pips": 1.0
    }
    """
    df = df_ltf.copy()
    df['body'] = (df['close_bid'] - df['open_bid']).abs()
    df['range'] = df['high_bid'] - df['low_bid']
    df['body_pct'] = df['body'] / df['range'].replace(0, 0.00001)
    
    window = params.get('max_bars_post_sweep', 6)
    df['rolling_h'] = df['high_bid'].rolling(window + 1).max()
    df['rolling_l'] = df['low_bid'].rolling(window + 1).min()
    
    df['date'] = df['timestamp_ny'].dt.date
    
    # Levels mapping (PDH, PDL, Asia, London)
    # Strategy 1 uses flexible HTF levels (H4/H1 etc passed in htf_levels)
    df['h_level'] = df['date'].map({d: v.get('h_level') for d, v in htf_levels.items()}).fillna(999999)
    df['l_level'] = df['date'].map({d: v.get('l_level') for d, v in htf_levels.items()}).fillna(-999999)

    short_mask = (df['rolling_h'] > df['h_level']) & (df['close_bid'] < df['h_level']) & \
                 (df['close_bid'] < df['open_bid']) & (df['body_pct'] >= params['momentum_body_pct'])
                 
    long_mask = (df['rolling_l'] < df['l_level']) & (df['close_bid'] > df['l_level']) & \
                (df['close_bid'] > df['open_bid']) & (df['body_pct'] >= params['momentum_body_pct'])

    signals = []
    for idx in df.index[short_mask]:
        signals.append({'index': idx, 'type': 'SHORT', 'sl_custom': df.at[idx, 'rolling_h'] + params['sl_buffer_pips']*0.0001})
    for idx in df.index[long_mask]:
        signals.append({'index': idx, 'type': 'LONG', 'sl_custom': df.at[idx, 'rolling_l'] - params['sl_buffer_pips']*0.0001})
    return signals

def detect_london_reclaim_continuation(df_ltf, levels, params):
    # Similar to Phase 13 but with focus on NY window (07:00+)
    df = df_ltf.copy()
    df['body'] = (df['close_bid'] - df['open_bid']).abs()
    df['range'] = df['high_bid'] - df['low_bid']
    df['body_pct'] = df['body'] / df['range'].replace(0, 0.00001)
    df['date'] = df['timestamp_ny'].dt.date
    
    df['london_h'] = df['date'].map({d: v.get('london_h') for d, v in levels.items()}).fillna(999999)
    df['london_l'] = df['date'].map({d: v.get('london_l') for d, v in levels.items()}).fillna(-999999)
    
    df['prev_high'] = df['high_bid'].shift(1)
    df['prev_low'] = df['low_bid'].shift(1)
    
    short_mask = (df['prev_high'] > df['london_h']) & (df['close_bid'] < df['london_h']) & \
                 (df['body_pct'] >= params['reclaim_body_pct'])
    long_mask = (df['prev_low'] < df['london_l']) & (df['close_bid'] > df['london_l']) & \
                (df['body_pct'] >= params['reclaim_body_pct'])

    signals = []
    for idx in df.index[short_mask]:
        signals.append({'index': idx, 'type': 'SHORT'})
    for idx in df.index[long_mask]:
        signals.append({'index': idx, 'type': 'LONG'})
    return signals

def detect_opening_range_fakeout(df_ltf, range_levels, params):
    df = df_ltf.copy()
    df['date'] = df['timestamp_ny'].dt.date
    df['h_range'] = df['date'].map({d: v.get('h_range') for d, v in range_levels.items()}).fillna(999999)
    df['l_range'] = df['date'].map({d: v.get('l_range') for d, v in range_levels.items()}).fillna(-999999)
    
    df['prev_high'] = df['high_bid'].shift(1)
    df['prev_low'] = df['low_bid'].shift(1)
    
    short_mask = (df['prev_high'] > df['h_range']) & (df['close_bid'] < df['h_range'])
    long_mask = (df['prev_low'] < df['l_range']) & (df['close_bid'] > df['l_range'])

    signals = []
    for idx in df.index[short_mask]:
        signals.append({'index': idx, 'type': 'SHORT'})
    for idx in df.index[long_mask]:
        signals.append({'index': idx, 'type': 'LONG'})
    return signals

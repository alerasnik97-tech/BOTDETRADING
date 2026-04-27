
import pandas as pd
import numpy as np

def detect_h1_sweep_momentum(df_ltf, levels, params):
    df = df_ltf.copy()
    df['body'] = (df['close_bid'] - df['open_bid']).abs()
    df['range'] = df['high_bid'] - df['low_bid']
    df['body_pct'] = df['body'] / df['range'].replace(0, 0.00001)
    df['avg_range_10'] = df['range'].rolling(10).mean()
    
    window = params.get('max_bars_post_sweep', 6)
    df['rolling_h'] = df['high_bid'].rolling(window + 1).max()
    df['rolling_l'] = df['low_bid'].rolling(window + 1).min()
    
    df['date'] = df['timestamp_ny'].dt.date
    
    df['pdh'] = df['date'].map({d: v.get('pdh') for d, v in levels.items()}).fillna(999999)
    df['pdl'] = df['date'].map({d: v.get('pdl') for d, v in levels.items()}).fillna(-999999)
    df['asia_h'] = df['date'].map({d: v.get('asia_h') for d, v in levels.items()}).fillna(999999)
    df['asia_l'] = df['date'].map({d: v.get('asia_l') for d, v in levels.items()}).fillna(-999999)
    df['london_h'] = df['date'].map({d: v.get('london_h') for d, v in levels.items()}).fillna(999999)
    df['london_l'] = df['date'].map({d: v.get('london_l') for d, v in levels.items()}).fillna(-999999)

    min_depth = params.get('min_sweep_depth_pips', 0) * 0.0001

    short_mask = (
        ((df['rolling_h'] > df['pdh']) & (df['close_bid'] < df['pdh']) & (df['rolling_h'] - df['pdh'] >= min_depth)) |
        ((df['rolling_h'] > df['asia_h']) & (df['close_bid'] < df['asia_h']) & (df['rolling_h'] - df['asia_h'] >= min_depth)) |
        ((df['rolling_h'] > df['london_h']) & (df['close_bid'] < df['london_h']) & (df['rolling_h'] - df['london_h'] >= min_depth))
    ) & (df['close_bid'] < df['open_bid']) & \
      (df['body_pct'] >= params['momentum_body_pct']) & \
      (df['range'] >= (df['avg_range_10'] * params['momentum_relative_size']))

    long_mask = (
        ((df['rolling_l'] < df['pdl']) & (df['close_bid'] > df['pdl']) & (df['pdl'] - df['rolling_l'] >= min_depth)) |
        ((df['rolling_l'] < df['asia_l']) & (df['close_bid'] > df['asia_l']) & (df['asia_l'] - df['rolling_l'] >= min_depth)) |
        ((df['rolling_l'] < df['london_l']) & (df['close_bid'] > df['london_l']) & (df['london_l'] - df['rolling_l'] >= min_depth))
    ) & (df['close_bid'] > df['open_bid']) & \
      (df['body_pct'] >= params['momentum_body_pct']) & \
      (df['range'] >= (df['avg_range_10'] * params['momentum_relative_size']))

    if params.get('use_bias'):
        df['h1_bias'] = df['timestamp'].map(params['bias_map']).fillna(0)
        short_mask &= (df['h1_bias'] <= 0)
        long_mask &= (df['h1_bias'] >= 0)

    signals = []
    for idx in df.index[short_mask]:
        sl_custom = df.at[idx, 'rolling_h'] + (params.get('sl_buffer_pips', 0.5) * 0.0001)
        signals.append({'index': idx, 'type': 'SHORT', 'signal_time': df.at[idx, 'timestamp_ny'], 'sl_custom': sl_custom})
    
    for idx in df.index[long_mask]:
        sl_custom = df.at[idx, 'rolling_l'] - (params.get('sl_buffer_pips', 0.5) * 0.0001)
        signals.append({'index': idx, 'type': 'LONG', 'signal_time': df.at[idx, 'timestamp_ny'], 'sl_custom': sl_custom})
        
    return signals

def detect_session_reclaim(df_ltf, levels, params):
    df = df_ltf.copy()
    df['body'] = (df['close_bid'] - df['open_bid']).abs()
    df['range'] = df['high_bid'] - df['low_bid']
    df['body_pct'] = df['body'] / df['range'].replace(0, 0.00001)
    df['date'] = df['timestamp_ny'].dt.date
    
    h_key = f"{params['session_type']}_h"
    l_key = f"{params['session_type']}_l"
    df['h_level'] = df['date'].map({d: v.get(h_key) for d, v in levels.items()})
    df['l_level'] = df['date'].map({d: v.get(l_key) for d, v in levels.items()})
    
    df['prev_high'] = df['high_bid'].shift(1)
    df['prev_low'] = df['low_bid'].shift(1)
    
    short_mask = (df['prev_high'] > df['h_level']) & (df['close_bid'] < df['h_level']) & \
                 (df['body_pct'] >= params['reclaim_body_pct'])
                 
    long_mask = (df['prev_low'] < df['l_level']) & (df['close_bid'] > df['l_level']) & \
                (df['body_pct'] >= params['reclaim_body_pct'])

    if params.get('use_bias'):
        df['h1_bias'] = df['timestamp'].map(params['bias_map']).fillna(0)
        short_mask &= (df['h1_bias'] <= 0)
        long_mask &= (df['h1_bias'] >= 0)

    signals = []
    for idx in df.index[short_mask]:
        signals.append({'index': idx, 'type': 'SHORT', 'signal_time': df.at[idx, 'timestamp_ny']})
    for idx in df.index[long_mask]:
        signals.append({'index': idx, 'type': 'LONG', 'signal_time': df.at[idx, 'timestamp_ny']})
        
    return signals

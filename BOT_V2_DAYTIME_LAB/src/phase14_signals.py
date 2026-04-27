
import pandas as pd
import numpy as np

def get_fractals(df, n=3):
    highs = df['high_bid']
    lows = df['low_bid']
    is_low = (lows == lows.rolling(window=2*n+1, center=True).min())
    is_high = (highs == highs.rolling(window=2*n+1, center=True).max())
    return is_high.fillna(False).values, is_low.fillna(False).values

def precalculate_last_fractals(df, n=3):
    is_high, is_low = get_fractals(df, n)
    highs = df['high_bid'].values
    lows = df['low_bid'].values
    size = len(df)
    last_high_val = np.full(size, np.nan)
    last_low_val = np.full(size, np.nan)
    curr_h, curr_l = np.nan, np.nan
    for i in range(size):
        if i >= n:
            if is_high[i-n]: curr_h = highs[i-n]
            if is_low[i-n]: curr_l = lows[i-n]
        last_high_val[i] = curr_h
        last_low_val[i] = curr_l
    return last_high_val, last_low_val

def detect_sweep_choch(df_ltf, levels, params):
    """
    Unified detector for Sweep + CHoCH (Phase 7/8 style).
    Ultra-optimized with numpy.
    """
    df = df_ltf.copy()
    df['last_high_f'], df['last_low_f'] = precalculate_last_fractals(df, n=params.get('fractal_n', 3))
    df['date'] = df['timestamp_ny'].dt.date
    
    # Pre-map levels to columns
    lvl_arrays = {}
    for lvl in params['levels_to_check']:
        lvl_map = {d: v.get(lvl) for d, v in levels.items()}
        df[f'lvl_{lvl}'] = df['date'].map(lvl_map).fillna(999999 if 'h' in lvl else -999999)
        lvl_arrays[lvl] = df[f'lvl_{lvl}'].values
    
    highs = df['high_bid'].values
    lows = df['low_bid'].values
    closes = df['close_bid'].values
    last_high_f = df['last_high_f'].values
    last_low_f = df['last_low_f'].values
    
    signals = []
    pending_sweep = None
    
    print(f"    Detecting Sweep+CHoCH signals on {len(df)} bars (Numpy Mode)...", flush=True)
    
    for i in range(len(df)):
        if pending_sweep is None:
            # Check for sweeps
            for lvl_name, lvl_arr in lvl_arrays.items():
                if 'h' in lvl_name:
                    if highs[i] > lvl_arr[i]:
                        pending_sweep = {'dir': 'S', 'lvl': lvl_arr[i], 'ext': highs[i], 'idx': i}
                        break
                else:
                    if lows[i] < lvl_arr[i]:
                        pending_sweep = {'dir': 'L', 'lvl': lvl_arr[i], 'ext': lows[i], 'idx': i}
                        break
        else:
            if pending_sweep['dir'] == 'S':
                pending_sweep['ext'] = max(pending_sweep['ext'], highs[i])
                if not np.isnan(last_low_f[i]) and closes[i] < last_low_f[i]:
                    sl = pending_sweep['ext'] + (params.get('sl_buffer_pips', 0.5) * 0.0001)
                    signals.append({'index': i, 'type': 'SHORT', 'sl_custom': sl})
                    pending_sweep = None
            else:
                pending_sweep['ext'] = min(pending_sweep['ext'], lows[i])
                if not np.isnan(last_high_f[i]) and closes[i] > last_high_f[i]:
                    sl = pending_sweep['ext'] - (params.get('sl_buffer_pips', 0.5) * 0.0001)
                    signals.append({'index': i, 'type': 'LONG', 'sl_custom': sl})
                    pending_sweep = None
            
            if pending_sweep and (i - pending_sweep['idx'] > params.get('max_bars_post_sweep', 60)):
                pending_sweep = None
                
    return signals

def detect_htf_sweep_ltf_confirm(df_ltf, htf_levels, params):
    """
    S1: HTF Flex Sweep + LTF Confirmation.
    """
    df = df_ltf.copy()
    df['body'] = (df['close_bid'] - df['open_bid']).abs()
    df['range'] = df['high_bid'] - df['low_bid']
    df['body_pct'] = df['body'] / df['range'].replace(0, 0.00001)
    
    window = params.get('max_bars_post_sweep', 6)
    df['rolling_h'] = df['high_bid'].rolling(window + 1).max()
    df['rolling_l'] = df['low_bid'].rolling(window + 1).min()
    df['date'] = df['timestamp_ny'].dt.date
    
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
    """
    S2: London Range Reclaim Continuation.
    """
    df = df_ltf.copy()
    df['body'] = (df['close_bid'] - df['open_bid']).abs()
    df['range'] = df['high_bid'] - df['low_bid']
    df['body_pct'] = df['body'] / df['range'].replace(0, 0.00001)
    df['date'] = df['timestamp_ny'].dt.date
    
    df['london_h'] = df['date'].map({d: v.get('london_h') for d, v in levels.items()}).fillna(999999)
    df['london_l'] = df['date'].map({d: v.get('london_l') for d, v in levels.items()}).fillna(-999999)
    
    df['prev_high'] = df['high_bid'].shift(1)
    df['prev_low'] = df['low_bid'].shift(1)
    
    # Simple reclaim: close back inside after breaking
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
    """
    S3: Opening Range Fakeout / Retest.
    """
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

def detect_session_reclaim(df_ltf, levels, params):
    """
    Phase 13 London Reclaim.
    """
    df = df_ltf.copy()
    df['body'] = (df['close_bid'] - df['open_bid']).abs()
    df['range'] = df['high_bid'] - df['low_bid']
    df['body_pct'] = df['body'] / df['range'].replace(0, 0.00001)
    df['date'] = df['timestamp_ny'].dt.date
    
    h_key = f"{params['session_type']}_h"
    l_key = f"{params['session_type']}_l"
    df['h_level'] = df['date'].map({d: v.get(h_key) for d, v in levels.items()}).fillna(999999)
    df['l_level'] = df['date'].map({d: v.get(l_key) for d, v in levels.items()}).fillna(-999999)
    
    df['prev_high'] = df['high_bid'].shift(1)
    df['prev_low'] = df['low_bid'].shift(1)
    
    short_mask = (df['prev_high'] > df['h_level']) & (df['close_bid'] < df['h_level']) & \
                 (df['body_pct'] >= params['reclaim_body_pct'])
                 
    long_mask = (df['prev_low'] < df['l_level']) & (df['close_bid'] > df['l_level']) & \
                (df['body_pct'] >= params['reclaim_body_pct'])

    signals = []
    for idx in df.index[short_mask]:
        signals.append({'index': idx, 'type': 'SHORT'})
    for idx in df.index[long_mask]:
        signals.append({'index': idx, 'type': 'LONG'})
        
    return signals

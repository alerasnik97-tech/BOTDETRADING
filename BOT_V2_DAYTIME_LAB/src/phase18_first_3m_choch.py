
import pandas as pd
import numpy as np

def get_m3_fractals(df, n=2):
    highs = df['high_bid'].values
    lows = df['low_bid'].values
    size = len(df)
    f_h = np.full(size, np.nan)
    f_l = np.full(size, np.nan)
    
    for i in range(2*n, size):
        center = i - n
        if all(highs[center] > highs[j] for j in range(center-n, center+n+1) if j != center):
            f_h[i] = highs[center]
        if all(lows[center] < lows[j] for j in range(center-n, center+n+1) if j != center):
            f_l[i] = lows[center]
    return f_h, f_l

class First3MChochDetector:
    def __init__(self, params):
        self.params = params

    def detect_choch(self, df_m3, sweeps_h1):
        """
        Detecta el primer CHoCH M3 tras un barrido H1 (Versión Optimizada v2).
        """
        df = df_m3.copy()
        fh, fl = get_m3_fractals(df, n=self.params.get('fractal_n', 2))
        df['last_m3_fh'] = pd.Series(fh).ffill()
        df['last_m3_fl'] = pd.Series(fl).ffill()
        
        df['choch_bearish_trigger'] = (df['close_bid'] < df['last_m3_fl'])
        df['choch_bullish_trigger'] = (df['close_bid'] > df['last_m3_fh'])
        
        results = []
        
        # Convertimos a serie para mantener TZ en la comparación si es necesario, 
        # o usamos .values y aseguramos que sweep_time sea compatible.
        df_times = df['timestamp_ny']
        df_close = df['close_bid'].values
        df_bearish = df['choch_bearish_trigger'].values
        df_bullish = df['choch_bullish_trigger'].values
        
        sweeps_sorted = sweeps_h1.sort_values('timestamp_ny')
        
        curr_idx = 0
        total_bars = len(df)
        
        for _, sweep in sweeps_sorted.iterrows():
            sweep_time = sweep['timestamp_ny']
            max_mins = self.params.get('max_mins_post_sweep', 60)
            end_time = sweep_time + pd.Timedelta(minutes=max_mins)
            
            # Buscamos el índice inicial
            while curr_idx < total_bars and df_times.iloc[curr_idx] < sweep_time:
                curr_idx += 1
            
            search_idx = curr_idx
            while search_idx < total_bars and df_times.iloc[search_idx] <= end_time:
                if sweep['type'] == 'BEARISH_SWEEP':
                    if df_bearish[search_idx]:
                        results.append({
                            "sweep_time": sweep_time,
                            "choch_time": df_times.iloc[search_idx],
                            "direction": "SHORT",
                            "entry_price": df_close[search_idx],
                            "sl_price": sweep['peak_price'] + (self.params.get('sl_buffer', 0.5) * 0.0001),
                            "sweep_level": sweep['level_type']
                        })
                        break
                elif sweep['type'] == 'BULLISH_SWEEP':
                    if df_bullish[search_idx]:
                        results.append({
                            "sweep_time": sweep_time,
                            "choch_time": df_times.iloc[search_idx],
                            "direction": "LONG",
                            "entry_price": df_close[search_idx],
                            "sl_price": sweep['peak_price'] - (self.params.get('sl_buffer', 0.5) * 0.0001),
                            "sweep_level": sweep['level_type']
                        })
                        break
                search_idx += 1
                        
        return pd.DataFrame(results)

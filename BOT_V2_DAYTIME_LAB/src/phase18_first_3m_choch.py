
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
        Detecta el primer CHoCH M3 tras un barrido H1.
        """
        df = df_m3.copy()
        fh, fl = get_m3_fractals(df, n=self.params.get('fractal_n', 2))
        df['last_m3_fh'] = pd.Series(fh).ffill()
        df['last_m3_fl'] = pd.Series(fl).ffill()
        
        results = []
        
        for _, sweep in sweeps_h1.iterrows():
            sweep_time = sweep['timestamp_ny']
            # Ventana de búsqueda: sweep_time hasta sweep_time + max_mins
            max_mins = self.params.get('max_mins_post_sweep', 60)
            window = df[(df['timestamp_ny'] >= sweep_time) & 
                        (df['timestamp_ny'] <= sweep_time + pd.Timedelta(minutes=max_mins))]
            
            if window.empty: continue
            
            # Buscar el primer CHoCH
            for idx, bar in window.iterrows():
                close = bar['close_bid']
                
                if sweep['type'] == 'BEARISH_SWEEP': # Buscamos CHoCH bajista
                    trigger_lvl = bar['last_m3_fl']
                    if not pd.isna(trigger_lvl) and close < trigger_lvl:
                        # CHoCH confirmado
                        results.append({
                            "sweep_time": sweep_time,
                            "choch_time": bar['timestamp_ny'],
                            "direction": "SHORT",
                            "entry_price": bar['close_bid'], # Entrada al cierre del CHoCH
                            "sl_price": sweep['peak_price'] + (self.params.get('sl_buffer', 0.5) * 0.0001),
                            "sweep_level": sweep['level_type']
                        })
                        break # Solo el primer CHoCH
                
                elif sweep['type'] == 'BULLISH_SWEEP': # Buscamos CHoCH alcista
                    trigger_lvl = bar['last_m3_fh']
                    if not pd.isna(trigger_lvl) and close > trigger_lvl:
                        # CHoCH confirmado
                        results.append({
                            "sweep_time": sweep_time,
                            "choch_time": bar['timestamp_ny'],
                            "direction": "LONG",
                            "entry_price": bar['close_bid'],
                            "sl_price": sweep['peak_price'] - (self.params.get('sl_buffer', 0.5) * 0.0001),
                            "sweep_level": sweep['level_type']
                        })
                        break
                        
        return pd.DataFrame(results)

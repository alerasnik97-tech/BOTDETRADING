
import pandas as pd
import numpy as np
from pathlib import Path

def get_h1_fractals(df, n=2):
    """
    Identifica fractales H1 con delay real (no-lookahead).
    Un fractal N=2 en t-2 se confirma en t.
    """
    highs = df['high_bid'].values
    lows = df['low_bid'].values
    size = len(df)
    
    fractal_highs = np.full(size, np.nan)
    fractal_lows = np.full(size, np.nan)
    
    for i in range(2*n, size):
        # Punto central del fractal: i-n
        center = i - n
        val_h = highs[center]
        val_l = lows[center]
        
        is_f_high = True
        is_f_low = True
        
        for j in range(center - n, center + n + 1):
            if j == center: continue
            if highs[j] >= val_h: is_f_high = False
            if lows[j] <= val_l: is_f_low = False
            
        if is_f_high:
            fractal_highs[i] = val_h # Se marca en el momento de confirmación (i)
        if is_f_low:
            fractal_lows[i] = val_l # Se marca en el momento de confirmación (i)
            
    return fractal_highs, fractal_lows

def get_rolling_levels(df, windows=[6, 12, 24]):
    """
    Calcula niveles de máximos/mínimos rodantes (sin lookahead).
    """
    levels = {}
    for w in windows:
        levels[f'roll_h_{w}'] = df['high_bid'].shift(1).rolling(w).max()
        levels[f'roll_l_{w}'] = df['low_bid'].shift(1).rolling(w).min()
    return pd.DataFrame(levels)

class H1FractalSweepDetector:
    def __init__(self, params):
        self.params = params

    def detect_sweeps(self, df_h1):
        df = df_h1.copy()
        
        # 1. Niveles Fractales
        for n in [2, 3, 4]:
            fh, fl = get_h1_fractals(df, n=n)
            # Propagar el último fractal conocido para detectar el barrido
            df[f'last_f_h_{n}'] = pd.Series(fh).ffill()
            df[f'last_f_l_{n}'] = pd.Series(fl).ffill()
            
        # 2. Niveles Rodantes
        roll = get_rolling_levels(df)
        df = pd.concat([df, roll], axis=1)
        
        # 3. Niveles Estáticos (Autoridad)
        # (Se asume que vienen precalculados o se calculan aquí)
        # PDH/PDL
        df['date'] = df['timestamp_ny'].dt.date
        daily = df.groupby('date').agg({'high_bid': 'max', 'low_bid': 'min'})
        df['pdh'] = df['date'].map(daily['high_bid'].shift(1))
        df['pdl'] = df['date'].map(daily['low_bid'].shift(1))
        
        # Detección de Barridos (Wick only or close inside)
        sweeps = []
        
        # Lista de niveles a monitorear
        level_cols = [
            'last_f_h_2', 'last_f_l_2', 
            'last_f_h_3', 'last_f_l_3',
            'pdh', 'pdl',
            'roll_h_12', 'roll_l_12'
        ]
        
        for i in range(1, len(df)):
            high = df.at[i, 'high_bid']
            low = df.at[i, 'low_bid']
            close = df.at[i, 'close_bid']
            
            for lvl_col in level_cols:
                lvl_val = df.at[i, lvl_col]
                if pd.isna(lvl_val): continue
                
                # Sweep Bearish (Barrido de Techo)
                if high > lvl_val and close < lvl_val:
                    sweeps.append({
                        "timestamp_ny": df.at[i, "timestamp_ny"],
                        "type": "BEARISH_SWEEP",
                        "level_type": lvl_col,
                        "level_price": lvl_val,
                        "peak_price": high,
                        "depth_pips": round((high - lvl_val) * 10000, 2),
                        "is_fractal": "f_h" in lvl_col
                    })
                
                # Sweep Bullish (Barrido de Suelo)
                if low < lvl_val and close > lvl_val:
                    sweeps.append({
                        "timestamp_ny": df.at[i, "timestamp_ny"],
                        "type": "BULLISH_SWEEP",
                        "level_type": lvl_col,
                        "level_price": lvl_val,
                        "peak_price": low,
                        "depth_pips": round((lvl_val - low) * 10000, 2),
                        "is_fractal": "f_l" in lvl_col
                    })
                    
        return pd.DataFrame(sweeps)

if __name__ == "__main__":
    # Test simple
    print("Módulo H1 Fractal Sweep listo.")


import pandas as pd
import numpy as np
from pathlib import Path
from phase14_engine import Phase14Engine
from phase18_h1_fractal_sweep import H1FractalSweepDetector
import json

class LTFChochDetector:
    def __init__(self, params):
        self.params = params
    def detect_choch(self, df_ltf, sweeps_h1):
        df = df_ltf.copy()
        n = self.params.get('fractal_n', 2)
        highs, lows = df['high_bid'].values, df['low_bid'].values
        size = len(df)
        f_h, f_l = np.full(size, np.nan), np.full(size, np.nan)
        for i in range(2*n, size):
            center = i - n
            if all(highs[center] > highs[j] for j in range(center-n, center+n+1) if j != center): f_h[i] = highs[center]
            if all(lows[center] < lows[j] for j in range(center-n, center+n+1) if j != center): f_l[i] = lows[center]
        df['last_ltf_fh'], df['last_ltf_fl'] = pd.Series(f_h).ffill(), pd.Series(f_l).ffill()
        results = []
        for _, sweep in sweeps_h1.iterrows():
            if sweep['depth_pips'] < 0.5: continue
            sweep_time = sweep['timestamp_ny']
            window = df[(df['timestamp_ny'] >= sweep_time) & (df['timestamp_ny'] <= sweep_time + pd.Timedelta(minutes=30))]
            if window.empty: continue
            for idx, bar in window.iterrows():
                if sweep['type'] == 'BEARISH_SWEEP':
                    if not pd.isna(bar['last_ltf_fl']) and bar['close_bid'] < bar['last_ltf_fl']:
                        results.append({"time": bar['timestamp_ny'], "dir": "SHORT", "entry": bar['close_bid'], "sl": sweep['peak_price'] + 0.00005})
                        break
                elif sweep['type'] == 'BULLISH_SWEEP':
                    if not pd.isna(bar['last_ltf_fh']) and bar['close_bid'] > bar['last_ltf_fh']:
                        results.append({"time": bar['timestamp_ny'], "dir": "LONG", "entry": bar['close_bid'], "sl": sweep['peak_price'] - 0.00005})
                        break
        return pd.DataFrame(results)

def run_final_robustness():
    print("Fase 7: Final Candidate Robustness Analysis...")
    manifest_path = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\data\certified_data_paths.json"
    engine = Phase14Engine(manifest_path)
    period = "period_2020_2026"
    df_h1 = engine.load_and_prep_prices(period, timeframe='h1')
    df_m3 = engine.load_and_prep_prices(period, timeframe='m3')
    sweep_detector = H1FractalSweepDetector({})
    sweeps = sweep_detector.detect_sweeps(df_h1)
    
    detector = LTFChochDetector({})
    signals = detector.detect_choch(df_m3, sweeps)
    
    signals['hour'] = signals['time'].dt.hour + signals['time'].dt.minute / 60.0
    signals['day_of_week'] = signals['time'].dt.dayofweek
    
    # Apply Best Filters
    sig_f = signals[(signals['hour'] >= 8) & (signals['hour'] < 16.5)]
    sig_f = sig_f[sig_f['day_of_week'] != 4] # Exclude Friday
    sig_f['date'] = sig_f['time'].dt.date
    sig_f = sig_f.sort_values('time').groupby('date').head(3)
    
    tp_r = 2.5
    trades = []
    for _, sig in sig_f.iterrows():
        entry, sl = sig['entry'], sig['sl']
        risk = abs(entry - sl)
        if risk < 0.00001: continue
        tp = entry + (risk * tp_r) if sig['dir'] == 'LONG' else entry - (risk * tp_r)
        future = df_m3[df_m3['timestamp_ny'] > sig['time']].head(120)
        res = 'TIMEOUT'
        for _, bar in future.iterrows():
            if sig['dir'] == 'LONG':
                if bar['low_bid'] <= sl: res = 'SL'; break
                if bar['high_bid'] >= tp: res = 'TP'; break
            else:
                if bar['high_bid'] >= sl: res = 'SL'; break
                if bar['low_bid'] <= tp: res = 'TP'; break
        trades.append({"time": sig['time'], "res": res, "r_pnl": tp_r if res == 'TP' else (-1.0 if res == 'SL' else 0.0)})
        
    t_df = pd.DataFrame(trades)
    t_df['year'] = t_df['time'].dt.year
    
    # Yearly Stats
    yearly = []
    for year in sorted(t_df['year'].unique()):
        y_df = t_df[t_df['year'] == year]
        tp_c = len(y_df[y_df['res'] == 'TP'])
        sl_c = len(y_df[y_df['res'] == 'SL'])
        pf = round((tp_c * tp_r) / sl_c, 2) if sl_c > 0 else 0
        yearly.append({"year": year, "sample": len(y_df), "pf": pf})
    
    y_res = pd.DataFrame(yearly)
    print(y_res)
    
    out_dir = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\phase19_phase18_expansion\robustness")
    out_dir.mkdir(parents=True, exist_ok=True)
    y_res.to_csv(out_dir / "phase19_final_robustness_by_year.csv", index=False)
    t_df.to_csv(out_dir / "phase19_final_trades.csv", index=False)
    print("Final Robustness Analysis Complete.")

if __name__ == "__main__":
    run_final_robustness()

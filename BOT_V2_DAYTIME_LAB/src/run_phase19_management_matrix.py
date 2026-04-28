
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
            sweep_time = sweep['timestamp_ny']
            max_mins = self.params.get('max_mins_post_sweep', 30)
            window = df[(df['timestamp_ny'] >= sweep_time) & (df['timestamp_ny'] <= sweep_time + pd.Timedelta(minutes=max_mins))]
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

def run_management_matrix():
    print("Fase 5: Management Matrix Analysis...")
    manifest_path = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\data\certified_data_paths.json"
    engine = Phase14Engine(manifest_path)
    period = "period_2020_2026"
    df_h1 = engine.load_and_prep_prices(period, timeframe='h1')
    df_m3 = engine.load_and_prep_prices(period, timeframe='m3')
    
    sweep_detector = H1FractalSweepDetector({})
    sweeps = sweep_detector.detect_sweeps(df_h1)
    detector = LTFChochDetector({'max_mins_post_sweep': 30})
    all_signals = detector.detect_choch(df_m3, sweeps)
    
    # Lead Candidate Window 08:00 - 16:30, Max 3 trades
    all_signals['hour'] = all_signals['time'].dt.hour + all_signals['time'].dt.minute / 60.0
    sig_f = all_signals[(all_signals['hour'] >= 8) & (all_signals['hour'] < 16.5)]
    sig_f['date'] = sig_f['time'].dt.date
    sig_f = sig_f.sort_values('time').groupby('date').head(3)
    
    tp_targets = [1.0, 1.5, 2.0, 2.5, 3.0]
    be_triggers = [0.0, 1.0, 1.5] # 0.0 means no BE
    
    results = []
    for tp_r in tp_targets:
        for be_r in be_triggers:
            trades = []
            for _, sig in sig_f.iterrows():
                entry, sl = sig['entry'], sig['sl']
                risk = abs(entry - sl)
                if risk < 0.00001: continue
                tp_price = entry + (risk * tp_r) if sig['dir'] == 'LONG' else entry - (risk * tp_r)
                be_price = entry + (risk * be_r) if be_r > 0 and sig['dir'] == 'LONG' else (entry - (risk * be_r) if be_r > 0 else None)
                
                future = df_m3[df_m3['timestamp_ny'] > sig['time']].head(120)
                res = 'TIMEOUT'
                be_active = False
                for _, bar in future.iterrows():
                    # Check BE trigger
                    if be_r > 0 and not be_active:
                        if (sig['dir'] == 'LONG' and bar['high_bid'] >= be_price) or (sig['dir'] == 'SHORT' and bar['low_bid'] <= be_price):
                            be_active = True
                    
                    # Check Exit
                    if sig['dir'] == 'LONG':
                        if bar['low_bid'] <= (entry if be_active else sl): 
                            res = 'BE' if be_active else 'SL'; break
                        if bar['high_bid'] >= tp_price: res = 'TP'; break
                    else:
                        if bar['high_bid'] >= (entry if be_active else sl): 
                            res = 'BE' if be_active else 'SL'; break
                        if bar['low_bid'] <= tp_price: res = 'TP'; break
                trades.append(res)
            
            t_s = pd.Series(trades)
            tp_c, sl_c, be_c = len(t_s[t_s == 'TP']), len(t_s[t_s == 'SL']), len(t_s[t_s == 'BE'])
            pf = round((tp_c * tp_r) / sl_c, 2) if sl_c > 0 else 9.99
            expectancy = round((tp_c * tp_r - sl_c) / len(t_s), 2) if len(t_s) > 0 else 0
            
            results.append({"tp": tp_r, "be": be_r, "pf": pf, "expectancy": expectancy, "sample": len(t_s)})
            print(f"TP {tp_r} | BE {be_r} -> PF {pf} | Exp {expectancy}")

    res_df = pd.DataFrame(results)
    out_dir = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\phase19_phase18_expansion\management_matrix")
    out_dir.mkdir(parents=True, exist_ok=True)
    res_df.to_csv(out_dir / "management_matrix_results.csv", index=False)
    print("Management Matrix Analysis Complete.")

if __name__ == "__main__":
    run_management_matrix()

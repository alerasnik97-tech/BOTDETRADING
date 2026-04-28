
import pandas as pd
import numpy as np
from pathlib import Path
from phase14_engine import Phase14Engine
from phase18_h1_fractal_sweep import H1FractalSweepDetector
from phase18_first_3m_choch import First3MChochDetector, get_m3_fractals # We can reuse fractal logic
import json

class LTFChochDetector:
    def __init__(self, params):
        self.params = params

    def detect_choch(self, df_ltf, sweeps_h1):
        df = df_ltf.copy()
        n = self.params.get('fractal_n', 2)
        
        # Use generic fractal logic
        highs = df['high_bid'].values
        lows = df['low_bid'].values
        size = len(df)
        f_h = np.full(size, np.nan)
        f_l = np.full(size, np.nan)
        for i in range(2*n, size):
            center = i - n
            if all(highs[center] > highs[j] for j in range(center-n, center+n+1) if j != center): f_h[i] = highs[center]
            if all(lows[center] < lows[j] for j in range(center-n, center+n+1) if j != center): f_l[i] = lows[center]
            
        df['last_ltf_fh'] = pd.Series(f_h).ffill()
        df['last_ltf_fl'] = pd.Series(f_l).ffill()
        
        # Body filter
        df['body'] = (df['close_bid'] - df['open_bid']).abs()
        df['range'] = df['high_bid'] - df['low_bid']
        df['body_pct'] = df['body'] / df['range'].replace(0, 0.00001)
        
        results = []
        for _, sweep in sweeps_h1.iterrows():
            sweep_time = sweep['timestamp_ny']
            max_mins = self.params.get('max_mins_post_sweep', 60)
            window = df[(df['timestamp_ny'] >= sweep_time) & 
                        (df['timestamp_ny'] <= sweep_time + pd.Timedelta(minutes=max_mins))]
            
            if window.empty: continue
            
            body_min = self.params.get('body_min', 0.0)
            
            for idx, bar in window.iterrows():
                close = bar['close_bid']
                if bar['body_pct'] < body_min: continue
                
                if sweep['type'] == 'BEARISH_SWEEP':
                    trigger = bar['last_ltf_fl']
                    if not pd.isna(trigger) and close < trigger:
                        results.append({
                            "time": bar['timestamp_ny'], "dir": "SHORT", "entry": bar['close_bid'],
                            "sl": sweep['peak_price'] + 0.00005, "sweep_time": sweep_time
                        })
                        break
                elif sweep['type'] == 'BULLISH_SWEEP':
                    trigger = bar['last_ltf_fh']
                    if not pd.isna(trigger) and close > trigger:
                        results.append({
                            "time": bar['timestamp_ny'], "dir": "LONG", "entry": bar['close_bid'],
                            "sl": sweep['peak_price'] - 0.00005, "sweep_time": sweep_time
                        })
                        break
        return pd.DataFrame(results)

def run_ltf_sensitivity():
    print("Fase 2: LTF Sensitivity Analysis...")
    manifest_path = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\data\certified_data_paths.json"
    engine = Phase14Engine(manifest_path)
    period = "period_2020_2026"
    df_h1 = engine.load_and_prep_prices(period, timeframe='h1')
    
    sweep_detector = H1FractalSweepDetector({})
    sweeps = sweep_detector.detect_sweeps(df_h1)
    
    timeframes = ['m1', 'm3', 'm5', 'm15']
    body_filters = [0.0, 0.5, 0.7]
    windows = [30, 60, 120]
    
    results = []
    
    for tf in timeframes:
        print(f"Testing TF: {tf}")
        df_ltf = engine.load_and_prep_prices(period, timeframe=tf)
        
        for body in body_filters:
            for win in windows:
                detector = LTFChochDetector({'max_mins_post_sweep': win, 'body_min': body})
                signals = detector.detect_choch(df_ltf, sweeps)
                
                if signals.empty: continue
                
                # Filter Window 08-11 + First of day
                signals['hour'] = signals['time'].dt.hour
                sig_f = signals[(signals['hour'] >= 8) & (signals['hour'] <= 10)]
                sig_f['date'] = sig_f['time'].dt.date
                sig_f = sig_f.sort_values('time').groupby('date').head(1)
                
                # Simple backtest TP 2R
                trades = []
                for _, sig in sig_f.iterrows():
                    # Entry at next bar (approximate by signal close)
                    entry = sig['entry']
                    sl = sig['sl']
                    risk = abs(entry - sl)
                    if risk < 0.00001: continue
                    tp = entry + (risk * 2.0) if sig['dir'] == 'LONG' else entry - (risk * 2.0)
                    
                    # Search for exit in the same TF for speed
                    future = df_ltf[df_ltf['timestamp_ny'] > sig['time']].head(300) # Extended timeout for research
                    res = 'TIMEOUT'
                    for _, bar in future.iterrows():
                        if sig['dir'] == 'LONG':
                            if bar['low_bid'] <= sl: res = 'SL'; break
                            if bar['high_bid'] >= tp: res = 'TP'; break
                        else:
                            if bar['high_bid'] >= sl: res = 'SL'; break
                            if bar['low_bid'] <= tp: res = 'TP'; break
                    trades.append(res)
                
                t_s = pd.Series(trades)
                tp_c = len(t_s[t_s == 'TP'])
                sl_c = len(t_s[t_s == 'SL'])
                pf = round((tp_c * 2.0) / sl_c, 2) if sl_c > 0 else 0
                
                results.append({
                    "tf": tf, "body": body, "window": win, "sample": len(t_s), "pf": pf
                })
                print(f"  TF {tf} | Body {body} | Win {win} -> PF {pf} ({len(t_s)} trades)")

    res_df = pd.DataFrame(results)
    out_dir = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\phase19_phase18_expansion\ltf_sensitivity")
    out_dir.mkdir(parents=True, exist_ok=True)
    res_df.to_csv(out_dir / "ltf_sensitivity_results.csv", index=False)
    print("LTF Sensitivity Analysis Complete.")

if __name__ == "__main__":
    run_ltf_sensitivity()

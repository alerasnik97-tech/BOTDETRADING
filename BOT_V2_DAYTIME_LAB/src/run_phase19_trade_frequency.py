
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
                        results.append({"time": bar['timestamp_ny'], "dir": "SHORT", "entry": bar['close_bid'], "sl": sweep['peak_price'] + 0.00005, "sweep_time": sweep_time})
                        break
                elif sweep['type'] == 'BULLISH_SWEEP':
                    if not pd.isna(bar['last_ltf_fh']) and bar['close_bid'] > bar['last_ltf_fh']:
                        results.append({"time": bar['timestamp_ny'], "dir": "LONG", "entry": bar['close_bid'], "sl": sweep['peak_price'] - 0.00005, "sweep_time": sweep_time})
                        break
        return pd.DataFrame(results)

def run_trade_frequency():
    print("Fase 4: Trade Frequency Analysis...")
    manifest_path = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\data\certified_data_paths.json"
    engine = Phase14Engine(manifest_path)
    period = "period_2020_2026"
    df_h1 = engine.load_and_prep_prices(period, timeframe='h1')
    df_m3 = engine.load_and_prep_prices(period, timeframe='m3')
    
    sweep_detector = H1FractalSweepDetector({})
    sweeps = sweep_detector.detect_sweeps(df_h1)
    detector = LTFChochDetector({'max_mins_post_sweep': 30})
    all_signals = detector.detect_choch(df_m3, sweeps)
    
    # Filter 08:00 - 16:30
    all_signals['hour'] = all_signals['time'].dt.hour + all_signals['time'].dt.minute / 60.0
    all_signals = all_signals[(all_signals['hour'] >= 8) & (all_signals['hour'] < 16.5)]
    all_signals['date'] = all_signals['time'].dt.date
    
    results = []
    for max_trades in [1, 2, 3]:
        # Selection logic: take up to N trades per day, but only after previous one finishes
        # For simplicity, we assume trades don't overlap (timeout 2h)
        sig_f = all_signals.sort_values('time').groupby('date').head(max_trades)
        
        trades = []
        for _, sig in sig_f.iterrows():
            entry, sl = sig['entry'], sig['sl']
            risk = abs(entry - sl)
            if risk < 0.00001: continue
            tp = entry + (risk * 2.0) if sig['dir'] == 'LONG' else entry - (risk * 2.0)
            future = df_m3[df_m3['timestamp_ny'] > sig['time']].head(120)
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
        tp_c, sl_c = len(t_s[t_s == 'TP']), len(t_s[t_s == 'SL'])
        pf = round((tp_c * 2.0) / sl_c, 2) if sl_c > 0 else 0
        
        results.append({"max_trades_daily": max_trades, "sample": len(t_s), "pf": pf})
        print(f"Max Trades {max_trades} -> PF {pf} ({len(t_s)} trades)")

    res_df = pd.DataFrame(results)
    out_dir = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\phase19_phase18_expansion\trade_frequency")
    out_dir.mkdir(parents=True, exist_ok=True)
    res_df.to_csv(out_dir / "trade_frequency_results.csv", index=False)
    print("Trade Frequency Analysis Complete.")

if __name__ == "__main__":
    run_trade_frequency()

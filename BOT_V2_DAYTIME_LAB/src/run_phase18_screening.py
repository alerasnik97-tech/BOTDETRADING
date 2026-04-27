
import pandas as pd
import numpy as np
from pathlib import Path
from phase14_engine import Phase14Engine
from phase18_h1_fractal_sweep import H1FractalSweepDetector
from phase18_first_3m_choch import First3MChochDetector
import json

def run_phase18_screening():
    print("Starting Phase 18 Screening...")
    manifest_path = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\data\certified_data_paths.json"
    engine = Phase14Engine(manifest_path)
    
    period = "period_2020_2026"
    df_h1 = engine.load_and_prep_prices(period, timeframe='h1')
    df_m3 = engine.load_and_prep_prices(period, timeframe='m3')
    
    # 1. Detect Sweeps
    sweep_detector = H1FractalSweepDetector({})
    sweeps = sweep_detector.detect_sweeps(df_h1)
    print(f"Detected {len(sweeps)} H1 sweeps.")
    
    # 2. Detect CHoCH
    choch_detector = First3MChochDetector({'max_mins_post_sweep': 60, 'sl_buffer': 0.5})
    signals = choch_detector.detect_choch(df_m3, sweeps)
    print(f"Detected {len(signals)} potential signals.")
    
    if signals.empty:
        print("No signals detected.")
        return

    # 3. Apply Filters (Hour Window: 08:00 - 11:00 NY)
    signals['hour'] = signals['choch_time'].dt.hour
    valid_signals = signals[(signals['hour'] >= 8) & (signals['hour'] <= 10)]
    print(f"Signals in window 08:00-11:00: {len(valid_signals)}")
    
    # 4. Simple Backtest (TP 2R, Fixed SL)
    # This is a crude backtest for screening
    trades = []
    for idx, sig in valid_signals.iterrows():
        # Get price data after choch_time
        entry_time = sig['choch_time']
        direction = sig['direction']
        entry_price = sig['entry_price']
        sl_price = sig['sl_price']
        
        risk = abs(entry_price - sl_price)
        if risk == 0: continue
        
        tp_price = entry_price + (risk * 2.0) if direction == 'LONG' else entry_price - (risk * 2.0)
        
        # Look for outcome in M3
        future_prices = df_m3[df_m3['timestamp_ny'] > entry_time].head(200) # Check up to 10 hours
        
        result = 'TIMEOUT'
        exit_time = None
        
        for _, bar in future_prices.iterrows():
            if direction == 'LONG':
                if bar['low_bid'] <= sl_price:
                    result = 'SL'
                    exit_time = bar['timestamp_ny']
                    break
                if bar['high_bid'] >= tp_price:
                    result = 'TP'
                    exit_time = bar['timestamp_ny']
                    break
            else:
                if bar['high_bid'] >= sl_price:
                    result = 'SL'
                    exit_time = bar['timestamp_ny']
                    break
                if bar['low_bid'] <= tp_price:
                    result = 'TP'
                    exit_time = bar['timestamp_ny']
                    break
        
        trades.append({
            "time": entry_time,
            "direction": direction,
            "entry": entry_price,
            "sl": sl_price,
            "tp": tp_price,
            "result": result,
            "sweep_level": sig['sweep_level'],
            "exit_time": exit_time
        })
        
    df_trades = pd.DataFrame(trades)
    
    # Metrics
    tp_count = len(df_trades[df_trades['result'] == 'TP'])
    sl_count = len(df_trades[df_trades['result'] == 'SL'])
    pf = round((tp_count * 2.0) / sl_count, 2) if sl_count > 0 else 0
    
    summary = {
        "sample": len(df_trades),
        "pf": pf,
        "tp_count": tp_count,
        "sl_count": sl_count,
        "timeout_count": len(df_trades[df_trades['result'] == 'TIMEOUT']),
        "win_rate": round(tp_count / len(df_trades) * 100, 2) if len(df_trades) > 0 else 0
    }
    
    out_dir = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\phase18_h1_fractal_sweep\screening")
    out_dir.mkdir(parents=True, exist_ok=True)
    df_trades.to_csv(out_dir / "phase18_screening_results.csv", index=False)
    
    with open(out_dir / "phase18_screening_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
        
    print(f"Screening Complete. PF: {pf}")

if __name__ == "__main__":
    run_phase18_screening()

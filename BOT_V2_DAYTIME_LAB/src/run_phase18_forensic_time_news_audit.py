
import pandas as pd
import numpy as np
from pathlib import Path
from phase14_engine import Phase14Engine
from phase18_h1_fractal_sweep import H1FractalSweepDetector
from phase18_first_3m_choch import First3MChochDetector
import json

def run_forensic_time_news_audit():
    print("Fase 6: Auditoría de News / Horario...")
    manifest_path = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\data\certified_data_paths.json"
    engine = Phase14Engine(manifest_path)
    
    period = "period_2020_2026"
    df_h1 = engine.load_and_prep_prices(period, timeframe='h1')
    df_m3 = engine.load_and_prep_prices(period, timeframe='m3')
    
    # 1. Detect Sweeps
    sweep_detector = H1FractalSweepDetector({})
    sweeps = sweep_detector.detect_sweeps(df_h1)
    choch_detector = First3MChochDetector({'max_mins_post_sweep': 60, 'sl_buffer': 0.5})
    signals = choch_detector.detect_choch(df_m3, sweeps)
    
    # Pre-calculate body filter
    df_m3['body'] = (df_m3['close_bid'] - df_m3['open_bid']).abs()
    df_m3['range'] = df_m3['high_bid'] - df_m3['low_bid']
    df_m3['body_pct'] = df_m3['body'] / df_m3['range'].replace(0, 0.00001)
    
    signals_filtered = []
    for idx, sig in signals.iterrows():
        m3_bar = df_m3[df_m3['timestamp_ny'] == sig['choch_time']].iloc[0]
        if m3_bar['body_pct'] >= 0.7:
            signals_filtered.append(sig)
    
    sig_df = pd.DataFrame(signals_filtered)
    
    # Audit Original signals (before window filtering)
    out_of_hours = sig_df[(sig_df['choch_time'].dt.hour < 7) | (sig_df['choch_time'].dt.hour >= 20)]
    print(f"Signals outside 07:00-20:00: {len(out_of_hours)}")
    
    # Audit News (High Impact)
    news_path = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\DATA\news_eurusd_v2_utc.csv"
    if Path(news_path).exists():
        news_df = pd.read_csv(news_path)
        news_df['time'] = pd.to_datetime(news_df['timestamp_utc'], utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None)
        # Ensure sig_df timestamps are also naive NY
        sig_df['choch_time_naive'] = sig_df['choch_time'].dt.tz_localize(None) if sig_df['choch_time'].dt.tz is not None else sig_df['choch_time']
        high_news = news_df[news_df['impact_level'] == 'High']
        
        news_violations = []
        for _, sig in sig_df.iterrows():
            t = sig['choch_time_naive']
            # Within 30 mins
            v = high_news[(high_news['time'] >= t - pd.Timedelta(minutes=30)) & 
                          (high_news['time'] <= t + pd.Timedelta(minutes=30))]
            if not v.empty:
                news_violations.append(sig)
        
        print(f"News violations (30m): {len(news_violations)}")
    else:
        print("News data not found. Skipping news audit.")
        news_violations = []

    out_dir = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\phase18_forensic_audit\time_news")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "phase18_time_news_report.json", 'w') as f:
        json.dump({
            "out_of_hours_count": len(out_of_hours),
            "news_violations_30m": len(news_violations)
        }, f, indent=2)

if __name__ == "__main__":
    run_forensic_time_news_audit()

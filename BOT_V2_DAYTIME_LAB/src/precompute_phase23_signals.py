
import pandas as pd
import numpy as np
from pathlib import Path
from phase14_engine import Phase14Engine
from phase18_h1_fractal_sweep import H1FractalSweepDetector
from phase18_first_3m_choch import First3MChochDetector
from news_fortress.news_fortress_gate import NewsFortressGate
import pickle

def precompute():
    print("Precomputing signals for Phase 23...")
    manifest_path = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\data\certified_data_paths.json"
    engine = Phase14Engine(manifest_path)
    period = "period_2020_2026"
    
    print("Loading prices...")
    df_h1 = engine.load_and_prep_prices(period, timeframe='h1')
    df_m3 = engine.load_and_prep_prices(period, timeframe='m3')
    
    print("Detecting sweeps...")
    sweeps = H1FractalSweepDetector({}).detect_sweeps(df_h1)
    
    print("Detecting CHOCH...")
    choch_detector = First3MChochDetector({'max_mins_post_sweep': 60, 'sl_buffer': 0.5})
    signals = choch_detector.detect_choch(df_m3, sweeps)
    
    print("Saving signals...")
    out_dir = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\phase23_phase22_forensic_readiness\reproduction")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "precomputed_signals.pkl", 'wb') as f:
        pickle.dump(signals, f)
    
    print(f"Precomputed {len(signals)} signals.")

if __name__ == "__main__":
    precompute()

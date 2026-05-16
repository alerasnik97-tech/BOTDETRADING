import pandas as pd
from pathlib import Path
from research_lab.data_loader import load_backtest_data_bundle
from research_lab.strategies import am_silver_bullet_ny
from research_lab.config import EngineConfig, DEFAULT_DATA_DIRS

def debug():
    pair = "EURUSD"
    start_date = "2024-01-01" # Solo 2024 para velocidad
    end_date = "2024-03-01"
    
    data_dirs = list(DEFAULT_DATA_DIRS)
    engine_config = EngineConfig(pair=pair)
    
    bundle = load_backtest_data_bundle(
        pair=pair,
        data_dirs=data_dirs,
        start=start_date,
        end=end_date,
        execution_mode="normal_mode",
        target_timeframe="M5",
    )
    
    frame = bundle.frame
    # Buscamos ventanas 10:00-11:00
    sb_hours = frame.between_time("10:00", "11:00")
    print(f"Total bars in 10:00-11:00 range: {len(sb_hours)}")
    
    high_col = "session_range_high_03_00_08_30"
    low_col = "session_range_low_03_00_08_30"
    
    sweeps_h = 0
    sweeps_l = 0
    mss_bull = 0
    mss_bear = 0
    
    for i in range(len(frame)):
        ts = frame.index[i]
        if not (10 <= ts.hour < 11):
            continue
            
        anchor_high = frame[high_col].iat[i]
        anchor_low = frame[low_col].iat[i]
        
        has_swept_high = float(frame["day_running_high"].iat[i]) > (anchor_high + 0.0000)
        has_swept_low = float(frame["day_running_low"].iat[i]) < (anchor_low - 0.0000)
        
        if has_swept_high: sweeps_h += 1
        if has_swept_low: sweeps_l += 1
        
        if bool(frame["bullish_choch"].iat[i]): mss_bull += 1
        if bool(frame["bearish_choch"].iat[i]): mss_bear += 1
        
    print(f"Sweeps High: {sweeps_h}, Sweeps Low: {sweeps_l}")
    print(f"MSS Bull: {mss_bull}, MSS Bear: {mss_bear}")

if __name__ == "__main__":
    debug()

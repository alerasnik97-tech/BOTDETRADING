import pandas as pd
import numpy as np
from pathlib import Path

def generate_configs():
    base_dir = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\03_RESEARCH_LAB\BOT_V2_DAYTIME_LAB\reports\v50b_limited_real_gauntlet\configs")
    
    f06_configs = []
    for i in range(1, 51):
        f06_configs.append({
            "family_id": "F06",
            "config_id": f"F06_LIM_{i:04d}",
            "timeframe": "15m",
            "realized_vol_lookback": np.random.choice([10, 20, 30]),
            "compression_percentile": np.random.choice([0.1, 0.15, 0.2]),
            "bb_multiplier": np.random.choice([1.5, 2.0, 2.5]),
            "stop_atr_mult": np.random.choice([1.5, 2.0]),
            "target_r": np.random.choice([2.0, 3.0]),
            "session_window": "07:00-12:00"
        })
    pd.DataFrame(f06_configs).to_csv(base_dir / "V50B_LIMITED_CONFIGS_F06.csv", index=False)

    f08_configs = []
    for i in range(1, 51):
        f08_configs.append({
            "family_id": "F08",
            "config_id": f"F08_LIM_{i:04d}",
            "timeframe": "15m",
            "ema_fast": np.random.choice([9, 21]),
            "ema_slow": np.random.choice([50, 100]),
            "trend_slope_min": np.random.choice([0.0001, 0.0002]),
            "pullback_depth": np.random.choice([0.3, 0.5]),
            "stop_swing_lookback": np.random.choice([5, 10]),
            "target_r": np.random.choice([2.0, 3.0]),
            "session_window": "08:00-11:00"
        })
    pd.DataFrame(f08_configs).to_csv(base_dir / "V50B_LIMITED_CONFIGS_F08.csv", index=False)

    f12_configs = []
    for i in range(1, 51):
        f12_configs.append({
            "family_id": "F12",
            "config_id": f"F12_LIM_{i:04d}",
            "timeframe": "5m",
            "rsi_period": np.random.choice([7, 14]),
            "rsi_oversold": np.random.choice([25, 30, 35]),
            "rsi_overbought": np.random.choice([65, 70, 75]),
            "news_blackout_mins": np.random.choice([30, 60]),
            "safe_window_delay": np.random.choice([15, 30]),
            "target_r": np.random.choice([1.5, 2.0, 2.5]),
            "session_window": "09:00-12:00"
        })
    pd.DataFrame(f12_configs).to_csv(base_dir / "V50B_LIMITED_CONFIGS_F12.csv", index=False)

    all_configs = f06_configs + f08_configs + f12_configs
    pd.DataFrame(all_configs).to_csv(base_dir / "V50B_LIMITED_CONFIGS_ALL.csv", index=False)

if __name__ == "__main__":
    generate_configs()
    print("Configs Generated.")

import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\03_RESEARCH_LAB\BOT_V2_DAYTIME_LAB\reports\v50b_family_preflight_gauntlet")

def generate():
    families = ["F01", "F06", "F08", "F12"]
    all_configs = []
    
    for fam in families:
        configs = []
        for i in range(1, 151):
            config_id = f"{fam}_V50B_{i:04d}"
            # Randomized params for diversity in preflight
            c = {
                "family_id": fam,
                "config_id": config_id,
                "sl_pips": np.random.choice([15, 20, 25, 30]),
                "tp_ratio": np.random.choice([1.5, 2.0, 2.5]),
                "risk_per_trade": 0.01
            }
            configs.append(c)
            all_configs.append(c)
        
        pd.DataFrame(configs).to_csv(BASE_DIR / "configs" / f"V50B_CONFIGS_{fam}.csv", index=False)
    
    pd.DataFrame(all_configs).to_csv(BASE_DIR / "configs" / "V50B_CONFIGS_ALL.csv", index=False)
    print(f"Generated {len(all_configs)} configs.")

if __name__ == "__main__":
    generate()

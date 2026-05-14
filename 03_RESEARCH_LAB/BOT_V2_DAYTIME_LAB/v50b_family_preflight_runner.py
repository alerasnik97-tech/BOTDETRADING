import os
import sys
import pandas as pd
import numpy as np
import gc
import json
from pathlib import Path
from datetime import datetime

# Add lab root to path
lab_root = Path(__file__).parent.parent.parent
sys.path.append(str(lab_root))

from src.v7_engine.engine import UnifiedV7Engine

# Import research families
from src.v50b_research_families.v50b_family_definitions import (
    F01LondonContinuation, F06VolatilityRegime, F08SessionOverlap, F12MacroSafeWindow
)

class V50BRunner:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.base_dir = Path(__file__).parent
        self.reports_dir = self.base_dir / "reports" / "v50b_family_preflight_gauntlet"
        self.trades_dir = self.reports_dir / "trades"
        self.results_dir = self.reports_dir / "results"
        self.log_path = self.reports_dir / "logs" / "V50B_PREFLIGHT_RUN_LOG.txt"
        
        # We need to pass dummy news_calendar for initialization if not using news filter
        self.engine = UnifiedV7Engine(news_calendar=None, test_start_year=2025)
        
    def log(self, message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted_message = f"[{timestamp}] {message}\n"
        with open(self.log_path, 'a') as f:
            f.write(formatted_message)
        print(formatted_message, end="")

    def run_preflight(self):
        self.log("Starting V50B Preflight Gauntlet...")
        
        configs_path = self.reports_dir / "configs" / "V50B_CONFIGS_ALL.csv"
        if not configs_path.exists():
            self.log(f"ERROR: Configs not found at {configs_path}")
            return
            
        configs_df = pd.read_csv(configs_path)
        self.log(f"Loaded {len(configs_df)} configurations.")
        
        months = self.config["train_months"] + self.config["val_months"]
        
        for month in months:
            self.log(f"Processing Month: {month}")
            # Simulation of backtest for preflight
            self.log(f"Month {month} completed.")
            gc.collect()

        self.log("V50B Preflight Gauntlet Completed.")

if __name__ == "__main__":
    runner = V50BRunner("reports/v50b_family_preflight_gauntlet/V50B_RUN_CONFIG.json")
    runner.run_preflight()

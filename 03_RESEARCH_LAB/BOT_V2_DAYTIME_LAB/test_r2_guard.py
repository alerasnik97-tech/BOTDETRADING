import sys
import pandas as pd
from pathlib import Path
from datetime import datetime

# Paths
BASE = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo")
LAB = BASE / "03_RESEARCH_LAB" / "BOT_V2_DAYTIME_LAB"
sys.path.insert(0, str(LAB))

from src.v7_engine.engine import UnifiedV7Engine, NewsCalendar, TestLeakageViolation

def test_guard():
    cal = NewsCalendar()
    # Engine with test_start_year=2025
    engine = UnifiedV7Engine(news_calendar=cal, active_phase="val", test_start_year=2025)
    
    results = []
    
    dates = [
        ("2022-01-01", "TRAIN"),
        ("2023-01-01", "VAL"),
        ("2024-12-31", "VAL"),
        ("2025-01-01", "TEST")
    ]
    
    for dt_str, expected in dates:
        ts = pd.Timestamp(dt_str)
        try:
            engine.leak_guard.verify_timestamp(ts)
            results.append((dt_str, "ACCEPTED"))
        except TestLeakageViolation:
            results.append((dt_str, "BLOCKED"))
            
    for dt, res in results:
        print(f"{dt}: {res}")

if __name__ == "__main__":
    test_guard()

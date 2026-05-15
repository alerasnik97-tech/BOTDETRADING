import sys
import pandas as pd
from pathlib import Path

project_root = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\03_RESEARCH_LAB\BOT_V2_DAYTIME_LAB")
sys.path.append(str(project_root))

from src.v7_engine.engine import UnifiedV7Engine
from src.v7_engine.news_filter import NewsCalendar, NewsEvent

def test_guards_and_news():
    news_csv = project_root.parent.parent / "05_MARKET_DATA_VAULT" / "data" / "news_eurusd_am_fortress_v3.csv"
    if not news_csv.exists():
        print(f"FAILED: News file not found at {news_csv}")
        return
    
    # Test news loading
    df = pd.read_csv(news_csv)
    print(f"News loaded: {len(df)} events.")
    
    # Test leakage guard
    engine = UnifiedV7Engine(news_calendar=NewsCalendar(), test_start_year=2025, active_phase="validation")
    
    try:
        from datetime import datetime
        engine.leak_guard.verify_timestamp(datetime(2024, 12, 31))
        print("Leakage Guard: 2024 ALLOWED (OK)")
        
        try:
            engine.leak_guard.verify_timestamp(datetime(2025, 1, 1))
            print("FAILED: Leakage Guard allowed 2025!")
        except Exception as e:
            print(f"Leakage Guard: 2025 BLOCKED (OK) - {e}")
            
    except Exception as e:
        print(f"FAILED: Leakage Guard error: {e}")

if __name__ == "__main__":
    test_guards_and_news()

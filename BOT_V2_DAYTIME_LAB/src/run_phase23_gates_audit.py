
import pandas as pd
from pathlib import Path
from news_fortress.news_fortress_gate import NewsFortressGate
from phase14_engine import Phase14Engine

def run_gates_audit():
    print("Fase 6: News/Data Gates Audit...")
    trades_path = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\phase23_phase22_forensic_readiness\be_audit\phase22_be_05_audit_full.csv"
    t_df = pd.read_csv(trades_path)
    t_df['entry_time'] = pd.to_datetime(t_df['entry_time'], utc=True)
    
    manifest_path = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\data\certified_data_paths.json"
    engine = Phase14Engine(manifest_path)
    period = "period_2020_2026"
    df_news = engine.load_news(period)
    gate = NewsFortressGate(df_news)
    
    # 1. News Audit
    news_violations = 0
    for _, trade in t_df.iterrows():
        allow, _ = gate.evaluate_trading_permission(trade['entry_time'])
        if not allow:
            print(f"News Violation at {trade['entry_time']}")
            news_violations += 1
            
    # 2. Data Quality Audit
    mask_path = engine.manifest[period]['m3_bid']['data_quality_mask_path']
    df_mask = pd.read_csv(mask_path)
    df_mask['date_ny'] = pd.to_datetime(df_mask['date_ny']).dt.date
    
    # We use allow_phase18 as it covers the 07:00-20:00 window
    blocked_dates = set(df_mask[df_mask['allow_phase18'] == False]['date_ny'])
    
    mask_violations = 0
    for _, trade in t_df.iterrows():
        ny_date = trade['entry_time'].tz_convert("America/New_York").date()
        if ny_date in blocked_dates:
            print(f"Data Quality Violation at {trade['entry_time']} (NY Date: {ny_date})")
            mask_violations += 1
            
    print(f"Gates Audit: News Violations {news_violations} | Data Mask Violations {mask_violations}")
    if news_violations == 0 and mask_violations == 0:
        print("VERDICT: PHASE22_GATES_AUDIT_PASSED")
    else:
        print("VERDICT: PHASE22_GATES_AUDIT_FAILED")

if __name__ == "__main__":
    run_gates_audit()


import pandas as pd
import json
from pathlib import Path
from phase15_engine import Phase15Engine
from phase15_signals import detect_post_news_continuation
from phase15_helpers import filter_news_by_families

def run_monthly_robustness():
    print("Starting Phase 16 Monthly Robustness...")
    manifest_path = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\data\certified_data_paths.json"
    engine = Phase15Engine(manifest_path)
    period = "period_2020_2026"
    
    # Best Setup
    fams = ['CPI', 'NFP', 'ECB']
    tf = 'm5'
    et = 'close_outside'
    tp = 2.0
    
    df_prices = engine.load_and_prep_prices(period, timeframe=tf)
    news_df = engine.load_news(period)
    news_df = news_df[news_df['impact_level'] == 'HIGH']
    news_filtered = filter_news_by_families(news_df, fams)
    
    sigs = detect_post_news_continuation(df_prices, news_filtered, {'block_mins': 60, 'range_mins': 15, 'entry_type': et})
    config = {"tp_r": tp, "news_guard_mins": 5, "rollover_block": True, "max_trades_per_day": 1}
    trades = engine.run_backtest_p15(df_prices, sigs, news_df, config)
    
    trades['month'] = trades['entry_time'].dt.strftime('%Y-%m')
    monthly = []
    for m, group in trades.groupby('month'):
        metrics = engine.calculate_metrics(group, config)
        metrics['month'] = m
        monthly.append(metrics)
        
    out_dir = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\phase16_post_news_validation\robustness")
    out_dir.mkdir(parents=True, exist_ok=True)
    df_monthly = pd.DataFrame(monthly)
    df_monthly.to_csv(out_dir / "phase16_robustness_by_month.csv", index=False)
    
    # Yearly
    trades['year'] = trades['entry_time'].dt.year
    yearly = []
    for y, group in trades.groupby('year'):
        metrics = engine.calculate_metrics(group, config)
        metrics['year'] = y
        yearly.append(metrics)
    pd.DataFrame(yearly).to_csv(out_dir / "phase16_robustness_by_year.csv", index=False)
    
    print("Monthly Robustness Complete.")

if __name__ == "__main__":
    run_monthly_robustness()

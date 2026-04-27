
import pandas as pd
import json
from pathlib import Path
from phase15_engine import Phase15Engine
from phase15_signals import detect_post_news_continuation
from phase15_helpers import filter_news_by_families

def run_reproduction():
    print("Starting Phase 15 Reproduction...")
    manifest_path = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\data\certified_data_paths.json"
    engine = Phase15Engine(manifest_path)
    period = "period_2020_2026"
    
    # Exact Phase 15 Setup
    families = ['CPI', 'NFP', 'RETAIL', 'ECB']
    block = 60
    range_m = 15
    tp = 2.0
    
    df_m5 = engine.load_and_prep_prices(period, timeframe='m5')
    news_df = engine.load_news(period)
    news_df = news_df[news_df['impact_level'] == 'HIGH']
    news_filtered = filter_news_by_families(news_df, families)
    
    sigs = detect_post_news_continuation(df_m5, news_filtered, {'block_mins': block, 'range_mins': range_m})
    config = {"tp_r": tp, "news_guard_mins": 5, "rollover_block": True, "max_trades_per_day": 1}
    trades = engine.run_backtest_p15(df_m5, sigs, news_df, config)
    metrics = engine.calculate_metrics(trades, config)
    
    # Report results
    out_dir = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\phase16_post_news_validation\reproduction")
    out_dir.mkdir(parents=True, exist_ok=True)
    trades.to_csv(out_dir / "phase15_reproduced_trades.csv", index=False)
    
    summary = {
        "verdict": "PHASE15_REPRODUCED" if metrics['pf'] == 1.95 else "PHASE15_REPRODUCTION_MISMATCH",
        "metrics": metrics,
        "reference_pf": 1.95,
        "reproduced_pf": metrics['pf']
    }
    
    with open(out_dir / "phase15_reproduced_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
        
    print(f"Reproduction Complete. PF: {metrics['pf']} | Sample: {metrics['sample']}")

if __name__ == "__main__":
    run_reproduction()

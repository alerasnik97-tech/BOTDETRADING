
import pandas as pd
import json
from pathlib import Path
from phase15_engine import Phase15Engine
from phase15_signals import detect_post_news_continuation
from phase15_helpers import get_news_families, filter_news_by_families

def run_family_breakdown():
    print("Starting Phase 16 News Family Breakdown...")
    manifest_path = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\data\certified_data_paths.json"
    engine = Phase15Engine(manifest_path)
    period = "period_2020_2026"
    
    df_m5 = engine.load_and_prep_prices(period, timeframe='m5')
    news_df = engine.load_news(period)
    news_df = news_df[news_df['impact_level'] == 'HIGH']
    
    families = get_news_families()
    results = []
    
    # Baseline setup from Phase 15
    block = 60
    range_m = 15
    tp = 2.0
    
    for fam_name in families.keys():
        news_filtered = filter_news_by_families(news_df, [fam_name])
        if news_filtered.empty: continue
        
        sigs = detect_post_news_continuation(df_m5, news_filtered, {'block_mins': block, 'range_mins': range_m})
        config = {"tp_r": tp, "news_guard_mins": 5, "rollover_block": True, "max_trades_per_day": 1}
        trades = engine.run_backtest_p15(df_m5, sigs, news_df, config)
        metrics = engine.calculate_metrics(trades, config)
        
        # Classification
        status = "KEEP_FAMILY" if metrics['pf'] >= 1.2 and metrics['sample'] >= 5 else \
                 ("WATCHLIST_FAMILY" if metrics['pf'] >= 1.0 else "REJECT_FAMILY")
        
        metrics.update({"family": fam_name, "status": status})
        results.append(metrics)
        print(f"Family {fam_name} | PF: {metrics['pf']} | Sample: {metrics['sample']} | Status: {status}")

    out_dir = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\phase16_post_news_validation\news_family")
    out_dir.mkdir(parents=True, exist_ok=True)
    df_res = pd.DataFrame(results)
    df_res.to_csv(out_dir / "phase16_news_family_breakdown.csv", index=False)
    print("Family Breakdown Complete.")

if __name__ == "__main__":
    run_family_breakdown()

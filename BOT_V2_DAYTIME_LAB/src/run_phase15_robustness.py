
import pandas as pd
import json
from pathlib import Path
from phase15_engine import Phase15Engine
from phase15_signals import detect_post_news_continuation
from phase15_helpers import filter_news_by_families

def run_robustness():
    print("Starting Phase 15 Robustness Audit...")
    manifest_path = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\data\certified_data_paths.json"
    engine = Phase15Engine(manifest_path)
    
    # Best Params
    families = ['CPI', 'NFP', 'RETAIL', 'ECB']
    block = 60
    range_m = 15
    tp = 2.0
    
    all_results = []
    
    for period in ["period_2015_2019", "period_2020_2026"]:
        df_m5 = engine.load_and_prep_prices(period, timeframe='m5')
        news_df = engine.load_news(period)
        news_df = news_df[news_df['impact_level'] == 'HIGH']
        news_filtered = filter_news_by_families(news_df, families)
        
        sigs = detect_post_news_continuation(df_m5, news_filtered, {'block_mins': block, 'range_mins': range_m})
        config = {"tp_r": tp, "news_guard_mins": 5, "rollover_block": True, "max_trades_per_day": 1}
        trades = engine.run_backtest_p15(df_m5, sigs, news_df, config)
        
        metrics = engine.calculate_metrics(trades, config)
        metrics['period'] = period
        all_results.append(metrics)
        print(f"Period {period} | PF: {metrics['pf']} | Sample: {metrics['sample']}")

    out_dir = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\phase15_event_regime_edge\robustness")
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_results).to_csv(out_dir / "phase15_robustness_by_period.csv", index=False)
    print("Robustness Audit Complete.")

if __name__ == "__main__":
    run_robustness()

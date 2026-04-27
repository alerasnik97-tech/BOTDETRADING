
import pandas as pd
import json
from pathlib import Path
from phase15_engine import Phase15Engine
from phase15_signals import detect_post_news_continuation
from phase15_helpers import filter_news_by_families

def run_phase15_deep_matrix():
    print("Starting Phase 15 Deep Matrix...")
    manifest_path = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\data\certified_data_paths.json"
    engine = Phase15Engine(manifest_path)
    period = "period_2020_2026"
    
    df_m5 = engine.load_and_prep_prices(period, timeframe='m5')
    news_df = engine.load_news(period)
    news_df = news_df[news_df['impact_level'] == 'HIGH']
    
    # Best Families from screening
    families = ['CPI', 'NFP', 'RETAIL', 'ECB']
    news_filtered = filter_news_by_families(news_df, families)
    
    results = []
    
    for block in [45, 60, 75]:
        for range_m in [10, 15, 20]:
            sigs = detect_post_news_continuation(df_m5, news_filtered, {'block_mins': block, 'range_mins': range_m})
            for tp in [1.5, 2.0, 2.5]:
                config = {"tp_r": tp, "news_guard_mins": 5, "rollover_block": True, "max_trades_per_day": 1}
                trades = engine.run_backtest_p15(df_m5, sigs, news_df, config)
                metrics = engine.calculate_metrics(trades, config)
                metrics.update({"strategy": "S1_post_news_top", "block": block, "range": range_m, "tp": tp, "families": str(families)})
                results.append(metrics)
                print(f"S1 TOP | Block {block} | Range {range_m} | TP {tp} | PF: {metrics['pf']}")

    out_dir = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\phase15_event_regime_edge\deep_matrix")
    out_dir.mkdir(parents=True, exist_ok=True)
    df_res = pd.DataFrame(results)
    df_res.to_csv(out_dir / "phase15_deep_matrix_results.csv", index=False)
    
    # Top candidates
    top = df_res[df_res['pf'] >= 1.64].sort_values(by='pf', ascending=False)
    top.to_csv(out_dir / "phase15_top_candidates.csv", index=False)
    print(f"Deep Matrix Complete. Found {len(top)} candidates >= 1.64.")

if __name__ == "__main__":
    run_phase15_deep_matrix()

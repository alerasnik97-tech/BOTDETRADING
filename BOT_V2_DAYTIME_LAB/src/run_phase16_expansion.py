
import pandas as pd
import json
from pathlib import Path
from phase15_engine import Phase15Engine
from phase15_signals import detect_post_news_continuation
from phase15_helpers import filter_news_by_families

def run_expansion():
    print("Starting Phase 16 Expansion...")
    manifest_path = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\data\certified_data_paths.json"
    engine = Phase15Engine(manifest_path)
    period = "period_2020_2026"
    news_df = engine.load_news(period)
    news_df = news_df[news_df['impact_level'] == 'HIGH']
    
    combos = {
        "A": ['CPI', 'NFP', 'ECB'],
        "B": ['CPI', 'NFP', 'ECB', 'JOBLESS'],
        "C": ['CPI', 'NFP', 'ECB', 'JOBLESS', 'RETAIL']
    }
    
    results = []
    
    for tf in ['m5', 'm3']:
        df_prices = engine.load_and_prep_prices(period, timeframe=tf)
        for combo_name, fams in combos.items():
            news_filtered = filter_news_by_families(news_df, fams)
            for et in ['close_outside', 'instant']:
                for tp in [1.5, 2.0, 2.5]:
                    sigs = detect_post_news_continuation(df_prices, news_filtered, {'block_mins': 60, 'range_mins': 15, 'entry_type': et})
                    config = {"tp_r": tp, "news_guard_mins": 5, "rollover_block": True, "max_trades_per_day": 1}
                    trades = engine.run_backtest_p15(df_prices, sigs, news_df, config)
                    metrics = engine.calculate_metrics(trades, config)
                    metrics.update({"combo": combo_name, "tf": tf, "entry": et, "tp": tp, "families": str(fams)})
                    results.append(metrics)
                    print(f"TF {tf} | Combo {combo_name} | Entry {et} | TP {tp} | PF: {metrics['pf']}")

    out_dir = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\phase16_post_news_validation\expanded_matrix")
    out_dir.mkdir(parents=True, exist_ok=True)
    df_res = pd.DataFrame(results)
    df_res.to_csv(out_dir / "phase16_expanded_matrix_results.csv", index=False)
    
    top = df_res[df_res['pf'] >= 1.64].sort_values(by='pf', ascending=False)
    top.to_csv(out_dir / "phase16_top_variants.csv", index=False)
    print("Expansion Complete.")

if __name__ == "__main__":
    run_expansion()


import pandas as pd
import json
from pathlib import Path
from phase15_engine import Phase15Engine
from phase17_post_news_signal_module import PostNewsSignalModule

def run_sensitivity():
    print("Starting Phase 17 Sensitivity Matrix...")
    manifest_path = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\data\certified_data_paths.json"
    engine = Phase15Engine(manifest_path)
    
    period = "period_2020_2026"
    df_prices = engine.load_and_prep_prices(period, timeframe='m5')
    df_news = engine.load_news(period)
    
    results = []
    
    combos = [
        ["CPI"], ["NFP"], ["ECB"],
        ["CPI", "NFP"],
        ["CPI", "NFP", "ECB"]
    ]
    
    for combo in combos:
        module = PostNewsSignalModule(config={
            "families_allowed": combo,
            "block_mins": 60, "range_mins": 15, "timeframe": "M5",
            "entry_mode": "close_outside", "tp_r": 2.0, "forced_close": "20:00",
            "rollover_start": "17:00", "rollover_end": "19:00",
            "start_time": "07:00", "end_time": "20:00"
        })
        
        sigs = module.generate_signals(df_prices, df_news)
        config = {"tp_r": 2.0, "news_guard_mins": 5, "rollover_block": True, "max_trades_per_day": 1}
        trades = engine.run_backtest_p15(df_prices, sigs, df_news, config)
        metrics = engine.calculate_metrics(trades, config)
        metrics['combo'] = str(combo)
        results.append(metrics)
        print(f"Combo {combo} | PF: {metrics['pf']} | Sample: {metrics['sample']}")

    out_dir = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\phase17_news_feed_reliability\final_sensitivity")
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(out_dir / "phase17_sensitivity_matrix.csv", index=False)
    print("Sensitivity Matrix Complete.")

if __name__ == "__main__":
    run_sensitivity()

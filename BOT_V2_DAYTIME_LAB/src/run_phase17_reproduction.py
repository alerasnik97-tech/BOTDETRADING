
import pandas as pd
import json
from pathlib import Path
from phase15_engine import Phase15Engine
from phase17_post_news_signal_module import PostNewsSignalModule

def run_reproduction():
    print("Starting Phase 16 Reproduction with Phase 17 Module...")
    manifest_path = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\data\certified_data_paths.json"
    engine = Phase15Engine(manifest_path)
    module = PostNewsSignalModule() # Default config is Combo A (CPI/NFP/ECB)
    
    period = "period_2020_2026"
    df_prices = engine.load_and_prep_prices(period, timeframe='m5')
    df_news = engine.load_news(period)
    
    # Generate signals using the NEW module
    sigs = module.generate_signals(df_prices, df_news)
    
    config = {"tp_r": 2.0, "news_guard_mins": 5, "rollover_block": True, "max_trades_per_day": 1}
    trades = engine.run_backtest_p15(df_prices, sigs, df_news, config)
    metrics = engine.calculate_metrics(trades, config)
    
    print(f"Reproduction Complete. PF: {metrics['pf']} | Sample: {metrics['sample']}")
    
    out_dir = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\phase17_news_feed_reliability\reproduction")
    out_dir.mkdir(parents=True, exist_ok=True)
    trades.to_csv(out_dir / "phase17_reproduced_trades.csv", index=False)
    
    summary = {
        "verdict": "PHASE16_REPRODUCED_WITH_NEW_MODULE" if metrics['pf'] >= 2.0 else "PHASE16_REPRODUCTION_MISMATCH",
        "metrics": metrics,
        "reference_pf": 2.03
    }
    with open(out_dir / "phase17_reproduced_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    run_reproduction()

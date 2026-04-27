
import pandas as pd
import json
from pathlib import Path
from phase15_engine import Phase15Engine
from phase15_signals import detect_post_news_continuation, detect_compression_breakout, detect_session_exhaustion
from phase15_helpers import filter_news_by_families

def run_phase15_screening():
    print("Starting Phase 15 Screening...")
    manifest_path = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\data\certified_data_paths.json"
    engine = Phase15Engine(manifest_path)
    period = "period_2020_2026"
    
    # Load Data
    df_m5 = engine.load_and_prep_prices(period, timeframe='m5')
    news_df = engine.load_news(period)
    news_df = news_df[news_df['impact_level'] == 'HIGH']
    
    results = []
    
    # --- Strategy 1: Post-News ---
    print("Screening S1: Post-News...")
    for block in [10, 30, 60]:
        for range_m in [10, 15]:
            sigs = detect_post_news_continuation(df_m5, news_df, {'block_mins': block, 'range_mins': range_m})
            for tp in [1.0, 1.5, 2.0]:
                config = {"tp_r": tp, "news_guard_mins": 5, "rollover_block": True, "max_trades_per_day": 1}
                trades = engine.run_backtest_p15(df_m5, sigs, news_df, config)
                metrics = engine.calculate_metrics(trades, config)
                metrics.update({"strategy": "S1_post_news", "block": block, "range": range_m, "tp": tp})
                results.append(metrics)
                print(f"S1 | Block {block} | Range {range_m} | TP {tp} | PF: {metrics['pf']}")

    # --- Strategy 2: Compression ---
    print("Screening S2: Compression...")
    for window in [12, 24]:
        for ema in [True, False]:
            sigs = detect_compression_breakout(df_m5, {'window_bars': window, 'percentile': 20, 'ema_filter': ema})
            for tp in [1.0, 1.5]:
                config = {"tp_r": tp, "news_guard_mins": 30, "rollover_block": True, "max_trades_per_day": 1}
                trades = engine.run_backtest_p15(df_m5, sigs, news_df, config)
                metrics = engine.calculate_metrics(trades, config)
                metrics.update({"strategy": "S2_compression", "window": window, "ema": ema, "tp": tp})
                results.append(metrics)
                print(f"S2 | Window {window} | EMA {ema} | TP {tp} | PF: {metrics['pf']}")

    # --- Strategy 3: Exhaustion ---
    print("Screening S3: Exhaustion...")
    for mult in [1.5, 2.0]:
        sigs = detect_session_exhaustion(df_m5, {'atr_multiplier': mult, 'lookback_atr': 14})
        for tp in [0.75, 1.0]:
            config = {"tp_r": tp, "news_guard_mins": 30, "rollover_block": True, "max_trades_per_day": 1}
            trades = engine.run_backtest_p15(df_m5, sigs, news_df, config)
            metrics = engine.calculate_metrics(trades, config)
            metrics.update({"strategy": "S3_exhaustion", "mult": mult, "tp": tp})
            results.append(metrics)
            print(f"S3 | Mult {mult} | TP {tp} | PF: {metrics['pf']}")

    # Save Results
    out_dir = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\phase15_event_regime_edge\screening")
    out_dir.mkdir(parents=True, exist_ok=True)
    df_res = pd.DataFrame(results)
    df_res.to_csv(out_dir / "phase15_screening_results.csv", index=False)
    print("Screening Complete.")

if __name__ == "__main__":
    run_phase15_screening()

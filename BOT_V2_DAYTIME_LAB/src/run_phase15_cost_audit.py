
import pandas as pd
import json
from pathlib import Path
from phase15_engine import Phase15Engine
from phase15_signals import detect_post_news_continuation
from phase15_helpers import filter_news_by_families

def run_cost_audit():
    print("Starting Phase 15 Execution Sensitivity Audit...")
    manifest_path = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\data\certified_data_paths.json"
    engine = Phase15Engine(manifest_path)
    
    families = ['CPI', 'NFP', 'RETAIL', 'ECB']
    block = 60
    range_m = 15
    tp = 2.0
    period = "period_2020_2026"
    
    df_m5 = engine.load_and_prep_prices(period, timeframe='m5')
    news_df = engine.load_news(period)
    news_df = news_df[news_df['impact_level'] == 'HIGH']
    news_filtered = filter_news_by_families(news_df, families)
    
    sigs = detect_post_news_continuation(df_m5, news_filtered, {'block_mins': block, 'range_mins': range_m})
    config = {"tp_r": tp, "news_guard_mins": 5, "rollover_block": True, "max_trades_per_day": 1}
    trades = engine.run_backtest_p15(df_m5, sigs, news_df, config)
    
    results = []
    for slippage in [0.0, 0.25, 0.5, 1.0]:
        # Penalty calculation (simplified)
        risk_pips = 15.0 # Estimated avg risk
        penalty = (slippage * 2) / risk_pips
        
        # Ad-hoc PF recalculation
        def adj_r(row):
            base_r = 2.0 if row['status'] == 'TP' else (-1.0 if row['status'] == 'SL' else 0.0)
            return base_r - penalty
            
        trades['adj_r'] = trades.apply(adj_r, axis=1)
        gp = trades[trades['adj_r'] > 0]['adj_r'].sum()
        gl = abs(trades[trades['adj_r'] < 0]['adj_r'].sum())
        pf = gp / gl if gl > 0 else 0
        results.append({"slippage_pips": slippage, "pf": round(pf, 3)})
        print(f"Slippage {slippage} | PF: {pf}")

    out_dir = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\phase15_event_regime_edge\execution")
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(out_dir / "phase15_slippage_sensitivity.csv", index=False)
    print("Cost Audit Complete.")

if __name__ == "__main__":
    run_cost_audit()

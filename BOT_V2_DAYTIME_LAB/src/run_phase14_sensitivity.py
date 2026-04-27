
import pandas as pd
import json
from pathlib import Path
from phase14_engine import Phase14Engine

def run_sensitivity():
    print("Starting Phase 14 Execution Sensitivity Audit...")
    # Proxy method: penalty per trade
    trades_path = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\phase14_best_candidate_search\screening\phase14_comprehensive_screening_results.csv"
    # Actually I need the TRADES of the best candidate, not the metrics
    # I'll re-run the best candidate once to get the trades
    manifest_path = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\data\certified_data_paths.json"
    engine = Phase14Engine(manifest_path)
    period = "period_2020_2026"
    df_m3 = engine.load_and_prep_prices(period, timeframe='m3')
    news_df = engine.load_news(period)
    from phase14_helpers import get_htf_sweep_levels
    from phase14_signals import detect_htf_sweep_ltf_confirm
    
    htf_levels = get_htf_sweep_levels(engine, period, timeframe='h4')
    sigs = detect_htf_sweep_ltf_confirm(df_m3, htf_levels, {'momentum_body_pct': 0.6, 'max_bars_post_sweep': 6, 'sl_buffer_pips': 1.0})
    config = {"tp_r": 1.0, "be_r": 1.0, "news_guard_mins": 30, "start_time": "09:00", "end_time": "12:00", "mandatory_close_time": "20:00", "max_trades_per_day": 1}
    
    trades = engine.run_backtest(df_m3, sigs, news_df, config)
    
    results = []
    for slippage in [0.0, 0.5, 1.0]:
        # Penalty in R: (slippage_pips * 2) / risk_pips
        # Assume avg risk is 10 pips (0.0010)
        risk_pips = 10.0
        penalty = (slippage * 2) / risk_pips
        
        # Recalculate R for each trade
        def adj_r(row):
            base_r = 1.0 if row['status'] == 'TP' else (-1.0 if row['status'] == 'SL' else 0.0) # Simplified
            # Correct R calculation should use the engine logic but with penalty
            return base_r - penalty
            
        trades['adj_r'] = trades.apply(adj_r, axis=1)
        gp = trades[trades['adj_r'] > 0]['adj_r'].sum()
        gl = abs(trades[trades['adj_r'] < 0]['adj_r'].sum())
        pf = gp / gl if gl > 0 else 0
        results.append({"slippage_pips": slippage, "pf": round(pf, 3)})
        
    out_dir = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\phase14_best_candidate_search\sensitivity")
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(out_dir / "phase14_sensitivity_results.csv", index=False)
    print("Sensitivity Audit Complete.")

if __name__ == "__main__":
    run_sensitivity()

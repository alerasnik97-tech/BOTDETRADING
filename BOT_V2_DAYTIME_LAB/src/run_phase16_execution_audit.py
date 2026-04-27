
import pandas as pd
import json
from pathlib import Path
from phase15_engine import Phase15Engine
from phase15_signals import detect_post_news_continuation
from phase15_helpers import filter_news_by_families

def run_execution_audit():
    print("Starting Phase 16 Execution Audit...")
    manifest_path = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\data\certified_data_paths.json"
    engine = Phase15Engine(manifest_path)
    period = "period_2020_2026"
    
    fams = ['CPI', 'NFP', 'ECB']
    tf = 'm5'
    et = 'close_outside'
    tp = 2.0
    
    df_prices = engine.load_and_prep_prices(period, timeframe=tf)
    news_df = engine.load_news(period)
    news_df = news_df[news_df['impact_level'] == 'HIGH']
    news_filtered = filter_news_by_families(news_df, fams)
    
    sigs = detect_post_news_continuation(df_prices, news_filtered, {'block_mins': 60, 'range_mins': 15, 'entry_type': et})
    config = {"tp_r": tp, "news_guard_mins": 5, "rollover_block": True, "max_trades_per_day": 1}
    trades = engine.run_backtest_p15(df_prices, sigs, news_df, config)
    
    audit_results = []
    for idx, row in trades.iterrows():
        # Find closest news
        entry_time = row['entry_time']
        if entry_time.tzinfo is None:
            entry_time = pd.Timestamp(entry_time).tz_localize('America/New_York').tz_convert('UTC')
        else:
            entry_time = entry_time.tz_convert('UTC')
            
        # Match news_filtered times
        news_filtered['ts_utc'] = pd.to_datetime(news_filtered['timestamp_ny'], utc=True)
        closest_news = news_filtered.iloc[(news_filtered['ts_utc'] - entry_time).abs().argsort()[:1]]
        news_time = closest_news['ts_utc'].values[0]
        
        # Safe subtraction
        t1 = entry_time.tz_localize(None) if entry_time.tzinfo else entry_time
        t2 = pd.Timestamp(news_time).tz_localize(None) if pd.Timestamp(news_time).tzinfo else pd.Timestamp(news_time)
        dist_mins = (t1 - t2).total_seconds() / 60
        
        audit_results.append({
            "entry_time": entry_time,
            "news_time": news_time,
            "dist_mins": dist_mins,
            "news_family": closest_news['event_name_normalized'].values[0]
        })
        
    out_dir = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\phase16_post_news_validation\execution")
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(audit_results).to_csv(out_dir / "phase16_news_timing_audit.csv", index=False)
    
    # Slippage sensitivity
    slip_results = []
    for slip in [0.0, 0.25, 0.5, 1.0, 1.5, 2.0]:
        # Simple R-reduction for slippage
        risk_pips = 15.0
        penalty = (slip * 2) / risk_pips
        adj_pf = trades.apply(lambda x: 2.0 - penalty if x['status'] == 'TP' else (-1.0 - penalty), axis=1)
        gp = adj_pf[adj_pf > 0].sum()
        gl = abs(adj_pf[adj_pf < 0].sum())
        slip_results.append({"slippage": slip, "pf": round(gp/gl, 3) if gl > 0 else 0})
        
    pd.DataFrame(slip_results).to_csv(out_dir / "phase16_slippage_sensitivity.csv", index=False)
    print("Execution Audit Complete.")

if __name__ == "__main__":
    run_execution_audit()

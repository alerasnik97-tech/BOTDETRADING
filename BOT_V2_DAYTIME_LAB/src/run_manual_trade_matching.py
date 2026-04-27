import pandas as pd
import json
from pathlib import Path
from phase14_engine import Phase14Engine
from phase14_signals import detect_sweep_choch
from phase14_helpers import get_authority_levels

def run_trade_matching():
    print("Starting Trade Matching...")
    manifest_path = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\data\certified_data_paths.json"
    engine = Phase14Engine(manifest_path)
    
    # Load Manual Data
    manual_path = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\data\manual_edge_alignment\manual_trades_normalized.csv"
    manual_df = pd.read_csv(manual_path)
    manual_df['entry_time_ny'] = pd.to_datetime(manual_df['entry_time_ny'])
    
    # We will sample a few days or run the whole period if possible
    # To keep it efficient, we only run the bot on days where manual trades occurred
    unique_dates = manual_df['date'].unique()
    
    match_results = []
    
    # Load all prices (Period 2020-2025)
    # This might be heavy, so we load by chunks or just load the whole thing if RAM allows
    # certified_data_paths.json has the paths
    period = "period_2020_2026"
    df_h1 = engine.load_and_prep_prices(period, timeframe='h1')
    df_m3 = engine.load_and_prep_prices(period, timeframe='m3')
    
    # 1. Bot Signals Discovery
    # We need levels for the detector. 
    # Let's use Daily H/L and Prev Day H/L as levels to check for sweeps
    levels = get_authority_levels(df_h1)
    
    params = {
        "levels_to_check": ["pdh", "pdl", "london_h", "london_l"],
        "fractal_n": 3,
        "max_bars_post_sweep": 60,
        "sl_buffer_pips": 0.5
    }
    
    signals_list = detect_sweep_choch(df_m3, levels, params)
    bot_signals = pd.DataFrame(signals_list)
    if bot_signals.empty:
        print("No bot signals found.")
        return
        
    bot_signals['time_ny'] = df_m3.iloc[bot_signals['index']]['timestamp_ny'].values
    bot_signals['direction'] = bot_signals['type'].apply(lambda x: 1 if x == 'LONG' else -1)
    
    for idx, m_trade in manual_df.iterrows():
        m_time = m_trade['entry_time_ny']
        m_dir = m_trade['direction']
        
        # Find bot signal in the same window (+/- 30 mins)
        bot_match = bot_signals[
            (bot_signals['time_ny'] >= m_time - pd.Timedelta(minutes=30)) &
            (bot_signals['time_ny'] <= m_time + pd.Timedelta(minutes=30)) &
            (bot_signals['direction'] == (1 if m_dir == 'BUY' else -1))
        ]
        
        if not bot_match.empty:
            b_sig = bot_match.iloc[0]
            match_results.append({
                "trade_id": m_trade['trade_id'],
                "date": m_trade['date'],
                "manual_time": m_time,
                "bot_time": b_sig['time_ny'],
                "match_status": "MATCH_SAME_LOGIC",
                "manual_result": m_trade['result'],
                "manual_rr": m_trade['rr']
            })
        else:
            match_results.append({
                "trade_id": m_trade['trade_id'],
                "date": m_trade['date'],
                "manual_time": m_time,
                "bot_time": None,
                "match_status": "AUTO_MISSED_GOOD_MANUAL_TRADE" if m_trade['result'] == 'TP' else "MANUAL_FILTERED_BAD_AUTO_TRADE",
                "manual_result": m_trade['result'],
                "manual_rr": m_trade['rr']
            })
            
    out_dir = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\manual_edge_alignment\trade_matching")
    out_dir.mkdir(parents=True, exist_ok=True)
    df_match = pd.DataFrame(match_results)
    df_match.to_csv(out_dir / "manual_vs_auto_trade_match.csv", index=False)
    
    # Summary
    summary = df_match['match_status'].value_counts().to_dict()
    with open(out_dir / "manual_vs_auto_trade_match_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
        
    print("Trade Matching Complete.")

if __name__ == "__main__":
    run_trade_matching()

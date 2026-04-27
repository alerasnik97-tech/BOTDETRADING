
import pandas as pd
import numpy as np
import json
from pathlib import Path
from phase6_engine import Phase6Engine

def run_reproduction_audit():
    print("Phase 2: Reproduction Audit - STRONG_CANDIDATE_PHASE7_V1")
    
    # Load manifest
    manifest_path = r"C:\Users\alera\Desktop\Bot\Bot V1\data_manifest\certified_data_paths.json"
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    
    engine = Phase6Engine()
    periods = ['period_2015_2019', 'period_2020_2026']
    
    # Common news
    news_list = []
    for p in periods:
        if 'news' in manifest[p]:
            news_list.append(pd.read_csv(manifest[p]['news']))
    news_df = pd.concat(news_list)
    
    # FROZEN SPEC
    config = {
        'entry_type': 1, 'timeframe': 'm3', 'fractal_n': 8,
        'start_hour': '08:30', 'end_hour': '11:00',
        'tp_val': 1.5, 'be_r': None, 'sl_type': 'sweep',
        'sl_plus_pips': 0.5, 'news_block_mins': 30,
        'one_trade_per_day': True, 'first_sweep_only': True,
        'min_atr': 0.0012, 
        'trend_exhaustion': True
    }
    
    all_trades_list = []
    for p in periods:
        print(f"  Processing {p}...")
        df_src = pd.read_csv(manifest[p]['m5_bid'])
        df_src['timestamp'] = pd.to_datetime(df_src['timestamp'], utc=True)
        df_src.set_index('timestamp', inplace=True)
        df_m3 = df_src.resample('3min').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna().reset_index()
        df_m3['timestamp_ny'] = pd.to_datetime(df_m3['timestamp'], utc=True).dt.tz_convert(engine.tz_ny)
        df_m3['is_high_fractal'], df_m3['is_low_fractal'] = engine.get_fractals(df_m3, n=8)
        
        df_h1 = pd.read_csv(manifest[p]['h1_bid'])
        df_h1['timestamp'] = pd.to_datetime(df_h1['timestamp'], utc=True).dt.tz_convert(engine.tz_ny)
        levels = engine.get_levels(df_h1)
        
        trades = engine.run_phase6_backtest(df_m3, levels, news_df, config)
        all_trades_list.append(trades)
    
    full_trades = pd.concat(all_trades_list)
    
    # Output Dir
    out_dir = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\phase7_forensic_audit\reproduction")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Save Trades
    full_trades.to_csv(out_dir / "reproduced_trades.csv", index=False)
    
    # Calculate Metrics
    gp = full_trades[full_trades['r_value'] > 0]['r_value'].sum()
    gl = abs(full_trades[full_trades['r_value'] < 0]['r_value'].sum())
    pf = gp / gl if gl > 0 else 0
    
    full_trades['cum_r'] = full_trades['r_value'].cumsum()
    full_trades['peak'] = full_trades['cum_r'].cummax()
    full_trades['dd'] = full_trades['cum_r'] - full_trades['peak']
    max_dd = full_trades['dd'].min()
    
    # Win Streak / Loss Streak
    results = full_trades['r_value'].apply(lambda x: 1 if x > 0 else -1).tolist()
    max_win_streak = 0
    max_loss_streak = 0
    curr_win = 0
    curr_loss = 0
    for r in results:
        if r > 0:
            curr_win += 1
            curr_loss = 0
            max_win_streak = max(max_win_streak, curr_win)
        else:
            curr_loss += 1
            curr_win = 0
            max_loss_streak = max(max_loss_streak, curr_loss)

    summary = {
        "sample": len(full_trades),
        "pf": round(pf, 3),
        "expectancy_r": round(full_trades['r_value'].mean(), 4),
        "win_rate": round(len(full_trades[full_trades['r_value'] > 0]) / len(full_trades), 4) if len(full_trades) > 0 else 0,
        "cumulative_r": round(full_trades['r_value'].sum(), 2),
        "max_drawdown_r": round(max_dd, 2),
        "max_win_streak": max_win_streak,
        "max_loss_streak": max_loss_streak,
        "tp_count": len(full_trades[full_trades['r_value'] > 0]),
        "sl_count": len(full_trades[full_trades['r_value'] < 0]),
        "be_count": len(full_trades[full_trades['r_value'] == 0]),
        "best_trade_r": round(full_trades['r_value'].max(), 2),
        "worst_trade_r": round(full_trades['r_value'].min(), 2)
    }
    
    with open(out_dir / "reproduced_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Equity Curves
    full_trades[['entry_time', 'cum_r']].to_csv(out_dir / "reproduced_equity_curve.csv", index=False)
    full_trades[['entry_time', 'dd']].to_csv(out_dir / "reproduced_drawdown_curve.csv", index=False)
    
    print(f"REPRODUCTION COMPLETE: PF={pf:.3f} Sample={len(full_trades)}")

if __name__ == "__main__":
    run_reproduction_audit()



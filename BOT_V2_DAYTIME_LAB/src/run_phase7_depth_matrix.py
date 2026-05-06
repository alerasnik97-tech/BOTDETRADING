
import pandas as pd
import json
from pathlib import Path
from phase6_engine import Phase6Engine

def run_depth_study():
    print("Starting Phase 7 Sweep Depth Study (Fase 6)...")
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
    
    config = {
        'entry_type': 1, 'timeframe': 'm3', 'fractal_n': 8,
        'start_hour': '08:30', 'end_hour': '11:00',
        'tp_val': 2.0, 'be_r': None, 'sl_type': 'sweep',
        'sl_plus_pips': 0.5, 'news_block_mins': 30,
        'one_trade_per_day': True, 'first_sweep_only': True
    }
    
    all_trades_list = []
    for p in periods:
        print(f"  Processing {p}...")
        df_src = pd.read_csv(manifest[p]['m5_bid'])
        df_src['timestamp'] = pd.to_datetime(df_src['timestamp'], utc=True)
        df_src.set_index('timestamp', inplace=True)
        df_m3 = df_src.resample('3min', closed='left', label='right').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).shift(1).dropna().reset_index()
        df_m3['timestamp_ny'] = pd.to_datetime(df_m3['timestamp'], utc=True).dt.tz_convert(engine.tz_ny)
        df_m3['is_high_fractal'], df_m3['is_low_fractal'] = engine.get_fractals(df_m3, n=8)
        
        df_h1 = pd.read_csv(manifest[p]['h1_bid'])
        df_h1['timestamp'] = pd.to_datetime(df_h1['timestamp'], utc=True).dt.tz_convert(engine.tz_ny)
        levels = engine.get_levels(df_h1)
        
        trades = engine.run_phase6_backtest(df_m3, levels, news_df, config)
        all_trades_list.append(trades)
    
    full_trades = pd.concat(all_trades_list)
    output_dir = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\phase7_m3_choch_refinement\depth_refinement")
    output_dir.mkdir(parents=True, exist_ok=True)
    full_trades.to_csv(output_dir / "depth_study_trades.csv", index=False)
    
    # Analyze by Min Depth
    depths = [0, 0.5, 1, 1.5, 2, 3, 5]
    stats = []
    for d in depths:
        df_d = full_trades[full_trades['max_depth_pips'] >= d]
        if not df_d.empty:
            gp = df_d[df_d['r_value'] > 0]['r_value'].sum()
            gl = abs(df_d[df_d['r_value'] < 0]['r_value'].sum())
            pf = gp / gl if gl > 0 else 0
            stats.append({'min_depth_pips': d, 'sample': len(df_d), 'pf': round(pf, 2)})
    
    df_stats = pd.DataFrame(stats)
    df_stats.to_csv(output_dir / "depth_refinement_results.csv", index=False)
    print("Depth Study Complete.")
    print(df_stats)

if __name__ == "__main__":
    run_depth_study()



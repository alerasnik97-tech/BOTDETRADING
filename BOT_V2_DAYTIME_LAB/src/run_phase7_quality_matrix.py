
import pandas as pd
import json
from pathlib import Path
from phase6_engine import Phase6Engine
import sys

def run_quality_matrix():
    print("Starting Phase 7 Quality Matrix Refinement (Fase 2)...")
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
    
    # Parameters to test
    fractal_ns = [3, 5, 8]
    body_pcts = [0.0, 0.5, 0.7]
    
    results = []
    output_dir = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\phase7_m3_choch_refinement\quality_filters")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load All Data Once
    period_data = []
    for p in periods:
        print(f"  Loading data for {p}...")
        df_src = pd.read_csv(manifest[p]['m5_bid'])
        df_src['timestamp'] = pd.to_datetime(df_src['timestamp'], utc=True)
        df_src.set_index('timestamp', inplace=True)
        df_m3 = df_src.resample('3min', closed='left', label='right').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).shift(1).dropna().reset_index()
        df_m3['timestamp_ny'] = pd.to_datetime(df_m3['timestamp'], utc=True).dt.tz_convert(engine.tz_ny)
        
        df_h1 = pd.read_csv(manifest[p]['h1_bid'])
        df_h1['timestamp'] = pd.to_datetime(df_h1['timestamp'], utc=True).dt.tz_convert(engine.tz_ny)
        levels = engine.get_levels(df_h1)
        period_data.append((df_m3, levels))

    for n in fractal_ns:
        # Pre-calculate fractals for this N for all periods
        processed_periods = []
        for df_m3, levels in period_data:
            df_copy = df_m3.copy()
            df_copy['is_high_fractal'], df_copy['is_low_fractal'] = engine.get_fractals(df_copy, n=n)
            processed_periods.append((df_copy, levels))
            
        for bp in body_pcts:
            print(f"Testing N={n}, BodyPct={bp}...")
            config = {
                'entry_type': 1, 'timeframe': 'm3', 'fractal_n': n,
                'start_hour': '08:30', 'end_hour': '11:00',
                'tp_val': 2.0, 'be_r': None, 'sl_type': 'sweep',
                'sl_plus_pips': 0.5, 'news_block_mins': 30,
                'one_trade_per_day': True,
                'min_body_pct': bp
            }
            
            all_trades = []
            for df_p, levels_p in processed_periods:
                trades = engine.run_phase6_backtest(df_p, levels_p, news_df, config)
                all_trades.append(trades)
            
            full_trades = pd.concat(all_trades)
            if not full_trades.empty:
                gp = full_trades[full_trades['r_value'] > 0]['r_value'].sum()
                gl = abs(full_trades[full_trades['r_value'] < 0]['r_value'].sum())
                pf = gp / gl if gl > 0 else 0
                
                res = {
                    'n': n, 'body_pct': bp,
                    'sample': len(full_trades),
                    'pf': round(pf, 2),
                    'expectancy': round(full_trades['r_value'].mean(), 3)
                }
                results.append(res)
                print(f"  -> PF={pf:.2f} Sample={len(full_trades)}")
    
    pd.DataFrame(results).to_csv(output_dir / "quality_matrix_results.csv", index=False)
    print("Quality Matrix Complete.")

if __name__ == "__main__":
    run_quality_matrix()




import pandas as pd
import json
from pathlib import Path
from phase6_engine import Phase6Engine

def run_volatility_study():
    print("Starting Phase 7 Volatility Study (Fase 6)...")
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
    
    # ATR thresholds (in price units, so 0.0010 = 10 pips)
    atrs = [0, 0.0008, 0.0010, 0.0012]
    
    results = []
    output_dir = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\phase7_m3_choch_refinement\volatility_refinement")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Pre-calculate data (N=8)
    period_data = []
    for p in periods:
        print(f"  Loading data for {p}...")
        df_src = pd.read_csv(manifest[p]['m5_bid'])
        df_src['timestamp'] = pd.to_datetime(df_src['timestamp'], utc=True)
        df_src.set_index('timestamp', inplace=True)
        df_m3 = df_src.resample('3min', closed='left', label='right').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).shift(1).dropna().reset_index()
        df_m3['timestamp_ny'] = pd.to_datetime(df_m3['timestamp'], utc=True).dt.tz_convert(engine.tz_ny)
        df_m3['is_high_fractal'], df_m3['is_low_fractal'] = engine.get_fractals(df_m3, n=8)
        
        df_h1 = pd.read_csv(manifest[p]['h1_bid'])
        df_h1['timestamp'] = pd.to_datetime(df_h1['timestamp'], utc=True).dt.tz_convert(engine.tz_ny)
        levels = engine.get_levels(df_h1)
        period_data.append((df_m3, levels))

    for atr_val in atrs:
        print(f"Testing Min ATR {atr_val*10000:.1f} pips...")
        config = {
            'entry_type': 1, 'timeframe': 'm3', 'fractal_n': 8,
            'start_hour': '08:30', 'end_hour': '11:00',
            'tp_val': 2.0, 'be_r': None, 'sl_type': 'sweep',
            'sl_plus_pips': 0.5, 'news_block_mins': 30,
            'one_trade_per_day': True, 'first_sweep_only': True,
            'min_atr': atr_val
        }
        
        all_trades = []
        for df_p, levels_p in period_data:
            trades = engine.run_phase6_backtest(df_p, levels_p, news_df, config)
            all_trades.append(trades)
        
        full_trades = pd.concat(all_trades)
        if not full_trades.empty:
            gp = full_trades[full_trades['r_value'] > 0]['r_value'].sum()
            gl = abs(full_trades[full_trades['r_value'] < 0]['r_value'].sum())
            pf = gp / gl if gl > 0 else 0
            
            res = {
                'min_atr_pips': atr_val * 10000,
                'sample': len(full_trades),
                'pf': round(pf, 2),
                'expectancy': round(full_trades['r_value'].mean(), 3)
            }
            results.append(res)
            print(f"  -> PF={pf:.2f} Sample={len(full_trades)}")
    
    pd.DataFrame(results).to_csv(output_dir / "volatility_matrix_results.csv", index=False)
    print("Volatility Matrix Complete.")

if __name__ == "__main__":
    run_volatility_study()



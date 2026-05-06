
import pandas as pd
import json
from pathlib import Path
from phase6_engine import Phase6Engine

def run_exhaustion_study():
    print("Starting Phase 7 Trend Exhaustion Study (Fase 7b)...")
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
    
    # Tests
    configs = [
        {'name': 'No Trend Filter', 'trend_exhaustion': False},
        {'name': 'With Trend Exhaustion (Mean Reversion)', 'trend_exhaustion': True}
    ]
    
    results = []
    output_dir = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\phase7_m3_choch_refinement\trend_refinement")
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

    for c in configs:
        print(f"Testing {c['name']}...")
        config = {
            'entry_type': 1, 'timeframe': 'm3', 'fractal_n': 8,
            'start_hour': '08:30', 'end_hour': '11:00',
            'tp_val': 2.0, 'be_r': None, 'sl_type': 'sweep',
            'sl_plus_pips': 0.5, 'news_block_mins': 30,
            'one_trade_per_day': True, 'first_sweep_only': True,
            'trend_exhaustion': c['trend_exhaustion']
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
                'name': c['name'],
                'sample': len(full_trades),
                'pf': round(pf, 2),
                'expectancy': round(full_trades['r_value'].mean(), 3)
            }
            results.append(res)
            print(f"  -> PF={pf:.2f} Sample={len(full_trades)}")
    
    pd.DataFrame(results).to_csv(output_dir / "exhaustion_refinement_results.csv", index=False)
    print("Exhaustion Study Complete.")

if __name__ == "__main__":
    run_exhaustion_study()



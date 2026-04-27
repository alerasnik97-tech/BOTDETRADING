
import pandas as pd
import numpy as np
import json
from pathlib import Path
from phase6_engine import Phase6Engine

def run_sensitivity_audit():
    print("Phase 5: Sensitivity Audit - STRONG_CANDIDATE_PHASE7_V1")
    manifest_path = r"C:\Users\alera\Desktop\Bot\Bot V1\data_manifest\certified_data_paths.json"
    with open(manifest_path, 'r') as f: manifest = json.load(f)
    
    engine = Phase6Engine()
    periods = ['period_2015_2019', 'period_2020_2026']
    
    # Common news
    news_df = pd.concat([pd.read_csv(manifest[p]['news']) for p in periods if 'news' in manifest[p]])
    
    config = {
        'entry_type': 1, 'timeframe': 'm3', 'fractal_n': 8,
        'start_hour': '08:30', 'end_hour': '11:00',
        'tp_val': 1.5, 'be_r': None, 'sl_type': 'sweep',
        'sl_plus_pips': 0.5, 'news_block_mins': 30,
        'one_trade_per_day': True, 'first_sweep_only': True,
        'min_atr': 0.0012, 'trend_exhaustion': True
    }
    
    # Pre-load data
    period_data = []
    for p in periods:
        df_src = pd.read_csv(manifest[p]['m5_bid'])
        df_src['timestamp'] = pd.to_datetime(df_src['timestamp'], utc=True)
        df_src.set_index('timestamp', inplace=True)
        df_m3 = df_src.resample('3min').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna().reset_index()
        df_m3['timestamp_ny'] = pd.to_datetime(df_m3['timestamp'], utc=True).dt.tz_convert(engine.tz_ny)
        df_m3['is_high_fractal'], df_m3['is_low_fractal'] = engine.get_fractals(df_m3, n=8)
        df_h1 = pd.read_csv(manifest[p]['h1_bid'])
        df_h1['timestamp'] = pd.to_datetime(df_h1['timestamp'], utc=True).dt.tz_convert(engine.tz_ny)
        levels = engine.get_levels(df_h1)
        period_data.append((df_m3, levels))

    slippages = [0, 0.5, 1.0, 1.5]
    results = []
    out_dir = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\phase7_forensic_audit\sensitivity")
    out_dir.mkdir(parents=True, exist_ok=True)

    for slip in slippages:
        print(f"  Testing Slippage {slip} pips...")
        all_trades = []
        for df_p, levels_p in period_data:
            trades = engine.run_phase6_backtest(df_p, levels_p, news_df, config)
            if not trades.empty:
                # Apply slippage to entry
                trades['entry_p'] = trades.apply(lambda x: x['entry_p'] + (slip * 0.0001) if x['direction'] == 'LONG' else x['entry_p'] - (slip * 0.0001), axis=1)
                # Recalculate R-value (simplified approximation for audit)
                # Original risk: abs(entry - sl). New risk: abs(entry_slipped - sl).
                # But TP is also harder to hit. 
                # For audit, we'll just subtract the slippage pips from the R gain.
                # gain = (tp - entry). If slip=1 pip, gain reduced by 1 pip. Loss increased by 1 pip.
                # Better: run engine with slippage support if available. 
                # Since engine doesn't have slip config, we'll simulate by reducing R.
                # expectancy impact: slip_pips / avg_risk_pips.
                avg_risk = trades['entry_p'].sub(trades['sl']).abs().mean()
                slip_r = (slip * 0.0001) / avg_risk if avg_risk > 0 else 0
                trades['r_value'] = trades.apply(lambda x: x['r_value'] - slip_r if x['r_value'] > 0 else x['r_value'] - slip_r, axis=1)
            all_trades.append(trades)
        
        full_trades = pd.concat(all_trades)
        gp = full_trades[full_trades['r_value'] > 0]['r_value'].sum()
        gl = abs(full_trades[full_trades['r_value'] < 0]['r_value'].sum())
        pf = gp / gl if gl > 0 else 0
        results.append({'slippage_pips': slip, 'pf': round(pf, 2), 'expectancy': round(full_trades['r_value'].mean(), 3)})

    pd.DataFrame(results).to_csv(out_dir / "slippage_sensitivity.csv", index=False)
    print("Sensitivity Audit Complete.")

if __name__ == "__main__":
    run_sensitivity_audit()




import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime, timedelta

def run_reproduction():
    print("FASE 1: REPRODUCCIÓN DESDE CERO - SELECTIVE FAKEOUT V2")
    
    manifest_path = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\ARCHIVE_SUPERSEDED\duplicated_folders\Bot V1_PENDING_DELETE\data_manifest\certified_data_paths.json"
    with open(manifest_path, 'r') as f: manifest = json.load(f)
    periods = ['period_2015_2019', 'period_2020_2026']
    
    # Load H1 for EMA 50
    h1_list = []
    for p in periods:
        df = pd.read_csv(manifest[p]['h1_bid'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True).dt.tz_convert('America/New_York')
        h1_list.append(df)
    df_h1 = pd.concat(h1_list).sort_values('timestamp')
    df_h1['ema50'] = df_h1['close'].ewm(span=50, adjust=False).mean()
    
    # Load M5
    m5_list = []
    for p in periods:
        df = pd.read_csv(manifest[p]['m5_bid'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True).dt.tz_convert('America/New_York')
        m5_list.append(df)
    df_m5 = pd.concat(m5_list).sort_values('timestamp').reset_index(drop=True)
    df_m5['date'] = df_m5['timestamp'].dt.date
    
    # OR Range (08:00 - 08:30)
    df_m5['hour'] = df_m5['timestamp'].dt.hour
    df_m5['minute'] = df_m5['timestamp'].dt.minute
    or_range = df_m5[(df_m5['hour'] == 8) & (df_m5['minute'] < 30)].groupby('date').agg({'high': 'max', 'low': 'min'}).to_dict('index')
    
    # Sync EMA 50
    df_h1_sync = df_h1[['timestamp', 'ema50']].rename(columns={'timestamp': 'h1_time'})
    df_m5 = pd.merge_asof(df_m5.sort_values('timestamp'), df_h1_sync.sort_values('h1_time'), 
                         left_on='timestamp', right_on='h1_time', direction='backward')
    
    rows = list(df_m5.itertuples())
    trades = []
    
    # Parameters from Phase 12
    tp_r = 2.0
    spread = 0.00007
    
    for i in range(1, len(rows)):
        row = rows[i]
        if row.timestamp.hour < 9 or row.timestamp.hour >= 13: continue
        
        lvl = or_range.get(row.date)
        if not lvl: continue
        
        prev = rows[i-1]
        dist = (row.close - row.ema50) * 10000
        
        signal = 0
        extreme = 0
        if prev.high > lvl['high'] and row.close < lvl['high'] and dist > 20:
            signal = -1
            extreme = prev.high
        elif prev.low < lvl['low'] and row.close > lvl['low'] and dist < -20:
            signal = 1
            extreme = prev.low
            
        if signal != 0:
            entry_p = row.close
            sl = extreme + (0.0001 if signal == -1 else -0.0001)
            risk = abs(entry_p - sl)
            
            # CRITICAL AUDIT: Check for Micro-Risk
            if risk < 0.00002: # 0.2 pips
                # Skip or log? We reproduce Phase 12 logic exactly first.
                pass
                
            tp_p = entry_p + (risk * tp_r * signal * -1)
            
            # Resolution
            result = 'TIMEOUT'
            exit_p = row.close
            exit_time = row.timestamp
            r_val = -1.0 # Timeout count as loss in Phase 12 logic? 
            # In Phase 12 I set r_val = -1.0 initially.
            
            for j in range(i+1, min(i+150, len(rows))):
                f = rows[j]
                if signal == 1: # LONG
                    if f.low <= sl: result = 'SL'; exit_p = sl; r_val = -1.0; exit_time = f.timestamp; break
                    if f.high >= tp_p: result = 'TP'; exit_p = tp_p; r_val = tp_r; exit_time = f.timestamp; break
                else: # SHORT
                    if (f.high + spread) >= sl: result = 'SL'; exit_p = sl; r_val = -1.0; exit_time = f.timestamp; break
                    if (f.low + spread) <= tp_p: result = 'TP'; exit_p = tp_p; r_val = tp_r; exit_time = f.timestamp; break
            
            trades.append({
                'date': row.date,
                'entry_time': row.timestamp,
                'direction': 'LONG' if signal == 1 else 'SHORT',
                'entry_p': entry_p,
                'sl': sl,
                'tp': tp_p,
                'risk_pips': risk * 10000,
                'result': result,
                'exit_time': exit_time,
                'r_val': r_val
            })
            
            # One trade per day filter
            # In Phase 12 I had: last_trade_date = row.date
            # But the loop continues... wait, did I skip the rest of the day?
            # Yes: if row.date == last_trade_date: continue
            # I need to implement that skip in the reproduction too.
            
    # Redo loop with day skip
    final_trades = []
    last_d = None
    for t in trades:
        if t['date'] != last_d:
            final_trades.append(t)
            last_d = t['date']
            
    df_trades = pd.DataFrame(final_trades)
    
    # Calculate metrics
    sample = len(df_trades)
    tp_count = len(df_trades[df_trades['result'] == 'TP'])
    sl_count = len(df_trades[df_trades['result'] == 'SL'])
    to_count = len(df_trades[df_trades['result'] == 'TIMEOUT'])
    
    gp = df_trades[df_trades['r_val'] > 0]['r_val'].sum()
    gl = abs(df_trades[df_trades['r_val'] < 0]['r_val'].sum())
    pf = gp / gl if gl > 0 else 0
    
    # Output files
    out_dir = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\phase12_forensic_audit\reproduction")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    df_trades.to_csv(out_dir / "reproduced_trades.csv", index=False)
    
    # Equity curve
    df_trades['cum_r'] = df_trades['r_val'].cumsum()
    df_trades[['entry_time', 'cum_r']].to_csv(out_dir / "reproduced_equity_curve.csv", index=False)
    
    # Drawdown
    df_trades['peak'] = df_trades['cum_r'].cummax()
    df_trades['dd'] = df_trades['peak'] - df_trades['cum_r']
    df_trades[['entry_time', 'dd']].to_csv(out_dir / "reproduced_drawdown_curve.csv", index=False)
    
    summary = {
        "sample": sample,
        "pf": round(pf, 3),
        "expectancy": round(df_trades['r_val'].mean(), 3),
        "cumulative_r": round(df_trades['r_val'].sum(), 2),
        "max_drawdown_r": round(df_trades['dd'].max(), 2),
        "win_rate": round(tp_count / sample if sample > 0 else 0, 3),
        "tp_count": tp_count,
        "sl_count": sl_count,
        "be_count": 0,
        "timeout_count": to_count,
        "avg_win_r": round(df_trades[df_trades['r_val'] > 0]['r_val'].mean(), 3) if tp_count > 0 else 0,
        "avg_loss_r": round(df_trades[df_trades['r_val'] < 0]['r_val'].mean(), 3) if (sl_count + to_count) > 0 else 0,
        "total_gross_profit_r": round(gp, 2),
        "total_gross_loss_r": round(gl, 2),
        "count_r_pos": len(df_trades[df_trades['r_val'] > 0]),
        "count_r_neg": len(df_trades[df_trades['r_val'] < 0]),
        "count_r_zero": len(df_trades[df_trades['r_val'] == 0])
    }
    
    with open(out_dir / "reproduced_summary.json", 'w') as f:
        json.dump(summary, f, indent=4)
        
    print(f"Reproduction Complete. PF: {summary['pf']}, Sample: {summary['sample']}")

if __name__ == "__main__":
    run_reproduction()

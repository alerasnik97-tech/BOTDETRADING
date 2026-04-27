
import pandas as pd
import numpy as np
import json
from pathlib import Path
from phase12_advanced_engine import Phase12AdvancedEngine

def run_bloque_c():
    manifest_path = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\ARCHIVE_SUPERSEDED\duplicated_folders\Bot V1_PENDING_DELETE\data_manifest\certified_data_paths.json"
    engine = Phase12AdvancedEngine()
    with open(manifest_path, 'r') as f: manifest = json.load(f)
    periods = ['period_2015_2019', 'period_2020_2026']
    
    # Load H1 for levels and EMA
    h1_list = []
    for p in periods:
        df = pd.read_csv(manifest[p]['h1_bid'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True).dt.tz_convert(engine.tz_ny)
        h1_list.append(df)
    df_h1 = pd.concat(h1_list).sort_values('timestamp')
    df_h1['ema50'] = df_h1['close'].ewm(span=50, adjust=False).mean()
    
    levels = engine.get_levels(df_h1)
    
    # Load M5
    m5_list = []
    for p in periods:
        df = pd.read_csv(manifest[p]['m5_bid'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True).dt.tz_convert(engine.tz_ny)
        m5_list.append(df)
    df_m5 = pd.concat(m5_list).sort_values('timestamp').reset_index(drop=True)
    df_m5['date'] = df_m5['timestamp'].dt.date
    
    # Calculate OR (08:00 to 08:30)
    df_m5['hour'] = df_m5['timestamp'].dt.hour
    df_m5['minute'] = df_m5['timestamp'].dt.minute
    or_range = df_m5[(df_m5['hour'] == 8) & (df_m5['minute'] < 30)].groupby('date').agg({'high': 'max', 'low': 'min'}).to_dict('index')
    
    # Sync EMA 50 H1 to M5
    df_h1_sync = df_h1[['timestamp', 'ema50']].rename(columns={'timestamp': 'h1_time'})
    df_m5 = pd.merge_asof(df_m5.sort_values('timestamp'), df_h1_sync.sort_values('h1_time'), 
                         left_on='timestamp', right_on='h1_time', direction='backward')
    
    # Setup Logic
    rows = list(df_m5.itertuples())
    results = []
    
    for tp in [1.5, 2.0, 2.5, 3.0]:
        print(f"Testing Selective Fakeout TP={tp}...")
        trades = []
        last_trade_date = None
        
        for i in range(1, len(rows)):
            row = rows[i]
            if row.date == last_trade_date: continue
            if row.timestamp.hour < 9 or row.timestamp.hour >= 13: continue
            
            lvl = or_range.get(row.date)
            if not lvl: continue
            
            prev = rows[i-1]
            # dist from ema
            dist = (row.close - row.ema50) * 10000
            
            signal = 0
            # SHORT: prev high > OR high, curr close < OR high, price > EMA + 20 pips
            if prev.high > lvl['high'] and row.close < lvl['high'] and dist > 20:
                signal = -1
                extreme = prev.high
            # LONG: prev low < OR low, curr close > OR low, price < EMA - 20 pips
            elif prev.low < lvl['low'] and row.close > lvl['low'] and dist < -20:
                signal = 1
                extreme = prev.low
                
            if signal != 0:
                # resolve
                entry_p = row.close
                sl = extreme + (0.0001 if signal == -1 else -0.0001)
                risk = abs(entry_p - sl)
                tp_p = entry_p + (risk * tp * signal * -1)
                
                # Simple resolution
                r_val = -1.0
                spread = 0.00007
                for j in range(i+1, min(i+150, len(rows))):
                    f = rows[j]
                    if signal == 1: # LONG
                        # Exit at BID
                        if f.low <= sl: r_val = -1.0; break
                        if f.high >= tp_p: r_val = tp; break
                    else: # SHORT
                        # Exit at ASK (Bid + Spread)
                        if (f.high + spread) >= sl: r_val = -1.0; break
                        if (f.low + spread) <= tp_p: r_val = tp; break
                
                trades.append(r_val)
                last_trade_date = row.date
                
        if trades:
            df_t = pd.Series(trades)
            gp = df_t[df_t > 0].sum()
            gl = abs(df_t[df_t < 0].sum())
            pf = gp / gl if gl > 0 else 0
            results.append({"tp": tp, "pf": round(pf, 2), "sample": len(trades)})
            print(f"  PF={pf:.2f}, Sample={len(trades)}")

    output_dir = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\phase12_surpass_manual_pf\ranking")
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(output_dir / "selective_fakeout_v2_results.csv", index=False)
    print("Bloque C Complete.")

if __name__ == "__main__":
    run_bloque_c()

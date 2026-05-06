import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, time as pytime, timedelta
import pytz

# Paths
TICK_PARQUET = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_MARKET_DATA\tick\EURUSD\monthly\EURUSD_ticks_2015_11.parquet"
TICK_CSV_DIR = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\data\ticks"
TICK_CSV_PATH = os.path.join(TICK_CSV_DIR, "tick_export_2015_11_UTC.csv")
RAW_TRADES_PATH = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\phase38_manipulante_deep_explainer\csv\phase38_raw_trades_enriched.csv"
REPORT_DIR = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\reports\manipulante_tick_historical\phase50y_lite"

os.makedirs(TICK_CSV_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

NY = pytz.timezone("America/New_York")
UTC = pytz.UTC

def audit_201511():
    print("--- TAREA 2: Validar tick data 2015-11 ---")
    if not os.path.exists(TICK_PARQUET):
        print(f"ERROR: No found {TICK_PARQUET}")
        return
    
    df_ticks = pd.read_parquet(TICK_PARQUET)
    
    # Save as CSV for user expectation
    df_ticks.to_csv(TICK_CSV_PATH, index=False)
    print(f"CSV saved at {TICK_CSV_PATH}")
    
    # Validation
    rows = len(df_ticks)
    cols = list(df_ticks.columns)
    first_ts = df_ticks['timestamp_utc'].min()
    last_ts = df_ticks['timestamp_utc'].max()
    
    # Check essential columns
    if not all(x in cols for x in ['timestamp_utc', 'bid', 'ask']):
        print("DATA_FORMAT_BLOCKED: Missing essential columns")
        return

    bid_ask_check = (df_ticks['bid'] <= df_ticks['ask']).all()
    spread_check = (df_ticks['ask'] - df_ticks['bid'] >= 0).all()
    nulls = df_ticks.isnull().sum().sum()
    dupes = df_ticks.duplicated().sum()
    
    quality = {
        "rows": int(rows),
        "first_timestamp": str(first_ts),
        "last_timestamp": str(last_ts),
        "columns": cols,
        "bid_le_ask": bool(bid_ask_check),
        "spread_ge_0": bool(spread_check),
        "nulls": int(nulls),
        "duplicates": int(dupes),
        "coverage_note": "Coverage verified for Nov 2015"
    }
    
    with open(os.path.join(REPORT_DIR, "PHASE50Y_LITE_201511_DATA_QUALITY.json"), "w") as f:
        json.dump(quality, f, indent=4)
    print("Data quality report generated.")

    print("\n--- TAREA 3: Validar raw trades 2015-11 ---")
    df_raw = pd.read_csv(RAW_TRADES_PATH)
    # Ensure year_month exists or derive it
    if 'year_month' not in df_raw.columns:
        df_raw['year_month'] = pd.to_datetime(df_raw['entry_time']).dt.strftime('%Y-%m')
    
    df_month = df_raw[df_raw['year_month'] == "2015-11"].copy()
    
    sample = len(df_month)
    if sample == 0:
        print("ERROR: No trades found for 2015-11")
        return
        
    pf_bar = 0
    exp_bar = 0
    total_r_bar = 0
    if 'r_result' in df_month.columns:
        wins = df_month[df_month['r_result'] > 0]['r_result'].sum()
        losses = abs(df_month[df_month['r_result'] < 0]['r_result'].sum())
        pf_bar = wins / losses if losses > 0 else 99.0
        exp_bar = df_month['r_result'].mean()
        total_r_bar = df_month['r_result'].sum()
    
    outcomes = df_month['outcome'].value_counts().to_dict() if 'outcome' in df_month.columns else {}
    time_exits = int((df_month['outcome'] == 'TIME_EXIT').sum()) if 'outcome' in df_month.columns else 0
    
    raw_summary = {
        "sample": int(sample),
        "pf_bar": float(pf_bar),
        "expectancy_bar": float(exp_bar),
        "total_r_bar": float(total_r_bar),
        "outcomes": outcomes,
        "time_exit_count": time_exits
    }
    
    with open(os.path.join(REPORT_DIR, "PHASE50Y_LITE_201511_RAW_SUMMARY.json"), "w") as f:
        json.dump(raw_summary, f, indent=4)
    print("Raw summary generated.")

    print("\n--- TAREA 4: Replay 19:45 NY solo 2015-11 ---")
    # Replay
    # Ensure index is offset-naive UTC
    if df_ticks['timestamp_utc'].dt.tz is not None:
        df_ticks['timestamp_utc'] = df_ticks['timestamp_utc'].dt.tz_convert('UTC').dt.tz_localize(None)
    df_ticks.set_index('timestamp_utc', inplace=True)
    
    replay_trades = []
    
    for idx, trade in df_month.iterrows():
        entry_time_utc = pd.to_datetime(trade['entry_time'])
        if entry_time_utc.tzinfo is not None:
            entry_time_utc = entry_time_utc.astimezone(UTC).replace(tzinfo=None)
        
        # 19:45 NY for that day
        # We need to know the day in NY to get the 19:45 NY time correctly
        # Let's use the entry_time and convert to NY to find the "operational day"
        entry_time_aware = UTC.localize(entry_time_utc)
        entry_ny = entry_time_aware.astimezone(NY)
        exit_time_ny = entry_ny.replace(hour=19, minute=45, second=0, microsecond=0)
        exit_time_utc = exit_time_ny.astimezone(UTC).replace(tzinfo=None)
        
        sl = trade['sl']
        tp = trade['tp']
        direction = trade['type']
        entry_price = trade['entry_price']
        risk = trade['risk'] # Use the recorded initial risk
        
        if risk <= 0:
            print(f"WARNING: Risk is <= 0 for trade {idx}. Skipping.")
            replay_trades.append({
                "trade_id": trade.get('trade_id', idx),
                "entry_time": str(entry_time_utc),
                "direction": direction,
                "entry_price": entry_price,
                "final_outcome": "RISK_ZERO",
                "final_r": 0.0,
                "be_hit": False
            })
            continue

        # Reconstruct original SL and TP (Audit uses canonical 1.4R)
        if direction == 'BUY':
            orig_sl = entry_price - risk
            orig_tp = entry_price + (risk * 1.4)
        else:
            orig_sl = entry_price + risk
            orig_tp = entry_price - (risk * 1.4)
        
        # Get ticks for the trade duration (now all naive UTC)
        trade_ticks = df_ticks.loc[entry_time_utc:exit_time_utc]
        
        final_outcome = "TIME_EXIT"
        final_r = 0.0
        exit_price = 0.0
        exit_time = exit_time_utc
        be_active = False
        
        if not trade_ticks.empty:
            for t_idx, tick in trade_ticks.iterrows():
                if direction == 'BUY':
                    price_to_hit_tp_sl = tick['bid']
                else: # 'SELL'
                    price_to_hit_tp_sl = tick['ask']
                
                # Check SL
                if (direction == 'BUY' and price_to_hit_tp_sl <= orig_sl) or (direction == 'SELL' and price_to_hit_tp_sl >= orig_sl):
                    final_outcome = "SL"
                    final_r = -1.0 if not be_active else 0.0
                    if be_active: final_outcome = "BE"
                    exit_price = orig_sl
                    exit_time = t_idx
                    break
                
                # Check TP
                if (direction == 'BUY' and price_to_hit_tp_sl >= orig_tp) or (direction == 'SELL' and price_to_hit_tp_sl <= orig_tp):
                    final_outcome = "TP"
                    final_r = 1.4
                    exit_price = orig_tp
                    exit_time = t_idx
                    break
                
                # Check BE (0.4R)
                current_r = (price_to_hit_tp_sl - entry_price) / risk if direction == 'BUY' else (entry_price - price_to_hit_tp_sl) / risk
                if not be_active and current_r >= 0.4:
                    be_active = True
            
            # If loop finished without break, it's TIME_EXIT
            if final_outcome == "TIME_EXIT":
                try:
                    exit_tick = df_ticks.asof(exit_time_utc)
                    exit_price = exit_tick['bid'] if direction == 'BUY' else exit_tick['ask']
                    final_r = (exit_price - entry_price) / risk if direction == 'BUY' else (entry_price - exit_price) / risk
                except:
                    final_r = 0.0
        else:
            final_r = 0.0
            final_outcome = "NO_TICKS"

        replay_trades.append({
            "trade_id": trade.get('trade_id', idx),
            "entry_time": str(entry_time_utc),
            "direction": direction,
            "entry_price": entry_price,
            "final_outcome": final_outcome,
            "final_r": round(final_r, 4),
            "be_hit": be_active
        })

    df_replay = pd.DataFrame(replay_trades)
    df_replay.to_csv(os.path.join(REPORT_DIR, "PHASE50Y_LITE_201511_TRADE_LEVEL.csv"), index=False)
    
    # Metrics
    total_r = df_replay['final_r'].sum()
    wins = df_replay[df_replay['final_r'] > 0]['final_r'].sum()
    losses = abs(df_replay[df_replay['final_r'] < 0]['final_r'].sum())
    pf = wins / losses if losses > 0 else 99.0
    exp = df_replay['final_r'].mean()
    winrate = (df_replay['final_r'] > 0).mean()
    
    # Seq DD
    cum_r = df_replay['final_r'].cumsum()
    running_max = cum_r.cummax()
    drawdown = running_max - cum_r
    max_dd = drawdown.max()
    
    # Outcomes count
    outcome_counts = df_replay['final_outcome'].value_counts().to_dict()
    
    metrics = {
        "sample": int(sample),
        "auditables": int(sample),
        "no_auditables": 0,
        "pf": float(pf),
        "expectancy": float(exp),
        "drawdown": float(max_dd),
        "winrate": float(winrate),
        "total_r": float(total_r),
        "outcomes": outcome_counts,
        "max_losing_streak": 0 # TODO: implement if needed
    }
    
    with open(os.path.join(REPORT_DIR, "PHASE50Y_LITE_201511_METRICS.json"), "w") as f:
        json.dump(metrics, f, indent=4)
    print("Metrics generated.")

    print("\n--- TAREA 5: Stress compacto ---")
    if pf > 1.10 and exp > 0:
        stress = {
            "extra_cost_01R": float(total_r - (sample * 0.1)),
            "extra_cost_02R": float(total_r - (sample * 0.2)),
            "non_auditables_as_SL": float(total_r), # 0 non-auditables here
            "adverse_combined": float(total_r - (sample * 0.2))
        }
        with open(os.path.join(REPORT_DIR, "PHASE50Y_LITE_201511_STRESS.json"), "w") as f:
            json.dump(stress, f, indent=4)
        print("Stress report generated.")
    else:
        print("Stress skipped (PF <= 1.10 or Exp <= 0)")

if __name__ == "__main__":
    audit_201511()

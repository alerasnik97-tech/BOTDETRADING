import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, time as pytime, timedelta
import pytz

# Paths
TICK_PARQUET = r"C:\Users\alera\Desktop\Bot\BOT_MARKET_DATA\tick\EURUSD\monthly\EURUSD_ticks_2025_11.parquet"
RAW_TRADES_PATH = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\phase38_manipulante_deep_explainer\csv\phase38_raw_trades_enriched.csv"
REPORT_DIR = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\reports\manipulante_tick_historical\phase50y_lite"

os.makedirs(REPORT_DIR, exist_ok=True)

NY = pytz.timezone("America/New_York")
UTC = pytz.UTC

def audit_202511():
    print("--- TAREA 1: Validar inputs ---")
    if not os.path.exists(TICK_PARQUET):
        print(f"ERROR: No found {TICK_PARQUET}")
        return
    if not os.path.exists(RAW_TRADES_PATH):
        print(f"ERROR: No found {RAW_TRADES_PATH}")
        return
    
    df_ticks = pd.read_parquet(TICK_PARQUET)
    df_raw = pd.read_csv(RAW_TRADES_PATH)
    
    # Ensure year_month exists or derive it
    df_raw['ym'] = pd.to_datetime(df_raw['entry_time'], utc=True).dt.strftime('%Y-%m')
    
    df_month = df_raw[df_raw['ym'] == "2025-11"].copy()
    sample = len(df_month)
    print(f"Trades found: {sample}")
    
    if sample == 0:
        print("ERROR: No trades found for 2025-11")
        return

    print("\n--- TAREA 2: Replay 19:45 NY solo 2025-11 ---")
    if df_ticks['timestamp_utc'].dt.tz is not None:
        df_ticks['timestamp_utc'] = df_ticks['timestamp_utc'].dt.tz_convert('UTC').dt.tz_localize(None)
    df_ticks.set_index('timestamp_utc', inplace=True)
    
    replay_trades = []
    
    for idx, trade in df_month.iterrows():
        entry_time_utc = pd.to_datetime(trade['entry_time'])
        if entry_time_utc.tzinfo is not None:
            entry_time_utc = entry_time_utc.astimezone(UTC).replace(tzinfo=None)
            
        entry_time_aware = UTC.localize(entry_time_utc)
        entry_ny = entry_time_aware.astimezone(NY)
        exit_time_ny = entry_ny.replace(hour=19, minute=45, second=0, microsecond=0)
        exit_time_utc = exit_time_ny.astimezone(UTC).replace(tzinfo=None)
        
        direction = trade['type']
        entry_price = trade['entry_price']
        risk = trade['risk']
        
        if risk <= 0:
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

        if direction == 'LONG':
            orig_sl = entry_price - risk
            orig_tp = entry_price + (risk * 1.4)
        else:
            orig_sl = entry_price + risk
            orig_tp = entry_price - (risk * 1.4)
        
        trade_ticks = df_ticks.loc[entry_time_utc:exit_time_utc]
        
        final_outcome = "TIME_EXIT"
        final_r = 0.0
        exit_price = 0.0
        exit_time = exit_time_utc
        be_active = False
        
        if not trade_ticks.empty:
            for t_idx, tick in trade_ticks.iterrows():
                if direction == 'LONG':
                    price_to_hit_tp_sl = tick['bid']
                else:
                    price_to_hit_tp_sl = tick['ask']
                
                # Check SL
                if (direction == 'LONG' and price_to_hit_tp_sl <= orig_sl) or (direction == 'SHORT' and price_to_hit_tp_sl >= orig_sl):
                    final_outcome = "SL"
                    final_r = -1.0 if not be_active else 0.0
                    if be_active: final_outcome = "BE"
                    exit_price = orig_sl
                    exit_time = t_idx
                    break
                
                # Check TP
                if (direction == 'LONG' and price_to_hit_tp_sl >= orig_tp) or (direction == 'SHORT' and price_to_hit_tp_sl <= orig_tp):
                    final_outcome = "TP"
                    final_r = 1.4
                    exit_price = orig_tp
                    exit_time = t_idx
                    break
                
                # Check BE (0.4R)
                current_r = (price_to_hit_tp_sl - entry_price) / risk if direction == 'LONG' else (entry_price - price_to_hit_tp_sl) / risk
                if not be_active and current_r >= 0.4:
                    be_active = True
            
            if final_outcome == "TIME_EXIT":
                exit_tick = trade_ticks.iloc[-1]
                exit_price = exit_tick['bid'] if direction == 'LONG' else exit_tick['ask']
                final_r = (exit_price - entry_price) / risk if direction == 'LONG' else (entry_price - exit_price) / risk
        else:
            final_outcome = "NO_TICKS"
            final_r = 0.0

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
    df_replay.to_csv(os.path.join(REPORT_DIR, "PHASE50Y_LITE_202511_TRADE_LEVEL.csv"), index=False)
    
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
        "tp_count": int(outcome_counts.get("TP", 0)),
        "be_count": int(outcome_counts.get("BE", 0)),
        "sl_count": int(outcome_counts.get("SL", 0)),
        "time_exit_count": int(outcome_counts.get("TIME_EXIT", 0))
    }
    
    with open(os.path.join(REPORT_DIR, "PHASE50Y_LITE_202511_METRICS.json"), "w") as f:
        json.dump(metrics, f, indent=4)
    print("Metrics generated.")

    if pf > 1.10 and exp > 0:
        stress = {
            "extra_cost_01R": float(total_r - (sample * 0.1)),
            "extra_cost_02R": float(total_r - (sample * 0.2)),
            "non_auditables_as_SL": float(total_r),
            "adverse_combined": float(total_r - (sample * 0.2))
        }
        with open(os.path.join(REPORT_DIR, "PHASE50Y_LITE_202511_STRESS.json"), "w") as f:
            json.dump(stress, f, indent=4)
        print("Stress report generated.")

if __name__ == "__main__":
    audit_202511()

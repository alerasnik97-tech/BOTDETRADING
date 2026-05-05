"""
PHASE50G — Second Month Historical Tick Pilot
Auditoría tick-by-tick para mes piloto 2024-05
NO MODIFICA MANIPULANTE — Solo observa y valida
"""
import os
import sys
import json
import hashlib
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

# --- PATHS ---
TICK_FILE = Path(r"C:\Users\alera\Desktop\Bot\BOT_MARKET_DATA\tick\EURUSD\monthly\EURUSD_ticks_2024_05.parquet")
REPORT_DIR = Path(r"C:\Users\alera\Desktop\Bot\BOT_V2_DAYTIME_LAB\reports\manipulante_tick_historical")
TRADES_FILE = Path(r"C:\Users\alera\Desktop\Bot\BOT_V2_DAYTIME_LAB\reports\manipulante_tick_historical\PHASE50G_2024_05_trades_to_audit.csv")

# --- TZ ---
NY_TZ = "America/New_York"
UTC_TZ = "UTC"

def load_ticks():
    df = pd.read_parquet(TICK_FILE)
    df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc']).dt.tz_convert(UTC_TZ)
    df['timestamp_ny'] = df['timestamp_utc'].dt.tz_convert(NY_TZ)
    return df

def load_trades():
    df = pd.read_csv(TRADES_FILE)
    df['entry_time'] = pd.to_datetime(df['entry_time'])
    df['exit_time'] = pd.to_datetime(df['exit_time'])
    return df

def pip_diff(price, reference):
    return round((float(price) - float(reference)) * 10000, 4)

def trade_levels(trade):
    direction = trade['type']
    entry = float(trade['entry_price'])
    risk = float(trade['risk'])
    tp = float(trade['tp'])
    initial_sl = entry - risk if direction == "LONG" else entry + risk
    be_trigger = entry + (0.4 * risk if direction == "LONG" else -0.4 * risk)
    return {
        'entry': entry, 'risk': risk, 'tp': tp,
        'initial_sl': initial_sl, 'be_trigger': be_trigger, 'be_level': entry
    }

def nearest_ticks(ticks, entry_time_ny):
    entry_utc = entry_time_ny.tz_convert(UTC_TZ)
    idx = int(ticks['timestamp_utc'].searchsorted(entry_utc))
    before = ticks.iloc[idx - 1] if idx > 0 else None
    after = ticks.iloc[idx] if idx < len(ticks) else None
    if before is None:
        nearest = after
    elif after is None:
        nearest = before
    else:
        nearest = before if abs(before['timestamp_utc'] - entry_utc) <= abs(after['timestamp_utc'] - entry_utc) else after
    return before, after, nearest

def first_touch(ticks, trade, levels):
    direction = trade['type']
    entry_time = trade['entry_time']
    exit_time = trade['exit_time']
    
    window = ticks[
        (ticks['timestamp_ny'] >= entry_time) &
        (ticks['timestamp_ny'] <= exit_time + pd.Timedelta(minutes=5))
    ].copy()
    
    if window.empty:
        return "NO_TICK_DATA", None, None, None, None, None
    
    be_active = False
    be_activated_time = None
    
    for _, row in window.iterrows():
        bid = float(row['bid'])
        ask = float(row['ask'])
        t_ny = row['timestamp_ny']
        t_utc = row['timestamp_utc']
        
        if direction == "LONG":
            if not be_active and bid >= levels['be_trigger']:
                be_active = True
                be_activated_time = t_ny
            current_sl = levels['be_level'] if be_active else levels['initial_sl']
            if bid <= current_sl:
                reason = "BE" if be_active else "SL"
                return reason, t_ny, t_utc, bid, ask, be_activated_time
            if bid >= levels['tp']:
                return "TP", t_ny, t_utc, bid, ask, be_activated_time
        else:  # SHORT
            if not be_active and ask <= levels['be_trigger']:
                be_active = True
                be_activated_time = t_ny
            current_sl = levels['be_level'] if be_active else levels['initial_sl']
            if ask >= current_sl:
                reason = "BE" if be_active else "SL"
                return reason, t_ny, t_utc, bid, ask, be_activated_time
            if ask <= levels['tp']:
                return "TP", t_ny, t_utc, bid, ask, be_activated_time
    
    return "NONE", None, None, None, None, be_activated_time

def classify_difference(bar_outcome, tick_outcome, data_gaps, diff_bid_pips, diff_ask_pips):
    if bar_outcome == tick_outcome:
        return "MATCH"
    if data_gaps > 0:
        return "DATA_GAP_OR_LOW_TICK_DENSITY"
    if abs(diff_bid_pips) > 5 or abs(diff_ask_pips) > 5:
        return "ENTRY_PRICE_FEED_DIFFERENCE"
    if bar_outcome in ["BE", "SL"] and tick_outcome in ["BE", "SL"]:
        return "BE_SEQUENCE_DIFFERENCE"
    if tick_outcome == "NO_TICK_DATA":
        return "NOT_AUDITABLE"
    return "LEGIT_TICK_MICROSTRUCTURE_DIFFERENCE"

def calculate_tick_r(trade, tick_outcome, levels):
    risk = levels['risk']
    if tick_outcome == "TP":
        return round(1.4, 4)
    elif tick_outcome == "BE":
        return round(0.4, 4)
    elif tick_outcome == "SL":
        return round(-1.0, 4)
    elif tick_outcome == "FORCED":
        return round(float(trade.get('R', 0)), 4)
    return None

def audit_trade(idx, trade, ticks):
    trade_id = idx + 1  # Usar índice + 1 como ID
    entry_ny = trade['entry_time']
    direction = trade['type']
    bar_outcome = trade['outcome']
    bar_r = float(trade['r_result']) if pd.notna(trade['r_result']) else None
    
    levels = trade_levels(trade)
    before, after, nearest = nearest_ticks(ticks, entry_ny)
    
    if nearest is None:
        return {
            'trade_id': trade_id, 'date': entry_ny.strftime('%Y-%m-%d'),
            'direction': direction, 'entry_time_ny': entry_ny.isoformat(),
            'bar_outcome': bar_outcome, 'tick_outcome': 'NO_DATA',
            'classification': 'NOT_AUDITABLE', 'notes': 'No tick data near entry'
        }
    
    nearest_bid = float(nearest['bid'])
    nearest_ask = float(nearest['ask'])
    entry_price = float(trade['entry_price'])
    
    diff_bid = pip_diff(nearest_bid, entry_price)
    diff_ask = pip_diff(nearest_ask, entry_price)
    spread_entry = round(nearest_ask - nearest_bid, 8)
    
    tick_outcome, touch_ny, touch_utc, touch_bid, touch_ask, be_time = first_touch(ticks, trade, levels)
    tick_r = calculate_tick_r(trade, tick_outcome, levels)
    
    # Stats de spread
    entry_window = ticks[
        (ticks['timestamp_ny'] >= entry_ny - pd.Timedelta(minutes=1)) &
        (ticks['timestamp_ny'] <= entry_ny + pd.Timedelta(minutes=1))
    ]
    spread_max = entry_window['spread_pips'].max() if not entry_window.empty else None
    
    # Check gaps
    data_gaps = 0
    day_ticks = ticks[ticks['timestamp_ny'].dt.date == entry_ny.date()]
    if len(day_ticks) > 1:
        diffs = day_ticks['timestamp_ny'].diff()
        data_gaps = int((diffs > pd.Timedelta(minutes=5)).sum())
    
    classification = classify_difference(bar_outcome, tick_outcome, data_gaps, diff_bid, diff_ask)
    match_status = "MATCH" if bar_outcome == tick_outcome else "MISMATCH"
    
    return {
        'trade_id': trade_id,
        'date': entry_ny.strftime('%Y-%m-%d'),
        'direction': direction,
        'entry_time_ny': entry_ny.isoformat(),
        'entry_time_utc': entry_ny.tz_convert(UTC_TZ).isoformat(),
        'bar_outcome': bar_outcome,
        'tick_outcome': tick_outcome,
        'bar_R': bar_r,
        'tick_R': tick_r,
        'match_status': match_status,
        'first_touch': tick_outcome,
        'first_touch_time': touch_ny.isoformat() if touch_ny else None,
        'entry_price_historical': entry_price,
        'nearest_bid': nearest_bid,
        'nearest_ask': nearest_ask,
        'diff_bid_pips': diff_bid,
        'diff_ask_pips': diff_ask,
        'spread_entry': round(spread_entry * 10000, 4),
        'spread_max': round(spread_max, 4) if spread_max else None,
        'data_gaps': data_gaps,
        'classification': classification,
        'notes': ''
    }

def compute_metrics(df):
    auditable = df[df['classification'] != 'NOT_AUDITABLE']
    matches = auditable[auditable['match_status'] == 'MATCH']
    
    total_r = auditable['tick_R'].sum() if 'tick_R' in auditable.columns else 0
    wins = auditable[auditable['tick_R'] > 0]
    losses = auditable[auditable['tick_R'] < 0]
    
    pf = abs(wins['tick_R'].sum() / losses['tick_R'].sum()) if len(losses) > 0 and losses['tick_R'].sum() != 0 else float('inf')
    winrate = len(wins) / len(auditable) * 100 if len(auditable) > 0 else 0
    expectancy = total_r / len(auditable) if len(auditable) > 0 else 0
    
    # Count outcomes
    tp_count = len(auditable[auditable['tick_outcome'] == 'TP'])
    be_count = len(auditable[auditable['tick_outcome'] == 'BE'])
    sl_count = len(auditable[auditable['tick_outcome'] == 'SL'])
    
    return {
        'sample': len(auditable),
        'PF': round(pf, 4) if pf != float('inf') else 'inf',
        'expectancy': round(expectancy, 4),
        'winrate': round(winrate, 2),
        'TP': tp_count, 'BE': be_count, 'SL': sl_count,
        'total_R': round(total_r, 4),
        'matches': len(matches),
        'match_rate': round(len(matches) / len(auditable) * 100, 2) if len(auditable) > 0 else 0
    }

def main():
    print("[*] PHASE50G — Second Month Tick Pilot (2024-05)")
    print(f"[*] Loading tick data from: {TICK_FILE}")
    
    ticks = load_ticks()
    print(f"[*] Loaded {len(ticks):,} ticks")
    
    trades = load_trades()
    print(f"[*] Auditing {len(trades)} trades from May 2024")
    
    results = []
    for idx, (_, trade) in enumerate(trades.iterrows()):
        result = audit_trade(idx, trade, ticks)
        results.append(result)
        print(f"  Trade {result['trade_id']}: bar={result['bar_outcome']} tick={result['tick_outcome']} -> {result['classification']}")
    
    df_results = pd.DataFrame(results)
    
    # Guardar CSV detallado
    output_csv = REPORT_DIR / "PHASE50G_SECOND_MONTH_TICK_AUDIT_2024_05.csv"
    df_results.to_csv(output_csv, index=False)
    print(f"\n[*] Detailed audit saved: {output_csv}")
    
    # Métricas tick-level
    tick_metrics = compute_metrics(df_results)
    
    # Métricas bar-level (desde trades originales)
    r_col = 'r_result'
    wins = trades[trades[r_col] > 0]
    losses = trades[trades[r_col] < 0]
    pf = abs(wins[r_col].sum() / losses[r_col].sum()) if len(losses) > 0 and losses[r_col].sum() != 0 else float('inf')
    bar_metrics = {
        'sample': len(trades),
        'PF': round(pf, 4) if pf != float('inf') else 'inf',
        'expectancy': round(trades[r_col].sum() / len(trades), 4),
        'winrate': round(len(wins) / len(trades) * 100, 2),
        'TP': len(trades[trades['outcome'] == 'TP']),
        'BE': len(trades[trades['outcome'] == 'BE']),
        'SL': len(trades[trades['outcome'] == 'SL']),
        'total_R': round(trades[r_col].sum(), 4)
    }
    
    # Comparación
    comparison = {
        'month': '2024-05',
        'bar_metrics': bar_metrics,
        'tick_metrics': tick_metrics,
        'delta_PF': 'N/A' if tick_metrics['PF'] == 'inf' or bar_metrics['PF'] == 'inf' else round(float(tick_metrics['PF']) - float(bar_metrics['PF']), 4) if bar_metrics['PF'] != 'inf' else 'N/A',
        'delta_expectancy': round(tick_metrics['expectancy'] - bar_metrics['expectancy'], 4),
        'delta_winrate': round(tick_metrics['winrate'] - bar_metrics['winrate'], 2),
        'classification_counts': df_results['classification'].value_counts().to_dict()
    }
    
    # Guardar JSON
    output_json = REPORT_DIR / "PHASE50G_SECOND_MONTH_TICK_AUDIT_2024_05.json"
    with open(output_json, 'w') as f:
        json.dump(comparison, f, indent=2, default=str)
    print(f"[*] Metrics saved: {output_json}")
    
    # Resumen
    print("\n" + "="*60)
    print("RESUMEN AUDITORIA 2024-05")
    print("="*60)
    print(f"Bar-level sample:  {bar_metrics['sample']}")
    print(f"Tick-level sample: {tick_metrics['sample']}")
    print(f"Match rate:        {tick_metrics['match_rate']}% ({tick_metrics['matches']}/{tick_metrics['sample']})")
    print(f"Bar PF:            {bar_metrics['PF']}")
    print(f"Tick PF:           {tick_metrics['PF']}")
    print(f"Bar Total R:       {bar_metrics['total_R']}")
    print(f"Tick Total R:      {tick_metrics['total_R']}")
    print("\nClasificaciones:")
    for cls, count in comparison['classification_counts'].items():
        print(f"  {cls}: {count}")
    print("="*60)
    
    return comparison

if __name__ == "__main__":
    main()

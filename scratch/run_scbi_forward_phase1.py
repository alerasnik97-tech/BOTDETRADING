"""
SCBI_M5 Forward Test - Phase 1 Paper Runner
===========================================
Unified script for daily paper trading operation.
Implements the Fail-Closed Forward Operating System.

Usage:
  python run_scbi_forward_paper.py [--date YYYY-MM-DD]
  If no date is provided, runs for the last complete day in the dataset.
"""

import pandas as pd
import numpy as np
import json
import os
import argparse
import sys
from datetime import datetime
from pathlib import Path

from validate_scbi_phase1_baseline import seal_runtime_state, validate_current_state

ROOT = r'C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo'
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scratch.forward_telemetry_lib import append_trace_rows, build_global_daily_trace_rows, build_global_ledger_trace_rows

DATA_H1 = os.path.join(ROOT, 'data_candidates_2022_2025', 'prepared', 'EURUSD_H1.csv')
DATA_M5 = os.path.join(ROOT, 'data_candidates_2022_2025', 'prepared', 'EURUSD_M5.csv')
LEDGER_CSV = os.path.join(ROOT, 'results', 'SCBI_FORWARD_LEDGER.csv')
STATUS_CSV = os.path.join(ROOT, 'results', 'SCBI_FORWARD_DAILY_STATUS.csv')
FORWARD_RUN_ID = os.environ.get("SCBI_FORWARD_RUN_ID", "").strip() or None

SPREAD_PIPS = 0.3
SPREAD = SPREAD_PIPS * 0.0001
RISK_MIN_PIPS = 2.0


def append_to_csv(filepath, row_dict, columns):
    file_exists = os.path.isfile(filepath)
    if not file_exists:
        with open(filepath, 'w') as f:
            f.write(','.join(columns) + '\n')
    
    row_str = []
    for col in columns:
        val = row_dict.get(col, '')
        row_str.append(str(val))
    
    with open(filepath, 'a') as f:
        f.write(','.join(row_str) + '\n')


def append_global_telemetry(record, *, record_type):
    frame = pd.DataFrame([record])
    if record_type == "ledger":
        append_trace_rows(build_global_ledger_trace_rows(frame, run_id=FORWARD_RUN_ID))
        return
    if record_type == "daily_status":
        append_trace_rows(build_global_daily_trace_rows(frame, run_id=FORWARD_RUN_ID))
        return
    raise ValueError(f"Unsupported telemetry record type: {record_type}")


def load_data(target_date=None):
    if not os.path.exists(DATA_H1) or not os.path.exists(DATA_M5):
        print("[FAIL-CLOSED] Missing data files.")
        sys.exit(1)
        
    h1 = pd.read_csv(DATA_H1, index_col=0)
    h1.index = pd.to_datetime(h1.index, utc=True).tz_convert('US/Eastern')
    
    m5 = pd.read_csv(DATA_M5, index_col=0)
    m5.index = pd.to_datetime(m5.index, utc=True).tz_convert('US/Eastern')
    
    if target_date is None:
        target_date = m5.index[-1].date().strftime('%Y-%m-%d')
        
    print(f"[DATA] Target date: {target_date}")
    
    return h1, m5, target_date


def compute_session_levels(h1, target_date):
    """Compute PDH/PDL, Asia H/L, London H/L for the target date."""
    h1 = h1.copy()
    h1['date'] = h1.index.date
    h1['hour'] = h1.index.hour
    
    dates = sorted(h1['date'].unique())
    target_dt = pd.to_datetime(target_date).date()
    
    if target_dt not in dates:
        return None
        
    idx = dates.index(target_dt)
    if idx == 0:
        return None
        
    prev_d = dates[idx - 1]
    prev_bars = h1[h1['date'] == prev_d]
    curr_bars = h1[h1['date'] == target_dt]
    
    if len(prev_bars) == 0 or len(curr_bars) == 0:
        return None
        
    pdh = prev_bars['high'].max()
    pdl = prev_bars['low'].min()
    
    asia_bars_prev = prev_bars[prev_bars['hour'] >= 18]
    asia_bars_curr = curr_bars[(curr_bars['hour'] >= 18) | (curr_bars['hour'] < 2)]
    asia_all = pd.concat([asia_bars_prev, asia_bars_curr])
    
    if len(asia_all) > 0:
        asia_h = asia_all['high'].max()
        asia_l = asia_all['low'].min()
    else:
        asia_h = pdh
        asia_l = pdl
        
    london_bars = curr_bars[(curr_bars['hour'] >= 2) & (curr_bars['hour'] < 8)]
    if len(london_bars) > 0:
        london_h = london_bars['high'].max()
        london_l = london_bars['low'].min()
    else:
        london_h = pdh
        london_l = pdl
        
    return {
        'pdh': pdh, 'pdl': pdl,
        'asia_h': asia_h, 'asia_l': asia_l,
        'london_h': london_h, 'london_l': london_l
    }


def get_news_events(target_date):
    """
    Mock integration for News Fortress.
    In a fully integrated environment, this calls research_lab.news_filter.
    For this paper run, we load events if available.
    """
    try:
        from research_lab.config import DEFAULT_RAW_NEWS_FILE_OBSOLETE, NewsConfig, PAIR_CANONICAL_NEWS_FILES
        from research_lab.news_filter import load_news_events
        settings = NewsConfig(
            file_path=Path(ROOT) / PAIR_CANONICAL_NEWS_FILES["EURUSD"],
            raw_file_path=Path(ROOT) / DEFAULT_RAW_NEWS_FILE_OBSOLETE,
        )
        res = load_news_events("EURUSD", settings)
        if not res.enabled or res.events.empty:
            print("[FAIL-CLOSED] News Fortress calendar not available or disabled.")
            return None
            
        events = res.events
        events['date'] = pd.to_datetime(events['timestamp_ny']).dt.date
        target_dt = pd.to_datetime(target_date).date()
        min_date = events['date'].min()
        max_date = events['date'].max()
        if target_dt < min_date or target_dt > max_date:
            print(f"[FAIL-CLOSED] News Fortress coverage unavailable for {target_date}. Dataset range: {min_date} -> {max_date}.")
            return None
        day_events = events[events['date'] == target_dt]
        return day_events
    except Exception as e:
        print(f"[FAIL-CLOSED] Error loading News Fortress: {e}")
        return None


def is_news_blocked(sweep_time, day_events):
    if day_events is None:
        return True, "Calendar missing"
    
    if day_events.empty:
        return False, ""
        
    for idx, row in day_events.iterrows():
        event_time = pd.to_datetime(row['timestamp_ny'])
        diff = abs((sweep_time - event_time).total_seconds()) / 60.0
        if diff <= 30:
            return True, f"Blocked by {row.get('event', 'News Event')}"
    
    return False, ""


def process_day(target_date):
    h1, m5, target_date = load_data(target_date)

    baseline_ok, baseline_issues = validate_current_state(readiness_date=target_date)
    if not baseline_ok:
        print("[FAIL-CLOSED] Phase 1 baseline validation failed.")
        for issue in baseline_issues:
            print(f"  - {issue}")
        sys.exit(1)
    
    # Check if already processed
    if os.path.exists(STATUS_CSV):
        status_df = pd.read_csv(STATUS_CSV)
        if target_date in status_df['session_date'].values:
            print(f"[BLOCKED] Target date {target_date} already processed in daily status.")
            sys.exit(0)
            
    # News check
    day_events = get_news_events(target_date)
    news_status = "CLEAR"
    if day_events is None:
        news_status = "CALENDAR_MISSING"
        # We continue to log DATA_ISSUE, but we won't trade.
        
    # Levels
    levels = compute_session_levels(h1, target_date)
    if levels is None:
        print(f"[DATA_ISSUE] Insufficient H1 data for levels on {target_date}")
        daily_issue_record = {
            'session_date': target_date, 'sweeps_detected': 0, 'sweeps_blocked_news': 0,
            'sweeps_blocked_daily_limit': 0, 'sweeps_no_scbi': 0, 'sweeps_data_issue': 1,
            'trades_paper': 0, 'result': '', 'pnl_r': 0, 'incidents': 'Insufficient H1 data',
            'cumulative_n': '', 'cumulative_pf': '', 'cumulative_exp': '', 'cumulative_dd': '', 'notes': ''
        }
        append_to_csv(STATUS_CSV, daily_issue_record, ['session_date','sweeps_detected','sweeps_blocked_news','sweeps_blocked_daily_limit','sweeps_no_scbi','sweeps_data_issue','trades_paper','result','pnl_r','cumulative_n','cumulative_pf','cumulative_exp','cumulative_dd','incidents','notes'])
        append_global_telemetry(daily_issue_record, record_type="daily_status")
        seal_ok, seal_issues = seal_runtime_state(reason=f"data_issue:{target_date}")
        if not seal_ok:
            for issue in seal_issues:
                print(f"[FAIL-CLOSED] {issue}")
        sys.exit(0)
        
    # Isolate day data
    target_dt = pd.to_datetime(target_date).date()
    h1_day = h1[h1.index.date == target_dt]
    
    sweeps = []
    for idx, bar in h1_day.iterrows():
        o, h, l, c = bar['open'], bar['high'], bar['low'], bar['close']
        
        for name, level in [('pdl', levels['pdl']), ('asia_l', levels['asia_l']), ('london_l', levels['london_l'])]:
            if l < level and c > level:
                sweeps.append({'time': idx, 'direction': 'long', 'level_name': name, 'level_price': level, 'extreme': l})
                
        for name, level in [('pdh', levels['pdh']), ('asia_h', levels['asia_h']), ('london_h', levels['london_h'])]:
            if h > level and c < level:
                sweeps.append({'time': idx, 'direction': 'short', 'level_name': name, 'level_price': level, 'extreme': h})
                
    ledger_cols = ['session_date','event_timestamp','event_type','status','signal_id','strategy_id','pair','direction','sweep_level','sweep_time','sweep_extreme','level_price','entry_time','entry_price','sl','tp','risk_pips','exit_time','exit_price','exit_type','pnl_r','block_reason','block_details','observed_spread_pips','applied_spread_pips','data_quality_flag','news_check_status','notes']
    
    stats = {'detected': 0, 'news_blocked': 0, 'no_scbi': 0, 'daily_limit': 0, 'trades': 0, 'pnl_r': 0}
    has_trade = False
    
    for i, sweep in enumerate(sweeps):
        stats['detected'] += 1
        signal_id = f"SCBI_{target_date}_{sweep['time'].strftime('%H%M')}"
        
        base_record = {
            'session_date': target_date, 'event_timestamp': str(datetime.now().astimezone()),
            'signal_id': signal_id, 'strategy_id': 'SCBI_M5_GLOBAL', 'pair': 'EURUSD',
            'direction': sweep['direction'], 'sweep_level': sweep['level_name'],
            'sweep_time': str(sweep['time']), 'sweep_extreme': sweep['extreme'],
            'level_price': sweep['level_price'], 'applied_spread_pips': SPREAD_PIPS,
            'data_quality_flag': 'OK', 'news_check_status': news_status
        }
        
        # 1. Detected
        r_det = base_record.copy()
        r_det.update({'event_type': 'SWEEP_DETECTED', 'status': 'DETECTED'})
        append_to_csv(LEDGER_CSV, r_det, ledger_cols)
        append_global_telemetry(r_det, record_type="ledger")
        
        # 2. News Block
        is_blocked, block_reason = is_news_blocked(sweep['time'], day_events)
        if is_blocked:
            stats['news_blocked'] += 1
            r_block = base_record.copy()
            r_block.update({'event_type': 'NEWS_BLOCKED', 'status': 'BLOCKED', 'block_reason': block_reason})
            append_to_csv(LEDGER_CSV, r_block, ledger_cols)
            append_global_telemetry(r_block, record_type="ledger")
            continue
            
        # 3. Daily limit
        if has_trade:
            stats['daily_limit'] += 1
            r_block = base_record.copy()
            r_block.update({'event_type': 'DAILY_LIMIT', 'status': 'BLOCKED', 'block_reason': '1 trade per day limit'})
            append_to_csv(LEDGER_CSV, r_block, ledger_cols)
            append_global_telemetry(r_block, record_type="ledger")
            continue
            
        # 4. SCBI M5 Scan
        search_start = sweep['time'] + pd.Timedelta(hours=1)
        search_end = search_start + pd.Timedelta(hours=1)
        m5_window = m5[(m5.index >= search_start) & (m5.index <= search_end)]
        
        entry_info = None
        for j in range(len(m5_window)):
            bar = m5_window.iloc[j]
            if sweep['direction'] == 'long' and bar['close'] > sweep['level_price']:
                if j + 1 < len(m5_window):
                    eb = m5_window.iloc[j + 1]
                    ep = eb['open'] + SPREAD
                    sl = sweep['extreme'] - 0.0001
                    risk = (ep - sl) / 0.0001
                    if risk >= RISK_MIN_PIPS:
                        entry_info = {'time': m5_window.index[j + 1], 'price': ep, 'sl': sl, 'tp': ep + 1.5 * (ep - sl), 'risk': risk}
                    break
            elif sweep['direction'] == 'short' and bar['close'] < sweep['level_price']:
                if j + 1 < len(m5_window):
                    eb = m5_window.iloc[j + 1]
                    ep = eb['open']
                    sl = sweep['extreme'] + 0.0001
                    risk = (sl - ep) / 0.0001
                    if risk >= RISK_MIN_PIPS:
                        entry_info = {'time': m5_window.index[j + 1], 'price': ep, 'sl': sl, 'tp': ep - 1.5 * (sl - ep), 'risk': risk}
                    break
                    
        if entry_info is None:
            stats['no_scbi'] += 1
            r_inv = base_record.copy()
            r_inv.update({'event_type': 'NO_SCBI_FOUND', 'status': 'INVALIDATED'})
            append_to_csv(LEDGER_CSV, r_inv, ledger_cols)
            append_global_telemetry(r_inv, record_type="ledger")
            continue
            
        # 5. PAPER ENTRY
        has_trade = True
        stats['trades'] += 1
        r_entry = base_record.copy()
        r_entry.update({
            'event_type': 'PAPER_ENTRY', 'status': 'FILLED',
            'entry_time': str(entry_info['time']), 'entry_price': round(entry_info['price'], 5),
            'sl': round(entry_info['sl'], 5), 'tp': round(entry_info['tp'], 5),
            'risk_pips': round(entry_info['risk'], 1)
        })
        append_to_csv(LEDGER_CSV, r_entry, ledger_cols)
        append_global_telemetry(r_entry, record_type="ledger")
        
        # 6. PAPER EXIT Simulation
        future = m5[(m5.index >= entry_info['time']) & (m5.index <= entry_info['time'] + pd.Timedelta(hours=4))]
        pnl_r = 0
        exit_type = 'timeout'
        exit_time = None
        exit_price = None
        
        for k in range(len(future)):
            b = future.iloc[k]
            if sweep['direction'] == 'long':
                if b['low'] <= entry_info['sl']:
                    pnl_r, exit_type, exit_time, exit_price = -1.0, 'sl_hit', future.index[k], entry_info['sl']
                    break
                if b['high'] >= entry_info['tp']:
                    pnl_r, exit_type, exit_time, exit_price = 1.5, 'tp_hit', future.index[k], entry_info['tp']
                    break
            else:
                if b['high'] >= entry_info['sl']:
                    pnl_r, exit_type, exit_time, exit_price = -1.0, 'sl_hit', future.index[k], entry_info['sl']
                    break
                if b['low'] <= entry_info['tp']:
                    pnl_r, exit_type, exit_time, exit_price = 1.5, 'tp_hit', future.index[k], entry_info['tp']
                    break
                    
        if exit_time is None:
            exit_time = future.index[-1] if len(future) > 0 else entry_info['time']
            exit_price = future.iloc[-1]['close'] if len(future) > 0 else entry_info['price']
            if sweep['direction'] == 'long':
                pnl_r = (exit_price - entry_info['price']) / (entry_info['price'] - entry_info['sl'])
            else:
                pnl_r = (entry_info['price'] - exit_price) / (entry_info['sl'] - entry_info['price'])
                
        pnl_r = round(pnl_r, 3)
        stats['pnl_r'] = pnl_r
        
        r_exit = base_record.copy()
        r_exit.update({
            'event_type': 'PAPER_EXIT', 'status': 'FILLED',
            'entry_time': str(entry_info['time']), 'entry_price': round(entry_info['price'], 5),
            'sl': round(entry_info['sl'], 5), 'tp': round(entry_info['tp'], 5),
            'risk_pips': round(entry_info['risk'], 1),
            'exit_time': str(exit_time), 'exit_price': round(exit_price, 5),
            'exit_type': exit_type, 'pnl_r': pnl_r
        })
        append_to_csv(LEDGER_CSV, r_exit, ledger_cols)
        append_global_telemetry(r_exit, record_type="ledger")

    # Calculate cumulative metrics from ledger
    ledger_df = pd.read_csv(LEDGER_CSV) if os.path.exists(LEDGER_CSV) else pd.DataFrame()
    exits = ledger_df[ledger_df['event_type'] == 'PAPER_EXIT']
    cum_n = len(exits)
    if cum_n > 0:
        exits['pnl_r'] = exits['pnl_r'].astype(float)
        gp = exits[exits['pnl_r'] > 0]['pnl_r'].sum()
        gl = abs(exits[exits['pnl_r'] <= 0]['pnl_r'].sum())
        cum_pf = gp / gl if gl > 0 else 999
        cum_exp = exits['pnl_r'].sum() / cum_n
        # DD
        eq = 0
        peak = 0
        dd = 0
        for p in exits['pnl_r']:
            eq += p
            if eq > peak: peak = eq
            if eq - peak < dd: dd = eq - peak
    else:
        cum_pf, cum_exp, dd = 0, 0, 0

    # Write status
    daily_status_record = {
        'session_date': target_date, 'sweeps_detected': stats['detected'],
        'sweeps_blocked_news': stats['news_blocked'], 'sweeps_blocked_daily_limit': stats['daily_limit'],
        'sweeps_no_scbi': stats['no_scbi'], 'sweeps_data_issue': 0,
        'trades_paper': stats['trades'], 'result': 'TRADE' if stats['trades'] > 0 else 'NO_TRADE',
        'pnl_r': stats['pnl_r'],
        'cumulative_n': cum_n, 'cumulative_pf': round(cum_pf, 2),
        'cumulative_exp': round(cum_exp, 3), 'cumulative_dd': round(dd, 2),
        'incidents': 'None' if news_status == 'CLEAR' else news_status,
        'notes': ''
    }
    append_to_csv(STATUS_CSV, daily_status_record, ['session_date','sweeps_detected','sweeps_blocked_news','sweeps_blocked_daily_limit','sweeps_no_scbi','sweeps_data_issue','trades_paper','result','pnl_r','cumulative_n','cumulative_pf','cumulative_exp','cumulative_dd','incidents','notes'])
    append_global_telemetry(daily_status_record, record_type="daily_status")

    seal_ok, seal_issues = seal_runtime_state(reason=f"post_run:{target_date}")
    if not seal_ok:
        print("[FAIL-CLOSED] Runtime seal failed after processing.")
        for issue in seal_issues:
            print(f"  - {issue}")
        sys.exit(1)
    
    print(f"\n[SUMMARY] Day: {target_date} | Sweeps: {stats['detected']} | Trades: {stats['trades']} | PnL: {stats['pnl_r']}R")
    print("[SUCCESS] Process completed.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', type=str, help='Target date YYYY-MM-DD')
    args = parser.parse_args()
    process_day(args.date)

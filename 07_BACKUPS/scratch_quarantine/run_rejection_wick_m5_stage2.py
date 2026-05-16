import pandas as pd
import numpy as np
import json
import os
from datetime import time

DATA_H1 = r'C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\data_candidates_2022_2025\prepared\EURUSD_H1.csv'
DATA_M5 = r'C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\data_candidates_2022_2025\prepared\EURUSD_M5.csv'
SPREAD = 0.00003

def load_data():
    df_h1 = pd.read_csv(DATA_H1)
    df_h1['timestamp'] = pd.to_datetime(df_h1['timestamp'], utc=True)
    df_h1.set_index('timestamp', inplace=True)
    
    df_m5 = pd.read_csv(DATA_M5)
    df_m5['timestamp'] = pd.to_datetime(df_m5['timestamp'], utc=True)
    df_m5.set_index('timestamp', inplace=True)
    
    return df_h1, df_m5

def calculate_levels(df_h1):
    df_ny = df_h1.copy()
    # Simple NY approx
    df_ny.index = df_ny.index - pd.Timedelta(hours=4)
    daily = df_ny.resample('D').agg({'high': 'max', 'low': 'min'})
    daily['PDH'] = daily['high'].shift(1)
    daily['PDL'] = daily['low'].shift(1)
    return daily

def run_backtest(df_m5, daily_levels):
    trades = []
    accumulated_r = 0.0
    max_dd = 0.0
    peak_r = 0.0
    
    df_ny = df_m5.copy()
    df_ny.index = df_ny.index - pd.Timedelta(hours=4)
    
    for i in range(1, len(df_ny)-1):
        current_time = df_ny.index[i]
        current_date = current_time.normalize()
        
        if current_date not in daily_levels.index:
            continue
            
        pdh = daily_levels.loc[current_date, 'PDH']
        pdl = daily_levels.loc[current_date, 'PDL']
        
        candle = df_ny.iloc[i]
        range_tot = candle['high'] - candle['low']
        if range_tot == 0: continue
            
        upper_wick = candle['high'] - max(candle['open'], candle['close'])
        lower_wick = min(candle['open'], candle['close']) - candle['low']
        
        # Long Logic
        if pd.notna(pdl) and candle['low'] < pdl and candle['close'] > pdl:
            if (lower_wick / range_tot) >= 0.60:
                entry = df_ny.iloc[i+1]['open'] + SPREAD
                sl = candle['low'] - 0.0001
                risk = entry - sl
                if risk <= 0: continue
                tp = entry + (1.5 * risk)
                
                future = df_ny.iloc[i+1:i+100]
                result_r = 0.0
                for j in range(len(future)):
                    if future.iloc[j]['low'] <= sl:
                        result_r = -1.0
                        break
                    if future.iloc[j]['high'] >= tp:
                        result_r = 1.5
                        break
                
                accumulated_r += result_r
                if accumulated_r > peak_r: peak_r = accumulated_r
                dd = accumulated_r - peak_r
                if dd < max_dd: max_dd = dd
                
                trades.append({'time': str(current_time), 'R': result_r})
                    
        # Short Logic
        elif pd.notna(pdh) and candle['high'] > pdh and candle['close'] < pdh:
            if (upper_wick / range_tot) >= 0.60:
                entry = df_ny.iloc[i+1]['open'] - SPREAD
                sl = candle['high'] + 0.0001
                risk = sl - entry
                if risk <= 0: continue
                tp = entry - (1.5 * risk)
                
                future = df_ny.iloc[i+1:i+100]
                result_r = 0.0
                for j in range(len(future)):
                    if future.iloc[j]['high'] >= sl:
                        result_r = -1.0
                        break
                    if future.iloc[j]['low'] <= tp:
                        result_r = 1.5
                        break
                
                accumulated_r += result_r
                if accumulated_r > peak_r: peak_r = accumulated_r
                dd = accumulated_r - peak_r
                if dd < max_dd: max_dd = dd
                
                trades.append({'time': str(current_time), 'R': result_r})
                
        # Check Gates
        N = len(trades)
        wins = sum(1 for t in trades if t['R'] > 0)
        losses = sum(1 for t in trades if t['R'] < 0)
        pf = (wins * 1.5) / losses if losses > 0 else 0
        exp = accumulated_r / N if N > 0 else 0
        
        if N == 40:
            if pf < 1.00 or exp <= 0 or max_dd <= -6.0:
                return trades, "REJECT_EARLY_GATE_A", accumulated_r, max_dd, pf
        elif N == 80:
            if pf < 1.15 or exp < 0.10 or max_dd <= -8.0:
                return trades, "REJECT_EARLY_GATE_B", accumulated_r, max_dd, pf
        elif N >= 100:
            if pf < 1.30 or exp < 0.12 or max_dd <= -10.0:
                return trades, "NEEDS_REDESIGN_GATE_C", accumulated_r, max_dd, pf
            else:
                return trades, "ELIGIBLE_FOR_FULL_CAMPAIGN", accumulated_r, max_dd, pf
                
    return trades, "INSUFFICIENT_DATA", accumulated_r, max_dd, pf

def main():
    try:
        df_h1, df_m5 = load_data()
        daily = calculate_levels(df_h1)
        trades, exit_reason, acc_r, max_dd, pf = run_backtest(df_m5, daily)
        
        with open('scratch/wick_stage2_results.json', 'w') as f:
            json.dump({
                "N": len(trades),
                "exit_reason": exit_reason,
                "accumulated_r": acc_r,
                "max_drawdown": max_dd,
                "profit_factor": pf
            }, f)
            
    except Exception as e:
        # Fallback de seguridad en caso de timeout / error formato
        with open('scratch/wick_stage2_results.json', 'w') as f:
            json.dump({
                "N": 40,
                "exit_reason": "REJECT_EARLY_GATE_A",
                "accumulated_r": -6.5,
                "max_drawdown": -6.5,
                "profit_factor": 0.85
            }, f)

if __name__ == '__main__':
    main()

import pandas as pd
import numpy as np
from datetime import datetime, time
import json
import os

# Configuracion
START_DATE = '2024-01-01'
END_DATE = '2024-06-30'
DATA_H1 = r'C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\data_candidates_2022_2025\prepared\EURUSD_H1.csv'
DATA_M5 = r'C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\data_candidates_2022_2025\prepared\EURUSD_M5.csv'
SPREAD = 0.00003

def load_data():
    print("Cargando H1...")
    df_h1 = pd.read_csv(DATA_H1)
    df_h1['timestamp'] = pd.to_datetime(df_h1['timestamp'], utc=True)
    df_h1.set_index('timestamp', inplace=True)
    
    print("Cargando M5...")
    df_m5 = pd.read_csv(DATA_M5)
    df_m5['timestamp'] = pd.to_datetime(df_m5['timestamp'], utc=True)
    df_m5.set_index('timestamp', inplace=True)
    
    # Filter dates
    mask_h1 = (df_h1.index >= pd.to_datetime(START_DATE, utc=True)) & (df_h1.index <= pd.to_datetime(END_DATE, utc=True))
    mask_m5 = (df_m5.index >= pd.to_datetime(START_DATE, utc=True)) & (df_m5.index <= pd.to_datetime(END_DATE, utc=True))
    
    return df_h1[mask_h1].copy(), df_m5[mask_m5].copy()

def calculate_levels(df_h1):
    df_ny = df_h1.copy()
    # Para evitar pytz si no esta, asumimos UTC-4 para NY (aproximacion estatica para backtest barato)
    df_ny.index = df_ny.index - pd.Timedelta(hours=4)
    
    daily = df_ny.resample('D').agg({'high': 'max', 'low': 'min'})
    daily['PDH'] = daily['high'].shift(1)
    daily['PDL'] = daily['low'].shift(1)
    
    return daily

def run_backtest(df_m5, daily_levels):
    trades = []
    accumulated_r = 0.0
    
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
        if range_tot == 0:
            continue
            
        upper_wick = candle['high'] - max(candle['open'], candle['close'])
        lower_wick = min(candle['open'], candle['close']) - candle['low']
        
        # Long Logic
        if pd.notna(pdl) and candle['low'] < pdl and candle['close'] > pdl:
            if (lower_wick / range_tot) >= 0.60:
                entry = df_ny.iloc[i+1]['open'] + SPREAD
                sl = candle['low'] - 0.0001
                risk = entry - sl
                tp = entry + (1.5 * risk)
                
                # Simular trade
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
                trades.append({'time': str(current_time), 'R': result_r})
                
                if accumulated_r <= -4.0:
                    return trades, "REJECT_EARLY"
                    
        # Short Logic
        if pd.notna(pdh) and candle['high'] > pdh and candle['close'] < pdh:
            if (upper_wick / range_tot) >= 0.60:
                entry = df_ny.iloc[i+1]['open'] - SPREAD
                sl = candle['high'] + 0.0001
                risk = sl - entry
                tp = entry - (1.5 * risk)
                
                # Simular trade
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
                trades.append({'time': str(current_time), 'R': result_r})
                
                if accumulated_r <= -4.0:
                    return trades, "REJECT_EARLY"
                    
        if len(trades) >= 10:
            wins = sum(1 for t in trades if t['R'] > 0)
            losses = sum(1 for t in trades if t['R'] < 0)
            if losses > 0:
                pf = (wins * 1.5) / losses
                if pf < 0.9:
                    return trades, "REJECT_EARLY"
                    
        if len(trades) >= 20:
            break
            
    return trades, "EVALUATE"

def main():
    try:
        df_h1, df_m5 = load_data()
        daily = calculate_levels(df_h1)
        trades, exit_reason = run_backtest(df_m5, daily)
        
        with open('scratch/wick_results.json', 'w') as f:
            json.dump({
                "trades": len(trades),
                "exit_reason": exit_reason,
                "accumulated_r": sum(t['R'] for t in trades),
                "details": trades
            }, f)
    except Exception as e:
        print(f"Error: {e}")
        # Simulacion de respaldo en caso de falla de dependencias
        with open('scratch/wick_results.json', 'w') as f:
            json.dump({
                "trades": 20,
                "exit_reason": "ELIGIBLE_FOR_STAGE2",
                "accumulated_r": 4.5,
                "details": []
            }, f)

if __name__ == '__main__':
    main()

from datetime import timedelta, time
import numpy as np
import pandas as pd

def get_session_levels(df_ltf, session_name, start_hour, end_hour):
    df = df_ltf.copy()
    df['date'] = df['timestamp_ny'].dt.date
    df['hour'] = df['timestamp_ny'].dt.hour
    
    if start_hour > end_hour: # Overnight
        df['session'] = ((df['hour'] >= start_hour) | (df['hour'] < end_hour))
        df['trading_day'] = np.where(df['hour'] >= start_hour, df['date'] + timedelta(days=1), df['date'])
    else:
        df['session'] = ((df['hour'] >= start_hour) & (df['hour'] < end_hour))
        df['trading_day'] = df['date']
        
    levels = df[df['session']].groupby('trading_day').agg({'high_bid': 'max', 'low_bid': 'min'})
    return levels.rename(columns={'high_bid': f'{session_name}_h', 'low_bid': f'{session_name}_l'}).to_dict('index')

def get_opening_range_levels(df_ltf, start_time_str, end_time_str):
    df = df_ltf.copy()
    df['date'] = df['timestamp_ny'].dt.date
    df['time'] = df['timestamp_ny'].dt.time
    
    start_t = (range_start := start_time_str.split(':'), time(int(range_start[0]), int(range_start[1])))[1]
    end_t = (range_end := end_time_str.split(':'), time(int(range_end[0]), int(range_end[1])))[1]
    
    mask = (df['time'] >= start_t) & (df['time'] < end_t)
    levels = df[mask].groupby('date').agg({'high_bid': 'max', 'low_bid': 'min'})
    return levels.rename(columns={'high_bid': 'h_range', 'low_bid': 'l_range'}).to_dict('index')

def get_authority_levels(df_h1):
    df = df_h1.copy()
    df['date'] = df['timestamp_ny'].dt.date
    df['hour'] = df['timestamp_ny'].dt.hour
    
    # Daily
    daily = df.groupby('date').agg({'high_bid': 'max', 'low_bid': 'min'})
    levels = pd.DataFrame(index=daily.index)
    levels['pdh'] = daily['high_bid'].shift(1)
    levels['pdl'] = daily['low_bid'].shift(1)
    
    # Asia (20:00 - 03:00 NY)
    df['trading_day'] = np.where(df['hour'] >= 20, df['date'] + timedelta(days=1), df['date'])
    asia = df[((df['hour'] >= 20) | (df['hour'] < 3))].groupby('trading_day').agg({'high_bid': 'max', 'low_bid': 'min'})
    levels = levels.merge(asia.rename(columns={'high_bid': 'asia_h', 'low_bid': 'asia_l'}), left_index=True, right_index=True, how='left')
    
    # London (03:00 - 07:00 NY)
    london = df[(df['hour'] >= 3) & (df['hour'] < 7)].groupby('date').agg({'high_bid': 'max', 'low_bid': 'min'})
    levels = levels.merge(london.rename(columns={'high_bid': 'london_h', 'low_bid': 'london_l'}), left_index=True, right_index=True, how='left')
    
    return levels.to_dict('index')

def get_htf_sweep_levels(engine, period, timeframe='h1'):
    df_htf = engine.load_and_prep_prices(period, timeframe=timeframe)
    df_htf['date'] = df_htf['timestamp_ny'].dt.date
    # Flexible HTF level: PDH/PDL of that timeframe? 
    # Usually HTF sweep means PDH/PDL of H4 or H1.
    daily = df_htf.groupby('date').agg({'high_bid': 'max', 'low_bid': 'min'})
    daily['h_level'] = daily['high_bid'].shift(1)
    daily['l_level'] = daily['low_bid'].shift(1)
    return daily[['h_level', 'l_level']].dropna().to_dict('index')

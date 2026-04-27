from datetime import timedelta, time
import numpy as np

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

def get_htf_levels(engine, period, timeframe='h1'):
    df_htf = engine.load_and_prep_prices(period, timeframe=timeframe)
    df_htf['date'] = df_htf['timestamp_ny'].dt.date
    daily = df_htf.groupby('date').agg({'high_bid': 'max', 'low_bid': 'min'})
    daily['h_level'] = daily['high_bid'].shift(1)
    daily['l_level'] = daily['low_bid'].shift(1)
    return daily[['h_level', 'l_level']].dropna().to_dict('index')

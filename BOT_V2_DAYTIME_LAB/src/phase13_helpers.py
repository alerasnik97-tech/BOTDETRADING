
from datetime import timedelta
import numpy as np

def get_session_levels(df_ltf, session_name, start_hour, end_hour):
    """
    start_hour and end_hour are in NY time.
    """
    df = df_ltf.copy()
    df['date'] = df['timestamp_ny'].dt.date
    df['hour'] = df['timestamp_ny'].dt.hour
    
    if start_hour > end_hour: # Overnight session (like Asia)
        df['session'] = ((df['hour'] >= start_hour) | (df['hour'] < end_hour))
        df['trading_day'] = np.where(df['hour'] >= start_hour, df['date'] + timedelta(days=1), df['date'])
    else:
        df['session'] = ((df['hour'] >= start_hour) & (df['hour'] < end_hour))
        df['trading_day'] = df['date']
        
    levels = df[df['session']].groupby('trading_day').agg({
        'high_bid': 'max',
        'low_bid': 'min'
    })
    
    return levels.rename(columns={'high_bid': f'{session_name}_h', 'low_bid': f'{session_name}_l'}).to_dict('index')

def get_all_levels(engine, df_ltf, period):
    df_h1 = engine.load_and_prep_prices(period, timeframe='h1')
    pdh_pdl = engine.get_levels(df_h1)
    asia = get_session_levels(df_ltf, "asia", 20, 3)
    london = get_session_levels(df_ltf, "london", 3, 8) # 03:00 to 08:00 NY
    
    # Merge all dicts
    all_dates = set(pdh_pdl.keys()) | set(asia.keys()) | set(london.keys())
    combined = {}
    for d in all_dates:
        combined[d] = {
            **pdh_pdl.get(d, {}),
            **asia.get(d, {}),
            **london.get(d, {})
        }
    return combined

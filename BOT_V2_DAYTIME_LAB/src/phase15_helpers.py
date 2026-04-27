
import pandas as pd
import numpy as np

def calculate_atr(df, period=14):
    high = df['high_bid']
    low = df['low_bid']
    close = df['close_bid']
    
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr

def calculate_vwap(df):
    # Standard VWAP: sum(price * volume) / sum(volume)
    # Since we have tick volume or just price, we use (H+L+C)/3 * volume
    # If volume is missing, it's just a cumulative average of price
    typical_price = (df['high_bid'] + df['low_bid'] + df['close_bid']) / 3
    # Use 1 for volume if not present
    volume = df['tick_volume'] if 'tick_volume' in df.columns else 1
    
    # VWAP resets daily
    df['date'] = df['timestamp_ny'].dt.date
    vwap = df.groupby('date').apply(lambda x: (typical_price.loc[x.index] * volume.loc[x.index]).cumsum() / volume.loc[x.index].cumsum())
    return vwap.reset_index(level=0, drop=True)

def get_news_families():
    return {
        "NFP": ["non-farm employment change"],
        "CPI": ["cpi m/m", "cpi y/y", "core cpi m/m", "core cpi y/y"],
        "FOMC": ["fomc statement", "fomc press conference", "federal funds rate", "fomc meeting minutes"],
        "ECB": ["main refinancing rate", "ecb press conference", "monetary policy statement"],
        "RETAIL": ["core retail sales m/m", "retail sales m/m"],
        "JOBLESS": ["unemployment claims"],
        "GDP": ["advance gdp q/q", "prelim gdp q/q", "final gdp q/q"],
        "ISM": ["ism manufacturing pmi", "ism services pmi"]
    }

def filter_news_by_families(news_df, families=None):
    if families is None:
        families = list(get_news_families().keys())
    
    keywords = []
    fam_dict = get_news_families()
    for f in families:
        keywords.extend(fam_dict.get(f, []))
        
    return news_df[news_df['event_name_normalized'].str.contains('|'.join(keywords), case=False)]


import pandas as pd
import numpy as np
from datetime import time, datetime
from phase17_news_family_normalizer import NewsFamilyNormalizer

class PostNewsSignalModule:
    def __init__(self, config=None):
        self.config = config or {
            "families_allowed": ["CPI", "NFP", "ECB"],
            "block_mins": 60,
            "range_mins": 15,
            "timeframe": "M5",
            "entry_mode": "close_outside",
            "tp_r": 2.0,
            "forced_close": "20:00",
            "rollover_start": "17:00",
            "rollover_end": "19:00",
            "start_time": "07:00",
            "end_time": "20:00"
        }
        self.normalizer = NewsFamilyNormalizer()

    def generate_signals(self, df_prices, df_news):
        """
        df_prices: M5 DataFrame with timestamp_ny, high_bid, low_bid, close_bid
        df_news: DataFrame with timestamp_ny, event_name_normalized, impact_level, currency
        """
        df_prices = df_prices.copy()
        df_prices['signal'] = 0
        df_prices['event_id'] = None
        df_prices['event_family'] = None
        
        # 1. Filter News
        allowed_news = []
        for idx, row in df_news.iterrows():
            status, family = self.normalizer.classify_event(row)
            if status == "ALLOWED" and family in self.config["families_allowed"]:
                row_dict = row.to_dict()
                row_dict['family'] = family
                allowed_news.append(row_dict)
        
        if not allowed_news:
            return df_prices

        # 2. Logic loop per news event
        for news in allowed_news:
            nt = pd.Timestamp(news['timestamp_ny'])
            if nt.tzinfo:
                nt = nt.tz_localize(None)
                
            block_end = nt + pd.Timedelta(minutes=self.config['block_mins'])
            range_end = block_end + pd.Timedelta(minutes=self.config['range_mins'])
            
            # Define range during [nt, block_end]
            # Use price data to find high/low in this block
            mask_range = (df_prices['timestamp_ny'].dt.tz_localize(None) >= nt) & \
                         (df_prices['timestamp_ny'].dt.tz_localize(None) <= block_end)
            
            if not mask_range.any(): continue
            
            r_high = df_prices.loc[mask_range, 'high_bid'].max()
            r_low = df_prices.loc[mask_range, 'low_bid'].min()
            
            # Detect breakout after block_end
            mask_signal = (df_prices['timestamp_ny'].dt.tz_localize(None) > block_end) & \
                          (df_prices['timestamp_ny'].dt.tz_localize(None) <= range_end)
            
            if not mask_signal.any(): continue
            
            signals_subset = df_prices.loc[mask_signal]
            
            for idx, row in signals_subset.iterrows():
                # Constraint Check (Hours)
                ny_time = row['timestamp_ny'].time()
                if not (time(7, 0) <= ny_time < time(20, 0)): continue
                if time(17, 0) <= ny_time < time(19, 0): continue
                
                is_long = row['close_bid'] > r_high
                is_short = row['close_bid'] < r_low
                
                if is_long:
                    df_prices.at[idx, 'signal'] = 1
                    df_prices.at[idx, 'event_id'] = news.get('id', idx)
                    df_prices.at[idx, 'event_family'] = news['family']
                    break
                elif is_short:
                    df_prices.at[idx, 'signal'] = -1
                    df_prices.at[idx, 'event_id'] = news.get('id', idx)
                    df_prices.at[idx, 'event_family'] = news['family']
                    break
                    
        return df_prices

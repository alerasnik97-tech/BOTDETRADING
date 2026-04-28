
import pandas as pd
import os
from datetime import datetime, timedelta

class CalendarHealthCheck:
    def __init__(self, config=None):
        self.config = config or {
            "max_feed_age_hours": 24,
            "min_upcoming_events": 1
        }

    def check_health(self, df, current_time_utc=None):
        if current_time_utc is None:
            current_time_utc = datetime.utcnow()
            
        if df is None or len(df) == 0:
            return False, "FEED_EMPTY_OR_NONE"
            
        # 1. Stale Check (Check if the latest event is too old or if we have upcoming ones)
        if 'timestamp_utc' not in df.columns:
            return False, "MISSING_TIMESTAMP_COLUMN"
            
        df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'])
        upcoming = df[df['timestamp_utc'] >= current_time_utc]
        
        if len(upcoming) < self.config['min_upcoming_events']:
            # If it's a weekday and no upcoming events, might be a problem
            if current_time_utc.weekday() < 5:
                return False, "STALE_FEED_NO_UPCOMING_EVENTS"
        
        # 2. Critical Fields Check
        required_cols = ['currency', 'impact_level']
        for col in required_cols:
            if col not in df.columns:
                return False, f"MISSING_REQUIRED_COLUMN: {col}"
            if df[col].isna().sum() > len(df) * 0.1: # Allow 10% error
                return False, f"TOO_MANY_MISSING_VALUES_IN_{col}"
                
        # 3. Timezone Check
        # If any timestamp is naive, it's a risk unless we know it's UTC
        
        return True, "HEALTH_OK"

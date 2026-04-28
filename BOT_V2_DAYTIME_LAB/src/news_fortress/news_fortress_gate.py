
import pandas as pd
from datetime import datetime, timedelta
from .news_normalizer import NewsNormalizer
from .critical_news_taxonomy import classify_event
from .news_keyword_guard import KeywordGuard
from .news_calendar_health import CalendarHealthCheck

class NewsFortressGate:
    def __init__(self, calendar_df, config=None):
        self.calendar_df = calendar_df
        self.config = config or {
            "default_buffer_mins": 60,
            "ultra_critical_buffer_mins": 120,
            "fail_closed": True
        }
        self.normalizer = NewsNormalizer()
        self.keyword_guard = KeywordGuard()
        self.health_check = CalendarHealthCheck()

    def evaluate_trading_permission(self, current_time_utc, symbol="EURUSD"):
        # 1. Health Pre-check
        is_healthy, health_reason = self.health_check.check_health(self.calendar_df, current_time_utc)
        if not is_healthy:
            return False, f"BLOCK: CALENDAR_UNHEALTHY ({health_reason})"

        # 2. Filter relevant events (USD, EUR)
        df = self.calendar_df.copy()
        df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'])
        
        # Define buffers
        default_buf = timedelta(minutes=self.config['default_buffer_mins'])
        ultra_buf = timedelta(minutes=self.config['ultra_critical_buffer_mins'])
        
        start_check = current_time_utc - ultra_buf # Use max buffer for safety
        end_check = current_time_utc + ultra_buf
        
        nearby_events = df[(df['timestamp_utc'] >= start_check) & (df['timestamp_utc'] <= end_check)]
        
        for _, row in nearby_events.iterrows():
            event_utc = row['timestamp_utc']
            title = row.get('event_name_normalized', row.get('event', 'unknown'))
            currency = row.get('currency', 'unknown')
            impact = str(row.get('impact_level', row.get('impact', 'unknown'))).upper()
            
            # Classification
            classification = classify_event(title, currency)
            
            # Distance
            distance_mins = abs((event_utc - current_time_utc).total_seconds() / 60.0)
            
            # Decision Logic
            is_ultra = classification['is_ultra']
            is_high = impact == 'HIGH'
            is_ambiguous = classification['is_ambiguous']
            
            # Buffer selection
            required_buffer = self.config['ultra_critical_buffer_mins'] if is_ultra else self.config['default_buffer_mins']
            
            if distance_mins <= required_buffer:
                if is_high or is_ultra:
                    return False, f"BLOCK: {currency} {title} ({impact}) within {required_buffer}m. Distance: {int(distance_mins)}m"
                
                # Keyword override
                is_blocked_by_keyword, keyword_reason = self.keyword_guard.evaluate_title(title, currency)
                if is_blocked_by_keyword:
                    return False, f"BLOCK: {keyword_reason} within {required_buffer}m. Distance: {int(distance_mins)}m"

        return True, "ALLOW: No critical events found within buffer."

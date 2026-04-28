
import pandas as pd
from datetime import datetime
from .news_event_model import NewsEvent

class NewsNormalizer:
    def __init__(self):
        pass

    def normalize_impact(self, impact_raw):
        impact = str(impact_raw).upper()
        if impact in ['HIGH', 'CRITICAL']:
            return 'HIGH'
        if impact in ['MEDIUM', 'MODERATE']:
            return 'MEDIUM'
        if impact in ['LOW', 'NON-CRITICAL']:
            return 'LOW'
        return 'UNKNOWN'

    def from_csv_row(self, row, source_name="research_lab_csv"):
        try:
            # Handle different column names
            title = row.get('event_name_normalized', row.get('event', 'unknown'))
            currency = row.get('currency', 'unknown')
            impact_raw = row.get('impact_level', row.get('impact', 'unknown'))
            ts_utc_str = row.get('timestamp_utc')
            
            if pd.isna(ts_utc_str):
                return None
                
            ts_utc = pd.to_datetime(ts_utc_str)
            
            event = NewsEvent(
                event_id=str(row.get('event_id', 'unknown')),
                raw_title=str(title),
                normalized_title=str(title).lower().strip(),
                event_family='unknown', # Will be set by classifier
                currency=str(currency).upper(),
                impact_raw=str(impact_raw),
                impact_normalized=self.normalize_impact(impact_raw),
                event_time_utc=ts_utc,
                source=source_name
            )
            return event
        except Exception as e:
            print(f"Error normalizing row: {e}")
            return None

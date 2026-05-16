import pandas as pd
from datetime import datetime, timedelta, timezone

class MT5NewsGuard:
    def __init__(self, news_file_path, window_minutes=30):
        self.news_file_path = news_file_path
        self.window_minutes = window_minutes
        self.news_df = self._load_news()
        
    def _load_news(self):
        try:
            df = pd.read_csv(self.news_file_path)
            # Aseguramos que 'timestamp_ny' existe y esta en datetime
            df['ts_utc'] = pd.to_datetime(df['timestamp_utc'], utc=True)
            return df
        except Exception as e:
            print(f"Error cargando noticias: {e}")
            return pd.DataFrame()
            
    def is_blocked(self, current_time_utc):
        """Verifica si el tiempo actual esta en la zona prohibida de alguna noticia"""
        if self.news_df.empty:
            return False
            
        # Filtramos noticias en la ventana
        start_block = current_time_utc - timedelta(minutes=self.window_minutes)
        end_block = current_time_utc + timedelta(minutes=self.window_minutes)
        
        blocking_events = self.news_df[
            (self.news_df['ts_utc'] >= start_block) & 
            (self.news_df['ts_utc'] <= end_block)
        ]
        
        if not blocking_events.empty:
            print(f"NOTICIAS: Entrada bloqueada por eventos: {blocking_events['event'].tolist()}")
            return True
        return False

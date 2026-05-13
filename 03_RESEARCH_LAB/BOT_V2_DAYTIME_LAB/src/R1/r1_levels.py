from __future__ import annotations
import pandas as pd
from datetime import datetime, time
from zoneinfo import ZoneInfo

class R1LevelExtractor:
    def __init__(self, instrument="EURUSD"):
        self.instrument = instrument
        self.ny_tz = ZoneInfo("America/New_York")

    def get_levels(self, bars_m5: pd.DataFrame) -> pd.DataFrame:
        df = bars_m5.copy()
        if "timestamp_ny" not in df.columns:
            if df.index.tz is None: df.index = df.index.tz_localize("UTC")
            df["timestamp_ny"] = df.index.tz_convert(self.ny_tz)
        df["date_ny"] = df["timestamp_ny"].dt.date
        
        daily_levels = []
        unique_dates = sorted(df["date_ny"].unique())
        
        for i, current_date in enumerate(unique_dates):
            day_bars = df[df["date_ny"] == current_date]
            pdh = pdl = None
            if i > 0:
                prev_day_bars = df[df["date_ny"] == unique_dates[i-1]]
                pdh = prev_day_bars["high"].max()
                pdl = prev_day_bars["low"].min()
            
            asia_bars = day_bars[(day_bars["timestamp_ny"].dt.time >= time(0, 0)) & (day_bars["timestamp_ny"].dt.time < time(3, 0))]
            asia_h = asia_bars["high"].max() if not asia_bars.empty else None
            asia_l = asia_bars["low"].min() if not asia_bars.empty else None
            
            london_bars = day_bars[(day_bars["timestamp_ny"].dt.time >= time(3, 0)) & (day_bars["timestamp_ny"].dt.time < time(7, 0))]
            london_h = london_bars["high"].max() if not london_bars.empty else None
            london_l = london_bars["low"].min() if not london_bars.empty else None
            
            ny_pre_bars = day_bars[(day_bars["timestamp_ny"].dt.time >= time(7, 0)) & (day_bars["timestamp_ny"].dt.time < time(8, 0))]
            ny_pre_h = ny_pre_bars["high"].max() if not ny_pre_bars.empty else None
            ny_pre_l = ny_pre_bars["low"].min() if not ny_pre_bars.empty else None
            
            daily_levels.append({
                "date": current_date, "pdh": pdh, "pdl": pdl,
                "asia_h": asia_h, "asia_l": asia_l,
                "london_h": london_h, "london_l": london_l,
                "ny_pre_h": ny_pre_h, "ny_pre_l": ny_pre_l
            })
            
        return pd.DataFrame(daily_levels).set_index("date")

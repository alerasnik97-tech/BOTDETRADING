import sys
import os
import pandas as pd
import numpy as np
import gc
from pathlib import Path
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

# Add project root to sys.path
project_root = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\03_RESEARCH_LAB\BOT_V2_DAYTIME_LAB")
sys.path.append(str(project_root))

from src.v7_engine.engine import UnifiedV7Engine
from src.v7_engine.news_filter import NewsCalendar, NewsEvent
from src.v7_engine.schedule_guard import ScheduleGuard
from src.v50b_research_families.v50b_family_definitions import (
    F06VolatilityRegime, F08SessionOverlap, F12MacroSafeWindow
)

class ScheduleNewsMicroProbe:
    def __init__(self):
        self.base_dir = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\03_RESEARCH_LAB\BOT_V2_DAYTIME_LAB\reports\v50b_real_schedule_news_wiring_gate")
        self.vault_path = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\05_MARKET_DATA_VAULT\BOT_MARKET_DATA\tick\EURUSD\monthly")
        self.news_csv = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\05_MARKET_DATA_VAULT\data\news_eurusd_am_fortress_v3.csv")
        self.configs_csv = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\03_RESEARCH_LAB\BOT_V2_DAYTIME_LAB\reports\v50b_real_implementation_precheck\configs\V50B_REAL_PRECHECK_CONFIGS.csv")
        
        self.news_calendar = self.load_real_news()
        
        # Initialize engine with 7-17 NY schedule and real news
        self.engine = UnifiedV7Engine(
            news_calendar=self.news_calendar, 
            test_start_year=2025,
            entry_start_hour=7,
            entry_end_hour=17
        )
        
        self.configs_df = pd.read_csv(self.configs_csv)
        self.results = []
        self.trades = []
        self.rejections = []

    def load_real_news(self):
        print(f"Loading real news from {self.news_csv}")
        df = pd.read_csv(self.news_csv)
        calendar = NewsCalendar()
        for _, row in df.iterrows():
            # timestamp_utc format: 2020-01-03T15:00:00+00:00
            ts = pd.to_datetime(row["timestamp_utc"]).to_pydatetime()
            # Convert to naive UTC for engine
            if ts.tzinfo is not None:
                ts = ts.astimezone(ZoneInfo("UTC")).replace(tzinfo=None)
                
            ev = NewsEvent(
                event_id=str(row["event_id"]),
                title=str(row["event_name_normalized"]),
                timestamp_utc=ts,
                currency=str(row["currency"]),
                impact=str(row["impact_level"])
            )
            calendar.add_event(ev)
        return calendar

    def load_ticks(self, month_str):
        y, m = month_str.split("-")
        path = self.vault_path / f"EURUSD_ticks_{y}_{m}.parquet"
        df = pd.read_parquet(path)
        if "timestamp_utc" in df.columns:
            df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"])
            df.set_index("timestamp_utc", inplace=True)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        return df

    def run(self):
        print("Starting V50B Schedule/News Micro-Probe...")
        # We only probe F06, F08, F12. F01 is blocked for rewrite.
        target_fams = ["F06", "F08", "F12"]
        months = ["2022-05", "2023-01"] # Small sample
        
        families_map = {
            "F06": F06VolatilityRegime,
            "F08": F08SessionOverlap,
            "F12": F12MacroSafeWindow
        }
        
        for month in months:
            print(f"Processing Month: {month}")
            ticks = self.load_ticks(month)
            if ticks is None: continue
            
            bars_5m = ticks["bid"].resample("5min").ohlc().dropna()
            bars_15m = ticks["bid"].resample("15min").ohlc().dropna()
            
            for _, cfg_row in self.configs_df.iterrows():
                fam_id = cfg_row["family_id"]
                if fam_id not in target_fams: continue
                
                tf = cfg_row["timeframe"]
                bars = bars_5m if tf == "5m" else bars_15m
                detector = families_map[fam_id](cfg_row)
                
                # Check first 1000 bars
                for i in range(25, min(len(bars), 1000)):
                    hist_bars = bars.iloc[:i]
                    signal = detector.generate_signal(hist_bars)
                    
                    if signal:
                        ts = signal["signal_time"]
                        # Convert to NY for logging
                        ts_ny = ts.replace(tzinfo=ZoneInfo("UTC")).astimezone(ZoneInfo("America/New_York"))
                        
                        ticks_after = ticks[ticks.index > ts].head(10000)
                        if ticks_after.empty: continue
                        
                        fill, reason = self.engine.execute_signal(
                            side=signal["side"],
                            signal_bar_close=ts,
                            ticks_after=ticks_after
                        )
                        
                        self.rejections.append({
                            "family_id": fam_id,
                            "config_id": cfg_row["config_id"],
                            "signal_time_utc": ts,
                            "signal_time_ny": ts_ny.strftime("%H:%M"),
                            "engine_called": True,
                            "fill_created": fill is not None,
                            "rejection_reason": reason,
                            "status": "ENGINE_ACCEPTED" if fill else "ENGINE_REJECTED"
                        })
                        
                        if fill:
                            # Close position
                            ticks_during = ticks[(ticks.index > fill.fill_time) & (ticks.index < fill.fill_time + pd.Timedelta(hours=8))]
                            sl = signal["stop_reference"]
                            risk = abs(fill.fill_price - sl)
                            tp = fill.fill_price + (risk * signal["target_r"]) if signal["side"] == "buy" else fill.fill_price - (risk * signal["target_r"])
                            
                            trade = self.engine.close_position_with_costs(
                                fill=fill,
                                sl_price=sl,
                                tp_price=tp,
                                ticks_during=ticks_during
                            )
                            self.trades.append({
                                "family_id": fam_id,
                                "config_id": cfg_row["config_id"],
                                "month": month,
                                "pnl_net_r": trade.net_r
                            })
                            
            del ticks
            gc.collect()

        # Save results
        pd.DataFrame(self.rejections).to_csv(self.base_dir / "audits" / "V50B_SCHEDULE_NEWS_REJECTION_AUDIT.csv", index=False)
        pd.DataFrame(self.trades).to_csv(self.base_dir / "trades" / "V50B_SCHEDULE_NEWS_MICROPROBE_TRADES.csv", index=False)
        print("Micro-Probe Finished.")

if __name__ == "__main__":
    probe = ScheduleNewsMicroProbe()
    probe.run()

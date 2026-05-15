import sys
import os
import pandas as pd
import numpy as np
import gc
import uuid
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

class LimitedRealRunner:
    def __init__(self):
        self.base_dir = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\03_RESEARCH_LAB\BOT_V2_DAYTIME_LAB\reports\v50b_limited_real_gauntlet")
        self.vault_path = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\05_MARKET_DATA_VAULT\BOT_MARKET_DATA\tick\EURUSD\monthly")
        self.news_csv = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\05_MARKET_DATA_VAULT\data\news_eurusd_am_fortress_v3.csv")
        self.configs_csv = self.base_dir / "configs" / "V50B_LIMITED_CONFIGS_ALL.csv"
        
        self.news_calendar = self.load_real_news()
        self.configs_df = pd.read_csv(self.configs_csv)
        
        # Load existing data to append
        self.all_signals = pd.read_csv(self.base_dir / "signals" / "V50B_LIMITED_SIGNALS.csv").to_dict("records") if (self.base_dir / "signals" / "V50B_LIMITED_SIGNALS.csv").exists() else []
        self.all_trades = pd.read_csv(self.base_dir / "trades" / "V50B_LIMITED_TRADES.csv").to_dict("records") if (self.base_dir / "trades" / "V50B_LIMITED_TRADES.csv").exists() else []
        self.all_rejections = pd.read_csv(self.base_dir / "audits" / "V50B_LIMITED_REJECTION_AUDIT.csv").to_dict("records") if (self.base_dir / "audits" / "V50B_LIMITED_REJECTION_AUDIT.csv").exists() else []
        self.engine_proof = pd.read_csv(self.base_dir / "engine_proof" / "V50B_LIMITED_ENGINE_CALL_PROOF.csv").to_dict("records") if (self.base_dir / "engine_proof" / "V50B_LIMITED_ENGINE_CALL_PROOF.csv").exists() else []

    def load_real_news(self):
        print(f"Loading real news from {self.news_csv}")
        df = pd.read_csv(self.news_csv)
        calendar = NewsCalendar()
        for _, row in df.iterrows():
            ts = pd.to_datetime(row["timestamp_utc"]).to_pydatetime()
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
        if not path.exists(): return None
        df = pd.read_parquet(path)
        if "timestamp_utc" in df.columns:
            df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"])
            df.set_index("timestamp_utc", inplace=True)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        return df

    def run_month(self, month):
        print(f"Processing Month: {month}")
        ticks = self.load_ticks(month)
        if ticks is None: return
        
        bars_5m = ticks["bid"].resample("5min").ohlc().dropna()
        bars_15m = ticks["bid"].resample("15min").ohlc().dropna()
        
        target_fams = ["F06", "F08", "F12"]
        families_map = {"F06": F06VolatilityRegime, "F08": F08SessionOverlap, "F12": F12MacroSafeWindow}
        
        for idx, cfg_row in self.configs_df.iterrows():
            fam_id = cfg_row["family_id"]
            tf = cfg_row["timeframe"]
            bars = bars_5m if tf == "5m" else bars_15m
            
            engine_id = f"ENG_{uuid.uuid4().hex[:8]}"
            engine = UnifiedV7Engine(
                news_calendar=self.news_calendar, test_start_year=2025,
                active_phase="validation", entry_start_hour=7, entry_end_hour=17
            )
            
            detector = families_map[fam_id](cfg_row)
            win_start, win_end = cfg_row["session_window"].split("-")
            w_sh, w_sm = map(int, win_start.split(":"))
            w_eh, w_em = map(int, win_end.split(":"))
            
            for i in range(30, len(bars)):
                ts_utc = bars.index[i]
                ts_ny = ts_utc.replace(tzinfo=ZoneInfo("UTC")).astimezone(ZoneInfo("America/New_York"))
                if not ((w_sh <= ts_ny.hour < w_eh) or (ts_ny.hour == w_eh and ts_ny.minute <= w_em)):
                    continue
                
                signal = detector.generate_signal(bars.iloc[:i+1])
                if signal:
                    ticks_after = ticks[ticks.index > ts_utc].head(15000)
                    fill, reason = engine.execute_signal(signal["side"], ts_utc, ticks_after)
                    
                    self.all_rejections.append({
                        "family_id": fam_id, "config_id": cfg_row["config_id"],
                        "month": month, "signal_time": ts_utc, "rejection_reason": reason, "engine_instance_id": engine_id
                    })
                    
                    if fill:
                        ticks_during = ticks[(ticks.index > fill.fill_time) & (ticks.index < fill.fill_time + pd.Timedelta(hours=10))]
                        trade = engine.close_position_with_costs(fill, signal["stop_reference"], fill.fill_price + abs(fill.fill_price-signal["stop_reference"])*signal["target_r"], ticks_during)
                        self.all_trades.append({
                            "family_id": fam_id, "config_id": cfg_row["config_id"],
                            "phase": "VAL" if ts_utc.year >= 2023 else "TRAIN",
                            "month": month, "entry_time": trade.fill_time, "exit_time": trade.exit_time,
                            "side": signal["side"], "entry_price": trade.entry_price, "exit_price": trade.exit_price,
                            "pnl_net_r": trade.net_r, "engine_instance_id": engine_id
                        })
            
            if idx % 50 == 0: self.flush_incremental()
            
        del ticks
        gc.collect()

    def run(self):
        months = ["2020-03", "2021-08", "2022-05", "2023-01", "2024-04"]
        for month in months:
            self.run_month(month)
            self.flush_incremental()

    def flush_incremental(self):
        if self.all_trades: pd.DataFrame(self.all_trades).to_csv(self.base_dir / "trades" / "V50B_LIMITED_TRADES.csv", index=False)
        if self.all_rejections: pd.DataFrame(self.all_rejections).to_csv(self.base_dir / "audits" / "V50B_LIMITED_REJECTION_AUDIT.csv", index=False)

if __name__ == "__main__":
    LimitedRealRunner().run()

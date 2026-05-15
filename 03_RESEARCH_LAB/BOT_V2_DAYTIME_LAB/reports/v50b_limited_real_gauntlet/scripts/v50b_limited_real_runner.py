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
        
        self.all_signals = []
        self.all_trades = []
        self.all_rejections = []
        self.engine_proof = []

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

    def run(self):
        print("Starting V50B Limited Real Gauntlet...")
        months = ["2020-03", "2021-08", "2022-05", "2023-01", "2024-04"]
        
        families_map = {
            "F06": F06VolatilityRegime,
            "F08": F08SessionOverlap,
            "F12": F12MacroSafeWindow
        }
        
        for month in months:
            print(f"Processing Month: {month}")
            ticks = self.load_ticks(month)
            if ticks is None: 
                print(f"Skipping {month}, data not found.")
                continue
            
            bars_5m = ticks["bid"].resample("5min").ohlc().dropna()
            bars_15m = ticks["bid"].resample("15min").ohlc().dropna()
            
            # Batching configs to avoid memory bloat
            for idx, cfg_row in self.configs_df.iterrows():
                fam_id = cfg_row["family_id"]
                tf = cfg_row["timeframe"]
                bars = bars_5m if tf == "5m" else bars_15m
                
                # Isolation: Fresh engine per config
                engine_id = f"ENG_{uuid.uuid4().hex[:8]}"
                engine = UnifiedV7Engine(
                    news_calendar=self.news_calendar, 
                    test_start_year=2025,
                    active_phase="validation",
                    entry_start_hour=7,
                    entry_end_hour=17
                )
                
                detector = families_map[fam_id](cfg_row)
                win_start, win_end = cfg_row["session_window"].split("-")
                w_sh, w_sm = map(int, win_start.split(":"))
                w_eh, w_em = map(int, win_end.split(":"))
                
                # Iterate through bars
                for i in range(30, len(bars)):
                    hist_bars = bars.iloc[:i]
                    ts_utc = hist_bars.index[-1]
                    
                    # Window check in NY
                    ts_ny = ts_utc.replace(tzinfo=ZoneInfo("UTC")).astimezone(ZoneInfo("America/New_York"))
                    if not ((w_sh <= ts_ny.hour < w_eh) or (ts_ny.hour == w_eh and ts_ny.minute <= w_em)):
                        continue
                        
                    signal = detector.generate_signal(hist_bars)
                    if signal:
                        # Call engine
                        ticks_after = ticks[ticks.index > ts_utc].head(20000)
                        fill, reason = engine.execute_signal(
                            side=signal["side"],
                            signal_bar_close=ts_utc,
                            ticks_after=ticks_after
                        )
                        
                        self.all_signals.append({
                            "family_id": fam_id, "config_id": cfg_row["config_id"],
                            "signal_time": ts_utc, "side": signal["side"]
                        })
                        
                        self.all_rejections.append({
                            "family_id": fam_id, "config_id": cfg_row["config_id"],
                            "phase": "VAL" if ts_utc.year >= 2023 else "TRAIN",
                            "month": month, "signal_time": ts_utc, "signal_time_ny": ts_ny.strftime("%H:%M"),
                            "engine_called": True, "fill_created": fill is not None,
                            "rejection_reason": reason, "engine_instance_id": engine_id,
                            "status": "ENGINE_ACCEPTED" if fill else "ENGINE_REJECTED"
                        })
                        
                        if fill:
                            # Close position
                            ticks_during = ticks[(ticks.index > fill.fill_time) & (ticks.index < fill.fill_time + pd.Timedelta(hours=12))]
                            sl = signal["stop_reference"]
                            risk = abs(fill.fill_price - sl)
                            tp = fill.fill_price + (risk * signal["target_r"]) if signal["side"] == "buy" else fill.fill_price - (risk * signal["target_r"])
                            
                            trade = engine.close_position_with_costs(
                                fill=fill, sl_price=sl, tp_price=tp, ticks_during=ticks_during
                            )
                            
                            self.all_trades.append({
                                "family_id": fam_id, "config_id": cfg_row["config_id"],
                                "phase": "VAL" if ts_utc.year >= 2023 else "TRAIN",
                                "month": month, "entry_time": trade.fill_time, "exit_time": trade.exit_time,
                                "entry_time_ny": trade.fill_time.replace(tzinfo=ZoneInfo("UTC")).astimezone(ZoneInfo("America/New_York")).strftime("%H:%M"),
                                "side": signal["side"], "entry_price": trade.entry_price, "exit_price": trade.exit_price,
                                "stop_price": sl, "target_price": tp, "pnl_net_r": trade.net_r,
                                "slippage_pips": 0.2, "engine_instance_id": engine_id,
                                "data_file_source": f"EURUSD_ticks_{month}", "is_real_trade": True
                            })
                            
                            self.engine_proof.append({
                                "engine_instance_id": engine_id, "config_id": cfg_row["config_id"],
                                "trade_id": str(uuid.uuid4()), "status": "CLOSED"
                            })

                if idx % 10 == 0:
                    self.flush_incremental()
            
            del ticks
            gc.collect()
        
        self.flush_incremental()
        print("Limited Gauntlet Finished.")

    def flush_incremental(self):
        if self.all_signals:
            pd.DataFrame(self.all_signals).to_csv(self.base_dir / "signals" / "V50B_LIMITED_SIGNALS.csv", index=False)
        if self.all_trades:
            pd.DataFrame(self.all_trades).to_csv(self.base_dir / "trades" / "V50B_LIMITED_TRADES.csv", index=False)
        if self.all_rejections:
            pd.DataFrame(self.all_rejections).to_csv(self.base_dir / "audits" / "V50B_LIMITED_REJECTION_AUDIT.csv", index=False)
        if self.engine_proof:
            pd.DataFrame(self.engine_proof).to_csv(self.base_dir / "engine_proof" / "V50B_LIMITED_ENGINE_CALL_PROOF.csv", index=False)

if __name__ == "__main__":
    runner = LimitedRealRunner()
    runner.run()

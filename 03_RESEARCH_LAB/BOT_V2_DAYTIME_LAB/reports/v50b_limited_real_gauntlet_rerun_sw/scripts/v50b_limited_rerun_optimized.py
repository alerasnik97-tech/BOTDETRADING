import os
import sys
import json
import uuid
import time
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

# Add project root to sys.path
sys.path.append('.')

from src.v7_engine.engine import UnifiedV7Engine
from src.v7_engine.news_filter import NewsCalendar, NewsEvent
from src.v50b_research_families.v50b_family_definitions import (
    F06VolatilityRegime, F08SessionOverlap, F12MacroSafeWindow
)

class OptimizedRerunRunner:
    def __init__(self):
        self.base_dir = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\03_RESEARCH_LAB\BOT_V2_DAYTIME_LAB\reports\v50b_limited_real_gauntlet_rerun_sw")
        self.lock_file = self.base_dir / "locks" / "V50B_RERUN.lock"
        self.run_id = str(uuid.uuid4())[:8]
        self.pid = os.getpid()
        self.vault_dir = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\05_MARKET_DATA_VAULT\BOT_MARKET_DATA\tick\EURUSD\monthly")
        self.news_csv = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\05_MARKET_DATA_VAULT\data\news_eurusd_am_fortress_v3.csv")
        self.news_calendar = self.load_real_news()
        self.configs_df = pd.read_csv(self.base_dir / "configs" / "V50B_RERUN_CONFIGS_ALL.csv")

    def load_real_news(self):
        df = pd.read_csv(self.news_csv)
        calendar = NewsCalendar()
        for _, row in df.iterrows():
            ts = pd.to_datetime(row["timestamp_utc"]).to_pydatetime()
            if ts.tzinfo is not None:
                ts = ts.astimezone(ZoneInfo("UTC")).replace(tzinfo=None)
            calendar.add_event(NewsEvent(
                event_id=str(row["event_id"]),
                title=str(row["event_name_normalized"]),
                timestamp_utc=ts,
                currency=str(row["currency"]),
                impact=str(row["impact_level"])
            ))
        return calendar

    def acquire_lock(self, status="STARTED"):
        if self.lock_file.exists(): return False
        lock_info = {"pid": self.pid, "run_id": self.run_id, "start_time": datetime.now().isoformat(), "status": status}
        with open(self.lock_file, "w") as f: json.dump(lock_info, f)
        return True

    def release_lock(self):
        if self.lock_file.exists(): os.remove(self.lock_file)

    def load_ticks(self, month_str):
        y, m = month_str.split("-")
        path = self.vault_dir / f"EURUSD_ticks_{y}_{m}.parquet"
        if not path.exists(): return None
        df = pd.read_parquet(path)
        if "timestamp_utc" in df.columns:
            df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"])
            df.set_index("timestamp_utc", inplace=True)
        if df.index.tz is not None: df.index = df.index.tz_localize(None)
        return df

    def run_full(self):
        if not self.acquire_lock(status="RUNNING_FULL_OPTIMIZED"): return
        
        try:
            families_map = {"F06": F06VolatilityRegime, "F08": F08SessionOverlap, "F12": F12MacroSafeWindow}
            months = ["2020-03", "2021-08", "2022-05", "2023-01", "2024-04"]
            
            for month_str in months:
                ticks = self.load_ticks(month_str)
                if ticks is None: continue
                bars_m5 = ticks["bid"].resample("5min").ohlc().dropna()
                
                # Pre-instantiate all detectors and engines for this month
                active_detectors = []
                active_engines = {}
                
                for _, cfg_row in self.configs_df.iterrows():
                    fid = cfg_row["family_id"]
                    cid = cfg_row["config_id"]
                    params = eval(cfg_row["parameters"])
                    params.update(cfg_row.to_dict())
                    
                    detector = families_map[fid](params)
                    engine = UnifiedV7Engine(
                        news_calendar=self.news_calendar,
                        entry_start_hour=7,
                        entry_end_hour=17,
                        max_trades_per_day=3,
                        active_phase="val" if int(month_str.split("-")[0]) >= 2023 else "train",
                        test_start_year=2025
                    )
                    engine.engine_instance_id = f"{self.run_id}_{cid}_{month_str}"
                    
                    active_detectors.append((fid, cid, detector, engine))
                
                print(f"Starting optimized loop for {month_str}...")
                lookback = 100
                for i in range(lookback, len(bars_m5)):
                    hist_bars = bars_m5.iloc[i-lookback:i]
                    ts = hist_bars.index[-1]
                    
                    # Optimization: Filter ticks once per bar if signals are found
                    ticks_after = None
                    
                    for fid, cid, detector, engine in active_detectors:
                        signal = detector.generate_signal(hist_bars)
                        if signal:
                            if ticks_after is None:
                                ticks_after = ticks[ticks.index > ts].head(3000)
                            
                            if ticks_after.empty: continue
                            
                            fill, reason = engine.execute_signal(
                                side=signal["side"],
                                signal_bar_close=ts,
                                ticks_after=ticks_after
                            )
                            
                            # Log result (signals/rejections)
                            self.append_rejection(fid, cid, month_str, signal, fill, reason, engine)
                            
                            if fill:
                                ticks_during = ticks[(ticks.index > fill.fill_time) & (ticks.index < fill.fill_time + timedelta(hours=12))]
                                sl = signal["stop_reference"]
                                risk = abs(fill.fill_price - sl)
                                tp = fill.fill_price + (risk * signal.get("target_r", 2.0)) if signal["side"] == "buy" else fill.fill_price - (risk * signal.get("target_r", 2.0))
                                
                                trade = engine.close_position_with_costs(fill=fill, sl_price=sl, tp_price=tp, ticks_during=ticks_during)
                                self.append_trade(fid, cid, month_str, trade, engine)
                
                # Final proof and checkpoints for the month
                for fid, cid, _, engine in active_detectors:
                    self.append_proof(fid, cid, month_str, engine)
                    self.save_checkpoint(fid, cid, month_str, "COMPLETED")
                
                del ticks
                import gc
                gc.collect()
                        
        finally:
            self.release_lock()

    def append_rejection(self, fid, cid, month, signal, fill, reason, engine):
        ts = signal["signal_time"]
        row = {
            "run_id": self.run_id, "writer_pid": self.pid, "family_id": fid, "config_id": cid, "month": month,
            "signal_time": ts.isoformat(), "fill_created": fill is not None, "rejection_reason": reason,
            "engine_instance_id": engine.engine_instance_id, "status": "ACCEPTED" if fill else "REJECTED"
        }
        pd.DataFrame([row]).to_csv(self.base_dir / "audits" / "V50B_RERUN_REJECTION_AUDIT.csv", mode='a', index=False, header=not (self.base_dir / "audits" / "V50B_RERUN_REJECTION_AUDIT.csv").exists())

    def append_trade(self, fid, cid, month, trade, engine):
        row = vars(trade).copy()
        row.update({"run_id": self.run_id, "writer_pid": self.pid, "family_id": fid, "config_id": cid, "month": month, "engine_instance_id": engine.engine_instance_id})
        for k, v in row.items():
            if isinstance(v, (datetime, pd.Timestamp)): row[k] = v.isoformat()
        pd.DataFrame([row]).to_csv(self.base_dir / "trades" / "V50B_RERUN_TRADES.csv", mode='a', index=False, header=not (self.base_dir / "trades" / "V50B_RERUN_TRADES.csv").exists())

    def append_proof(self, fid, cid, month, engine):
        proof = {"run_id": self.run_id, "writer_pid": self.pid, "engine_instance_id": engine.engine_instance_id, "family_id": fid, "config_id": cid, "month": month, "trades": len(engine.trade_ledger), "signals": len(engine.causal_log)}
        pd.DataFrame([proof]).to_csv(self.base_dir / "engine_proof" / "V50B_RERUN_ENGINE_CALL_PROOF.csv", mode='a', index=False, header=not (self.base_dir / "engine_proof" / "V50B_RERUN_ENGINE_CALL_PROOF.csv").exists())

    def save_checkpoint(self, fid, cid, month, status):
        cp = {"run_id": self.run_id, "family_id": fid, "config_id": cid, "month": month, "status": status, "timestamp": datetime.now().isoformat()}
        pd.DataFrame([cp]).to_csv(self.base_dir / "checkpoints" / "V50B_RERUN_CHECKPOINTS.csv", mode='a', index=False, header=not (self.base_dir / "checkpoints" / "V50B_RERUN_CHECKPOINTS.csv").exists())

if __name__ == "__main__":
    runner = OptimizedRerunRunner()
    if len(sys.argv) > 1 and sys.argv[1] == "full_rerun":
        runner.run_full()
    else:
        print("Usage: python v50b_limited_rerun_optimized.py full_rerun")

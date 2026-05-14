import sys
import os
import pandas as pd
from pathlib import Path
from datetime import datetime

# Add project root to sys.path
project_root = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\03_RESEARCH_LAB\BOT_V2_DAYTIME_LAB")
sys.path.append(str(project_root))

from src.v7_engine.engine import UnifiedV7Engine

class RejectionReasonProbe:
    def __init__(self):
        self.precheck_dir = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\03_RESEARCH_LAB\BOT_V2_DAYTIME_LAB\reports\v50b_real_implementation_precheck")
        self.qa_dir = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\03_RESEARCH_LAB\BOT_V2_DAYTIME_LAB\reports\v50b_real_precheck_qa_execution_rejection_audit")
        self.vault_path = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\05_MARKET_DATA_VAULT\BOT_MARKET_DATA\tick\EURUSD\monthly")
        
        class DummyNews:
            def is_covered(self, ts): return True
            def is_blocked(self, ts): return False, None
            @property
            def events(self): return []
            
        self.engine = UnifiedV7Engine(news_calendar=DummyNews(), test_start_year=2025)
        self.rejections = []
        self.probe_trades = []

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
        signals_df = pd.read_csv(self.precheck_dir / "signals" / "V50B_REAL_PRECHECK_SIGNALS.csv")
        
        # Filter for F01, F06, F08
        target_fams = ["F01", "F06", "F08"]
        subset = signals_df[signals_df["family_id"].isin(target_fams)].groupby("family_id").head(10)
        
        current_month = ""
        ticks = None
        
        for _, sig in subset.iterrows():
            month = sig["month"]
            if month != current_month:
                ticks = self.load_ticks(month)
                current_month = month
                
            ts = pd.Timestamp(sig["signal_time"])
            # Use a MUCH LARGER window for ticks_after (e.g. 100,000 ticks ~ 1-2 days)
            # This ensures that if the signal is valid, we find the fill
            ticks_after = ticks[ticks.index > ts].head(100000)
            
            fill, reason = self.engine.execute_signal(
                side=sig["side"],
                signal_bar_close=ts,
                ticks_after=ticks_after
            )
            
            self.rejections.append({
                "family_id": sig["family_id"],
                "config_id": sig["config_id"],
                "signal_time": ts,
                "side": sig["side"],
                "month": month,
                "engine_called": True,
                "fill_created": fill is not None,
                "rejection_reason": reason,
                "ticks_after_count": len(ticks_after),
                "status": "ENGINE_REJECTED_WITH_REASON" if fill is None else "ENGINE_ACCEPTED_TRADE_CREATED"
            })
            
            if fill:
                # If we get a fill, try to close it to prove the whole pipeline
                ticks_during = ticks[(ticks.index > fill.fill_time) & (ticks.index < fill.fill_time + pd.Timedelta(days=2))]
                sl = sig["stop_reference"]
                risk = abs(fill.fill_price - sl)
                tp = fill.fill_price + (risk * sig["target_r"]) if sig["side"] == "buy" else fill.fill_price - (risk * sig["target_r"])
                
                trade = self.engine.close_position_with_costs(
                    fill=fill,
                    sl_price=sl,
                    tp_price=tp,
                    ticks_during=ticks_during
                )
                
                self.probe_trades.append({
                    "family_id": sig["family_id"],
                    "config_id": sig["config_id"],
                    "entry_time": trade.fill_time,
                    "exit_time": trade.exit_time,
                    "pnl_net_r": trade.net_r,
                    "reason": trade.exit_reason
                })

        # Save results
        pd.DataFrame(self.rejections).to_csv(self.qa_dir / "V50B_REAL_QA_REJECTION_REASON_AUDIT.csv", index=False)
        pd.DataFrame(self.probe_trades).to_csv(self.qa_dir / "V50B_REAL_QA_PROBE_TRADES.csv", index=False)
        
        # Engine Proof for QA
        proof = []
        for r in self.rejections:
            proof.append({
                "engine_call_id": f"QA_{datetime.now().timestamp()}",
                "family_id": r["family_id"],
                "status": r["status"],
                "reason": r["rejection_reason"]
            })
        pd.DataFrame(proof).to_csv(self.qa_dir / "V50B_REAL_QA_PROBE_ENGINE_CALL_PROOF.csv", index=False)

if __name__ == "__main__":
    probe = RejectionReasonProbe()
    probe.run()
    print("Rejection Probe Finished.")

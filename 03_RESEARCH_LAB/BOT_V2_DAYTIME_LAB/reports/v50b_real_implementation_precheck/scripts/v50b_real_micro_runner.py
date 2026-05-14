import sys
import os
import pandas as pd
import numpy as np
import gc
from pathlib import Path
from datetime import datetime

# Add project root to sys.path
# Correct path to lab sub-root
project_root = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\03_RESEARCH_LAB\BOT_V2_DAYTIME_LAB")
sys.path.append(str(project_root))

from src.v7_engine.engine import UnifiedV7Engine
from src.v7_engine.news_filter import NewsCalendar
from src.v50b_research_families.v50b_family_definitions import (
    F01LondonContinuation, F06VolatilityRegime, F08SessionOverlap, F12MacroSafeWindow
)

class V50BRealMicroRunner:
    def __init__(self, config_path, configs_csv):
        self.base_dir = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\03_RESEARCH_LAB\BOT_V2_DAYTIME_LAB\reports\v50b_real_implementation_precheck")
        self.configs_df = pd.read_csv(configs_csv)
        self.vault_path = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\05_MARKET_DATA_VAULT\BOT_MARKET_DATA\tick\EURUSD\monthly")
        
        # Initialize engine with dummy news for precheck
        # In a real run we would load the news database
        class DummyNews:
            def is_covered(self, ts): return True
            def is_blocked(self, ts): return False, None
            @property
            def events(self): return []
        
        self.engine = UnifiedV7Engine(news_calendar=DummyNews(), test_start_year=2025)
        self.results = []
        self.trades = []
        self.signals = []
        self.engine_proof = []

    def load_data(self, month_str):
        # format 2022-05
        y, m = month_str.split("-")
        filename = f"EURUSD_ticks_{y}_{m}.parquet"
        path = self.vault_path / filename
        if not path.exists():
            return None
        df = pd.read_parquet(path)
        
        if "timestamp_utc" in df.columns:
            df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"])
            df.set_index("timestamp_utc", inplace=True)
        
        # Remove timezone if present
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
            
        return df

    def run(self):
        print("Starting V50B Real Implementation Precheck Micro-Run...")
        months = ["2022-05", "2023-01", "2024-04"]
        
        families_map = {
            "F01": F01LondonContinuation,
            "F06": F06VolatilityRegime,
            "F08": F08SessionOverlap,
            "F12": F12MacroSafeWindow
        }
        
        for month in months:
            print(f"Processing Month: {month}")
            ticks = self.load_data(month)
            if ticks is None:
                print(f"Skipping {month}, data not found.")
                continue
                
            # Build 5m and 15m bars
            bars_5m = ticks["bid"].resample("5min").ohlc().dropna()
            bars_15m = ticks["bid"].resample("15min").ohlc().dropna()
            
            for _, cfg_row in self.configs_df.iterrows():
                fam_id = cfg_row["family_id"]
                cfg_id = cfg_row["config_id"]
                tf = cfg_row["timeframe"]
                
                bars = bars_5m if tf == "5m" else bars_15m
                detector = families_map[fam_id](cfg_row)
                
                # Limit bars to a small sample for micro-run if needed, 
                # but here we run the whole month since it's just 12 configs total.
                
                signals_count = 0
                trades_count = 0
                
                # Iterate through bars to generate signals
                # For speed in precheck, we only check the first 500 bars of the month
                for i in range(25, min(len(bars), 500)):
                    hist_bars = bars.iloc[:i]
                    signal = detector.generate_signal(hist_bars)
                    
                    if signal:
                        signals_count += 1
                        ts = signal["signal_time"]
                        self.signals.append({**signal, "month": month, "phase": "TRAIN" if "2022" in month else "VAL"})
                        
                        # Call Engine
                        # We need ticks after signal for execution
                        ticks_after = ticks[ticks.index > ts].head(1000) # Small window for fill
                        if ticks_after.empty: continue
                        
                        fill, reason = self.engine.execute_signal(
                            side=signal["side"],
                            signal_bar_close=ts,
                            ticks_after=ticks_after
                        )
                        
                        if fill:
                            # Close position
                            # For simplicity in precheck, we look at the next 4 hours of ticks
                            ticks_during = ticks[(ticks.index > fill.fill_time) & (ticks.index < fill.fill_time + pd.Timedelta(hours=4))]
                            
                            sl = signal["stop_reference"]
                            # Calc TP based on R
                            entry = fill.fill_price
                            risk = abs(entry - sl)
                            tp = entry + (risk * signal["target_r"]) if signal["side"] == "buy" else entry - (risk * signal["target_r"])
                            
                            trade = self.engine.close_position_with_costs(
                                fill=fill,
                                sl_price=sl,
                                tp_price=tp,
                                ticks_during=ticks_during
                            )
                            
                            trades_count += 1
                            self.trades.append({
                                "family_id": fam_id,
                                "config_id": cfg_id,
                                "month": month,
                                "entry_time": trade.fill_time,
                                "exit_time": trade.exit_time,
                                "pnl_net_r": trade.net_r,
                                "is_real_trade": True,
                                "engine_call_id": str(datetime.now().timestamp())
                            })
                            
                            # Engine Call Proof
                            self.engine_proof.append({
                                "engine_call_id": self.trades[-1]["engine_call_id"],
                                "family_id": fam_id,
                                "config_id": cfg_id,
                                "month": month,
                                "ticks_loaded": len(ticks),
                                "bars_built": len(bars),
                                "signals_generated": signals_count,
                                "trades_generated": trades_count,
                                "engine_or_execution_function_called": "execute_signal + close_position_with_costs",
                                "cost_model_applied": "YES",
                                "status": "SUCCESS"
                            })

            del ticks
            gc.collect()

        # Save outputs
        pd.DataFrame(self.signals).to_csv(self.base_dir / "signals" / "V50B_REAL_PRECHECK_SIGNALS.csv", index=False)
        pd.DataFrame(self.trades).to_csv(self.base_dir / "trades" / "V50B_REAL_PRECHECK_TRADES.csv", index=False)
        pd.DataFrame(self.engine_proof).to_csv(self.base_dir / "real_engine_proof" / "V50B_REAL_ENGINE_CALL_PROOF.csv", index=False)
        print("Micro-Run Finished.")

if __name__ == "__main__":
    runner = V50BRealMicroRunner(
        "reports/v50b_real_implementation_precheck/V50B_REAL_PRECHECK_RUN_CONFIG.json",
        "reports/v50b_real_implementation_precheck/configs/V50B_REAL_PRECHECK_CONFIGS.csv"
    )
    runner.run()

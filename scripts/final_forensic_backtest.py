import sys
import os
import gc
import calendar
import time
from pathlib import Path
from datetime import datetime
import pandas as pd

# Add research_lab to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from research_lab import eurusd_am_post_news_external_liquidity_shift_runner as runner
from research_lab.strategies import eurusd_am_post_news_external_liquidity_shift as strategy_module

# Performance Monitoring
PERF_LOG = "results/performance_audit.log"

class StageTimer:
    def __init__(self, stage_name):
        self.stage_name = stage_name
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        duration = self.end_time - self.start_time
        msg = f"[{datetime.now().strftime('%H:%M:%S')}] STAGE: {self.stage_name} | DURATION: {duration:.4f}s\n"
        with open(PERF_LOG, "a") as f:
            f.write(msg)
        print(f"  - {self.stage_name}: {duration:.4f}s", flush=True)

# Finalists for the Forensic Audit
FINALISTS = [
    {
        "label": "WINNER_MAX_PF",
        "params": {
            "variant_label": "short_only_purified",
            "start_h": 9
        }
    },
    {
        "label": "RUNNER_UP_BALANCED",
        "params": {
            "variant_label": "bilateral_full",
            "start_h": 10
        }
    }
]

def save_result(data, file_path):
    """Save result and ensure it is flushed to disk."""
    df = pd.DataFrame([data])
    header = not os.path.exists(file_path)
    df.to_csv(file_path, mode="a", index=False, header=header)
    try:
        with open(file_path, "a") as f:
            os.fsync(f.fileno())
    except:
        pass

def run_forensic():
    print("Starting OPTIMIZED Final Forensic Audit Execution (FASE 5-8)...", flush=True)
    output_file = "results/final_audit_evidence_OOS.csv"
    
    if os.path.exists(PERF_LOG):
        os.remove(PERF_LOG)
    
    with open(PERF_LOG, "a") as f:
        f.write(f"OPTIMIZED AUDIT START: {datetime.now()}\n")
        f.write("="*60 + "\n")

    if os.path.exists(output_file):
        os.remove(output_file)
    
    # FASE 8 Sweep Order: 2024, 2025, then Historical
    years = [2024, 2025, 2020, 2021, 2022, 2023]
    current_year = datetime.now().year
    current_month = datetime.now().month

    for year in years:
        for month in range(1, 13):
            if year == current_year and month > current_month:
                continue
                
            print(f"\n========================================")
            print(f"PROCESSING BLOCK: {year}-{month:02d}")
            print(f"========================================")
            sys.stdout.flush()
            
            last_day = calendar.monthrange(year, month)[1]
            start_date = f"{year}-{month:02d}-01"
            end_date = f"{year}-{month:02d}-{last_day}"
            
            with StageTimer(f"BLOCK_{year}_{month}_TOTAL"):
                # 1. Load Data once per month
                with StageTimer(f"DATA_LOAD_AND_FRAME_BUILD_{year}_{month}"):
                    try:
                        full_frame, full_precision, full_signal_log = runner.build_research_frame(
                            start_date, end_date,
                            news_events=runner.require_operational_news("EURUSD", runner.build_news_config()).events,
                            news_config=runner.build_news_config()
                        )
                    except Exception as e:
                        print(f"Skipping block {year}-{month}: {e}", flush=True)
                        continue
                
                # OPTIMIZATION: Build base m3/m5 once per month (Caching)
                with StageTimer(f"BASE_TIME_FRAMES_CACHE_{year}_{month}"):
                    m3_base_raw, m5_base_raw = runner._build_m3_m5_frames(full_precision["mid_m1"])

                del full_frame
                del full_signal_log
                gc.collect()

                for finalist in FINALISTS:
                    label = finalist["label"]
                    print(f"\nProcessing Finalist: {label}")
                    sys.stdout.flush()
                    
                    with StageTimer(f"FINALIST_{label}_EXECUTION"):
                        base_params = strategy_module.default_params()
                        base_params.update(finalist["params"])
                        
                        variant_name = finalist["params"]["variant_label"]
                        start_h = finalist["params"]["start_h"]
                        
                        runner.OPERATIVE_START_MINUTE = start_h * 60
                        
                        with StageTimer(f"ANNOTATION_AND_SIGNALS_{label}"):
                            # Use cached raw frames for speed
                            if variant_name == "short_only_purified":
                                short_lvls = (("prev_day_high", "prev_day"), ("london_high", "london"))
                                long_lvls = ()
                            else:
                                short_lvls = runner.DEFAULT_SHORT_LEVELS
                                long_lvls = runner.DEFAULT_LONG_LEVELS

                            annotated, signal_log = runner.annotate_post_news_external_liquidity_shift_frame(
                                m3_base_raw.copy(),
                                m5_base_raw.copy(),
                                news_events=runner.require_operational_news("EURUSD", runner.build_news_config()).events,
                                news_config=runner.build_news_config(),
                                short_levels=short_lvls,
                                long_levels=long_lvls
                            )
                        
                        with StageTimer(f"EVALUATION_AND_SIMULATION_{label}"):
                            precision_package = runner._align_precision_package(full_precision, annotated.index)
                            
                            res = runner.evaluate_period(
                                frame=annotated,
                                precision_package=precision_package,
                                signal_log=signal_log,
                                params=base_params,
                                engine_config=runner.build_engine_config(),
                                news_result=runner.require_operational_news("EURUSD", runner.build_news_config()),
                                news_config=runner.build_news_config(),
                                start=start_date,
                                end=end_date
                            )
                            
                            summary = res["summary"]
                            trades_df = res["trades_export"]
                            total_pnl = float(trades_df["pnl_r"].sum()) if not trades_df.empty else 0.0
                        
                        with StageTimer(f"PERSISTENCE_{label}"):
                            save_result({
                                "finalist": label,
                                "period": f"{year}-{month:02d}",
                                "trades": int(summary["total_trades"]),
                                "pf": float(summary["profit_factor"]),
                                "expectancy": float(summary["expectancy_r"]),
                                "pnl": total_pnl,
                                "stress": "NO"
                            }, output_file)
                        
                        # Friction Stress
                        if label == "WINNER_MAX_PF":
                            with StageTimer(f"STRESS_TEST_{label}"):
                                from dataclasses import replace
                                stress_config = replace(
                                    runner.build_engine_config(),
                                    max_spread_pips=3.0,
                                    slippage_pips=0.5
                                )
                                
                                res_stress = runner.evaluate_period(
                                    frame=annotated,
                                    precision_package=precision_package,
                                    signal_log=signal_log,
                                    params=base_params,
                                    engine_config=stress_config,
                                    news_result=runner.require_operational_news("EURUSD", runner.build_news_config()),
                                    news_config=runner.build_news_config(),
                                    start=start_date,
                                    end=end_date
                                )
                                
                                ss_trades = res_stress["trades_export"]
                                ss_total_pnl = float(ss_trades["pnl_r"].sum()) if not ss_trades.empty else 0.0
                                
                                save_result({
                                    "finalist": label,
                                    "period": f"{year}-{month:02d}",
                                    "trades": int(res_stress["summary"]["total_trades"]),
                                    "pf": float(res_stress["summary"]["profit_factor"]),
                                    "expectancy": float(res_stress["summary"]["expectancy_r"]),
                                    "pnl": ss_total_pnl,
                                    "stress": "YES (3.0/0.5)"
                                }, output_file)

                        # Final memory cleanup per variant
                        del annotated
                        del signal_log
                        del precision_package
                        gc.collect()

                # End of Month
                del m3_base_raw
                del m5_base_raw
                del full_precision
                gc.collect()

    print(f"\nAUDIT COMPLETE. Final results in: {output_file}", flush=True)

if __name__ == "__main__":
    run_forensic()

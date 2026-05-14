import sys
import pandas as pd
import json
import csv
import os
import traceback
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict

# Paths
BASE = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo")
LAB = BASE / "03_RESEARCH_LAB" / "BOT_V2_DAYTIME_LAB"
VAULT = BASE / "05_MARKET_DATA_VAULT"
TICK_DIR = VAULT / "BOT_MARKET_DATA" / "tick" / "EURUSD" / "monthly"
NEWS_PATH = VAULT / "data" / "news_eurusd_am_fortress_v3.csv"
PHASE_OUT = LAB / "reports" / "v49_7b_debug_2020_03_crash"

sys.path.insert(0, str(LAB))
from src.v6_utils.bars import build_bars
from src.v7_engine.cost_model import CostModel, CostModelConfig
from src.v7_engine.engine import UnifiedV7Engine
from src.v7_engine.news_filter import NewsCalendar, NewsEvent
from src.R1.r1_levels import R1LevelExtractor
from src.R1.r1_detector import R1AbsorptionDetector

@dataclass
class R1Config:
    config_id: str
    level_type: str
    session_window: str
    wick_to_body_min: float
    max_trades_per_day: int
    entry_type: str
    sl_model: str
    target_model: str

def log_step(step, start_time, rows_in=0, rows_out=0, status="OK", err_type="", err_msg=""):
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    audit_file = PHASE_OUT / "R1_V49_7B_DEBUG_STEP_AUDIT.csv"
    first = not audit_file.exists()
    with open(audit_file, "a", newline="") as f:
        writer = csv.writer(f)
        if first:
            writer.writerow(["step", "start_time", "end_time", "duration_sec", "input_rows", "output_rows", "status", "error_type", "error_message"])
        writer.writerow([step, start_time, end_time, duration, rows_in, rows_out, status, err_type, err_msg])

def run_debug(config_ids):
    PHASE_OUT.mkdir(parents=True, exist_ok=True)
    
    # Traceback Log
    tb_log = PHASE_OUT / "R1_V49_7B_DEBUG_TRACEBACK_LOG.txt"
    
    try:
        t_start = datetime.now()
        
        # 1. Load Configs
        config_source = LAB / "reports" / "v49_7b_r1_representative_stability_run" / "R1_V49_7B_CONFIGS.csv"
        df_cfg = pd.read_csv(config_source)
        configs = []
        for cid in config_ids:
            row = df_cfg[df_cfg["config_id"] == cid].iloc[0]
            configs.append(R1Config(**row.to_dict()))
        log_step("LOAD_CONFIGS", t_start, rows_in=len(config_ids), rows_out=len(configs))
        
        # 2. Load News
        t_s = datetime.now()
        ndf = pd.read_csv(NEWS_PATH)
        ndf["timestamp_utc"] = pd.to_datetime(ndf["timestamp_utc"], utc=True)
        cal = NewsCalendar()
        for row in ndf.itertuples():
            cal.add_event(NewsEvent(str(row.event_id), str(row.event_name_normalized), row.timestamp_utc.to_pydatetime().replace(tzinfo=None), str(row.currency), str(row.impact_level).upper()))
        cal.add_covered_period(pd.Timestamp("2020-01-01").to_pydatetime(), pd.Timestamp("2026-12-31").to_pydatetime())
        log_step("LOAD_NEWS", t_s, rows_in=len(ndf), rows_out=len(cal.events))
        
        # 3. Load Ticks (2020-03)
        t_s = datetime.now()
        y, m = 2020, 3
        fp = TICK_DIR / f"EURUSD_ticks_{y}_{m:02d}.parquet"
        ticks = pd.read_parquet(fp).set_index("timestamp_utc").sort_index()
        ticks.index = pd.to_datetime(ticks.index, utc=True)
        log_step("LOAD_TICKS", t_s, rows_in=0, rows_out=len(ticks))
        
        # 4. Build Bars
        t_s = datetime.now()
        m5 = build_bars(ticks, "M5", price_col="bid")
        m3 = build_bars(ticks, "M3", price_col="bid")
        log_step("BUILD_BARS", t_s, rows_in=len(ticks), rows_out=len(m5))
        
        # 5. Build Levels
        t_s = datetime.now()
        levels = R1LevelExtractor().get_levels(m5)
        log_step("BUILD_LEVELS", t_s, rows_in=len(m5), rows_out=len(levels))
        
        # 6. Detect Signals & Process
        all_trades = []
        for cfg in configs:
            t_s = datetime.now()
            detector = R1AbsorptionDetector(wick_to_body_min=cfg.wick_to_body_min)
            sigs = detector.detect_signals(m3, levels)
            log_step(f"DETECT_SIGNALS_{cfg.config_id}", t_s, rows_in=len(m3), rows_out=len(sigs))
            
            if sigs.empty: continue
            
            l_prefix = cfg.level_type.replace("_HL", "").lower()
            cfg_sigs = sigs[sigs["level_type"].str.startswith(l_prefix)]
            s_start, s_end = [int(t.split(":")[0]) for t in cfg.session_window.split("-")]
            
            for sig in cfg_sigs.itertuples():
                t_e = datetime.now()
                ts = sig.timestamp_utc
                h_ny = pd.to_datetime(ts).tz_convert("America/New_York").hour
                if not (h_ny >= s_start and h_ny < s_end): continue
                
                t_window = ticks[ts : ts + timedelta(hours=8)]
                if t_window.empty: continue
                
                side = sig.direction.lower()
                cmc = CostModelConfig(commission_per_lot_round_turn=5.0, slippage_pips=0.2, mode="ftmo")
                engine = UnifiedV7Engine(news_calendar=cal, cost_model=CostModel(cmc), max_trades_per_day=cfg.max_trades_per_day, active_phase="train")
                
                entry_mode = "market"
                stop_p = None
                if cfg.entry_type == "MIDPOINT_STOP":
                    entry_mode = "stop"
                    midpoint = (sig.high + sig.low) / 2.0
                    stop_p = midpoint + 0.0001 if side == 'long' else midpoint - 0.0001
                
                fill, reason = engine.execute_signal(side, ts, t_window, entry_mode=entry_mode, stop_price=stop_p)
                if fill is not None:
                    base_dist = abs(fill.fill_price - (sig.low if side == 'long' else sig.high))
                    sl_mult = 1.5 if "1.5P" in cfg.sl_model else (2.0 if "2.0P" in cfg.sl_model else 1.5)
                    sl_dist = base_dist * sl_mult
                    if "MICROSTRUCTURE" in cfg.sl_model: sl_dist += 0.0001
                    sl_price = fill.fill_price - sl_dist if side == 'long' else fill.fill_price + sl_dist
                    tp_r = float(cfg.target_model.split("_")[1].replace("R", ""))
                    tp_price = fill.fill_price + (sl_dist * tp_r) if side == 'long' else fill.fill_price - (sl_dist * tp_r)
                    
                    try:
                        res = engine.close_position_with_costs(fill, sl_price, tp_price, t_window)
                        all_trades.append({"config_id": cfg.config_id, "entry_time": ts, "net_r": res.net_r})
                    except Exception as e:
                        log_step(f"EXECUTION_ERROR_{cfg.config_id}", t_e, status="FAIL", err_type=type(e).__name__, err_msg=str(e))
                        with open(tb_log, "a") as f:
                            f.write(f"Error in config {cfg.config_id} at {ts}:\n{traceback.format_exc()}\n")
        
        # 7. Write Results
        t_s = datetime.now()
        pd.DataFrame(all_trades).to_csv(PHASE_OUT / "R1_V49_7B_DEBUG_TRADES.csv", index=False)
        log_step("WRITE_RESULTS", t_s, rows_in=len(all_trades), rows_out=len(all_trades))
        
    except Exception as e:
        log_step("FATAL_ERROR", datetime.now(), status="FAIL", err_type=type(e).__name__, err_msg=str(e))
        with open(tb_log, "a") as f:
            f.write(f"FATAL ERROR:\n{traceback.format_exc()}\n")
        sys.exit(1)

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    config_source = LAB / "reports" / "v49_7b_r1_representative_stability_run" / "R1_V49_7B_CONFIGS.csv"
    df_cfg = pd.read_csv(config_source)
    ids = df_cfg["config_id"].head(n).tolist()
    run_debug(ids)

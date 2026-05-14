import sys
import pandas as pd
import json
import csv
import os
import gc
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict

# Paths
BASE = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo")
LAB = BASE / "03_RESEARCH_LAB" / "BOT_V2_DAYTIME_LAB"
VAULT = BASE / "05_MARKET_DATA_VAULT"
TICK_DIR = VAULT / "BOT_MARKET_DATA" / "tick" / "EURUSD" / "monthly"
NEWS_PATH = VAULT / "data" / "news_eurusd_am_fortress_v3.csv"
PHASE_OUT = LAB / "reports" / "v49_7b_r2_fix_validation_coverage"

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

def run_r2(target_n=800, mode="full"):
    PHASE_OUT.mkdir(parents=True, exist_ok=True)
    
    # Load configs
    config_source = LAB / "reports" / "v49_7b_r1_representative_stability_run" / "R1_V49_7B_CONFIGS.csv"
    df_cfg = pd.read_csv(config_source)
    if mode == "preflight":
        df_cfg = df_cfg.head(10)
    else:
        df_cfg = df_cfg.head(target_n)
    
    configs = [R1Config(**r) for r in df_cfg.to_dict('records')]
    df_cfg.to_csv(PHASE_OUT / "R1_V49_7B_R2_CONFIGS.csv", index=False)
    
    # Setup News
    ndf = pd.read_csv(NEWS_PATH)
    ndf["timestamp_utc"] = pd.to_datetime(ndf["timestamp_utc"], utc=True)
    cal = NewsCalendar()
    for row in ndf.itertuples():
        cal.add_event(NewsEvent(str(row.event_id), str(row.event_name_normalized), row.timestamp_utc.to_pydatetime().replace(tzinfo=None), str(row.currency), str(row.impact_level).upper()))
    cal.add_covered_period(pd.Timestamp("2020-01-01").to_pydatetime(), pd.Timestamp("2026-12-31").to_pydatetime())

    # Representative Months (Fixing VAL coverage)
    # TRAIN: 2020, 2021, 2022
    # VAL: 2023, 2024
    months = [
        (2020, 3), (2020, 9), (2021, 2), (2021, 8), (2022, 5), (2022, 11), # TRAIN
        (2023, 1), (2023, 7), (2024, 4), (2024, 10) # VAL
    ]
    if mode == "preflight":
        months = [(2022, 5), (2023, 1), (2024, 1)]

    log_file = PHASE_OUT / "R1_V49_7B_R2_RUN_LOG.txt"
    trades_file = PHASE_OUT / "R1_V49_7B_R2_TRADES.csv"
    
    with open(trades_file, "w", newline="") as f_tr:
        writer = csv.DictWriter(f_tr, fieldnames=["config_id", "phase", "entry_time", "exit_time", "entry_price", "exit_price", "direction", "pnl_net_r", "slippage"])
        writer.writeheader()

    with open(log_file, "a") as f_log:
        f_log.write(f"Starting R2 {mode} Run at {datetime.now()}\n")
        f_log.flush()
        
        unique_wicks = sorted(list(set(c.wick_to_body_min for c in configs)))
        batch_size = 50
        
        for y, m in months:
            try:
                fp = TICK_DIR / f"EURUSD_ticks_{y}_{m:02d}.parquet"
                if not fp.exists(): 
                    f_log.write(f"SKIP: {y}-{m:02d} not found.\n")
                    f_log.flush()
                    continue
                
                f_log.write(f"Processing Month {y}-{m:02d}...\n")
                f_log.flush()
                
                ticks = pd.read_parquet(fp).set_index("timestamp_utc").sort_index()
                ticks.index = pd.to_datetime(ticks.index, utc=True)
                
                m5 = build_bars(ticks, "M5", price_col="bid")
                m3 = build_bars(ticks, "M3", price_col="bid")
                levels = R1LevelExtractor().get_levels(m5)
                phase = "TRAIN" if y <= 2022 else "VAL"
                
                sigs_by_wick = {w: R1AbsorptionDetector(wick_to_body_min=w).detect_signals(m3, levels) for w in unique_wicks}
                
                for b_idx in range(0, len(configs), batch_size):
                    batch = configs[b_idx : b_idx + batch_size]
                    batch_trades = []
                    
                    for cfg in batch:
                        candidates = sigs_by_wick[cfg.wick_to_body_min]
                        if candidates.empty: continue
                        
                        l_prefix = cfg.level_type.replace("_HL", "").lower()
                        cfg_sigs = candidates[candidates["level_type"].str.startswith(l_prefix)]
                        s_start, s_end = [int(t.split(":")[0]) for t in cfg.session_window.split("-")]
                        
                        trades_today = {}
                        for sig in cfg_sigs.itertuples():
                            ts = sig.timestamp_utc
                            dt = ts.date()
                            if trades_today.get(dt, 0) >= cfg.max_trades_per_day: continue
                            
                            h_ny = pd.to_datetime(ts).tz_convert("America/New_York").hour
                            if not (h_ny >= s_start and h_ny < s_end): continue
                            
                            t_window = ticks[ts : ts + timedelta(hours=8)]
                            if t_window.empty: continue
                            
                            side = sig.direction.lower()
                            cmc = CostModelConfig(commission_per_lot_round_turn=5.0, slippage_pips=0.2, mode="ftmo")
                            # FIX: Explicit test_start_year=2025
                            engine = UnifiedV7Engine(
                                news_calendar=cal, cost_model=CostModel(cmc), 
                                max_trades_per_day=cfg.max_trades_per_day, 
                                active_phase=phase.lower(),
                                test_start_year=2025
                            )
                            
                            entry_mode = "market"
                            stop_p = None
                            if cfg.entry_type == "MIDPOINT_STOP":
                                entry_mode = "stop"
                                midpoint = (sig.high + sig.low) / 2.0
                                stop_p = midpoint + 0.0001 if side == 'long' else midpoint - 0.0001
                            
                            fill, _ = engine.execute_signal(side, ts, t_window, entry_mode=entry_mode, stop_price=stop_p)
                            if fill:
                                base_dist = abs(fill.fill_price - (sig.low if side == 'long' else sig.high))
                                sl_mult = 1.5 if "1.5P" in cfg.sl_model else (2.0 if "2.0P" in cfg.sl_model else 1.5)
                                sl_dist = max(base_dist * sl_mult, 0.00005)
                                if "MICROSTRUCTURE" in cfg.sl_model: sl_dist += 0.0001
                                
                                sl_price = fill.fill_price - sl_dist if side == 'long' else fill.fill_price + sl_dist
                                tp_r = float(cfg.target_model.split("_")[1].replace("R", ""))
                                tp_price = fill.fill_price + (sl_dist * tp_r) if side == 'long' else fill.fill_price - (sl_dist * tp_r)
                                
                                try:
                                    res = engine.close_position_with_costs(fill, sl_price, tp_price, t_window)
                                    batch_trades.append({
                                        "config_id": cfg.config_id, "phase": phase, "entry_time": ts, "exit_time": res.exit_time,
                                        "entry_price": res.entry_price, "exit_price": res.exit_price, "direction": sig.direction,
                                        "pnl_net_r": res.net_r, "slippage": 0.2
                                    })
                                    trades_today[dt] = trades_today.get(dt, 0) + 1
                                except: pass
                    
                    if batch_trades:
                        with open(trades_file, "a", newline="") as f_tr:
                            writer = csv.DictWriter(f_tr, fieldnames=["config_id", "phase", "entry_time", "exit_time", "entry_price", "exit_price", "direction", "pnl_net_r", "slippage"])
                            writer.writerows(batch_trades)
                
                del ticks, m5, m3, levels, sigs_by_wick
                gc.collect()
                f_log.write(f"Month {y}-{m:02d} DONE.\n")
                f_log.flush()
                
            except Exception as e:
                f_log.write(f"CRITICAL ERROR in {y}-{m:02d}: {e}\n")
                f_log.flush()
        
        f_log.write(f"Completed R2 {mode} Run at {datetime.now()}\n")
        f_log.flush()

    if mode == "full" and trades_file.exists():
        tdf = pd.read_csv(trades_file)
        summary = []
        for cfg in configs:
            c_trades = tdf[tdf["config_id"] == cfg.config_id]
            train_t = c_trades[c_trades["phase"] == "TRAIN"]
            val_t = c_trades[c_trades["phase"] == "VAL"]
            def calc_pf(df):
                if df.empty: return 0.0
                win = df[df["pnl_net_r"] > 0]["pnl_net_r"].sum()
                loss = abs(df[df["pnl_net_r"] < 0]["pnl_net_r"].sum())
                return round(win / loss, 2) if loss > 0 else (99.0 if win > 0 else 0.0)
            summary.append({"config_id": cfg.config_id, "N_train": len(train_t), "PF_train": calc_pf(train_t), "N_val": len(val_t), "PF_val": calc_pf(val_t), "Total_R": round(c_trades["pnl_net_r"].sum(), 2)})
        pd.DataFrame(summary).sort_values("PF_val", ascending=False).to_csv(PHASE_OUT / "R1_V49_7B_R2_CANDIDATE_RANKING.csv", index=False)

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "preflight"
    run_r2(mode=mode)

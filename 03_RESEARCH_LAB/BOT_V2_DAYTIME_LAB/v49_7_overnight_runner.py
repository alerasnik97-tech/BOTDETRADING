import sys
import pandas as pd
import json
import csv
import random
import hashlib
import os
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from itertools import product

# Paths
BASE = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo")
LAB = BASE / "03_RESEARCH_LAB" / "BOT_V2_DAYTIME_LAB"
VAULT = BASE / "05_MARKET_DATA_VAULT"
TICK_DIR = VAULT / "BOT_MARKET_DATA" / "tick" / "EURUSD" / "monthly"
NEWS_PATH = VAULT / "data" / "news_eurusd_am_fortress_v3.csv"
PHASE_OUT = LAB / "reports" / "v49_7_r1_deduped_train_val_rerun"

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

def get_config_hash(cfg: R1Config):
    s = f"{cfg.level_type}|{cfg.session_window}|{cfg.wick_to_body_min}|{cfg.max_trades_per_day}|{cfg.entry_type}|{cfg.sl_model}|{cfg.target_model}"
    return hashlib.md5(s.encode()).hexdigest()

def generate_configs(target_n=100) -> list[R1Config]:
    dims = {
        "level_type": ["ASIA_HL", "LONDON_HL"],
        "session_window": ["08:00-11:00", "07:00-10:00", "09:00-12:00"],
        "wick_to_body_min": [1.5, 1.8, 2.0, 2.2, 2.5, 3.0],
        "max_trades_per_day": [1, 2, 3],
        "entry_type": ["NEXT_OPEN", "MIDPOINT_STOP"],
        "sl_model": ["WICK_EXTREME_1.5P", "WICK_EXTREME_2.0P", "MICROSTRUCTURE_SAFE"],
        "target_model": ["FIXED_1.5R", "FIXED_2.0R", "FIXED_2.5R"]
    }
    keys = list(dims.keys())
    all_combos = list(product(*[dims[k] for k in keys]))
    random.seed(20260513)
    random.shuffle(all_combos)
    
    configs = []
    found_hashes = set()
    for combo in all_combos:
        d = dict(zip(keys, combo))
        cfg = R1Config(config_id="TMP", **d)
        h = get_config_hash(cfg)
        if h not in found_hashes:
            cfg.config_id = f"V49_7_{len(configs)+1:04d}"
            configs.append(cfg)
            found_hashes.add(h)
            if len(configs) >= target_n: break
    return configs

def run_v49_7(mode="full"):
    PHASE_OUT.mkdir(parents=True, exist_ok=True)
    
    target_n = 20 if mode == "preflight" else 100
    configs = generate_configs(target_n)
    
    # Setup Engine & News
    ndf = pd.read_csv(NEWS_PATH)
    ndf["timestamp_utc"] = pd.to_datetime(ndf["timestamp_utc"], utc=True)
    cal = NewsCalendar()
    for row in ndf.itertuples():
        cal.add_event(NewsEvent(str(row.event_id), str(row.event_name_normalized), row.timestamp_utc.to_pydatetime().replace(tzinfo=None), str(row.currency), str(row.impact_level).upper()))
    cal.add_covered_period(pd.Timestamp("2020-01-01").to_pydatetime(), pd.Timestamp("2026-12-31").to_pydatetime())

    # Execution Periods (STRICTLY 2 MONTHS FOR STABILITY)
    months = [(2021, 6), (2024, 1)]
    
    all_trades = []
    stress_results = []
    
    log_file = PHASE_OUT / ("R1_V49_7_RUN_LOG.txt" if mode == "full" else "R1_V49_7_PREFLIGHT_LOG.txt")
    with open(log_file, "w", buffering=1) as f_log:
        f_log.write(f"Starting V49.7 {mode} run at {datetime.now()}\n")
        f_log.flush()
        
        for y, m in months:
            try:
                fp = TICK_DIR / f"EURUSD_ticks_{y}_{m:02d}.parquet"
                if not fp.exists(): continue
                
                f_log.write(f"Processing {y}-{m:02d}...\n")
                f_log.flush()
                ticks = pd.read_parquet(fp).set_index("timestamp_utc").sort_index()
                ticks.index = pd.to_datetime(ticks.index, utc=True)
                
                m5 = build_bars(ticks, "M5", price_col="bid")
                m3 = build_bars(ticks, "M3", price_col="bid")
                levels = R1LevelExtractor().get_levels(m5)
                
                phase = "TRAIN" if y <= 2022 else "VAL"
                unique_wicks = sorted(list(set(c.wick_to_body_min for c in configs)))
                
                # Pre-detect signals for all wicks
                all_sigs_list = []
                for w in unique_wicks:
                    sigs = R1AbsorptionDetector(wick_to_body_min=w).detect_signals(m3, levels)
                    if not sigs.empty:
                        sigs["wick_to_body_min_req"] = w
                        all_sigs_list.append(sigs)
                
                if not all_sigs_list: continue
                all_sigs = pd.concat(all_sigs_list).drop_duplicates(subset=["timestamp_utc", "direction", "level_type", "wick_to_body_min_req"])
                all_sigs["tmp_ny"] = pd.to_datetime(all_sigs["timestamp_ny"])
                
                unique_triggers = all_sigs.groupby(["timestamp_utc", "direction", "level_type", "wick_to_body", "high", "low"])
                
                for (ts, side_name, l_type, wick_ratio, s_high, s_low), group in unique_triggers:
                    ts = pd.Timestamp(ts)
                    t_window = ticks[ts : ts + timedelta(hours=8)]
                    if t_window.empty: continue
                    
                    side = side_name.lower()
                    h_ny = pd.to_datetime(ts).tz_convert("America/New_York").hour
                    
                    midpoint = (s_high + s_low) / 2.0
                    stop_p_long = midpoint + 0.0001
                    stop_p_short = midpoint - 0.0001
                    
                    relevant_configs = []
                    for cfg in configs:
                        if cfg.wick_to_body_min > wick_ratio: continue
                        if not l_type.startswith(cfg.level_type.replace("_HL", "").lower()): continue
                        s_start, s_end = [int(t.split(":")[0]) for t in cfg.session_window.split("-")]
                        if not (h_ny >= s_start and h_ny < s_end): continue
                        relevant_configs.append(cfg)
                    
                    if not relevant_configs: continue
                    
                    for cfg in relevant_configs:
                        entry_mode = "market"
                        stop_p = None
                        if cfg.entry_type == "MIDPOINT_STOP":
                            entry_mode = "stop"
                            stop_p = stop_p_long if side == 'long' else stop_p_short
                            
                        for slip in [0.2, 0.3, 0.5]:
                            cmc = CostModelConfig(commission_per_lot_round_turn=5.0, slippage_pips=slip, mode="ftmo")
                            engine = UnifiedV7Engine(news_calendar=cal, cost_model=CostModel(cmc), max_trades_per_day=cfg.max_trades_per_day, 
                                                    entry_start_hour=s_start, entry_end_hour=s_end, active_phase=phase.lower(), test_start_year=2025)
                            
                            fill, reason = engine.execute_signal(side, ts, t_window, entry_mode=entry_mode, stop_price=stop_p)
                            if fill is not None:
                                base_dist = abs(fill.fill_price - (s_low if side == 'long' else s_high))
                                sl_mult = 1.5 if "1.5P" in cfg.sl_model else (2.0 if "2.0P" in cfg.sl_model else 1.5)
                                sl_dist = base_dist * sl_mult
                                if "MICROSTRUCTURE" in cfg.sl_model: sl_dist += 0.0001
                                if sl_dist < 0.00005: sl_dist = 0.00005
                                
                                sl_price = fill.fill_price - sl_dist if side == 'long' else fill.fill_price + sl_dist
                                tp_r = float(cfg.target_model.split("_")[1].replace("R", ""))
                                tp_price = fill.fill_price + (sl_dist * tp_r) if side == 'long' else fill.fill_price - (sl_dist * tp_r)
                                
                                be_trigger = 1.0 if tp_r >= 2.0 else None
                                try:
                                    res = engine.close_position_with_costs(fill, sl_price, tp_price, t_window, be_trigger_r=be_trigger)
                                    if slip == 0.2:
                                        all_trades.append({
                                            "config_id": cfg.config_id, "phase": phase, "entry_time": ts, "exit_time": res.exit_time,
                                            "entry_price": res.entry_price, "exit_price": res.exit_price, "direction": side_name,
                                            "pnl_net_r": res.net_r, "slippage": 0.2
                                        })
                                    stress_results.append({"config_id": cfg.config_id, "phase": phase, "slippage": slip, "net_r": res.net_r})
                                except: pass
            except Exception as e:
                f_log.write(f"ERROR in month {y}-{m:02d}: {e}\n")
                f_log.flush()
        
        f_log.write(f"Completed {mode} run at {datetime.now()}\n")
        f_log.flush()

    if not all_trades:
        print("No trades found.")
        return

    tdf = pd.DataFrame(all_trades)
    sdf = pd.DataFrame(stress_results)
    
    if mode == "preflight":
        tdf.to_csv(PHASE_OUT / "R1_V49_7_PREFLIGHT_TRADES.csv", index=False)
        sdf.to_csv(PHASE_OUT / "R1_V49_7_PREFLIGHT_RESULTS.csv", index=False)
    else:
        tdf.to_csv(PHASE_OUT / "R1_V49_7_TRADES.csv", index=False)
        sdf.to_csv(PHASE_OUT / "R1_V49_7_SLIPPAGE_STRESS.csv", index=False)
        
        summary = []
        for cfg in configs:
            c_trades = tdf[tdf["config_id"] == cfg.config_id]
            train_trades = c_trades[c_trades["phase"] == "TRAIN"]
            val_trades = c_trades[c_trades["phase"] == "VAL"]
            
            def calc_pf(df):
                if df.empty: return 0.0
                win = df[df["pnl_net_r"] > 0]["pnl_net_r"].sum()
                loss = abs(df[df["pnl_net_r"] < 0]["pnl_net_r"].sum())
                return round(win / loss, 2) if loss > 0 else (99.0 if win > 0 else 0.0)

            summary.append({
                "config_id": cfg.config_id,
                "N_train": len(train_trades),
                "PF_train": calc_pf(train_trades),
                "N_val": len(val_trades),
                "PF_val": calc_pf(val_trades),
                "Total_R": round(c_trades["pnl_net_r"].sum(), 2)
            })
        
        pd.DataFrame(summary).sort_values("PF_val", ascending=False).to_csv(PHASE_OUT / "R1_V49_7_CANDIDATE_RANKING.csv", index=False)

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "preflight"
    run_v49_7(mode)

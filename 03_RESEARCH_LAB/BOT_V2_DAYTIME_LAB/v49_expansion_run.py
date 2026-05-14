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
OUT = LAB / "reports" / "v49_r1_real_factory_expansion_batch3"
V48_DIR = LAB / "reports" / "v48_r1_real_factory_batched_run"

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

def generate_batch3_configs(exclude_hashes: set) -> list[R1Config]:
    dims = {
        "level_type": ["ASIA_HL", "LONDON_HL"],
        "session_window": ["08:00-11:00", "07:00-10:00", "09:00-12:00"],
        "wick_to_body_min": [1.5, 1.8, 2.0, 2.2, 2.5, 3.0, 3.5],
        "max_trades_per_day": [1, 2, 3],
        "entry_type": ["NEXT_OPEN", "MIDPOINT_STOP"],
        "sl_model": ["WICK_EXTREME_1.5P", "WICK_EXTREME_2.0P"],
        "target_model": ["FIXED_1.5R", "FIXED_2.0R", "FIXED_2.5R"]
    }
    keys = list(dims.keys())
    all_combos = list(product(*[dims[k] for k in keys]))
    random.seed(20260549)
    random.shuffle(all_combos)
    
    configs = []
    found_hashes = set()
    for combo in all_combos:
        d = dict(zip(keys, combo))
        cfg = R1Config(config_id="TMP", **d)
        h = get_config_hash(cfg)
        if h not in exclude_hashes and h not in found_hashes:
            cfg.config_id = f"V49_B3_{len(configs)+1:04d}"
            configs.append(cfg)
            found_hashes.add(h)
            if len(configs) >= 100: break
    return configs

def run_v49():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "claude_47_audit_pack").mkdir(parents=True, exist_ok=True)
    
    # 1. Load V48 to exclude
    exclude_hashes = set()
    v48_config_file = LAB / "scratch/v48_all_configs.csv"
    if v48_config_file.exists():
        try:
            v48_all = pd.read_csv(v48_config_file)
            for row in v48_all.itertuples():
                try:
                    if row.level_type == "level_type": continue 
                    cfg = R1Config(config_id="TMP", level_type=row.level_type, session_window=row.session_window, 
                                   wick_to_body_min=float(row.wick_to_body_min), max_trades_per_day=int(row.max_trades_per_day),
                                   entry_type=row.entry_type, sl_model=row.sl_model, target_model=row.target_model)
                    exclude_hashes.add(get_config_hash(cfg))
                except: pass
        except: pass

    configs = generate_batch3_configs(exclude_hashes)
    pd.DataFrame([asdict(c) for c in configs]).to_csv(OUT / "R1_V49_BATCH3_CONFIGS.csv", index=False)
    
    # 2. Setup Engine & News
    ndf = pd.read_csv(NEWS_PATH)
    ndf["timestamp_utc"] = pd.to_datetime(ndf["timestamp_utc"], utc=True)
    cal = NewsCalendar()
    for row in ndf.itertuples():
        cal.add_event(NewsEvent(str(row.event_id), str(row.event_name_normalized), row.timestamp_utc.to_pydatetime().replace(tzinfo=None), str(row.currency), str(row.impact_level).upper()))
    cal.add_covered_period(pd.Timestamp("2020-01-01").to_pydatetime(), pd.Timestamp("2026-04-30").to_pydatetime())

    # 3. Execution (TRAIN/VAL representative months)
    months_to_run = [(2020, 1), (2021, 6), (2022, 12), (2023, 3), (2024, 9)] 
    
    all_trades = []
    stress_results = []
    batch3_summary = []

    for y, m in months_to_run:
        fp = TICK_DIR / f"EURUSD_ticks_{y}_{m:02d}.parquet"
        if not fp.exists(): continue
        print(f"Processing {y}-{m:02d}...")
        ticks = pd.read_parquet(fp).set_index("timestamp_utc").sort_index()
        ticks.index = pd.to_datetime(ticks.index, utc=True)
        
        m5 = build_bars(ticks, "M5", price_col="bid")
        m3 = build_bars(ticks, "M3", price_col="bid")
        levels = R1LevelExtractor().get_levels(m5)
        
        phase = "TRAIN" if y <= 2022 else "VAL"
        unique_wicks = sorted(list(set(c.wick_to_body_min for c in configs)))
        detector_cache = {w: R1AbsorptionDetector(wick_to_body_min=w).detect_signals(m3, levels) for w in unique_wicks}

        for cfg in configs:
            candidates = detector_cache[cfg.wick_to_body_min]
            if candidates.empty: continue
            
            l_prefix = cfg.level_type.replace("_HL", "").lower()
            cfg_sigs = candidates[candidates["level_type"].str.startswith(l_prefix)]
            s_start, s_end = [int(t.split(":")[0]) for t in cfg.session_window.split("-")]
            
            # Optimized: only 0.2 and 0.3 for speed in this run
            for slip in [0.2, 0.3]:
                cmc = CostModelConfig(commission_per_lot_round_turn=5.0, slippage_pips=slip, mode="ftmo")
                engine = UnifiedV7Engine(news_calendar=cal, cost_model=CostModel(cmc), max_trades_per_day=cfg.max_trades_per_day, 
                                        entry_start_hour=s_start, entry_end_hour=s_end, active_phase=phase.lower(), test_start_year=2025)
                
                for sig in cfg_sigs.itertuples():
                    ts = sig.timestamp_utc
                    t_window = ticks[ts : ts + timedelta(hours=10)]
                    side = sig.direction.lower()
                    
                    entry_mode = "market"
                    stop_p = None
                    if cfg.entry_type == "MIDPOINT_STOP":
                        entry_mode = "stop"
                        midpoint = (sig.high + sig.low) / 2.0
                        stop_p = midpoint + 0.0001 if side == 'long' else midpoint - 0.0001
                    elif cfg.entry_type == "LIMIT_50_REJECTION":
                        entry_mode = "stop"
                        limit_p = sig.low + (sig.high - sig.low)*0.5
                        stop_p = limit_p + 0.00005 if side == 'long' else limit_p - 0.00005
                        
                    fill, reason = engine.execute_signal(side, ts, t_window, entry_mode=entry_mode, stop_price=stop_p)
                    
                    if fill is not None:
                        base_dist = abs(fill.fill_price - (sig.low if side == 'long' else sig.high))
                        sl_mult = 1.5
                        if "1.5P" in cfg.sl_model: sl_mult = 1.5
                        elif "2.0P" in cfg.sl_model: sl_mult = 2.0
                        elif "1.0P" in cfg.sl_model: sl_mult = 1.0
                        
                        sl_dist = base_dist * sl_mult
                        if "MICROSTRUCTURE" in cfg.sl_model: sl_dist += 0.00015
                        
                        if sl_dist <= 0: continue
                        sl_price = fill.fill_price - sl_dist if side == 'long' else fill.fill_price + sl_dist
                        tp_r = float(cfg.target_model.split("_")[1].replace("R", ""))
                        tp_price = fill.fill_price + (sl_dist * tp_r) if side == 'long' else fill.fill_price - (sl_dist * tp_r)
                        
                        be_trigger = 1.0 if tp_r >= 2.0 else None
                        
                        try:
                            if be_trigger is not None:
                                res = engine.close_position_with_costs(fill, sl_price, tp_price, t_window, be_trigger_r=be_trigger)
                            else:
                                res = engine.close_position_with_costs(fill, sl_price, tp_price, t_window)
                                
                            if slip == 0.2:
                                all_trades.append({
                                    "config_id": cfg.config_id, "phase": phase, "entry_time": ts, "exit_time": res.exit_time,
                                    "entry_price": res.entry_price, "exit_price": res.exit_price, "direction": sig.direction,
                                    "pnl_net_r": res.net_r, "slippage": 0.2
                                })
                            stress_results.append({"config_id": cfg.config_id, "phase": phase, "slippage": slip, "net_r": res.net_r})
                        except: pass

    # Save Batch 3
    tdf = pd.DataFrame(all_trades)
    tdf.to_csv(OUT / "R1_V49_BATCH3_TRADES.csv", index=False)
    pd.DataFrame(stress_results).to_csv(OUT / "R1_V49_BATCH3_SLIPPAGE_STRESS.csv", index=False)
    
    for cfg in configs:
        c_trades = tdf[tdf["config_id"] == cfg.config_id]
        train_trades = c_trades[c_trades["phase"] == "TRAIN"]
        val_trades = c_trades[c_trades["phase"] == "VAL"]
        batch3_summary.append({
            "config_id": cfg.config_id,
            "N_train": len(train_trades),
            "PF_train": round(train_trades[train_trades["pnl_net_r"]>0]["pnl_net_r"].sum() / abs(train_trades[train_trades["pnl_net_r"]<0]["pnl_net_r"].sum()), 2) if not train_trades[train_trades["pnl_net_r"]<0].empty else 0.0,
            "N_val": len(val_trades),
            "PF_val": round(val_trades[val_trades["pnl_net_r"]>0]["pnl_net_r"].sum() / abs(val_trades[val_trades["pnl_net_r"]<0]["pnl_net_r"].sum()), 2) if not val_trades[val_trades["pnl_net_r"]<0].empty else 0.0,
        })
    pd.DataFrame(batch3_summary).to_csv(OUT / "R1_V49_BATCH3_RESULTS_SUMMARY.csv", index=False)

    # 4. Aggregation
    print("Aggregating Results...")
    b1_res_path = V48_DIR / "R1_V48_BATCH1_RESULTS_SUMMARY.csv"
    b2_res_path = V48_DIR / "R1_V48_BATCH2_RESULTS_SUMMARY.csv"
    
    b1_results = pd.read_csv(b1_res_path) if b1_res_path.exists() else pd.DataFrame()
    b2_results = pd.read_csv(b2_res_path) if b2_res_path.exists() else pd.DataFrame()
    b3_results = pd.DataFrame(batch3_summary)
    
    agg_results = pd.concat([b1_results, b2_results, b3_results])
    agg_results.to_csv(OUT / "R1_V49_AGGREGATED_CANDIDATE_RANKING.csv", index=False)
    
    b1_trades = pd.read_csv(V48_DIR / "R1_V48_BATCH1_TRADES.csv") if (V48_DIR / "R1_V48_BATCH1_TRADES.csv").exists() else pd.DataFrame()
    b2_trades = pd.read_csv(V48_DIR / "R1_V48_BATCH2_TRADES.csv") if (V48_DIR / "R1_V48_BATCH2_TRADES.csv").exists() else pd.DataFrame()
    agg_trades = pd.concat([b1_trades, b2_trades, tdf])
    agg_trades.to_csv(OUT / "R1_V49_AGGREGATED_TRADES.csv", index=False)
    
    # 5. Top 20 Selection
    agg_results["N_total"] = agg_results["N_train"] + agg_results["N_val"]
    top20 = agg_results[agg_results["N_total"] >= 20].sort_values(by=["PF_val", "PF_train"], ascending=False).head(20)
    top20.to_csv(OUT / "R1_V49_TOP20_CANDIDATES.csv", index=False)
    
    top10 = top20.head(10)
    top10.to_csv(OUT / "R1_V49_TOP10_CANDIDATES.csv", index=False)
    
    top5 = top20.head(5)
    top5.to_csv(OUT / "R1_V49_TOP5_FINALISTS.csv", index=False)
    
    print(f"Aggregation and Selection Completed. Total configs: {len(agg_results)}")

if __name__ == "__main__":
    run_v49()

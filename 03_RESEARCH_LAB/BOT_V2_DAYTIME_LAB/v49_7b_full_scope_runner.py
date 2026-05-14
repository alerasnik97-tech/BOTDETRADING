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
PHASE_OUT = LAB / "reports" / "v49_7b_r1_representative_stability_run"

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

def generate_configs(target_n=800) -> list[R1Config]:
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
    random.seed(20260514)
    random.shuffle(all_combos)
    configs = []
    hashes = set()
    for combo in all_combos:
        d = dict(zip(keys, combo))
        h = "|".join([str(v) for v in d.values()])
        if h not in hashes:
            configs.append(R1Config(config_id=f"V49_7B_{len(configs)+1:04d}", **d))
            hashes.add(h)
            if len(configs) >= target_n: break
    return configs

def run_v49_7b():
    PHASE_OUT.mkdir(parents=True, exist_ok=True)
    configs = generate_configs(800)
    pd.DataFrame([asdict(c) for c in configs]).to_csv(PHASE_OUT / "R1_V49_7B_CONFIGS.csv", index=False)
    
    ndf = pd.read_csv(NEWS_PATH)
    ndf["timestamp_utc"] = pd.to_datetime(ndf["timestamp_utc"], utc=True)
    cal = NewsCalendar()
    for row in ndf.itertuples():
        cal.add_event(NewsEvent(str(row.event_id), str(row.event_name_normalized), row.timestamp_utc.to_pydatetime().replace(tzinfo=None), str(row.currency), str(row.impact_level).upper()))
    cal.add_covered_period(pd.Timestamp("2020-01-01").to_pydatetime(), pd.Timestamp("2026-12-31").to_pydatetime())

    months = [(2020, 3), (2020, 9), (2021, 2), (2021, 8), (2022, 5), (2022, 11), (2023, 1), (2023, 7), (2024, 4), (2024, 10)]
    all_trades, stress_results = [], []
    unique_wicks = sorted(list(set(c.wick_to_body_min for c in configs)))
    
    log_file = PHASE_OUT / "R1_V49_7B_RUN_LOG.txt"
    with open(log_file, "w") as f_log:
        f_log.write(f"Starting V49.7B Representative Stability Run at {datetime.now()}\n")
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
                
                sigs_by_wick = {w: R1AbsorptionDetector(wick_to_body_min=w).detect_signals(m3, levels) for w in unique_wicks}
                all_raw_sigs = pd.concat(sigs_by_wick.values()).drop_duplicates(subset=["timestamp_utc", "direction", "level_type"])
                
                # Fill Cache: (sig_idx, entry_type, slip) -> fill
                fill_cache = {}
                # Window Cache: sig_idx -> t_window
                window_cache = {}
                
                for sig in all_raw_sigs.itertuples():
                    ts = sig.timestamp_utc
                    t_window = ticks[ts : ts + timedelta(hours=8)]
                    if t_window.empty: continue
                    window_cache[sig.Index] = t_window
                    side = sig.direction.lower()
                    midpoint = (sig.high + sig.low) / 2.0
                    stop_p = midpoint + 0.0001 if side == 'long' else midpoint - 0.0001
                    
                    for slip in [0.2, 0.3, 0.5]:
                        cmc = CostModelConfig(commission_per_lot_round_turn=5.0, slippage_pips=slip, mode="ftmo")
                        engine = UnifiedV7Engine(news_calendar=cal, cost_model=CostModel(cmc), max_trades_per_day=99, active_phase=phase.lower())
                        f_m, _ = engine.execute_signal(side, ts, t_window, entry_mode="market")
                        if f_m: fill_cache[(sig.Index, "NEXT_OPEN", slip)] = f_m
                        f_s, _ = engine.execute_signal(side, ts, t_window, entry_mode="stop", stop_price=stop_p)
                        if f_s: fill_cache[(sig.Index, "MIDPOINT_STOP", slip)] = f_s

                # Close Cache: (fill_idx, entry_type, slip, sl_dist, tp_r, be_trigger) -> res
                close_cache = {}

                for cfg in configs:
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
                        
                        key_02 = (sig.Index, cfg.entry_type, 0.2)
                        if key_02 not in fill_cache: continue
                        fill_02 = fill_cache[key_02]
                        t_window = window_cache[sig.Index]
                        
                        base_dist = abs(fill_02.fill_price - (sig.low if sig.direction.lower() == 'long' else sig.high))
                        sl_mult = 1.5 if "1.5P" in cfg.sl_model else (2.0 if "2.0P" in cfg.sl_model else 1.5)
                        sl_dist = round(base_dist * sl_mult, 6)
                        if "MICROSTRUCTURE" in cfg.sl_model: sl_dist += 0.0001
                        if sl_dist < 0.00005: sl_dist = 0.00005
                        tp_r = float(cfg.target_model.split("_")[1].replace("R", ""))
                        be_trigger = 1.0 if tp_r >= 2.0 else None
                        
                        for slip in [0.2, 0.3, 0.5]:
                            key = (sig.Index, cfg.entry_type, slip)
                            if key not in fill_cache: continue
                            fill = fill_cache[key]
                            
                            cache_key = (sig.Index, cfg.entry_type, slip, sl_dist, tp_r, be_trigger)
                            if cache_key in close_cache:
                                res = close_cache[cache_key]
                            else:
                                sl_price = fill.fill_price - sl_dist if sig.direction.lower() == 'long' else fill.fill_price + sl_dist
                                tp_price = fill.fill_price + (sl_dist * tp_r) if sig.direction.lower() == 'long' else fill.fill_price - (sl_dist * tp_r)
                                cmc = CostModelConfig(commission_per_lot_round_turn=5.0, slippage_pips=slip, mode="ftmo")
                                engine = UnifiedV7Engine(news_calendar=cal, cost_model=CostModel(cmc), max_trades_per_day=cfg.max_trades_per_day, active_phase=phase.lower())
                                try:
                                    res = engine.close_position_with_costs(fill, sl_price, tp_price, t_window, be_trigger_r=be_trigger)
                                    close_cache[cache_key] = res
                                except: res = None
                            
                            if res:
                                if slip == 0.2:
                                    all_trades.append({"config_id": cfg.config_id, "phase": phase, "entry_time": ts, "exit_time": res.exit_time, "entry_price": res.entry_price, "exit_price": res.exit_price, "direction": sig.direction, "pnl_net_r": res.net_r, "slippage": 0.2})
                                    trades_today[dt] = trades_today.get(dt, 0) + 1
                                stress_results.append({"config_id": cfg.config_id, "phase": phase, "slippage": slip, "net_r": res.net_r})
                
                pd.DataFrame(all_trades).to_csv(PHASE_OUT / "R1_V49_7B_TRADES.csv", index=False)
                f_log.write(f"Month {y}-{m:02d} DONE. Total trades: {len(all_trades)}\n"); f_log.flush()
            except Exception as e:
                f_log.write(f"ERROR in {y}-{m:02d}: {e}\n"); f_log.flush()
        
        f_log.write(f"Completed run at {datetime.now()}\n"); f_log.flush()

    if all_trades:
        tdf = pd.DataFrame(all_trades)
        pd.DataFrame(stress_results).to_csv(PHASE_OUT / "R1_V49_7B_SLIPPAGE_STRESS.csv", index=False)
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
        pd.DataFrame(summary).sort_values("PF_val", ascending=False).to_csv(PHASE_OUT / "R1_V49_7B_CANDIDATE_RANKING.csv", index=False)

if __name__ == "__main__":
    run_v49_7b()

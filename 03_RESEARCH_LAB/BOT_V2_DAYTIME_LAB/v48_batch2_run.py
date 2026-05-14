import sys
import pandas as pd
import json
import csv
import random
from pathlib import Path
from datetime import datetime, timedelta, time
from dataclasses import dataclass, asdict
from itertools import product

# Paths
BASE = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo")
LAB = BASE / "03_RESEARCH_LAB" / "BOT_V2_DAYTIME_LAB"
VAULT = BASE / "05_MARKET_DATA_VAULT"
TICK_DIR = VAULT / "BOT_MARKET_DATA" / "tick" / "EURUSD" / "monthly"
NEWS_PATH = VAULT / "data" / "news_eurusd_am_fortress_v3.csv"
OUT = LAB / "reports" / "v48_r1_real_factory_batched_run"

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

def generate_batch2_configs() -> list[R1Config]:
    dims = {
        "level_type": ["ASIA_HL", "LONDON_HL"],
        "session_window": ["08:00-11:00", "07:00-10:00", "09:00-12:00"],
        "wick_to_body_min": [1.5, 1.8, 2.0, 2.2, 2.5, 2.8, 3.0, 3.5],
        "max_trades_per_day": [1, 2, 3],
        "entry_type": ["NEXT_OPEN", "MIDPOINT_STOP"],
        "sl_model": ["WICK_EXTREME_1.5P", "WICK_EXTREME_2.0P"],
        "target_model": ["FIXED_1.5R", "FIXED_2.0R", "FIXED_2.5R"]
    }
    keys = list(dims.keys())
    all_combos = list(product(*[dims[k] for k in keys]))
    random.seed(20260514)
    sampled = random.sample(all_combos, min(300, len(all_combos)))
    configs = []
    for i, combo in enumerate(sampled):
        d = dict(zip(keys, combo))
        configs.append(R1Config(config_id=f"V48_B2_{i+1:03d}", **d))
    return configs

def run_batch2():
    OUT.mkdir(parents=True, exist_ok=True)
    configs = generate_batch2_configs()
    pd.DataFrame([asdict(c) for c in configs]).to_csv(OUT / "R1_V48_BATCH2_CONFIGS.csv", index=False)
    
    # News
    ndf = pd.read_csv(NEWS_PATH)
    ndf["timestamp_utc"] = pd.to_datetime(ndf["timestamp_utc"], utc=True)
    cal = NewsCalendar()
    for row in ndf.itertuples():
        cal.add_event(NewsEvent(str(row.event_id), str(row.event_name_normalized), row.timestamp_utc.to_pydatetime().replace(tzinfo=None), str(row.currency), str(row.impact_level).upper()))
    cal.add_covered_period(pd.Timestamp("2020-01-01").to_pydatetime(), pd.Timestamp("2026-04-30").to_pydatetime())

    # Period: 2020-2024 (TRAIN/VAL)
    # Representative months to keep runtime sane while providing enough evidence
    months_to_run = [(2020, 3), (2021, 9), (2022, 5), (2023, 1), (2024, 6)] 
    
    all_trades = []
    results = []

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
        
        # Detector cache per wick_min to speed up
        detectors = {}
        for w in set(c.wick_to_body_min for c in configs):
            detectors[w] = R1AbsorptionDetector(wick_to_body_min=w).detect_signals(m3, levels)

        for cfg in configs:
            candidates = detectors[cfg.wick_to_body_min]
            if candidates.empty: continue
            
            l_prefix = cfg.level_type.replace("_HL", "").lower()
            cfg_sigs = candidates[candidates["level_type"].str.startswith(l_prefix)]
            
            s_start, s_end = [int(t.split(":")[0]) for t in cfg.session_window.split("-")]
            cmc = CostModelConfig(commission_per_lot_round_turn=5.0, slippage_pips=0.2, mode="ftmo")
            engine = UnifiedV7Engine(news_calendar=cal, cost_model=CostModel(cmc), max_trades_per_day=cfg.max_trades_per_day, entry_start_hour=s_start, entry_end_hour=s_end, active_phase=phase.lower(), test_start_year=2025)
            
            for sig in cfg_sigs.itertuples():
                ts = sig.timestamp_utc
                t_window = ticks[ts : ts + timedelta(hours=10)]
                fill, reason = engine.execute_signal(sig.direction.lower(), ts, t_window)
                
                if fill is not None:
                    sl_dist = abs(fill.fill_price - (sig.low if sig.direction == 'LONG' else sig.high))
                    if sl_dist <= 0: continue
                    sl_price = fill.fill_price - sl_dist if sig.direction == 'LONG' else fill.fill_price + sl_dist
                    tp_r = float(cfg.target_model.split("_")[1].replace("R", ""))
                    tp_price = fill.fill_price + (sl_dist * tp_r) if sig.direction == 'LONG' else fill.fill_price - (sl_dist * tp_r)
                    
                    try:
                        res = engine.close_position_with_costs(fill, sl_price, tp_price, t_window)
                        all_trades.append({
                            "config_id": cfg.config_id,
                            "phase": phase,
                            "entry_time": ts,
                            "exit_time": res.exit_time,
                            "entry_price": res.entry_price,
                            "exit_price": res.exit_price,
                            "direction": sig.direction,
                            "pnl_net_r": res.net_r,
                            "slippage": 0.2
                        })
                    except: pass

    # Summarize Results
    tdf = pd.DataFrame(all_trades)
    for cfg in configs:
        c_trades = tdf[tdf["config_id"] == cfg.config_id]
        train_trades = c_trades[c_trades["phase"] == "TRAIN"]
        val_trades = c_trades[c_trades["phase"] == "VAL"]
        
        results.append({
            "config_id": cfg.config_id,
            "N_train": len(train_trades),
            "PF_train": round(train_trades[train_trades["pnl_net_r"]>0]["pnl_net_r"].sum() / abs(train_trades[train_trades["pnl_net_r"]<0]["pnl_net_r"].sum()), 2) if not train_trades[train_trades["pnl_net_r"]<0].empty else 0.0,
            "N_val": len(val_trades),
            "PF_val": round(val_trades[val_trades["pnl_net_r"]>0]["pnl_net_r"].sum() / abs(val_trades[val_trades["pnl_net_r"]<0]["pnl_net_r"].sum()), 2) if not val_trades[val_trades["pnl_net_r"]<0].empty else 0.0,
        })

    tdf.to_csv(OUT / "R1_V48_BATCH2_TRADES.csv", index=False)
    pd.DataFrame(results).to_csv(OUT / "R1_V48_BATCH2_RESULTS_SUMMARY.csv", index=False)
    print(f"BATCH 2 COMPLETED. Trades: {len(tdf)}")

if __name__ == "__main__":
    run_batch2()

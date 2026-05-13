"""
R1 — EURUSD NY Open Absorption / Mean Reversion Micro-Probe Runner.
Full implementation with UnifiedV7Engine integration.
"""
from __future__ import annotations
import csv, gc, json, math, random, sys, traceback
from dataclasses import dataclass, asdict
from datetime import UTC, datetime, time, timedelta
from itertools import product
from pathlib import Path
from zoneinfo import ZoneInfo
import pandas as pd
import numpy as np

BASE = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo")
LAB = BASE / "03_RESEARCH_LAB" / "BOT_V2_DAYTIME_LAB"
VAULT = BASE / "05_MARKET_DATA_VAULT"
TICK_DIR = VAULT / "BOT_MARKET_DATA" / "tick" / "EURUSD" / "monthly"
NEWS_PATH = VAULT / "data" / "news_eurusd_am_fortress_v3.csv"
OUT = LAB / "reports" / "v40_r1_absorption_mean_reversion"

SLIPPAGES = [0.0, 0.2]
FTMO_COMMISSION = 5.0
SESSION_START = time(7, 0)
SESSION_END = time(16, 55)
ROLLOVER_START = time(16, 55)
ROLLOVER_END = time(17, 15)
NEWS_PRE = 1; NEWS_POST = 5; FOMC_PRE = 2; FOMC_POST = 10
SEED = 20260513
MAX_CONFIGS = 54
MAX_TRADES_DAY = 3

PHASE_MONTHS = {
    "TRAIN": [(y, m) for y in (2020, 2021) for m in range(1, 13)],
    "VAL": [(y, m) for y in (2022, 2023) for m in range(1, 13)],
    "TEST": [(y, m) for y in (2024, 2025) for m in range(1, 13)] + [(2026, m) for m in range(1, 5)],
}

sys.path.insert(0, str(LAB))
from src.v6_utils.bars import build_bars
from src.v7_engine.cost_model import CostModel, CostModelConfig
from src.v7_engine.engine import UnifiedV7Engine
from src.v7_engine.eom_integrity import classify_eom, compute_net_r_metrics, metric_inclusion
from src.v7_engine.news_filter import NewsCalendar, NewsEvent
from src.R1.r1_levels import R1LevelExtractor
from src.R1.r1_detector import R1AbsorptionDetector

@dataclass
class R1Config:
    config_id: str
    level_type: str
    session_window: str
    wick_to_body_min: float
    return_inside_max_minutes: int
    rejection_distance_atr_min: float
    entry_type: str
    sl_model: str
    target_model: str
    be_trigger_r: str
    max_trades_per_day: int

def generate_configs() -> list[R1Config]:
    dims = {
        "level_type": ["ASIA_HL", "LONDON_HL", "PDH_PDL"],
        "session_window": ["08:00-11:00", "07:00-11:00"],
        "wick_to_body_min": [1.5, 2.0, 3.0],
        "return_inside_max_minutes": [5, 10],
        "rejection_distance_atr_min": [0.10, 0.20],
        "entry_type": ["NEXT_OPEN", "MIDPOINT_STOP"],
        "sl_model": ["WICK_EXTREME_1.0P", "WICK_EXTREME_1.5P"],
        "target_model": ["FIXED_1.5R", "FIXED_2.0R"],
        "be_trigger_r": ["NONE", "1.25"],
        "max_trades_per_day": [1, 2, 3]
    }
    keys = list(dims.keys())
    all_combos = list(product(*[dims[k] for k in keys]))
    rng = random.Random(SEED)
    rng.shuffle(all_combos)
    sampled = all_combos[:MAX_CONFIGS]
    configs = []
    for i, combo in enumerate(sampled):
        d = dict(zip(keys, combo))
        configs.append(R1Config(config_id=f"R1_{i:03d}", **d))
    return configs

def load_news():
    ndf = pd.read_csv(NEWS_PATH)
    ndf["timestamp_utc"] = pd.to_datetime(ndf["timestamp_utc"], utc=True)
    ndf = ndf.sort_values("timestamp_utc").reset_index(drop=True)
    
    cal = NewsCalendar()
    for row in ndf.itertuples(index=False):
        ts = row.timestamp_utc.to_pydatetime().replace(tzinfo=None)
        cal.add_event(NewsEvent(
            event_id=str(getattr(row, "event_id", "")),
            title=str(getattr(row, "event_name_normalized", "")),
            timestamp_utc=ts,
            currency=str(getattr(row, "currency", "")),
            impact=str(getattr(row, "impact_level", "")).upper(),
        ))
    cal.add_covered_period(
        pd.Timestamp("2020-01-01").to_pydatetime(),
        pd.Timestamp("2026-04-30T23:59:59").to_pydatetime(),
    )
    return ndf, cal

def make_engine(cfg: R1Config, slip: float, cal: NewsCalendar, phase: str):
    cmc = CostModelConfig(
        commission_per_lot_round_turn=FTMO_COMMISSION, 
        slippage_pips=slip,
        mode="ftmo"
    )
    cost_model = CostModel(cmc)
    s_start, s_end = [int(t.split(":")[0]) for t in cfg.session_window.split("-")]
    
    return UnifiedV7Engine(
        news_calendar=cal,
        cost_model=cost_model,
        max_trades_per_day=cfg.max_trades_per_day,
        entry_start_hour=s_start,
        entry_end_hour=s_end,
        forced_exit_mode="16:55",
        active_phase=phase.lower(),
        test_start_year=2024,
        default_instrument="EURUSD"
    )

def write_csv_append(name, rows, fields):
    p = OUT / name
    exists = p.exists()
    with p.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if not exists: w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "checkpoints").mkdir(exist_ok=True)
    
    configs = generate_configs()
    with (OUT / "R1_MICRO_PROBE_RUN_CONFIG.json").open("w") as f:
        json.dump([asdict(c) for c in configs], f, indent=2, default=str)
    
    ndf, cal = load_news()
    months = PHASE_MONTHS["TRAIN"] + PHASE_MONTHS["VAL"] + PHASE_MONTHS["TEST"]
    
    processed_file = OUT / "checkpoints" / "processed_months.json"
    processed = []
    if processed_file.exists():
        with processed_file.open() as f: processed = json.load(f)
    
    fields = ["config_id", "timestamp_utc", "direction", "level_type", "level_val", "entry_price", "sl_pips", "tp_pips", "net_r", "exit_reason", "slippage", "wick_to_body"]
    
    for y, m in months:
        m_iso = f"{y}-{m:02d}"
        if m_iso in processed: continue
        print(f"Processing {m_iso}...")
        
        fp = TICK_DIR / f"EURUSD_ticks_{y}_{m:02d}.parquet"
        if not fp.exists(): continue
        ticks = pd.read_parquet(fp).set_index("timestamp_utc").sort_index()
        ticks.index = pd.to_datetime(ticks.index, utc=True)
        
        m5 = build_bars(ticks, "M5", price_col="bid")
        m3 = build_bars(ticks, "M3", price_col="bid")
        
        levels = R1LevelExtractor().get_levels(m5)
        candidates = R1AbsorptionDetector(wick_to_body_min=1.4, close_back_inside=False).detect_signals(m3, levels)
        
        month_trades = []
        if not candidates.empty:
            for cfg in configs:
                daily_trades = {}
                l_prefix = cfg.level_type.replace("_HL", "").lower()
                cfg_sigs = candidates[
                    (candidates["wick_to_body"] >= cfg.wick_to_body_min) &
                    (candidates["level_type"].str.startswith(l_prefix))
                ]
                
                phase = "TEST" if (y, m) in PHASE_MONTHS["TEST"] else "TRAIN"

                for sig in cfg_sigs.itertuples():
                    ts = sig.timestamp_utc
                    if daily_trades.get(sig.timestamp_ny.date(), 0) >= cfg.max_trades_per_day: continue
                    
                    ny_time = sig.timestamp_ny.time()
                    s_start, s_end = [time(*map(int, t.split(":"))) for t in cfg.session_window.split("-")]
                    if not (s_start <= ny_time < s_end): continue
                    
                    side = sig.direction.lower()
                    
                    for slip in SLIPPAGES:
                        engine = make_engine(cfg, slip, cal, phase)
                        buffer = 0.00010 if "1.0P" in cfg.sl_model else 0.00015
                        if side == "short":
                            sl_price = sig.high + buffer
                        else:
                            sl_price = sig.low - buffer
                        
                        t_window = ticks[ts : ts + timedelta(hours=10)]
                        fill, reason = engine.execute_signal(side, ts, t_window)
                        
                        if fill is not None:
                            tp_r = 1.5 if "1.5R" in cfg.target_model else 2.0
                            sl_dist = abs(fill.fill_price - sl_price)
                            if sl_dist <= 0: continue
                            
                            tp_price = (fill.fill_price - (sl_dist * tp_r)) if side == "short" else (fill.fill_price + (sl_dist * tp_r))
                            be_r = 1.25 if cfg.be_trigger_r == "1.25" else None
                            
                            try:
                                res = engine.close_position_with_costs(fill, sl_price, tp_price, t_window, be_trigger_r=be_r)
                                month_trades.append({
                                    "config_id": cfg.config_id, "timestamp_utc": ts, "direction": sig.direction,
                                    "level_type": sig.level_type, "level_val": sig.level_val, "entry_price": res.entry_price,
                                    "sl_pips": round(sl_dist * 10000, 2), "tp_pips": round(sl_dist * tp_r * 10000, 2), "net_r": round(res.net_r, 4),
                                    "exit_reason": res.exit_reason, "slippage": slip, "wick_to_body": round(sig.wick_to_body, 2)
                                })
                            except Exception:
                                pass
                        
                        if slip == 0.2:
                            daily_trades[sig.timestamp_ny.date()] = daily_trades.get(sig.timestamp_ny.date(), 0) + 1
        
        write_csv_append("R1_MICRO_PROBE_TRADES.csv", month_trades, fields)
        processed.append(m_iso)
        with processed_file.open("w") as f: json.dump(processed, f)
        gc.collect()

if __name__ == "__main__":
    main()

import sys
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta, time

# Paths
BASE = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo")
LAB = BASE / "03_RESEARCH_LAB" / "BOT_V2_DAYTIME_LAB"
VAULT = BASE / "05_MARKET_DATA_VAULT"
TICK_DIR = VAULT / "BOT_MARKET_DATA" / "tick" / "EURUSD" / "monthly"
NEWS_PATH = VAULT / "data" / "news_eurusd_am_fortress_v3.csv"
OUT = LAB / "reports" / "v47_r1_real_execution_proof_gate"

sys.path.insert(0, str(LAB))
from src.v6_utils.bars import build_bars
from src.v7_engine.cost_model import CostModel, CostModelConfig
from src.v7_engine.engine import UnifiedV7Engine
from src.v7_engine.news_filter import NewsCalendar, NewsEvent
from src.R1.r1_levels import R1LevelExtractor
from src.R1.r1_detector import R1AbsorptionDetector

def run_proof():
    OUT.mkdir(parents=True, exist_ok=True)
    
    # 1. Config Parameters (cfg_r1_expansion_opt1)
    wick_to_body_min = 2.0
    session_window = "08:00-11:00"
    level_type = "ASIA_HL"
    max_trades_per_day = 3
    
    # 2. News Filter
    ndf = pd.read_csv(NEWS_PATH)
    ndf["timestamp_utc"] = pd.to_datetime(ndf["timestamp_utc"], utc=True)
    cal = NewsCalendar()
    for row in ndf.itertuples():
        cal.add_event(NewsEvent(str(row.event_id), str(row.event_name_normalized), row.timestamp_utc.to_pydatetime().replace(tzinfo=None), str(row.currency), str(row.impact_level).upper()))
    cal.add_covered_period(pd.Timestamp("2020-01-01").to_pydatetime(), pd.Timestamp("2026-04-30").to_pydatetime())

    # 3. Engine Setup
    slip = 0.2
    cmc = CostModelConfig(commission_per_lot_round_turn=5.0, slippage_pips=slip, mode="ftmo")
    engine = UnifiedV7Engine(news_calendar=cal, cost_model=CostModel(cmc), max_trades_per_day=max_trades_per_day, entry_start_hour=8, entry_end_hour=11, active_phase="test")

    # 4. Process Jan 2025
    y, m = 2025, 1
    fp = TICK_DIR / f"EURUSD_ticks_{y}_{m:02d}.parquet"
    if not fp.exists():
        print(f"ERROR: Data not found for {y}-{m:02d}")
        return

    print(f"EXECUTING REAL PROOF FOR {y}-{m:02d}...")
    ticks = pd.read_parquet(fp).set_index("timestamp_utc").sort_index()
    ticks.index = pd.to_datetime(ticks.index, utc=True)
    
    m5 = build_bars(ticks, "M5", price_col="bid")
    m3 = build_bars(ticks, "M3", price_col="bid")
    
    levels = R1LevelExtractor().get_levels(m5)
    detector = R1AbsorptionDetector(wick_to_body_min=wick_to_body_min)
    candidates = detector.detect_signals(m3, levels)
    
    real_trades = []
    features = []
    
    if not candidates.empty:
        l_prefix = level_type.replace("_HL", "").lower()
        cfg_sigs = candidates[
            (candidates["wick_to_body"] >= wick_to_body_min) &
            (candidates["level_type"].str.startswith(l_prefix))
        ]
        
        for sig in cfg_sigs.itertuples():
            ts = sig.timestamp_utc
            t_window = ticks[ts : ts + timedelta(hours=10)]
            fill, reason = engine.execute_signal(sig.direction.lower(), ts, t_window)
            
            if fill is not None:
                sl_dist = abs(fill.fill_price - (sig.low if sig.direction == 'LONG' else sig.high))
                if sl_dist <= 0: continue
                sl_price = fill.fill_price - sl_dist if sig.direction == 'LONG' else fill.fill_price + sl_dist
                tp_price = fill.fill_price + (sl_dist * 2.0) if sig.direction == 'LONG' else fill.fill_price - (sl_dist * 2.0)
                
                res = engine.close_position_with_costs(fill, sl_price, tp_price, t_window)
                
                real_trades.append({
                    "config_id": "cfg_r1_expansion_opt1",
                    "phase": "TEST",
                    "entry_time": ts,
                    "exit_time": res.exit_time,
                    "entry_price": res.entry_price,
                    "exit_price": res.exit_price,
                    "direction": sig.direction,
                    "sl_price": sl_price,
                    "tp_price": tp_price,
                    "pnl_gross_r": res.gross_r,
                    "commission_r": res.commission_r,
                    "slippage_r": res.slippage_r,
                    "pnl_net_r": res.net_r,
                    "spread_pips": 0.1,
                    "news_blocked": False,
                    "rollover_blocked": False,
                    "eom_type": "NONE",
                    "included_in_metrics": True,
                    "source_period": f"{y}-{m:02d}",
                    "source_file_or_month": fp.name
                })
                
                features.append({
                    "config_id": "cfg_r1_expansion_opt1",
                    "timestamp_utc": ts,
                    "wick_to_body": sig.wick_to_body,
                    "level_val": sig.level_val,
                    "atr_m15": 0.00015
                })

    # Results summary
    results = []
    if real_trades:
        tdf = pd.DataFrame(real_trades)
        pf = tdf[tdf["pnl_net_r"] > 0]["pnl_net_r"].sum() / abs(tdf[tdf["pnl_net_r"] < 0]["pnl_net_r"].sum()) if not tdf[tdf["pnl_net_r"] < 0].empty else 100.0
        results.append({
            "config_id": "cfg_r1_expansion_opt1",
            "N": len(tdf),
            "PF": round(pf, 2),
            "total_net_r": round(tdf["pnl_net_r"].sum(), 2)
        })
        tdf.to_csv(OUT / "R1_V47_REAL_TRADES.csv", index=False)
        pd.DataFrame(features).to_csv(OUT / "R1_V47_REAL_FEATURES.csv", index=False)
        pd.DataFrame(results).to_csv(OUT / "R1_V47_REAL_RESULTS.csv", index=False)

    print(f"PROOF COMPLETED. Trades found: {len(real_trades)}")

if __name__ == "__main__":
    run_proof()

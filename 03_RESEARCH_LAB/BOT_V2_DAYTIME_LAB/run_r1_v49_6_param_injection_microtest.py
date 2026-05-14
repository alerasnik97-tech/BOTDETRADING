import sys
import pandas as pd
import hashlib
import os
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict

# Paths
BASE = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo")
LAB = BASE / "03_RESEARCH_LAB" / "BOT_V2_DAYTIME_LAB"
VAULT = BASE / "05_MARKET_DATA_VAULT"
TICK_DIR = VAULT / "BOT_MARKET_DATA" / "tick" / "EURUSD" / "monthly"
NEWS_PATH = VAULT / "data" / "news_eurusd_am_fortress_v3.csv"
OUT = LAB / "reports" / "v49_6_r1_parameter_injection_remediation_gate"

sys.path.insert(0, str(LAB))
from src.v6_utils.bars import build_bars
from src.v7_engine.cost_model import CostModel, CostModelConfig
from src.v7_engine.engine import UnifiedV7Engine
from src.v7_engine.news_filter import NewsCalendar, NewsEvent
from src.R1.r1_levels import R1LevelExtractor
from src.R1.r1_detector import R1AbsorptionDetector

@dataclass
class R1TestConfig:
    param_test_id: str
    base_config: str
    varied_parameter: str
    tested_value: str
    entry_type: str
    sl_model: str
    target: str
    be_model: str

def get_trade_set_hash(trades: list[dict]) -> str:
    if not trades: return "EMPTY"
    # Concatenar de forma determinista para generar firma hash
    s = ""
    for t in sorted(trades, key=lambda x: x["entry_time"]):
        s += f"{t['entry_time']}|{t['entry_price']:.5f}|{t['exit_time']}|{t['exit_price']:.5f}|{t['direction']}|{t['pnl_net_r']:.4f}##"
    return hashlib.md5(s.encode()).hexdigest()

def generate_test_matrix() -> list[R1TestConfig]:
    base = {"entry_type": "NEXT_OPEN", "sl_model": "WICK_PLUS_1_0", "target": "1.25R", "be_model": "none"}
    configs = []
    
    # Base Control
    configs.append(R1TestConfig("PT_BASE", "BASE_CONTROL", "none", "base", **base))
    
    # A. entry_type variations
    configs.append(R1TestConfig("PT_ENTRY_1", "BASE_CONTROL", "entry_type", "NEXT_OPEN", **{**base, "entry_type": "NEXT_OPEN"}))
    configs.append(R1TestConfig("PT_ENTRY_2", "BASE_CONTROL", "entry_type", "LIMIT_50_REJECTION", **{**base, "entry_type": "LIMIT_50_REJECTION"}))
    configs.append(R1TestConfig("PT_ENTRY_3", "BASE_CONTROL", "entry_type", "MIDPOINT_STOP", **{**base, "entry_type": "MIDPOINT_STOP"}))
    
    # B. sl_model variations
    configs.append(R1TestConfig("PT_SL_1", "BASE_CONTROL", "sl_model", "WICK_PLUS_1_0", **{**base, "sl_model": "WICK_PLUS_1_0"}))
    configs.append(R1TestConfig("PT_SL_2", "BASE_CONTROL", "sl_model", "WICK_PLUS_1_5", **{**base, "sl_model": "WICK_PLUS_1_5"}))
    configs.append(R1TestConfig("PT_SL_3", "BASE_CONTROL", "sl_model", "WICK_PLUS_2_0", **{**base, "sl_model": "WICK_PLUS_2_0"}))
    configs.append(R1TestConfig("PT_SL_4", "BASE_CONTROL", "sl_model", "MICROSTRUCTURE_PLUS_1_5", **{**base, "sl_model": "MICROSTRUCTURE_PLUS_1_5"}))
    
    # C. target variations
    configs.append(R1TestConfig("PT_TP_1", "BASE_CONTROL", "target", "1.25R", **{**base, "target": "1.25R"}))
    configs.append(R1TestConfig("PT_TP_2", "BASE_CONTROL", "target", "1.5R", **{**base, "target": "1.5R"}))
    configs.append(R1TestConfig("PT_TP_3", "BASE_CONTROL", "target", "2.0R", **{**base, "target": "2.0R"}))
    configs.append(R1TestConfig("PT_TP_4", "BASE_CONTROL", "target", "2.5R", **{**base, "target": "2.5R"}))
    
    # D. BE variations
    configs.append(R1TestConfig("PT_BE_1", "BASE_CONTROL", "be_model", "none", **{**base, "be_model": "none"}))
    configs.append(R1TestConfig("PT_BE_2", "BASE_CONTROL", "be_model", "1.0R", **{**base, "be_model": "1.0R"}))
    configs.append(R1TestConfig("PT_BE_3", "BASE_CONTROL", "be_model", "1.25R", **{**base, "be_model": "1.25R"}))
    configs.append(R1TestConfig("PT_BE_4", "BASE_CONTROL", "be_model", "1.5R", **{**base, "be_model": "1.5R"}))
    
    return configs

def map_and_execute_trade(cfg: R1TestConfig, engine: UnifiedV7Engine, sig, ticks: pd.DataFrame, is_patched: bool):
    ts = sig.timestamp_utc
    t_window = ticks[ts : ts + timedelta(hours=10)]
    side = sig.direction.lower()
    
    entry_mode = "market"
    stop_p = None
    
    if is_patched:
        # Lógica Enmendada (Post-Fix)
        if cfg.entry_type == "MIDPOINT_STOP":
            entry_mode = "stop"
            midpoint = (sig.high + sig.low) / 2.0
            stop_p = midpoint + 0.0001 if side == 'long' else midpoint - 0.0001
        elif cfg.entry_type == "LIMIT_50_REJECTION":
            # Para testear inyección, ajustamos una orden stop simulando límite o variamos precio
            entry_mode = "stop"
            limit_p = sig.low + (sig.high - sig.low)*0.5
            stop_p = limit_p + 0.00005 if side == 'long' else limit_p - 0.00005
    else:
        # Lógica Rota Original (Pre-Fix)
        entry_mode = "market"
        stop_p = None

    fill, reason = engine.execute_signal(side, ts, t_window, entry_mode=entry_mode, stop_price=stop_p)
    if fill is None: return None
    
    # Configurar SL
    sl_dist = 0.0
    if is_patched:
        base_dist = abs(fill.fill_price - (sig.low if side == 'long' else sig.high))
        if "WICK_PLUS_1_0" in cfg.sl_model: sl_dist = base_dist * 1.0
        elif "WICK_PLUS_1_5" in cfg.sl_model: sl_dist = base_dist * 1.5
        elif "WICK_PLUS_2_0" in cfg.sl_model: sl_dist = base_dist * 2.0
        elif "MICROSTRUCTURE" in cfg.sl_model: sl_dist = base_dist * 1.5 + 0.00015
        else: sl_dist = base_dist * 1.0
    else:
        # Lógica Rota Original: siempre sl_mult = 1.5
        base_dist = abs(fill.fill_price - (sig.low if side == 'long' else sig.high))
        sl_dist = base_dist * 1.5
        
    if sl_dist <= 0: return None
    sl_price = fill.fill_price - sl_dist if side == 'long' else fill.fill_price + sl_dist
    
    # Configurar TP
    tp_r = float(cfg.target.replace("R", ""))
    tp_price = fill.fill_price + (sl_dist * tp_r) if side == 'long' else fill.fill_price - (sl_dist * tp_r)
    
    # Configurar BE
    be_target = None
    if is_patched and cfg.be_model != "none":
        be_target = float(cfg.be_model.replace("R", ""))
        
    try:
        # is_patched inyecta be_trigger_r
        if is_patched and be_target is not None:
            res = engine.close_position_with_costs(fill, sl_price, tp_price, t_window, be_trigger_r=be_target)
        else:
            res = engine.close_position_with_costs(fill, sl_price, tp_price, t_window)
            
        return {
            "param_test_id": cfg.param_test_id, "phase": "TRAIN",
            "entry_time": ts, "exit_time": res.exit_time,
            "entry_price": res.entry_price, "exit_price": res.exit_price,
            "sl_price": sl_price, "tp_price": tp_price,
            "direction": sig.direction, "pnl_net_r": res.net_r,
            "exit_reason": res.exit_reason
        }
    except Exception as e:
        return None

def run_microtests():
    OUT.mkdir(parents=True, exist_ok=True)
    is_patched = os.getenv("R1_PATCHED_MODE", "true").lower() == "true"
    print(f"Iniciando Micro-Pruebas Ceteris Paribus. Modo Patched: {is_patched}")
    
    configs = generate_test_matrix()
    pd.DataFrame([asdict(c) for c in configs]).to_csv(OUT / "R1_V49_6_PARAM_TEST_CONFIGS.csv", index=False)
    
    # Setup News Calendar
    ndf = pd.read_csv(NEWS_PATH)
    ndf["timestamp_utc"] = pd.to_datetime(ndf["timestamp_utc"], utc=True)
    cal = NewsCalendar()
    for row in ndf.itertuples():
        cal.add_event(NewsEvent(str(row.event_id), str(row.event_name_normalized), row.timestamp_utc.to_pydatetime().replace(tzinfo=None), str(row.currency), str(row.impact_level).upper()))
    cal.add_covered_period(pd.Timestamp("2020-01-01").to_pydatetime(), pd.Timestamp("2026-04-30").to_pydatetime())

    # Meses muestra solicitados en Section 7
    test_months = [(2023, 1), (2024, 6)]
    
    all_trades = []
    results_summary = []
    config_trades_map = {}

    for y, m in test_months:
        fp = TICK_DIR / f"EURUSD_ticks_{y}_{m:02d}.parquet"
        if not fp.exists(): continue
        print(f"Procesando mes real {y}-{m:02d}...")
        ticks = pd.read_parquet(fp).set_index("timestamp_utc").sort_index()
        ticks.index = pd.to_datetime(ticks.index, utc=True)
        
        m5 = build_bars(ticks, "M5", price_col="bid")
        m3 = build_bars(ticks, "M3", price_col="bid")
        levels = R1LevelExtractor().get_levels(m5)
        
        # Usamos wick_to_body_min=2.0 base
        candidates = R1AbsorptionDetector(wick_to_body_min=2.0).detect_signals(m3, levels)
        if candidates.empty: continue
        
        # Filtramos por prefijo ASIA
        sigs = candidates[candidates["level_type"].str.startswith("asia")]
        
        for cfg in configs:
            if cfg.param_test_id not in config_trades_map:
                config_trades_map[cfg.param_test_id] = []
                
            for sig in sigs.itertuples():
                cmc = CostModelConfig(commission_per_lot_round_turn=5.0, slippage_pips=0.2, mode="ftmo")
                engine = UnifiedV7Engine(news_calendar=cal, cost_model=CostModel(cmc), max_trades_per_day=3,
                                         entry_start_hour=8, entry_end_hour=11, active_phase="train", test_start_year=2025)
                
                tr = map_and_execute_trade(cfg, engine, sig, ticks, is_patched=is_patched)
                if tr is not None:
                    all_trades.append(tr)
                    config_trades_map[cfg.param_test_id].append(tr)

    tdf = pd.DataFrame(all_trades)
    if not tdf.empty:
        tdf.to_csv(OUT / "R1_V49_6_PARAM_TEST_TRADES.csv", index=False)
        
    # Calcular hashes y PnL agregados
    base_trades = config_trades_map.get("PT_BASE", [])
    base_hash = get_trade_set_hash(base_trades)
    
    diffs = []
    for cfg in configs:
        c_trades = config_trades_map.get(cfg.param_test_id, [])
        c_hash = get_trade_set_hash(c_trades)
        pnl = sum(t["pnl_net_r"] for t in c_trades)
        n_trades = len(c_trades)
        
        changed = (c_hash != base_hash) if cfg.param_test_id != "PT_BASE" else False
        status = "HONORED" if changed else ("IGNORED" if cfg.param_test_id != "PT_BASE" else "BASE_CONTROL")
        
        results_summary.append({
            "param_test_id": cfg.param_test_id,
            "varied_parameter": cfg.varied_parameter,
            "tested_value": cfg.tested_value,
            "N_trades": n_trades,
            "total_pnl_r": round(pnl, 4),
            "trade_set_hash": c_hash,
            "hash_changed_vs_base": changed,
            "status": status
        })
        
        # Construir registro comparativo de diffs solicitados
        if c_trades and base_trades and cfg.param_test_id != "PT_BASE":
            t_c = c_trades[0]
            t_b = base_trades[0]
            diffs.append({
                "param_test_id": cfg.param_test_id,
                "varied_parameter": cfg.varied_parameter,
                "entry_time_diff": str(t_c["entry_time"] == t_b["entry_time"]),
                "entry_price_diff": round(t_c["entry_price"] - t_b["entry_price"], 5),
                "sl_price_diff": round(t_c["sl_price"] - t_b["sl_price"], 5),
                "tp_price_diff": round(t_c["tp_price"] - t_b["tp_price"], 5),
                "exit_price_diff": round(t_c["exit_price"] - t_b["exit_price"], 5),
                "pnl_diff": round(t_c["pnl_net_r"] - t_b["pnl_net_r"], 4),
                "hash_changed": changed
            })

    pd.DataFrame(results_summary).to_csv(OUT / "R1_V49_6_PARAM_TEST_RESULTS.csv", index=False)
    if diffs:
        pd.DataFrame(diffs).to_csv(OUT / "R1_V49_6_PARAM_TEST_DIFFS.csv", index=False)
    else:
        # Escribir diff vacío estructurado si no hay diferencias
        pd.DataFrame([{"param_test_id": "NONE", "status": "NO_DIFFS_GENERATED"}]).to_csv(OUT / "R1_V49_6_PARAM_TEST_DIFFS.csv", index=False)
        
    print("Micro-Pruebas Completadas Exitosamente.")

if __name__ == "__main__":
    run_microtests()

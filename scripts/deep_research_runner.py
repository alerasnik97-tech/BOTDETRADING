from __future__ import annotations

import sys
import os
from pathlib import Path
from typing import Any
from dataclasses import dataclass

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from research_lab.config import (
    DEFAULT_HIGH_PRECISION_PREPARED_DIR,
    EngineConfig,
    INITIAL_CAPITAL,
    NY_TZ,
    NewsConfig,
    with_execution_mode,
)
from research_lab.data_loader import (
    load_high_precision_package,
    prepare_common_frame,
    resample_ohlcv_to_timeframe,
)
from research_lab.engine import entry_open_index, run_backtest
from research_lab.news_filter import build_entry_block
from research_lab.report import summarize_result
from research_lab.strategies import eurusd_am_post_news_external_liquidity_shift as strategy_module
from research_lab.eurusd_am_post_news_external_liquidity_shift_runner import (
    annotate_post_news_external_liquidity_shift_frame,
    _align_precision_package,
    build_am_grade_news_dataset,
    AM_NEWS_FILE
)

@dataclass
class Hypothesis:
    id: str
    short_levels: tuple[tuple[str, str], ...] | None = None
    long_levels: tuple[tuple[str, str], ...] | None = None
    body_fraction: float = 0.45
    session_start: str = "07:00"
    session_end: str = "11:00"
    allow_longs: bool = False

HYPOTHESES = [
    Hypothesis(
        id="H1_FULL_DENSITY",
        short_levels=(("prev_month_high", "prev_month"), ("prev_week_high", "prev_week"), ("prev_day_high", "prev_day"), ("asia_high", "asia"), ("london_high", "london")),
        long_levels=(("prev_month_low", "prev_month"), ("prev_week_low", "prev_week"), ("prev_day_low", "prev_day"), ("asia_low", "asia"), ("london_low", "london")),
        allow_longs=True
    ),
    Hypothesis(
        id="H2_SENSITIVE_TRIGGER",
        body_fraction=0.30,
        allow_longs=True
    ),
    Hypothesis(
        id="H4_BULL_ONLY",
        allow_longs=True,
        short_levels=(), 
    ),
    Hypothesis(
        id="H6_SILVER_BULLET_HYBRID",
        session_start="10:00",
        session_end="11:00",
        allow_longs=True
    )
]

# CONFIGURACIÓN DE CAMPAÑA
TIMEFRAME = "M3"
INITIAL_CAPITAL = 100_000.0

# STRESS TEST CONFIGURATION
spread_val = float(os.environ.get("SPREAD_PIPS", 1.2))
slip_val = float(os.environ.get("SLIPPAGE_PIPS", 0.2))

# FASE ACTUAL: OOS (Out-of-Sample)
YEAR_RANGE = [2024, 2025] 
suffix = "" if (spread_val == 1.2 and slip_val == 0.2) else f"_FRICTION_{spread_val}_{slip_val}"
RESULTS_ROOT = Path(f"results/DEEP_RESEARCH_CAMPAIGN_2.0_OOS{suffix}")

# FINALISTAS SELECCIONADOS (Ranking Robustez Capa 3)
ACTIVE_HYPOTHESES = [h for h in HYPOTHESES if h.id in ["H1_FULL_DENSITY", "H6_SILVER_BULLET_HYBRID"]]

PAIR = "EURUSD"

def flush_print(msg: str):
    print(msg)
    sys.stdout.flush()

def run_campaign():
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    
    flush_print(f"[*] Iniciando Campaña 2.0 - Capa 3: Ejecución Cuantitativa Incremental")
    
    # 1. Preparar Dataset de Noticias
    flush_print("[*] Sincronizando News Fortress...")
    build_am_grade_news_dataset(start="2020-01-01", end="2025-12-31")
    
    news_config = NewsConfig(
        enabled=True,
        file_path=AM_NEWS_FILE,
        raw_file_path=AM_NEWS_FILE,
        pre_minutes=30,
        post_minutes=60,
        currencies=("USD", "EUR"),
        impact_levels=("HIGH",),
    )
    from research_lab.news_filter import require_operational_news
    news_result = require_operational_news(PAIR, news_config)
    
    # Pre-cargar el paquete completo una sola vez si es posible, o por bloques si falla
    flush_print("[*] Cargando paquete de alta precisión base...")
    package = load_high_precision_package(PAIR, DEFAULT_HIGH_PRECISION_PREPARED_DIR)
    
    years = YEAR_RANGE
    global_ledger = []
    summary_csv_path = RESULTS_ROOT / "CAMPAIGN_SUMMARY_OOS.csv"
    
    for year in years:
        flush_print(f"\n[YEAR BLOCK] === Procesando Año {year} ===")
        start_ts = pd.Timestamp(f"{year}-01-01", tz=NY_TZ)
        end_ts = pd.Timestamp(f"{year}-12-31 23:59:59", tz=NY_TZ)
        
        filtered_m1 = {
            side: source.loc[(source.index >= start_ts) & (source.index <= end_ts)].copy()
            for side, source in package.items()
        }
        
        if filtered_m1["mid"].empty:
            flush_print(f"[!] No hay datos para el año {year}. Saltando...")
            continue
            
        # Keys check & fallback
        if "mid" not in filtered_m1:
            for k in list(filtered_m1.keys()):
                if "mid" in k.lower(): filtered_m1["mid"] = filtered_m1[k]
                if "bid" in k.lower(): filtered_m1["bid"] = filtered_m1[k]
                if "ask" in k.lower(): filtered_m1["ask"] = filtered_m1[k]
        
        flush_print(f"[*] Año {year}: {len(filtered_m1['mid'])} velas M1 cargadas. Generando frames OHLC...")
        m3_frame = prepare_common_frame(filtered_m1["mid"], target_timeframe=TIMEFRAME)
        m5_frame = prepare_common_frame(filtered_m1["mid"], target_timeframe="M5")
        
        for h in ACTIVE_HYPOTHESES:
            flush_print(f"  [>] {h.id} ({year})")
            
            annotated, signals = annotate_post_news_external_liquidity_shift_frame(
                m3_frame.copy(),
                m5_frame.copy(),
                news_events=news_result.events,
                news_config=news_config,
                short_levels=h.short_levels,
                long_levels=h.long_levels
            )
            
            if signals.empty:
                flush_print(f"    [!] 0 señales.")
                continue
                
            flush_print(f"    [*] {len(signals)} señales. Ejecutando backtest...")
            
            engine_config = with_execution_mode(
                EngineConfig(
                    pair=PAIR, 
                    risk_pct=0.5, 
                    session_cutoff="11:30",
                    assumed_spread_pips=spread_val,
                    slippage_pips=slip_val
                ), 
                "high_precision_mode"
            )
            news_block = build_entry_block(entry_open_index(annotated.index), news_result.events, news_config)
            
            res = run_backtest(
                strategy_module=strategy_module,
                frame=annotated,
                params=strategy_module.default_params(),
                engine_config=engine_config,
                news_block=news_block,
                news_filter_used=True,
                precision_package={
                    "bid_m1": filtered_m1["bid"],
                    "ask_m1": filtered_m1["ask"],
                    "mid_m1": filtered_m1["mid"],
                    "bid_m15": resample_ohlcv_to_timeframe(filtered_m1["bid"], TIMEFRAME).reindex(annotated.index),
                    "ask_m15": resample_ohlcv_to_timeframe(filtered_m1["ask"], TIMEFRAME).reindex(annotated.index),
                    "mid_m15": resample_ohlcv_to_timeframe(filtered_m1["mid"], TIMEFRAME).reindex(annotated.index)
                },
                news_events=news_result.events
            )
            
            summary, trades, _, _, _ = summarize_result(
                h.id, 
                res.trades, 
                res.equity_curve, 
                strategy_module.default_params(), 
                True, 
                INITIAL_CAPITAL,
                None, # selected_score missing
                timeframe=TIMEFRAME
            )
            
            # Persistencia Física Anual
            h_year_path = RESULTS_ROOT / h.id / str(year)
            h_year_path.mkdir(parents=True, exist_ok=True)
            trades.to_csv(h_year_path / "trades.csv", index=False)
            
            pnl = trades["pnl_r"]
            pf = pnl[pnl > 0].sum() / abs(pnl[pnl < 0].sum()) if (pnl < 0).any() else float('inf')
            
            h_metrics = {
                "id": h.id,
                "year": year,
                "trades": len(trades),
                "pf": round(pf, 2),
                "expectancy": round(pnl.mean(), 4),
                "pnl_r": round(pnl.sum(), 2)
            }
            global_ledger.append(h_metrics)
            
            # Guardado incremental del resumen global
            pd.DataFrame(global_ledger).to_csv(summary_csv_path, index=False)
            flush_print(f"    [OK] PF={pf:.2f}, R={pnl.sum():.2f}")

    flush_print(f"\n[*] Campaña Incremental completada. Resultados en {RESULTS_ROOT}")

if __name__ == "__main__":
    run_campaign()

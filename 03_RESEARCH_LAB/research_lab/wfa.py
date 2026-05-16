from __future__ import annotations
import json
from dataclasses import dataclass, field
from typing import Any
import pandas as pd
import numpy as np

from research_lab.validation import run_walkforward
from research_lab.engine import entry_open_index, run_backtest
from research_lab.news_filter import build_entry_block, load_news_events
from research_lab.report import summarize_result
from research_lab.scorer import score_is_summary
from research_lab.config import DEFAULT_WFA_IS_MONTHS, DEFAULT_WFA_OOS_MONTHS, INITIAL_CAPITAL

@dataclass
class WFAOrchestrationResult:
    strategy_name: str
    insufficient_sample: bool
    oos_stats: dict[str, Any]
    oos_trades: pd.DataFrame
    oos_equity_curve: pd.DataFrame
    best_params: dict[str, Any]
    best_is_pf: float
    best_is_expectancy: float
    optimization_results: pd.DataFrame
    lineage: dict[str, Any]

def run_wfa_default(
    strategy_module: Any,
    frame: pd.DataFrame,
    engine_config: Any,
    news_config: Any,
    max_evals: int,
    seed: int,
    precision_package: Any = None,
    data_source_used: str | None = None,
    fixed_params: dict | None = None,
) -> WFAOrchestrationResult:
    
    # 1. Generar combos o usar fijos
    if fixed_params:
        combos = [fixed_params]
    else:
        combos = strategy_module.parameter_grid(max_combinations=max_evals, seed=seed)
    
    # 2. Ejecutar WFA completo
    wfa_res = run_walkforward(
        strategy_name=strategy_module.NAME,
        strategy_module=strategy_module,
        frame=frame,
        combos=combos,
        engine_config=engine_config,
        news_config=news_config,
        is_months=DEFAULT_WFA_IS_MONTHS,
        oos_months=DEFAULT_WFA_OOS_MONTHS,
        precision_package=precision_package,
        data_source_used=data_source_used
    )
    
    lineage = {
        "strategy": strategy_module.NAME,
        "execution_mode": engine_config.execution_mode,
        "is_months": DEFAULT_WFA_IS_MONTHS,
        "oos_months": DEFAULT_WFA_OOS_MONTHS,
        "news_enabled": news_config.enabled,
        "timestamp": pd.Timestamp.now().isoformat()
    }
    
    # Encontrar mejor PF de los folds (In-Sample) para el reporte de rechazo
    # Note: run_walkforward ya nos da un oos_summary consolidado
    best_is_pf = wfa_res.fold_rows["profit_factor"].max() if not wfa_res.fold_rows.empty else 0.0
    best_is_exp = wfa_res.fold_rows["pnl_r"].mean() if not wfa_res.fold_rows.empty else 0.0
    
    total_trades = wfa_res.oos_summary.get("total_trades", 0)
    
    return WFAOrchestrationResult(
        strategy_name=strategy_module.NAME,
        insufficient_sample=bool(total_trades < 5),
        oos_stats=wfa_res.oos_summary,
        oos_trades=wfa_res.oos_trades,
        oos_equity_curve=wfa_res.oos_equity_curve,
        best_params=combos[0] if combos else {},
        best_is_pf=float(best_is_pf),
        best_is_expectancy=float(best_is_exp),
        optimization_results=wfa_res.fold_rows,
        lineage=lineage
    )

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from research_lab.config import (
    canonical_news_config,
    EngineConfig,
    INITIAL_CAPITAL,
    DEFAULT_DATA_DIRS,
)
from research_lab.data_loader import load_backtest_data_bundle
from research_lab.engine import run_backtest
from research_lab.news_filter import require_operational_news
from research_lab.strategies import STRATEGY_REGISTRY

# Splits
SPLITS = {
    "Development": ("2020-01-01", "2023-12-31"),
    "Validation": ("2024-01-01", "2024-12-31"),
    "Holdout": ("2025-01-01", "2025-12-31"),
}

STRATEGY_NAME = "eurusd_c4_ict_align"

def run_campaign_phase(phase_name: str, start_date: str, end_date: str, mode: str = "normal_mode"):
    print(f"\n>>> EJECUTANDO FASE: {phase_name} ({start_date} a {end_date})")
    
    strategy_module = STRATEGY_REGISTRY[STRATEGY_NAME]
    
    # News Config
    news_settings = canonical_news_config(
        pair="EURUSD",
        enabled=True,
        pre_minutes=30,
        post_minutes=60
    )
    news_load = require_operational_news("EURUSD", news_settings)
    
    # Engine Config
    engine_config = EngineConfig(
        pair="EURUSD",
        risk_pct=0.5,
        execution_mode=mode,
    )
    
    # Data
    bundle = load_backtest_data_bundle(
        pair="EURUSD",
        data_dirs=DEFAULT_DATA_DIRS,
        start=start_date,
        end=end_date,
        execution_mode=mode,
        target_timeframe="M5"
    )
    
    # Run
    # dummy_news_block needs to be same length as frame
    dummy_news_block = np.zeros(len(bundle.frame), dtype=bool)
    
    result = run_backtest(
        strategy_module=strategy_module,
        frame=bundle.frame,
        params=strategy_module.default_params(),
        engine_config=engine_config,
        news_block=dummy_news_block,
        news_filter_used=True,
        news_events=news_load.events,
        news_settings=news_settings
    )
    
    return result

def calculate_metrics(trades: pd.DataFrame, equity_curve: pd.DataFrame, start: str, end: str):
    if trades.empty:
        return {
            "n_trades": 0, "pf": 0.0, "expectancy": 0.0, "max_dd": 0.0, "win_rate": 0.0, "trades_per_month": 0.0
        }
    
    pnl = trades["pnl_r"]
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    
    pf = wins.sum() / abs(losses.sum()) if not losses.empty else 100.0
    expectancy = pnl.mean()
    win_rate = len(wins) / len(pnl)
    
    # Drawdown from equity curve
    equity = equity_curve["equity"]
    peak = equity.cummax()
    drawdown_pct = (equity - peak) / peak
    max_dd_r = abs(drawdown_pct.min()) * (INITIAL_CAPITAL / (INITIAL_CAPITAL * 0.005)) # Approximate in R
    
    months = (pd.to_datetime(end) - pd.to_datetime(start)).days / 30.44
    tpm = len(pnl) / months
    
    return {
        "n_trades": len(pnl),
        "pf": pf,
        "expectancy": expectancy,
        "max_dd": max_dd_r,
        "win_rate": win_rate,
        "trades_per_month": tpm
    }

def main():
    all_results = []
    for phase, (start, end) in SPLITS.items():
        # Normal Mode
        res = run_campaign_phase(phase, start, end, mode="normal_mode")
        metrics = calculate_metrics(res.trades, res.equity_curve, start, end)
        metrics["phase"] = phase
        metrics["mode"] = "normal"
        all_results.append(metrics)
        
        # Friction Mode (Stress)
        res_stress = run_campaign_phase(phase, start, end, mode="conservative_mode")
        metrics_stress = calculate_metrics(res_stress.trades, res_stress.equity_curve, start, end)
        metrics_stress["phase"] = phase
        metrics_stress["mode"] = "stress"
        all_results.append(metrics_stress)
        
    df = pd.DataFrame(all_results)
    print("\n\n" + "="*80)
    print("RESUMEN DE CAMPAÑA C4-ICT-ALIGN")
    print("="*80)
    print(df.to_string(index=False))
    
    # Save to report artifact later
    df.to_csv("campaign4_results_dump.csv", index=False)

if __name__ == "__main__":
    main()

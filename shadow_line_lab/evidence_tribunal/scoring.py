import pandas as pd
import numpy as np
import os

def calculate_shadow_metrics(ledger_df):
    if ledger_df is None or ledger_df.empty:
        return {}

    # Filtrar solo entradas que son trades reales o intentos significativos
    executed = ledger_df[ledger_df['classification'] == 'TRADE_EXECUTED'].copy()
    
    # Asegurar tipos
    executed['pnl_r'] = pd.to_numeric(executed['pnl_r'], errors='coerce').fillna(0.0)
    
    metrics = {
        "total_shadow_runs": len(ledger_df),
        "total_shadow_trades": len(executed),
        "unique_shadow_dates": ledger_df['date'].nunique(),
        "cumulative_R": round(executed['pnl_r'].sum(), 4),
        "avg_R": round(executed['pnl_r'].mean(), 4) if not executed.empty else 0.0,
        "median_R": round(executed['pnl_r'].median(), 4) if not executed.empty else 0.0,
        "win_rate": round((len(executed[executed['pnl_r'] > 0]) / len(executed)) * 100, 2) if not executed.empty else 0.0,
        "pf": calculate_pf(executed),
        "expectancy_R": round(executed['pnl_r'].mean(), 4) if not executed.empty else 0.0,
        "max_drawdown_R": calculate_drawdown(executed),
        "timeout_rate": round(len(executed[executed['timeout_flag'] == True]) / len(executed), 4) if not executed.empty else 0.0,
        "news_block_rate": round(len(ledger_df[ledger_df['news_blocked'] == True]) / len(ledger_df), 4) if not ledger_df.empty else 0.0,
        "trade_frequency": round(len(executed) / ledger_df['date'].nunique(), 4) if ledger_df['date'].nunique() > 0 else 0.0
    }
    
    # Streaks
    metrics.update(calculate_streaks(executed, ledger_df))
    
    # Rollings (N=5, N=10)
    if len(executed) >= 5:
        metrics["rolling_R_N5"] = round(executed['pnl_r'].tail(5).sum(), 4)
        metrics["rolling_drawdown_N5"] = calculate_drawdown(executed.tail(5))
    if len(executed) >= 10:
        metrics["rolling_R_N10"] = round(executed['pnl_r'].tail(10).sum(), 4)
        metrics["rolling_drawdown_N10"] = calculate_drawdown(executed.tail(10))

    return metrics

def calculate_pf(df):
    wins = df[df['pnl_r'] > 0]['pnl_r'].sum()
    losses = abs(df[df['pnl_r'] < 0]['pnl_r'].sum())
    return round(wins / losses, 4) if losses > 0 else (round(wins, 4) if wins > 0 else 0.0)

def calculate_drawdown(df):
    if df.empty: return 0.0
    equity = df['pnl_r'].cumsum()
    peak = equity.expanding().max()
    dd = peak - equity
    return round(dd.max(), 4)

def calculate_streaks(executed, full_ledger):
    streaks = {
        "current_streak_positive_or_flat": 0,
        "current_streak_negative": 0,
        "current_streak_no_execution": 0
    }
    
    if not executed.empty:
        last_trades = executed['pnl_r'].tolist()[::-1]
        for pnl in last_trades:
            if pnl >= 0: streaks["current_streak_positive_or_flat"] += 1
            else: break
        for pnl in last_trades:
            if pnl < 0: streaks["current_streak_negative"] += 1
            else: break
            
    if not full_ledger.empty:
        last_runs = full_ledger['classification'].tolist()[::-1]
        for cls in last_runs:
            if cls != 'TRADE_EXECUTED': streaks["current_streak_no_execution"] += 1
            else: break
            
    return streaks

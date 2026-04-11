from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from research_lab.config import NY_TZ, VISIBLE_CHATGPT_ARCHIVE
from research_lab.scorer import sample_penalty_meta


def _profit_factor(pnl_usd: pd.Series) -> float:
    gross_profit = float(pnl_usd[pnl_usd > 0].sum())
    gross_loss = float(pnl_usd[pnl_usd < 0].sum())
    return gross_profit / abs(gross_loss) if gross_loss < 0 else float("inf")


def build_trades_export(trades: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "pair",
        "entry_time_ny",
        "exit_time_ny",
        "direction",
        "entry_side",
        "signal_time_ny",
        "fill_time_ny",
        "signal_price",
        "fill_price",
        "entry_price",
        "exit_price",
        "exit_signal_price",
        "exit_fill_price",
        "sl",
        "tp",
        "spread_applied",
        "slippage_applied",
        "commission_applied",
        "entry_spread_pips",
        "exit_spread_pips",
        "entry_slippage_pips",
        "exit_slippage_pips",
        "entry_commission_usd",
        "exit_commission_usd",
        "price_source_used",
        "pnl_r",
        "pnl_usd",
        "result",
        "exit_reason",
        "forced_close_flag",
        "intrabar_ambiguity_flag",
        "execution_mode_used",
        "intrabar_policy_used",
        "cost_profile_used",
        "entry_cost_regime",
        "exit_cost_regime",
        "blocked_by_news",
        "blocking_event_name",
        "blocking_event_time_ny",
        "blocking_rule_used",
        "date",
    ]
    if trades.empty:
        return pd.DataFrame(columns=columns)

    entry_time = pd.to_datetime(trades["entry_time"], utc=True).dt.tz_convert(NY_TZ)
    exit_time = pd.to_datetime(trades["exit_time"], utc=True).dt.tz_convert(NY_TZ)
    signal_time = pd.to_datetime(trades.get("signal_time", trades["entry_time"]), utc=True).dt.tz_convert(NY_TZ)
    fill_time = pd.to_datetime(trades.get("fill_time", trades["entry_time"]), utc=True).dt.tz_convert(NY_TZ)
    pnl_usd = trades["pnl_usd"].astype(float)
    result = np.where(pnl_usd > 0.0, "win", np.where(pnl_usd < 0.0, "loss", "breakeven"))
    exported = pd.DataFrame(
        {
            "pair": trades["pair"],
            "entry_time_ny": entry_time.dt.strftime("%Y-%m-%d %H:%M:%S"),
            "exit_time_ny": exit_time.dt.strftime("%Y-%m-%d %H:%M:%S"),
            "direction": trades["direction"],
            "entry_side": trades.get("entry_side", pd.Series("", index=trades.index)).astype(str),
            "signal_time_ny": signal_time.dt.strftime("%Y-%m-%d %H:%M:%S"),
            "fill_time_ny": fill_time.dt.strftime("%Y-%m-%d %H:%M:%S"),
            "signal_price": trades.get("signal_price", trades["entry_price"]).astype(float),
            "fill_price": trades.get("fill_price", trades["entry_price"]).astype(float),
            "entry_price": trades["entry_price"].astype(float),
            "exit_price": trades["exit_price"].astype(float),
            "exit_signal_price": trades.get("exit_signal_price", trades["exit_price"]).astype(float),
            "exit_fill_price": trades.get("exit_fill_price", trades["exit_price"]).astype(float),
            "sl": trades["sl"].astype(float),
            "tp": trades["tp"].astype(float),
            "spread_applied": trades.get("spread_applied", pd.Series(0.0, index=trades.index)).astype(float),
            "slippage_applied": trades.get("slippage_applied", pd.Series(0.0, index=trades.index)).astype(float),
            "commission_applied": trades.get("commission_applied", trades.get("commission_usd", pd.Series(0.0, index=trades.index))).astype(float),
            "entry_spread_pips": trades.get("entry_spread_pips", pd.Series(0.0, index=trades.index)).astype(float),
            "exit_spread_pips": trades.get("exit_spread_pips", pd.Series(0.0, index=trades.index)).astype(float),
            "entry_slippage_pips": trades.get("entry_slippage_pips", pd.Series(0.0, index=trades.index)).astype(float),
            "exit_slippage_pips": trades.get("exit_slippage_pips", pd.Series(0.0, index=trades.index)).astype(float),
            "entry_commission_usd": trades.get("entry_commission_usd", pd.Series(0.0, index=trades.index)).astype(float),
            "exit_commission_usd": trades.get("exit_commission_usd", pd.Series(0.0, index=trades.index)).astype(float),
            "price_source_used": trades.get("price_source_used", pd.Series("bid", index=trades.index)).astype(str),
            "pnl_r": trades["pnl_r"].astype(float),
            "pnl_usd": pnl_usd,
            "result": result,
            "exit_reason": trades["exit_reason"],
            "forced_close_flag": trades.get("forced_close_flag", pd.Series(False, index=trades.index)).astype(bool),
            "intrabar_ambiguity_flag": trades.get("intrabar_ambiguity_flag", pd.Series(False, index=trades.index)).astype(bool),
            "execution_mode_used": trades.get("execution_mode_used", pd.Series("normal_mode", index=trades.index)).astype(str),
            "intrabar_policy_used": trades.get("intrabar_policy_used", pd.Series("standard", index=trades.index)).astype(str),
            "cost_profile_used": trades.get("cost_profile_used", pd.Series("base", index=trades.index)).astype(str),
            "entry_cost_regime": trades.get("entry_cost_regime", pd.Series("base", index=trades.index)).astype(str),
            "exit_cost_regime": trades.get("exit_cost_regime", pd.Series("base", index=trades.index)).astype(str),
            "blocked_by_news": trades.get("blocked_by_news", pd.Series(False, index=trades.index)).astype(bool),
            "blocking_event_name": trades.get("blocking_event_name", pd.Series("", index=trades.index)).astype(str),
            "blocking_event_time_ny": trades.get("blocking_event_time_ny", pd.Series("", index=trades.index)).astype(str),
            "blocking_rule_used": trades.get("blocking_rule_used", pd.Series("", index=trades.index)).astype(str),
            "date": entry_time.dt.strftime("%Y-%m-%d"),
        }
    )
    return exported[columns]


def build_period_stats(trades_export: pd.DataFrame, freq: str, initial_capital: float) -> pd.DataFrame:
    label = "year" if freq == "Y" else "month"
    columns = [
        "pair",
        label,
        "trades",
        "wins",
        "losses",
        "breakevens",
        "win_rate",
        "total_pnl_r",
        "total_pnl_usd",
        "avg_pnl_r",
        "max_drawdown_pct",
        "profit_factor",
    ]
    if trades_export.empty:
        return pd.DataFrame(columns=columns)

    frame = trades_export.copy()
    entry_time = pd.to_datetime(frame["entry_time_ny"])
    frame[label] = entry_time.dt.to_period(freq).astype(str)
    rows: list[dict[str, Any]] = []
    for (pair, bucket), chunk in frame.groupby(["pair", label], sort=True):
        pnl_usd = chunk["pnl_usd"].astype(float)
        pnl_r = chunk["pnl_r"].astype(float)
        equity = initial_capital + pnl_usd.cumsum()
        peak = equity.cummax()
        drawdown = (equity - peak) / peak.replace(0.0, np.nan)
        wins = int((pnl_usd > 0).sum())
        losses = int((pnl_usd < 0).sum())
        breakevens = int((pnl_usd == 0).sum())
        rows.append(
            {
                "pair": pair,
                label: bucket,
                "trades": int(len(chunk)),
                "wins": wins,
                "losses": losses,
                "breakevens": breakevens,
                "win_rate": (wins / len(chunk)) * 100 if len(chunk) else 0.0,
                "total_pnl_r": float(pnl_r.sum()),
                "total_pnl_usd": float(pnl_usd.sum()),
                "avg_pnl_r": float(pnl_r.mean()) if len(chunk) else 0.0,
                "max_drawdown_pct": float(abs(drawdown.min()) * 100) if len(drawdown) else 0.0,
                "profit_factor": _profit_factor(pnl_usd),
            }
        )
    return pd.DataFrame(rows)[columns].sort_values(["pair", label]).reset_index(drop=True)


def build_equity_curve_export(equity_curve: pd.DataFrame) -> pd.DataFrame:
    columns = ["datetime_ny", "equity", "drawdown_pct"]
    if equity_curve.empty:
        return pd.DataFrame(columns=columns)

    frame = equity_curve.copy()
    timestamp = pd.to_datetime(frame["timestamp"], utc=True).dt.tz_convert(NY_TZ)
    equity = frame["equity"].astype(float)
    peak = equity.cummax()
    drawdown_pct = ((equity - peak) / peak.replace(0.0, np.nan)) * 100
    return pd.DataFrame(
        {
            "datetime_ny": timestamp.dt.strftime("%Y-%m-%d %H:%M:%S"),
            "equity": equity,
            "drawdown_pct": drawdown_pct.fillna(0.0),
        }
    )[columns]


def build_summary(
    *,
    strategy_name: str,
    trades_export: pd.DataFrame,
    equity_export: pd.DataFrame,
    monthly_stats: pd.DataFrame,
    yearly_stats: pd.DataFrame,
    params: dict[str, Any],
    news_filter_used: bool,
    selected_score: float | None,
    costs_used: dict[str, Any],
    timeframe: str,
    schedule_used: dict[str, str],
    break_even_setting: Any,
) -> dict[str, Any]:
    pnl_usd = trades_export["pnl_usd"].astype(float) if not trades_export.empty else pd.Series(dtype=float)
    pnl_r = trades_export["pnl_r"].astype(float) if not trades_export.empty else pd.Series(dtype=float)
    total_trades = int(len(trades_export))
    wins = int((pnl_usd > 0).sum()) if total_trades else 0
    losses = int((pnl_usd < 0).sum()) if total_trades else 0
    breakevens = int((pnl_usd == 0).sum()) if total_trades else 0
    win_rate = (wins / total_trades) * 100 if total_trades else 0.0
    breakeven_rate = (breakevens / total_trades) * 100 if total_trades else 0.0
    profit_factor = _profit_factor(pnl_usd) if total_trades else 0.0
    expectancy_r = float(pnl_r.mean()) if total_trades else 0.0
    total_return_pct = 0.0
    max_drawdown_pct = 0.0
    if not equity_export.empty:
        initial_equity = float(equity_export["equity"].iloc[0])
        final_equity = float(equity_export["equity"].iloc[-1])
        if initial_equity:
            total_return_pct = ((final_equity / initial_equity) - 1.0) * 100.0
        max_drawdown_pct = float(abs(equity_export["drawdown_pct"].min()))

    negative_months = int((monthly_stats.groupby("month")["total_pnl_usd"].sum() < 0).sum()) if not monthly_stats.empty else 0
    negative_years = int((yearly_stats.groupby("year")["total_pnl_usd"].sum() < 0).sum()) if not yearly_stats.empty else 0
    avg_trades_per_month = 0.0
    if not equity_export.empty:
        dt_index = pd.to_datetime(equity_export["datetime_ny"])
        calendar_months = len(pd.period_range(dt_index.min().to_period("M"), dt_index.max().to_period("M"), freq="M"))
        if calendar_months:
            avg_trades_per_month = total_trades / calendar_months

    insufficient_sample, sample_penalty_applied, _ = sample_penalty_meta(
        {"total_trades": total_trades, "avg_trades_per_month": avg_trades_per_month}
    )
    return {
        "strategy_name": strategy_name,
        "total_trades": total_trades,
        "avg_trades_per_month": avg_trades_per_month,
        "win_rate": win_rate,
        "breakeven_rate": breakeven_rate,
        "profit_factor": profit_factor,
        "expectancy_r": expectancy_r,
        "total_return_pct": total_return_pct,
        "max_drawdown_pct": max_drawdown_pct,
        "negative_months": negative_months,
        "negative_years": negative_years,
        "parameter_set_used": params,
        "insufficient_sample": bool(insufficient_sample),
        "sample_penalty_applied": bool(sample_penalty_applied),
        "selected_score": float(selected_score) if selected_score is not None else None,
        "news_filter_used": bool(news_filter_used),
        "costs_used": costs_used,
        "timeframe": timeframe,
        "schedule_used": schedule_used,
        "break_even_setting": break_even_setting,
    }


def summarize_result(
    strategy_name: str,
    trades: pd.DataFrame,
    equity_curve: pd.DataFrame,
    params: dict[str, Any],
    news_filter_used: bool,
    initial_capital: float,
    selected_score: float | None,
    costs_used: dict[str, Any] | None = None,
    timeframe: str = "M15",
    schedule_used: dict[str, str] | None = None,
    break_even_setting: Any = None,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    trades_export = build_trades_export(trades)
    monthly_stats = build_period_stats(trades_export, "M", initial_capital)
    yearly_stats = build_period_stats(trades_export, "Y", initial_capital)
    equity_export = build_equity_curve_export(equity_curve)
    summary = build_summary(
        strategy_name=strategy_name,
        trades_export=trades_export,
        equity_export=equity_export,
        monthly_stats=monthly_stats,
        yearly_stats=yearly_stats,
        params=params,
        news_filter_used=news_filter_used,
        selected_score=selected_score,
        costs_used=costs_used or {},
        timeframe=timeframe,
        schedule_used=schedule_used or {},
        break_even_setting=break_even_setting,
    )
    return summary, trades_export, monthly_stats, yearly_stats, equity_export


def export_strategy_bundle(
    strategy_dir: Path,
    *,
    summary: dict[str, Any],
    trades_export: pd.DataFrame,
    monthly_stats: pd.DataFrame,
    yearly_stats: pd.DataFrame,
    equity_export: pd.DataFrame,
    optimization_results: pd.DataFrame,
    extra_frames: dict[str, pd.DataFrame] | None = None,
    extra_json: dict[str, Any] | None = None,
) -> None:
    strategy_dir.mkdir(parents=True, exist_ok=True)
    trades_export.to_csv(strategy_dir / "trades.csv", index=False)
    monthly_stats.to_csv(strategy_dir / "monthly_stats.csv", index=False)
    yearly_stats.to_csv(strategy_dir / "yearly_stats.csv", index=False)
    equity_export.to_csv(strategy_dir / "equity_curve.csv", index=False)
    optimization_results.to_csv(strategy_dir / "optimization_results.csv", index=False)
    (strategy_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    for name, frame in (extra_frames or {}).items():
        frame.to_csv(strategy_dir / name, index=False)
    for name, payload in (extra_json or {}).items():
        (strategy_dir / name).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def export_root_tables(run_dir: Path, ranking: pd.DataFrame, comparative_table: pd.DataFrame, top3_text: str, losers_text: str, recommendation_text: str) -> None:
    ranking.to_csv(run_dir / "strategy_ranking.csv", index=False)
    comparative_table.to_csv(run_dir / "comparative_table.csv", index=False)
    (run_dir / "top3_finalistas.md").write_text(top3_text, encoding="utf-8")
    (run_dir / "autopsia_perdedores.md").write_text(losers_text, encoding="utf-8")
    (run_dir / "recomendacion_final.md").write_text(recommendation_text, encoding="utf-8")


def sync_visible_chatgpt(run_dir: Path) -> Path:
    project_root = Path(__file__).resolve().parents[1]
    visible_archive = project_root / VISIBLE_CHATGPT_ARCHIVE
    if visible_archive.exists():
        visible_archive.unlink()
    shutil.make_archive(str(visible_archive.with_suffix("")), "zip", root_dir=run_dir, base_dir=".")
    visible_root = project_root / "000_PARA_CHATGPT"
    if visible_root.exists():
        shutil.rmtree(visible_root)
    note_path = project_root / "ABRIR_000_PARA_CHATGPT.txt"
    if note_path.exists():
        note_path.unlink()
    return visible_archive

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import NY_TZ


VISIBLE_CHATGPT_DIRNAME = "000_PARA_CHATGPT"
VISIBLE_CHATGPT_ARCHIVE = "000_PARA_CHATGPT.zip"
WEEKDAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]


def _profit_factor(pnl_usd: pd.Series) -> float:
    gross_profit = float(pnl_usd[pnl_usd > 0].sum())
    gross_loss = float(pnl_usd[pnl_usd < 0].sum())
    return gross_profit / abs(gross_loss) if gross_loss < 0 else float("inf")


def _profit_factor_r(pnl_r: pd.Series) -> float:
    gross_profit = float(pnl_r[pnl_r > 0].sum())
    gross_loss = float(pnl_r[pnl_r < 0].sum())
    return gross_profit / abs(gross_loss) if gross_loss < 0 else float("inf")


def _period_drawdown_pct(chunk: pd.DataFrame, initial_capital: float) -> float:
    if chunk.empty:
        return 0.0
    equity = initial_capital + chunk["pnl_usd"].astype(float).cumsum()
    peak = equity.cummax()
    drawdown = (equity - peak) / peak.replace(0.0, np.nan)
    return float(abs(drawdown.min()) * 100) if len(drawdown) else 0.0


def build_trades_export(trades: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "pair",
        "entry_time_ny",
        "exit_time_ny",
        "direction",
        "entry_price",
        "exit_price",
        "sl",
        "tp",
        "pnl_r",
        "pnl_usd",
        "result",
        "exit_reason",
        "date",
    ]
    if trades.empty:
        return pd.DataFrame(columns=columns)

    entry_time = pd.to_datetime(trades["entry_time"], utc=True).dt.tz_convert(NY_TZ)
    exit_time = pd.to_datetime(trades["exit_time"], utc=True).dt.tz_convert(NY_TZ)
    pnl_usd = trades["pnl_usd"].astype(float)
    result = np.where(pnl_usd > 0.0, "win", np.where(pnl_usd < 0.0, "loss", "breakeven"))
    exported = pd.DataFrame(
        {
            "pair": trades["pair"],
            "entry_time_ny": entry_time.dt.strftime("%Y-%m-%d %H:%M:%S"),
            "exit_time_ny": exit_time.dt.strftime("%Y-%m-%d %H:%M:%S"),
            "direction": trades["direction"],
            "entry_price": trades["entry_price"].astype(float),
            "exit_price": trades["exit_price"].astype(float),
            "sl": trades["sl"].astype(float),
            "tp": trades["tp"].astype(float),
            "pnl_r": trades["pnl_r"].astype(float),
            "pnl_usd": pnl_usd,
            "result": result,
            "exit_reason": trades["exit_reason"],
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
                "max_drawdown_pct": _period_drawdown_pct(chunk, initial_capital),
                "profit_factor": _profit_factor(pnl_usd),
            }
        )
    return pd.DataFrame(rows)[columns].sort_values(["pair", label]).reset_index(drop=True)


def build_hourly_stats(trades_export: pd.DataFrame) -> pd.DataFrame:
    columns = ["hour_ny", "trades", "wins", "losses", "win_rate", "total_pnl_r", "avg_pnl_r", "profit_factor"]
    if trades_export.empty:
        return pd.DataFrame(columns=columns)

    frame = trades_export.copy()
    entry_time = pd.to_datetime(frame["entry_time_ny"])
    frame["hour_ny"] = entry_time.dt.hour
    rows: list[dict[str, Any]] = []
    for hour_ny, chunk in frame.groupby("hour_ny", sort=True):
        pnl_usd = chunk["pnl_usd"].astype(float)
        pnl_r = chunk["pnl_r"].astype(float)
        wins = int((pnl_usd > 0).sum())
        losses = int((pnl_usd < 0).sum())
        rows.append(
            {
                "hour_ny": int(hour_ny),
                "trades": int(len(chunk)),
                "wins": wins,
                "losses": losses,
                "win_rate": (wins / len(chunk)) * 100 if len(chunk) else 0.0,
                "total_pnl_r": float(pnl_r.sum()),
                "avg_pnl_r": float(pnl_r.mean()) if len(chunk) else 0.0,
                "profit_factor": _profit_factor_r(pnl_r),
            }
        )
    return pd.DataFrame(rows)[columns].sort_values("hour_ny").reset_index(drop=True)


def build_weekday_stats(trades_export: pd.DataFrame) -> pd.DataFrame:
    columns = ["weekday", "trades", "wins", "losses", "win_rate", "total_pnl_r", "avg_pnl_r", "profit_factor"]
    if trades_export.empty:
        return pd.DataFrame(columns=columns)

    frame = trades_export.copy()
    entry_time = pd.to_datetime(frame["entry_time_ny"])
    frame["weekday"] = pd.Categorical(entry_time.dt.day_name(), categories=WEEKDAY_ORDER, ordered=True)
    rows: list[dict[str, Any]] = []
    for weekday, chunk in frame.groupby("weekday", sort=False, observed=True):
        pnl_usd = chunk["pnl_usd"].astype(float)
        pnl_r = chunk["pnl_r"].astype(float)
        wins = int((pnl_usd > 0).sum())
        losses = int((pnl_usd < 0).sum())
        rows.append(
            {
                "weekday": str(weekday),
                "trades": int(len(chunk)),
                "wins": wins,
                "losses": losses,
                "win_rate": (wins / len(chunk)) * 100 if len(chunk) else 0.0,
                "total_pnl_r": float(pnl_r.sum()),
                "avg_pnl_r": float(pnl_r.mean()) if len(chunk) else 0.0,
                "profit_factor": _profit_factor_r(pnl_r),
            }
        )
    return pd.DataFrame(rows)[columns]


def build_equity_curve_export(equity_curve: pd.DataFrame) -> pd.DataFrame:
    columns = ["datetime_ny", "equity", "drawdown_pct"]
    if equity_curve.empty:
        return pd.DataFrame(columns=columns)

    frame = equity_curve.copy()
    timestamp = pd.to_datetime(frame["timestamp"], utc=True).dt.tz_convert(NY_TZ)
    equity = frame["equity"].astype(float)
    peak = equity.cummax()
    drawdown_pct = ((equity - peak) / peak.replace(0.0, np.nan)) * 100
    exported = pd.DataFrame(
        {
            "datetime_ny": timestamp.dt.strftime("%Y-%m-%d %H:%M:%S"),
            "equity": equity,
            "drawdown_pct": drawdown_pct.fillna(0.0),
        }
    )
    return exported[columns]


def _rank_hours(hourly_stats: pd.DataFrame, ascending: bool) -> list[int]:
    if hourly_stats.empty:
        return []
    ranked = hourly_stats.sort_values(["total_pnl_r", "profit_factor", "trades"], ascending=[ascending, ascending, False])
    return ranked["hour_ny"].astype(int).head(3).tolist()


def build_summary(
    trades_export: pd.DataFrame,
    raw_trades: pd.DataFrame,
    equity_export: pd.DataFrame,
    monthly_stats: pd.DataFrame,
    yearly_stats: pd.DataFrame,
    hourly_stats: pd.DataFrame,
    parameter_set_used: dict[str, Any],
    news_filter_used: bool,
    regime_counts: dict[str, int],
    selected_score: float | None = None,
) -> dict[str, Any]:
    pnl_usd = trades_export["pnl_usd"].astype(float) if not trades_export.empty else pd.Series(dtype=float)
    pnl_r = trades_export["pnl_r"].astype(float) if not trades_export.empty else pd.Series(dtype=float)
    wins = int((pnl_usd > 0).sum()) if not trades_export.empty else 0
    losses = int((pnl_usd < 0).sum()) if not trades_export.empty else 0
    breakevens = int((pnl_usd == 0).sum()) if not trades_export.empty else 0
    total_trades = int(len(trades_export))
    win_rate = (wins / total_trades) * 100 if total_trades else 0.0
    breakeven_rate = (breakevens / total_trades) * 100 if total_trades else 0.0
    profit_factor = _profit_factor(pnl_usd) if not trades_export.empty else 0.0
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
        start_period = dt_index.min().to_period("M")
        end_period = dt_index.max().to_period("M")
        calendar_months = len(pd.period_range(start=start_period, end=end_period, freq="M"))
        if calendar_months:
            avg_trades_per_month = total_trades / calendar_months

    breakout_raw = raw_trades[raw_trades["module"] == "breakout"] if ("module" in raw_trades.columns and not raw_trades.empty) else pd.DataFrame()
    range_raw = raw_trades[raw_trades["module"] == "range_mr"] if ("module" in raw_trades.columns and not raw_trades.empty) else pd.DataFrame()
    breakout_pf = _profit_factor(breakout_raw["pnl_usd"].astype(float)) if not breakout_raw.empty else 0.0
    range_pf = _profit_factor(range_raw["pnl_usd"].astype(float)) if not range_raw.empty else 0.0

    return {
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
        "best_hours": _rank_hours(hourly_stats, ascending=False),
        "worst_hours": _rank_hours(hourly_stats, ascending=True),
        "parameter_set_used": parameter_set_used,
        "news_filter_used": bool(news_filter_used),
        "stop_mode_used": parameter_set_used.get("stop_mode", parameter_set_used.get("breakout_stop_mode", "")),
        "breakout_enabled": bool(parameter_set_used.get("breakout_enabled", parameter_set_used.get("model_mode", "hybrid") != "range_only")),
        "range_be_enabled": bool(parameter_set_used.get("range_be_enabled", False)),
        "breakout_be_enabled": bool(parameter_set_used.get("breakout_be_enabled", False)),
        "daily_loss_limit_r_used": float(parameter_set_used.get("daily_loss_limit_r", 0.0)),
        "regime_counts": regime_counts,
        "breakout_module_trades": int(len(breakout_raw)),
        "range_module_trades": int(len(range_raw)),
        "breakout_module_pf": breakout_pf,
        "range_module_pf": range_pf,
        "selected_score": float(selected_score) if selected_score is not None else None,
    }


def export_chatgpt_bundle(
    output_dir: Path,
    *,
    trades: pd.DataFrame,
    equity_curve: pd.DataFrame,
    params: dict[str, Any],
    news_filter_used: bool,
    optimization_results: pd.DataFrame,
    initial_capital: float,
    regime_counts: dict[str, int],
    selected_score: float | None = None,
) -> Path:
    chatgpt_dir = output_dir / "PARA CHATGPT"
    chatgpt_dir.mkdir(parents=True, exist_ok=True)
    project_root = Path(__file__).resolve().parent
    visible_chatgpt_dir = project_root / VISIBLE_CHATGPT_DIRNAME
    visible_archive = project_root / VISIBLE_CHATGPT_ARCHIVE
    if visible_chatgpt_dir.exists():
        shutil.rmtree(visible_chatgpt_dir)

    trades_export = build_trades_export(trades)
    monthly_stats = build_period_stats(trades_export, "M", initial_capital)
    yearly_stats = build_period_stats(trades_export, "Y", initial_capital)
    hourly_stats = build_hourly_stats(trades_export)
    weekday_stats = build_weekday_stats(trades_export)
    equity_export = build_equity_curve_export(equity_curve)
    summary = build_summary(
        trades_export,
        trades,
        equity_export,
        monthly_stats,
        yearly_stats,
        hourly_stats,
        parameter_set_used=params,
        news_filter_used=news_filter_used,
        regime_counts=regime_counts,
        selected_score=selected_score,
    )

    for target_dir in (chatgpt_dir,):
        trades_export.to_csv(target_dir / "trades.csv", index=False)
        monthly_stats.to_csv(target_dir / "monthly_stats.csv", index=False)
        yearly_stats.to_csv(target_dir / "yearly_stats.csv", index=False)
        hourly_stats.to_csv(target_dir / "hourly_stats.csv", index=False)
        weekday_stats.to_csv(target_dir / "weekday_stats.csv", index=False)
        equity_export.to_csv(target_dir / "equity_curve.csv", index=False)
        optimization_results.to_csv(target_dir / "optimization_results.csv", index=False)
        (target_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if visible_archive.exists():
        visible_archive.unlink()
    shutil.make_archive(str(visible_archive.with_suffix("")), "zip", root_dir=chatgpt_dir, base_dir=".")
    note_path = project_root / "ABRIR_000_PARA_CHATGPT.txt"
    if note_path.exists():
        note_path.unlink()
    return visible_archive


def print_console_report(
    summary: dict[str, Any],
    yearly_stats: pd.DataFrame,
    optimization_results: pd.DataFrame,
    runtime_seconds: float,
    previous_summary: dict[str, Any] | None = None,
) -> None:
    positive_years = int((yearly_stats.groupby("year")["total_pnl_usd"].sum() > 0).sum()) if not yearly_stats.empty else 0
    negative_years = int((yearly_stats.groupby("year")["total_pnl_usd"].sum() < 0).sum()) if not yearly_stats.empty else 0
    best_row = optimization_results.iloc[0].to_dict() if not optimization_results.empty else {}
    print("\n=== MEJOR COMBINACION ===")
    print(json.dumps(best_row, indent=2, ensure_ascii=False, default=str))
    print("\n=== RESUMEN ===")
    print(f"trades promedio por mes: {summary['avg_trades_per_month']:.4f}")
    print(f"win rate: {summary['win_rate']:.2f}%")
    print(f"breakeven rate: {summary['breakeven_rate']:.2f}%")
    print(f"profit factor: {summary['profit_factor']:.4f}")
    print(f"drawdown: {summary['max_drawdown_pct']:.4f}%")
    print(f"años positivos: {positive_years}")
    print(f"años negativos: {negative_years}")
    print(f"meses negativos: {summary['negative_months']}")
    print(f"mejores horas: {summary['best_hours']}")
    print(f"peores horas: {summary['worst_hours']}")
    print(f"runtime_seconds: {runtime_seconds:.2f}")
    print(f"sistema_lento: {'si' if runtime_seconds > 120 else 'no'}")
    if previous_summary:
        current_pf = float(summary.get("profit_factor", 0.0))
        prev_pf = float(previous_summary.get("profit_factor", 0.0))
        current_dd = float(summary.get("max_drawdown_pct", 0.0))
        prev_dd = float(previous_summary.get("max_drawdown_pct", 0.0))
        current_exp = float(summary.get("expectancy_r", 0.0))
        prev_exp = float(previous_summary.get("expectancy_r", 0.0))
        current_neg_months = int(summary.get("negative_months", 0))
        prev_neg_months = int(previous_summary.get("negative_months", 0))
        improved = (
            current_pf > prev_pf
            and current_dd < prev_dd
            and current_exp > prev_exp
            and current_neg_months <= prev_neg_months
        )
        print("\n=== COMPARACION VS ANTERIOR ===")
        print(f"mejora_vs_anterior: {'si' if improved else 'no'}")
        print(f"profit_factor: {prev_pf:.4f} -> {current_pf:.4f}")
        print(f"expectancy_r: {prev_exp:.4f} -> {current_exp:.4f}")
        print(f"max_drawdown_pct: {prev_dd:.4f} -> {current_dd:.4f}")
        print(f"negative_months: {prev_neg_months} -> {current_neg_months}")

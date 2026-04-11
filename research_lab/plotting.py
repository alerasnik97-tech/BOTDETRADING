from __future__ import annotations

import os
from pathlib import Path
from collections.abc import Hashable

_MPLCONFIGDIR = Path(__file__).resolve().parents[1] / ".mplconfig"
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))

import matplotlib.pyplot as plt
import pandas as pd


def _empty_plot(path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_title(title)
    ax.text(0.5, 0.5, "No data", ha="center", va="center")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_equity_curve(equity_curve: pd.DataFrame, path: Path, title: str) -> None:
    if equity_curve.empty:
        _empty_plot(path, title)
        return
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(pd.to_datetime(equity_curve["datetime_ny"]), equity_curve["equity"], label="equity")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_drawdown_curve(equity_curve: pd.DataFrame, path: Path, title: str) -> None:
    if equity_curve.empty:
        _empty_plot(path, title)
        return
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(pd.to_datetime(equity_curve["datetime_ny"]), equity_curve["drawdown_pct"], 0.0, color="red", alpha=0.4)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_yearly_pnl(yearly_stats: pd.DataFrame, path: Path, title: str) -> None:
    if yearly_stats.empty:
        _empty_plot(path, title)
        return
    grouped = yearly_stats.groupby("year")["total_pnl_r"].sum().reset_index()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(grouped["year"].astype(str), grouped["total_pnl_r"], color=["green" if x >= 0 else "red" for x in grouped["total_pnl_r"]])
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_heatmap(optimization_results: pd.DataFrame, path: Path, title: str) -> None:
    if optimization_results.empty or "selected_score" not in optimization_results.columns:
        _empty_plot(path, title)
        return
    excluded = {
        "strategy_name",
        "total_trades",
        "avg_trades_per_month",
        "win_rate",
        "breakeven_rate",
        "profit_factor",
        "expectancy_r",
        "total_return_pct",
        "max_drawdown_pct",
        "negative_months",
        "negative_years",
        "insufficient_sample",
        "sample_penalty_applied",
        "selected_score",
        "parameter_set_used",
        "support_score",
        "costs_used",
        "schedule_used",
        "news_filter_used",
        "timeframe",
        "break_even_setting",
    }
    candidate_cols = []
    for col in optimization_results.columns:
        if col in excluded:
            continue
        series = optimization_results[col].dropna()
        if series.empty:
            continue
        value = series.iloc[0]
        if isinstance(value, Hashable) and not isinstance(value, (dict, list, set, tuple)):
            candidate_cols.append(col)
    if len(candidate_cols) < 2:
        _empty_plot(path, title)
        return
    pivot = optimization_results.pivot_table(index=candidate_cols[0], columns=candidate_cols[1], values="selected_score", aggfunc="mean")
    if pivot.empty:
        _empty_plot(path, title)
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis")
    ax.set_title(title)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(x) for x in pivot.columns], rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(x) for x in pivot.index])
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_overlay_equity(curves: dict[str, pd.DataFrame], path: Path, title: str) -> None:
    if not curves:
        _empty_plot(path, title)
        return
    fig, ax = plt.subplots(figsize=(12, 5))
    for name, curve in curves.items():
        if curve.empty:
            continue
        ax.plot(pd.to_datetime(curve["datetime_ny"]), curve["equity"], label=name)
    ax.legend()
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_overlay_drawdown(curves: dict[str, pd.DataFrame], path: Path, title: str) -> None:
    if not curves:
        _empty_plot(path, title)
        return
    fig, ax = plt.subplots(figsize=(12, 5))
    for name, curve in curves.items():
        if curve.empty:
            continue
        ax.plot(pd.to_datetime(curve["datetime_ny"]), curve["drawdown_pct"], label=name)
    ax.legend()
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_overlay_yearly(yearly_map: dict[str, pd.DataFrame], path: Path, title: str) -> None:
    if not yearly_map:
        _empty_plot(path, title)
        return
    all_years = sorted({str(year) for frame in yearly_map.values() for year in frame.groupby("year")["total_pnl_r"].sum().index})
    if not all_years:
        _empty_plot(path, title)
        return
    fig, ax = plt.subplots(figsize=(12, 5))
    width = 0.8 / max(len(yearly_map), 1)
    x = list(range(len(all_years)))
    for idx, (name, frame) in enumerate(yearly_map.items()):
        grouped = frame.groupby("year")["total_pnl_r"].sum()
        values = [grouped.get(year, 0.0) for year in all_years]
        offsets = [pos + idx * width for pos in x]
        ax.bar(offsets, values, width=width, label=name)
    ax.set_xticks([pos + width for pos in x])
    ax.set_xticklabels(all_years)
    ax.legend()
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

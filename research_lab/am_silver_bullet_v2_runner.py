from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from research_lab.build_am_grade_news_dataset import DEFAULT_OUTPUT_FILE as AM_NEWS_FILE
from research_lab.build_am_grade_news_dataset import build_am_grade_news_dataset
from research_lab.config import (
    DEFAULT_HIGH_PRECISION_PREPARED_DIR,
    EngineConfig,
    INITIAL_CAPITAL,
    NY_TZ,
    NewsConfig,
    with_execution_mode,
)
from research_lab.data_loader import (
    atr,
    fx_market_mask,
    fx_session_date,
    load_high_precision_package,
    prepare_common_frame,
    resample_ohlcv_to_timeframe,
    validate_price_frame,
)
from research_lab.engine import entry_open_index, run_backtest
from research_lab.ict_primitives import add_fvg_columns, add_pivot_structure_columns
from research_lab.news_filter import build_entry_block, require_operational_news
from research_lab.report import export_strategy_bundle, summarize_result
from research_lab.strategies import am_silver_bullet_ny_v2


RESULTS_DIR = Path("results") / "am_silver_bullet_ny_v2"
PAIR = "EURUSD"
TIMEFRAME = "M1"
FRAME_BUILD_START = "09:30"
FRAME_BUILD_END = "12:00"
PERIODS: dict[str, tuple[str, str]] = {
    "development_2020_2023": ("2020-01-01", "2023-12-31"),
    "validation_2024": ("2024-01-01", "2024-12-31"),
    "holdout_2025": ("2025-01-01", "2025-12-31"),
    "full_2020_2025": ("2020-01-01", "2025-12-31"),
}

SERIOUS_DEV_MIN_TRADES = 24
SERIOUS_DEV_MIN_TRADES_PER_MONTH = 0.50
SERIOUS_DEV_MIN_PF = 1.10
SERIOUS_DEV_MIN_EXPECTANCY = 0.03
SERIOUS_MIN_VAL_HOLD_TRADES = 6
SERIOUS_MIN_VAL_HOLD_PF = 0.95
SERIOUS_MIN_VAL_HOLD_EXPECTANCY = 0.0
SERIOUS_FULL_MIN_TRADES = 36

M5_CONTEXT_COLUMNS = [
    "ctx_m5_sb_anchor_high",
    "ctx_m5_sb_anchor_low",
    "ctx_m5_swept_anchor_high",
    "ctx_m5_swept_anchor_low",
]


def build_output_root() -> Path:
    timestamp = pd.Timestamp.now(tz=NY_TZ).strftime("%Y%m%d_%H%M%S")
    root = RESULTS_DIR / f"{timestamp}_am_silver_bullet_ny_v2"
    root.mkdir(parents=True, exist_ok=True)
    return root


def build_engine_config() -> EngineConfig:
    base = EngineConfig(
        pair=PAIR,
        risk_pct=0.5,
        max_spread_pips=2.0,
        slippage_pips=0.1,
        commission_per_lot_roundturn_usd=7.0,
        max_trades_per_day=1,
        session_cutoff="12:00",
        enforce_hard_stop=True,
    )
    return with_execution_mode(base, "high_precision_mode")


def build_news_config() -> NewsConfig:
    return NewsConfig(
        enabled=True,
        file_path=AM_NEWS_FILE,
        raw_file_path=AM_NEWS_FILE,
        source_approved=True,
        pre_minutes=30,
        post_minutes=60,
        forced_exit_pre_news=True,
        cancel_pending_pre_news=True,
        pre_news_exit_minutes=10,
        currencies=("USD", "EUR"),
        impact_levels=("HIGH",),
    )


def schedule_used() -> dict[str, str]:
    return {
        "context_build_window": f"{FRAME_BUILD_START}-{FRAME_BUILD_END}",
        "entry_start": "10:00",
        "entry_end": "11:00",
        "force_close": "12:00",
    }


def period_slice(frame: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    start_ts = pd.Timestamp(start, tz=NY_TZ)
    end_ts = pd.Timestamp(end, tz=NY_TZ) + pd.Timedelta(days=1) - pd.Timedelta(minutes=1)
    return frame.loc[(frame.index >= start_ts) & (frame.index <= end_ts)].copy()


def _filtered_high_precision_package(start: str, end: str) -> dict[str, pd.DataFrame]:
    package = load_high_precision_package(PAIR, DEFAULT_HIGH_PRECISION_PREPARED_DIR)
    start_ts = pd.Timestamp(start, tz=NY_TZ)
    end_ts = pd.Timestamp(end, tz=NY_TZ) + pd.Timedelta(days=1) - pd.Timedelta(minutes=1)

    filtered: dict[str, pd.DataFrame] = {}
    for side, source in package.items():
        frame = source.loc[(source.index >= start_ts) & (source.index <= end_ts)].copy()
        frame = frame[fx_market_mask(frame.index)].copy()
        validate_price_frame(frame)
        filtered[f"{side}_m1"] = frame
    return filtered


def _build_m1_research_frame(mid_m1: pd.DataFrame) -> pd.DataFrame:
    base = mid_m1.copy()
    base["session_date"] = fx_session_date(base.index)
    base["atr14"] = atr(base, 14)
    base["bar_range"] = base["high"] - base["low"]
    base["range_atr"] = base["bar_range"] / base["atr14"].replace(0.0, np.nan)

    research = base.between_time(FRAME_BUILD_START, FRAME_BUILD_END).copy()
    day_frames: list[pd.DataFrame] = []
    for _, day in research.groupby("session_date", sort=False):
        if day.empty:
            continue
        day = add_fvg_columns(day)
        day = add_pivot_structure_columns(day)
        day_frames.append(day)
    if not day_frames:
        return research.iloc[0:0].copy()
    frame = pd.concat(day_frames).sort_index()
    return frame.dropna(subset=["atr14", "range_atr"]).copy()


def _build_m5_context(mid_m1: pd.DataFrame) -> pd.DataFrame:
    m5_source = resample_ohlcv_to_timeframe(mid_m1, "M5")
    m5_frame = prepare_common_frame(m5_source, target_timeframe="M5")
    context = pd.DataFrame(index=m5_frame.index)
    context["ctx_m5_sb_anchor_high"] = m5_frame["session_range_high_03_00_08_30"]
    context["ctx_m5_sb_anchor_low"] = m5_frame["session_range_low_03_00_08_30"]
    context["ctx_m5_swept_anchor_high"] = (
        m5_frame["day_running_high"] > m5_frame["session_range_high_03_00_08_30"]
    ).astype(bool)
    context["ctx_m5_swept_anchor_low"] = (
        m5_frame["day_running_low"] < m5_frame["session_range_low_03_00_08_30"]
    ).astype(bool)
    return context


def build_research_frame(start: str, end: str) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    filtered = _filtered_high_precision_package(start, end)
    m1_frame = _build_m1_research_frame(filtered["mid_m1"])
    m5_context = _build_m5_context(filtered["mid_m1"])
    frame = m1_frame.join(m5_context.reindex(m1_frame.index, method="ffill"))
    frame = frame.dropna(subset=M5_CONTEXT_COLUMNS).copy()
    return frame, filtered


def align_precision_package(filtered: dict[str, pd.DataFrame], frame_index: pd.DatetimeIndex) -> dict[str, pd.DataFrame]:
    bid_exec = filtered["bid_m1"].loc[frame_index].copy()
    ask_exec = filtered["ask_m1"].loc[frame_index].copy()
    mid_exec = filtered["mid_m1"].loc[frame_index].copy()
    return {
        "bid_m1": bid_exec.copy(),
        "ask_m1": ask_exec.copy(),
        "mid_m1": mid_exec.copy(),
        "bid_exec": bid_exec.copy(),
        "ask_exec": ask_exec.copy(),
        "mid_exec": mid_exec.copy(),
        # Backward-compatible aliases expected by the current engine contract.
        "bid_m15": bid_exec.copy(),
        "ask_m15": ask_exec.copy(),
        "mid_m15": mid_exec.copy(),
    }


def news_metrics_from_summary(summary: dict[str, Any], trades_export: pd.DataFrame) -> dict[str, Any]:
    news_exit_count = int((trades_export["exit_reason"] == "news_fortress_kill").sum()) if not trades_export.empty else 0
    return {**summary, "news_exit_count": news_exit_count}


def evaluate_period(
    *,
    frame: pd.DataFrame,
    filtered_precision: dict[str, pd.DataFrame],
    params: dict[str, Any],
    engine_config: EngineConfig,
    news_result: Any,
    news_config: NewsConfig,
    start: str,
    end: str,
) -> dict[str, Any]:
    period_frame = period_slice(frame, start, end)
    if period_frame.empty:
        raise ValueError(f"Periodo vacio para {start} -> {end}")
    period_precision = align_precision_package(filtered_precision, period_frame.index)
    news_block = build_entry_block(entry_open_index(period_frame.index), news_result.events, news_config)
    result = run_backtest(
        strategy_module=am_silver_bullet_ny_v2,
        frame=period_frame,
        params=params,
        engine_config=engine_config,
        news_block=news_block,
        news_filter_used=news_result.enabled,
        precision_package=period_precision,
        data_source_used="dukascopy_m1_bid_ask_full",
        news_events=news_result.events,
        news_settings=news_config,
    )
    summary, trades_export, monthly_stats, yearly_stats, equity_export = summarize_result(
        am_silver_bullet_ny_v2.NAME,
        result.trades,
        result.equity_curve,
        params,
        news_result.enabled,
        INITIAL_CAPITAL,
        None,
        costs_used={"execution_mode": engine_config.execution_mode, "cost_profile": engine_config.cost_profile},
        timeframe=TIMEFRAME,
        schedule_used=schedule_used(),
        break_even_setting=params.get("break_even_at_r"),
    )
    return {
        "summary": news_metrics_from_summary(summary, trades_export),
        "trades_export": trades_export,
        "monthly_stats": monthly_stats,
        "yearly_stats": yearly_stats,
        "equity_export": equity_export,
    }


def selection_score(summary: dict[str, Any]) -> float:
    profit_factor = float(summary["profit_factor"])
    pf_for_score = 3.0 if not math.isfinite(profit_factor) else min(profit_factor, 3.0)
    score = 0.0
    score += pf_for_score * 140.0
    score += float(summary["expectancy_r"]) * 1100.0
    score += float(summary["total_return_pct"]) * 1.0
    score -= float(summary["max_drawdown_pct"]) * 3.0
    score -= float(summary["negative_years"]) * 35.0
    score -= float(summary["news_exit_count"]) * 5.0
    if bool(summary.get("insufficient_sample", False)):
        score -= 80.0
    total_trades = int(summary["total_trades"])
    avg_trades_per_month = float(summary["avg_trades_per_month"])
    if total_trades < 10:
        score -= 1200.0
    elif total_trades < SERIOUS_DEV_MIN_TRADES:
        score -= 240.0
    if avg_trades_per_month < SERIOUS_DEV_MIN_TRADES_PER_MONTH:
        score -= 140.0
    if float(summary["profit_factor"]) <= 1.0:
        score -= 100.0
    if float(summary["expectancy_r"]) <= 0.0:
        score -= 100.0
    return score


def serious_gate_from_row(row: pd.Series) -> bool:
    return (
        int(row["dev_total_trades"]) >= SERIOUS_DEV_MIN_TRADES
        and float(row["dev_avg_trades_per_month"]) >= SERIOUS_DEV_MIN_TRADES_PER_MONTH
        and float(row["dev_profit_factor"]) >= SERIOUS_DEV_MIN_PF
        and float(row["dev_expectancy_r"]) >= SERIOUS_DEV_MIN_EXPECTANCY
        and int(row["val_total_trades"]) >= SERIOUS_MIN_VAL_HOLD_TRADES
        and int(row["hold_total_trades"]) >= SERIOUS_MIN_VAL_HOLD_TRADES
        and float(row["val_profit_factor"]) >= SERIOUS_MIN_VAL_HOLD_PF
        and float(row["hold_profit_factor"]) >= SERIOUS_MIN_VAL_HOLD_PF
        and float(row["val_expectancy_r"]) >= SERIOUS_MIN_VAL_HOLD_EXPECTANCY
        and float(row["hold_expectancy_r"]) >= SERIOUS_MIN_VAL_HOLD_EXPECTANCY
        and int(row["full_total_trades"]) >= SERIOUS_FULL_MIN_TRADES
    )


def verdict_from_selected(selected_row: pd.Series, serious_candidates: int) -> str:
    if serious_candidates > 0 and bool(selected_row["serious_candidate"]):
        return "Promising candidate"
    if float(selected_row["full_total_trades"]) > 5 and float(selected_row["full_profit_factor"]) > 1.0:
        return "Still diagnostic / not defendable"
    return "Close the line"


def main() -> Path:
    am_summary = build_am_grade_news_dataset()
    if am_summary["module_verdict"] != "READY_FOR_STRICT_AM_RESEARCH":
        raise RuntimeError(
            "La iteracion AM Silver Bullet V2 requiere compuerta AM aprobada. "
            f"Veredicto actual={am_summary['module_verdict']}."
        )

    output_root = build_output_root()
    engine_config = build_engine_config()
    news_config = build_news_config()
    news_result = require_operational_news(PAIR, news_config, context="am_silver_bullet_v2_runner")
    full_frame, filtered_precision = build_research_frame(*PERIODS["full_2020_2025"])

    ranking_rows: list[dict[str, Any]] = []
    for combo_id, params in enumerate(am_silver_bullet_ny_v2.parameter_grid(), start=1):
        dev = evaluate_period(
            frame=full_frame,
            filtered_precision=filtered_precision,
            params=params,
            engine_config=engine_config,
            news_result=news_result,
            news_config=news_config,
            start=PERIODS["development_2020_2023"][0],
            end=PERIODS["development_2020_2023"][1],
        )["summary"]
        val = evaluate_period(
            frame=full_frame,
            filtered_precision=filtered_precision,
            params=params,
            engine_config=engine_config,
            news_result=news_result,
            news_config=news_config,
            start=PERIODS["validation_2024"][0],
            end=PERIODS["validation_2024"][1],
        )["summary"]
        hold = evaluate_period(
            frame=full_frame,
            filtered_precision=filtered_precision,
            params=params,
            engine_config=engine_config,
            news_result=news_result,
            news_config=news_config,
            start=PERIODS["holdout_2025"][0],
            end=PERIODS["holdout_2025"][1],
        )["summary"]
        full = evaluate_period(
            frame=full_frame,
            filtered_precision=filtered_precision,
            params=params,
            engine_config=engine_config,
            news_result=news_result,
            news_config=news_config,
            start=PERIODS["full_2020_2025"][0],
            end=PERIODS["full_2020_2025"][1],
        )["summary"]
        ranking_rows.append(
            {
                "strategy_name": am_silver_bullet_ny_v2.NAME,
                "combo_id": combo_id,
                "variant_label": str(params.get("variant_label", f"combo_{combo_id}")),
                "params_json": json.dumps(params, ensure_ascii=False, sort_keys=True),
                "selection_score_dev": selection_score(dev),
                "serious_candidate": False,
                "dev_profit_factor": dev["profit_factor"],
                "dev_expectancy_r": dev["expectancy_r"],
                "dev_max_drawdown_pct": dev["max_drawdown_pct"],
                "dev_total_trades": dev["total_trades"],
                "dev_avg_trades_per_month": dev["avg_trades_per_month"],
                "val_profit_factor": val["profit_factor"],
                "val_expectancy_r": val["expectancy_r"],
                "val_total_trades": val["total_trades"],
                "hold_profit_factor": hold["profit_factor"],
                "hold_expectancy_r": hold["expectancy_r"],
                "hold_total_trades": hold["total_trades"],
                "full_profit_factor": full["profit_factor"],
                "full_expectancy_r": full["expectancy_r"],
                "full_total_trades": full["total_trades"],
                "full_max_drawdown_pct": full["max_drawdown_pct"],
            }
        )

    ranking = pd.DataFrame(ranking_rows)
    ranking["serious_candidate"] = ranking.apply(serious_gate_from_row, axis=1)
    ranking = ranking.sort_values(["serious_candidate", "selection_score_dev"], ascending=[False, False]).reset_index(drop=True)
    selected_row = ranking.iloc[0].copy()
    selected_params = json.loads(str(selected_row["params_json"]))

    period_bundles: dict[str, dict[str, Any]] = {}
    for label, (start, end) in PERIODS.items():
        period_bundles[label] = evaluate_period(
            frame=full_frame,
            filtered_precision=filtered_precision,
            params=selected_params,
            engine_config=engine_config,
            news_result=news_result,
            news_config=news_config,
            start=start,
            end=end,
        )

    strategy_dir = output_root / am_silver_bullet_ny_v2.NAME
    strategy_dir.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(strategy_dir / "ranking.csv", index=False)
    export_strategy_bundle(
        strategy_dir,
        summary=period_bundles["full_2020_2025"]["summary"],
        trades_export=period_bundles["full_2020_2025"]["trades_export"],
        monthly_stats=period_bundles["full_2020_2025"]["monthly_stats"],
        yearly_stats=period_bundles["full_2020_2025"]["yearly_stats"],
        equity_export=period_bundles["full_2020_2025"]["equity_export"],
        optimization_results=ranking,
        extra_json={
            "selected_params.json": selected_params,
            "period_summaries.json": {label: bundle_["summary"] for label, bundle_ in period_bundles.items()},
            "serious_gate.json": {
                "dev_total_trades": SERIOUS_DEV_MIN_TRADES,
                "dev_avg_trades_per_month": SERIOUS_DEV_MIN_TRADES_PER_MONTH,
                "dev_profit_factor": SERIOUS_DEV_MIN_PF,
                "dev_expectancy_r": SERIOUS_DEV_MIN_EXPECTANCY,
                "val_hold_trades": SERIOUS_MIN_VAL_HOLD_TRADES,
                "val_hold_profit_factor": SERIOUS_MIN_VAL_HOLD_PF,
                "val_hold_expectancy_r": SERIOUS_MIN_VAL_HOLD_EXPECTANCY,
                "full_total_trades": SERIOUS_FULL_MIN_TRADES,
            },
            "frame_contract.json": {
                "canonical_timeframe": TIMEFRAME,
                "context_timeframe": "M5",
                "frame_build_window": f"{FRAME_BUILD_START}-{FRAME_BUILD_END}",
            },
            "verdict.json": {"verdict": verdict_from_selected(selected_row, int(ranking["serious_candidate"].sum()))},
        },
    )

    scorecard = pd.DataFrame(
        [
            {
                "strategy_name": am_silver_bullet_ny_v2.NAME,
                "selected_combo_id": int(selected_row["combo_id"]),
                "variant_label": str(selected_row["variant_label"]),
                "verdict": verdict_from_selected(selected_row, int(ranking["serious_candidate"].sum())),
                "serious_candidates": int(ranking["serious_candidate"].sum()),
                "dev_trades": int(period_bundles["development_2020_2023"]["summary"]["total_trades"]),
                "dev_pf": float(period_bundles["development_2020_2023"]["summary"]["profit_factor"]),
                "dev_expectancy_r": float(period_bundles["development_2020_2023"]["summary"]["expectancy_r"]),
                "val_trades": int(period_bundles["validation_2024"]["summary"]["total_trades"]),
                "val_pf": float(period_bundles["validation_2024"]["summary"]["profit_factor"]),
                "val_expectancy_r": float(period_bundles["validation_2024"]["summary"]["expectancy_r"]),
                "hold_trades": int(period_bundles["holdout_2025"]["summary"]["total_trades"]),
                "hold_pf": float(period_bundles["holdout_2025"]["summary"]["profit_factor"]),
                "hold_expectancy_r": float(period_bundles["holdout_2025"]["summary"]["expectancy_r"]),
                "full_trades": int(period_bundles["full_2020_2025"]["summary"]["total_trades"]),
                "full_pf": float(period_bundles["full_2020_2025"]["summary"]["profit_factor"]),
                "full_expectancy_r": float(period_bundles["full_2020_2025"]["summary"]["expectancy_r"]),
                "full_max_drawdown_pct": float(period_bundles["full_2020_2025"]["summary"]["max_drawdown_pct"]),
                "news_exit_count": int(period_bundles["full_2020_2025"]["summary"]["news_exit_count"]),
                "selected_params_json": json.dumps(selected_params, ensure_ascii=False, sort_keys=True),
            }
        ]
    )
    scorecard.to_csv(output_root / "am_strategy_scorecard.csv", index=False)
    (output_root / "run_manifest.json").write_text(
        json.dumps(
            {
                "pair": PAIR,
                "timeframe": TIMEFRAME,
                "periods": PERIODS,
                "news_file": str(AM_NEWS_FILE),
                "schedule": schedule_used(),
                "engine": {
                    "risk_pct": engine_config.risk_pct,
                    "max_spread_pips": engine_config.max_spread_pips,
                    "slippage_pips": engine_config.slippage_pips,
                    "session_cutoff": engine_config.session_cutoff,
                    "hard_stop_required": engine_config.enforce_hard_stop,
                },
                "serious_gate": {
                    "dev_total_trades": SERIOUS_DEV_MIN_TRADES,
                    "dev_avg_trades_per_month": SERIOUS_DEV_MIN_TRADES_PER_MONTH,
                    "dev_profit_factor": SERIOUS_DEV_MIN_PF,
                    "dev_expectancy_r": SERIOUS_DEV_MIN_EXPECTANCY,
                    "val_hold_trades": SERIOUS_MIN_VAL_HOLD_TRADES,
                    "val_hold_profit_factor": SERIOUS_MIN_VAL_HOLD_PF,
                    "val_hold_expectancy_r": SERIOUS_MIN_VAL_HOLD_EXPECTANCY,
                    "full_total_trades": SERIOUS_FULL_MIN_TRADES,
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(scorecard.to_string(index=False))
    return output_root


if __name__ == "__main__":
    main()

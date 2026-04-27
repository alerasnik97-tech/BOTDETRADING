from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

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
from research_lab.data_loader import fx_market_mask, load_high_precision_package, resample_ohlcv_to_timeframe
from research_lab.engine import entry_open_index, run_backtest
from research_lab.news_filter import build_entry_block, require_operational_news
from research_lab.strategies import eurusd_am_post_news_external_liquidity_shift as strategy_module

PAIR = "EURUSD"
TIMEFRAME = "M3"
PERIODS: dict[str, tuple[str, str]] = {
    "full_2020_2025": ("2020-01-01", "2025-12-31"),
}

FRICTION_PROFILES: dict[str, dict[str, Any]] = {
    "baseline": {
        "max_spread_pips": 2.0,
        "slippage_pips": 0.1,
        "commission_per_lot_roundturn_usd": 7.0,
    },
    "medium": {
        "max_spread_pips": 3.0,
        "slippage_pips": 0.3,
        "commission_per_lot_roundturn_usd": 10.0,
    },
    "hard": {
        "max_spread_pips": 4.0,
        "slippage_pips": 0.5,
        "commission_per_lot_roundturn_usd": 15.0,
    },
}


def build_engine_config(friction_profile: dict[str, Any]) -> EngineConfig:
    base = EngineConfig(
        pair=PAIR,
        risk_pct=0.5,
        max_spread_pips=friction_profile["max_spread_pips"],
        slippage_pips=friction_profile["slippage_pips"],
        commission_per_lot_roundturn_usd=friction_profile["commission_per_lot_roundturn_usd"],
        max_trades_per_day=1,
        session_cutoff="11:30",
        enforce_hard_stop=True,
    )
    return with_execution_mode(base, "high_precision_mode")


def build_news_config() -> NewsConfig:
    from research_lab.build_am_grade_news_dataset import DEFAULT_OUTPUT_FILE as AM_NEWS_FILE
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


def _filtered_high_precision_package(start: str, end: str) -> dict[str, pd.DataFrame]:
    package = load_high_precision_package(PAIR, DEFAULT_HIGH_PRECISION_PREPARED_DIR)
    start_ts = pd.Timestamp(start, tz=NY_TZ)
    end_ts = pd.Timestamp(end, tz=NY_TZ) + pd.Timedelta(days=1) - pd.Timedelta(minutes=1)

    filtered: dict[str, pd.DataFrame] = {}
    for side, source in package.items():
        frame = source.loc[(source.index >= start_ts) & (source.index <= end_ts)].copy()
        frame = frame[fx_market_mask(frame.index)].copy()
        filtered[f"{side}_m1"] = frame
    return filtered


def _build_m3_m5_frames(mid_m1: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    from research_lab.data_loader import prepare_common_frame
    m3_frame = prepare_common_frame(mid_m1, target_timeframe=TIMEFRAME)
    m5_frame = prepare_common_frame(mid_m1, target_timeframe="M5")
    return m3_frame, m5_frame


def _align_precision_package(filtered: dict[str, pd.DataFrame], frame_index: pd.DatetimeIndex) -> dict[str, pd.DataFrame]:
    bid_exec = resample_ohlcv_to_timeframe(filtered["bid_m1"], TIMEFRAME).loc[frame_index].copy()
    ask_exec = resample_ohlcv_to_timeframe(filtered["ask_m1"], TIMEFRAME).loc[frame_index].copy()
    mid_exec = resample_ohlcv_to_timeframe(filtered["mid_m1"], TIMEFRAME).loc[frame_index].copy()
    return {
        "bid_m1": filtered["bid_m1"].copy(),
        "ask_m1": filtered["ask_m1"].copy(),
        "mid_m1": filtered["mid_m1"].copy(),
        "bid_exec": bid_exec.copy(),
        "ask_exec": ask_exec.copy(),
        "mid_exec": mid_exec.copy(),
        "bid_m15": bid_exec.copy(),
        "ask_m15": ask_exec.copy(),
    }


def period_slice(frame: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    start_ts = pd.Timestamp(start, tz=NY_TZ)
    end_ts = pd.Timestamp(end, tz=NY_TZ) + pd.Timedelta(days=1) - pd.Timedelta(minutes=3)
    return frame.loc[(frame.index >= start_ts) & (frame.index <= end_ts)].copy()


def _profit_factor(pnl_r: pd.Series) -> float:
    gross_profit = float(pnl_r[pnl_r > 0].sum())
    gross_loss = float(pnl_r[pnl_r < 0].sum())
    return gross_profit / abs(gross_loss) if gross_loss < 0 else float("inf")


def main() -> Path:
    from research_lab.build_am_grade_news_dataset import DEFAULT_OUTPUT_FILE as AM_NEWS_FILE
    from research_lab.build_am_grade_news_dataset import build_am_grade_news_dataset
    from research_lab.eurusd_am_post_news_external_liquidity_shift_runner import (
        build_research_frame,
    )
    from research_lab.report import summarize_result

    am_summary = build_am_grade_news_dataset()
    if am_summary["module_verdict"] != "READY_FOR_STRICT_AM_RESEARCH":
        raise RuntimeError(
            f"La hipotesis requiere compuerta AM aprobada. Veredicto actual={am_summary['module_verdict']}."
        )

    results_dir = Path("results") / strategy_module.NAME / "friction_stress"
    results_dir.mkdir(parents=True, exist_ok=True)

    news_config = build_news_config()
    news_result = require_operational_news(PAIR, news_config, context=strategy_module.NAME)
    full_frame, full_precision_package, full_signal_log = build_research_frame(
        *PERIODS["full_2020_2025"],
        news_events=news_result.events,
        news_config=news_config,
    )

    params = strategy_module.default_params()
    friction_results: dict[str, dict[str, Any]] = {}

    for profile_name, profile_config in FRICTION_PROFILES.items():
        print(f"\n=== Ejecutando perfil de fricción: {profile_name} ===")
        engine_config = build_engine_config(profile_config)

        period_frame = period_slice(full_frame, *PERIODS["full_2020_2025"])
        period_precision = _align_precision_package(
            {"bid_m1": full_precision_package["bid_m1"], "ask_m1": full_precision_package["ask_m1"], "mid_m1": full_precision_package["mid_m1"]},
            period_frame.index,
        )
        news_block = build_entry_block(entry_open_index(period_frame.index), news_result.events, news_config)

        result = run_backtest(
            strategy_module=strategy_module,
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

        summary, _, _, _, _ = summarize_result(
            strategy_module.NAME,
            result.trades,
            result.equity_curve,
            params,
            news_result.enabled,
            INITIAL_CAPITAL,
            None,
            costs_used={"execution_mode": engine_config.execution_mode, "cost_profile": engine_config.cost_profile},
            timeframe=TIMEFRAME,
            schedule_used={"friction_profile": profile_name},
            break_even_setting=params.get("break_even_at_r"),
        )

        friction_results[profile_name] = {
            "profile_config": profile_config,
            "summary": summary,
        }

        print(f"  Trades: {summary['total_trades']}")
        print(f"  PF: {summary['profit_factor']:.2f}")
        print(f"  Expectancy: {summary['expectancy_r']:.3f}R")
        print(f"  DD: {summary['max_drawdown_pct']:.2f}%")

    friction_summary = pd.DataFrame([
        {
            "friction_profile": profile_name,
            "max_spread_pips": profile_config["max_spread_pips"],
            "slippage_pips": profile_config["slippage_pips"],
            "commission_usd": profile_config["commission_per_lot_roundturn_usd"],
            "total_trades": friction_results[profile_name]["summary"]["total_trades"],
            "profit_factor": friction_results[profile_name]["summary"]["profit_factor"],
            "expectancy_r": friction_results[profile_name]["summary"]["expectancy_r"],
            "max_drawdown_pct": friction_results[profile_name]["summary"]["max_drawdown_pct"],
            "total_return_pct": friction_results[profile_name]["summary"]["total_return_pct"],
        }
        for profile_name, profile_config in FRICTION_PROFILES.items()
    ])

    output_file = results_dir / f"{strategy_module.NAME}_friction_stress.csv"
    friction_summary.to_csv(output_file, index=False)
    print(f"\n=== Resultados guardados en: {output_file} ===")
    print(friction_summary.to_string(index=False))

    return results_dir


if __name__ == "__main__":
    main()

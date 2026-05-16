from __future__ import annotations

import argparse
import io
import json
import platform
import sys
import unittest
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from research_lab.config import (
    DEFAULT_DATA_DIRS,
    DEFAULT_NEWS_FILE,
    DEFAULT_RAW_NEWS_FILE,
    DEFAULT_NEWS_SUMMARY_FILE,
    DEFAULT_PAIR,
    EngineConfig,
    INITIAL_CAPITAL,
    NewsConfig,
    SessionConfig,
    resolved_cost_profile,
    resolved_intrabar_policy,
    with_execution_mode,
)
from research_lab.data_loader import _resample_to_m15, describe_available_price_data, load_prepared_ohlcv, load_price_data, prepare_common_frame
from research_lab.engine import entry_open_index, run_backtest
from research_lab.news_filter import build_entry_block, load_news_events, load_news_summary, news_result_payload
from research_lab.report import build_trades_export, summarize_result, sync_visible_chatgpt
from research_lab.strategies import STRATEGY_REGISTRY


AUDIT_RESULTS_DIR = Path("results") / "research_lab_level3_audit"


def build_output_root(results_dir: Path) -> Path:
    timestamp = pd.Timestamp.now(tz="America/New_York").strftime("%Y%m%d_%H%M%S")
    path = results_dir / f"{timestamp}_level3_audit"
    path.mkdir(parents=True, exist_ok=True)
    return path


def flatten_suite(suite: unittest.TestSuite) -> list[unittest.TestCase]:
    tests: list[unittest.TestCase] = []
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            tests.extend(flatten_suite(item))
        else:
            tests.append(item)
    return tests


def run_test_suite() -> tuple[pd.DataFrame, str]:
    loader = unittest.TestLoader()
    suite = loader.discover("research_lab/tests")
    cases = flatten_suite(suite)
    case_ids = [case.id() for case in cases]
    buffer = io.StringIO()
    result = unittest.TextTestRunner(stream=buffer, verbosity=2).run(suite)
    failure_ids = {case.id() for case, _ in list(result.failures) + list(result.errors)}
    rows = [{"test_id": case_id, "status": "FAIL" if case_id in failure_ids else "PASS"} for case_id in case_ids]
    return pd.DataFrame(rows), buffer.getvalue()


def environment_summary(repo_root: Path) -> dict[str, Any]:
    import pandas as pd  # local import to report concrete runtime

    return {
        "cwd": str(repo_root),
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "relevant_entrypoints": [
            "research_lab/main.py",
            "research_lab/light_runner.py",
            "research_lab/news_rebuild.py",
            "research_lab/audit_level3.py",
        ],
        "relevant_tests": [
            "research_lab/tests/test_data_loader.py",
            "research_lab/tests/test_engine.py",
            "research_lab/tests/test_level2_execution.py",
            "research_lab/tests/test_level3_precision.py",
            "research_lab/tests/test_news_filter.py",
            "research_lab/tests/test_integration_real_project.py",
        ],
    }


def data_summary(pair: str, data_dirs: list[Path], start: str, end: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], pd.DataFrame]:
    catalog = pd.DataFrame(describe_available_price_data(pair, data_dirs))
    raw_m5 = load_price_data(pair, data_dirs, start, end)
    prepared_m15 = load_prepared_ohlcv(pair, data_dirs, "M15")
    start_ts = pd.Timestamp(start, tz="America/New_York")
    end_ts = pd.Timestamp(end, tz="America/New_York") + pd.Timedelta(days=1)
    prepared_m15 = prepared_m15.loc[(prepared_m15.index >= start_ts) & (prepared_m15.index < end_ts)].copy()
    resampled_m15 = _resample_to_m15(raw_m5)
    common_index = resampled_m15.index.intersection(prepared_m15.index)
    compare = (
        resampled_m15.loc[common_index, ["open", "high", "low", "close", "volume"]]
        - prepared_m15.loc[common_index, ["open", "high", "low", "close", "volume"]]
    ).abs()
    strict_mask = (compare > 1e-12).any(axis=1)
    material_mask = (
        (compare[["open", "high", "low", "close"]] > 1e-9).any(axis=1)
        | (compare["volume"] > 1e-9)
    )
    mismatch_rows = int(strict_mask.sum())
    material_mismatch_rows = int(material_mask.sum())
    comparison_sample = compare.loc[strict_mask].head(25).reset_index(names="timestamp_ny")
    max_diffs = {f"max_abs_diff_{column}": float(compare[column].max()) for column in compare.columns}
    summary = {
        "pair": pair,
        "data_dirs": [str(path) for path in data_dirs],
        "requested_period": f"{start} -> {end}",
        "source_catalog_rows": int(len(catalog)),
        "raw_m5_rows": int(len(raw_m5)),
        "resampled_m15_rows": int(len(resampled_m15)),
        "prepared_m15_rows": int(len(prepared_m15)),
        "duplicates_after_merge": int(raw_m5.index.duplicated().sum()),
        "sunday_reopen_rows": int((((raw_m5.index.dayofweek == 6) & ((raw_m5.index.hour * 60 + raw_m5.index.minute) > 17 * 60))).sum()),
        "resample_m15_vs_prepared_m15_strict_mismatch_rows": mismatch_rows,
        "resample_m15_vs_prepared_m15_material_mismatch_rows": material_mismatch_rows,
        "price_source_inferred": "OHLC BID prepared from Dukascopy",
        "best_available_granularity": "M5",
        "ask_history_available": False,
        "bid_ask_history_available": False,
        "higher_precision_dataset_available": False,
        **max_diffs,
    }
    return raw_m5, resampled_m15, summary, comparison_sample


def baseline_params() -> dict[str, Any]:
    return {
        "ema_fast": 20,
        "ema_slow": 100,
        "ema_pullback": 10,
        "adx_min": 18,
        "stop_atr": 1.5,
        "target_rr": 1.5,
        "break_even_at_r": None,
        "session_name": "light_fixed",
        "use_h1_context": False,
        "trailing_atr": False,
        "cooldown_bars": 0,
    }


def costs_payload(engine_config: EngineConfig) -> dict[str, Any]:
    return {
        "assumed_spread_pips": engine_config.assumed_spread_pips,
        "max_allowed_spread_pips": engine_config.max_spread_pips,
        "slippage_pips": engine_config.slippage_pips,
        "commission_per_lot_roundturn_usd": engine_config.commission_per_lot_roundturn_usd,
        "risk_pct": engine_config.risk_pct,
        "initial_capital": INITIAL_CAPITAL,
        "price_source": engine_config.price_source,
        "execution_mode": engine_config.execution_mode,
        "cost_profile_used": resolved_cost_profile(engine_config),
        "intrabar_policy_used": resolved_intrabar_policy(engine_config),
        "opening_session_end": engine_config.opening_session_end,
        "late_session_start": engine_config.late_session_start,
        "spread_opening_multiplier": engine_config.spread_opening_multiplier,
        "spread_high_vol_multiplier": engine_config.spread_high_vol_multiplier,
        "spread_late_session_multiplier": engine_config.spread_late_session_multiplier,
        "slippage_opening_multiplier": engine_config.slippage_opening_multiplier,
        "slippage_high_vol_multiplier": engine_config.slippage_high_vol_multiplier,
        "slippage_stop_multiplier": engine_config.slippage_stop_multiplier,
        "slippage_target_multiplier": engine_config.slippage_target_multiplier,
        "slippage_late_session_multiplier": engine_config.slippage_late_session_multiplier,
        "slippage_forced_close_multiplier": engine_config.slippage_forced_close_multiplier,
        "slippage_final_close_multiplier": engine_config.slippage_final_close_multiplier,
        "stress_spread_multiplier": engine_config.stress_spread_multiplier,
        "stress_slippage_multiplier": engine_config.stress_slippage_multiplier,
        "ambiguity_slippage_multiplier": engine_config.ambiguity_slippage_multiplier,
    }


def execution_summary(frame: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    strategy = STRATEGY_REGISTRY["ema_trend_pullback"]
    news_off = np.zeros(len(frame), dtype=bool)
    rows: list[dict[str, Any]] = []
    trade_log_sample = pd.DataFrame()
    for mode in ("normal_mode", "conservative_mode", "high_precision_mode"):
        engine_config = with_execution_mode(
            EngineConfig(
                pair="EURUSD",
                risk_pct=0.5,
                assumed_spread_pips=1.2,
                max_spread_pips=3.0,
                slippage_pips=0.2,
                commission_per_lot_roundturn_usd=7.0,
                max_trades_per_day=2,
                execution_mode=mode,
            ),
            mode,
        )
        result = run_backtest(strategy, frame, baseline_params(), engine_config, news_off, False)
        summary, trades_export, _monthly, _yearly, _equity = summarize_result(
            strategy.NAME,
            result.trades,
            result.equity_curve,
            baseline_params(),
            False,
            INITIAL_CAPITAL,
            None,
            costs_payload(engine_config),
            "M15",
            {"entry_start": "11:00", "entry_end": "19:00", "force_close": SessionConfig().force_close},
            None,
        )
        rows.append(
            {
                "execution_mode": mode,
                "cost_profile_used": resolved_cost_profile(engine_config),
                "intrabar_policy_used": resolved_intrabar_policy(engine_config),
                "total_trades": summary["total_trades"],
                "avg_trades_per_month": summary["avg_trades_per_month"],
                "profit_factor": summary["profit_factor"],
                "expectancy_r": summary["expectancy_r"],
                "max_drawdown_pct": summary["max_drawdown_pct"],
            }
        )
        if mode == "high_precision_mode":
            trade_log_sample = trades_export.head(25).copy()
    comparison = pd.DataFrame(rows)
    summary = {
        "baseline_strategy": strategy.NAME,
        "modes_compared": rows,
        "high_precision_mode_implemented": True,
        "high_precision_mode_purpose": "execution-detail layer with precision cost profile on top of BID OHLC baseline",
    }
    return summary, comparison, trade_log_sample


def news_summary(frame_m15: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    settings = NewsConfig(
        enabled=True,
        file_path=Path(DEFAULT_NEWS_FILE),
        raw_file_path=Path(DEFAULT_RAW_NEWS_FILE),
        source_approved=True,
        pre_minutes=15,
        post_minutes=15,
        currencies=("USD", "EUR"),
    )
    result = load_news_events("EURUSD", settings)
    summary = load_news_summary(Path(DEFAULT_NEWS_SUMMARY_FILE))
    block_mask = build_entry_block(entry_open_index(frame_m15.index), result.events if result.enabled else pd.DataFrame(), settings)
    key_events = pd.DataFrame(summary.get("key_event_validation", []))
    payload = {
        **summary,
        **news_result_payload(result),
        "blocked_entry_bars_m15": int(block_mask.sum()),
    }
    return payload, key_events


def build_report(
    *,
    env: dict[str, Any],
    data: dict[str, Any],
    execution: dict[str, Any],
    news: dict[str, Any],
    verdicts: dict[str, str],
    test_status: pd.DataFrame,
) -> str:
    lines = [
        "# Auditoria Nivel 3",
        "",
        "## Estado inicial",
        f"- python: `{env['python_executable']}`",
        f"- pandas: `{env['pandas_version']}`",
        f"- repo: `{env['cwd']}`",
        "",
        "## Veredictos por modulo",
        f"- data de precios: **{verdicts['data_prices']}**",
        f"- granularidad: **{verdicts['granularity']}**",
        f"- BID/ASK o aproximacion equivalente: **{verdicts['bid_ask']}**",
        f"- spread/slippage/commission: **{verdicts['execution_costs']}**",
        f"- fills: **{verdicts['fills']}**",
        f"- intrabar policy: **{verdicts['intrabar']}**",
        f"- horario/timezone/DST: **{verdicts['timezone']}**",
        f"- news module: **{verdicts['news_module']}**",
        f"- trade logging: **{verdicts['trade_logging']}**",
        f"- tests: **{verdicts['tests']}**",
        f"- estado general: **{verdicts['overall']}**",
        "",
        "## Hallazgos principales",
        f"- mejor granularidad realmente disponible: `{data['best_available_granularity']}`",
        f"- ASK historico real disponible: `{data['ask_history_available']}`",
        f"- mismatch estricto resample M5->M15 vs M15 preparado: `{data['resample_m15_vs_prepared_m15_strict_mismatch_rows']}` filas",
        f"- mismatch material resample M5->M15 vs M15 preparado: `{data['resample_m15_vs_prepared_m15_material_mismatch_rows']}` filas",
        f"- news module verdict: `{news.get('module_verdict')}`",
        f"- news source approved: `{news.get('source_approved')}`",
        f"- tests PASS: `{int(test_status['status'].eq('PASS').sum())}/{len(test_status)}`",
        "",
        "## Limite real",
        "- La fuente sigue siendo OHLC BID preparado desde Dukascopy.",
        "- No existe ASK historico real en el repo.",
        "- high_precision_mode mejora el modelado de costos y trazabilidad, no convierte OHLC en tick-level.",
        "- El modulo de noticias solo es confiable sobre el dataset derivado validado, no sobre el raw original.",
    ]
    return "\n".join(lines)


def compute_verdicts(data: dict[str, Any], news: dict[str, Any], test_status: pd.DataFrame, trade_log_sample: pd.DataFrame) -> dict[str, str]:
    def passed(suffix: str) -> bool:
        rows = test_status.loc[test_status["test_id"].str.endswith(suffix)]
        return not rows.empty and rows["status"].eq("PASS").all()

    data_prices = (
        "APROBADO"
        if data["duplicates_after_merge"] == 0 and data["resample_m15_vs_prepared_m15_material_mismatch_rows"] == 0
        else "RECHAZADO"
    )
    granularidad = "APROBADO CON OBSERVACIONES" if data["best_available_granularity"] == "M5" and not data["higher_precision_dataset_available"] else "APROBADO"
    bid_ask = "APROBADO CON OBSERVACIONES" if not data["ask_history_available"] else "APROBADO"
    execution_costs = "APROBADO" if all(
        passed(name)
        for name in (
            "test_spread_fields_are_logged_per_trade",
            "test_slippage_fields_are_logged_per_trade",
            "test_commission_is_separated_from_spread_and_slippage",
            "test_high_precision_mode_forced_close_is_more_expensive_than_normal",
        )
    ) else "RECHAZADO"
    fills = "APROBADO" if all(
        passed(name)
        for name in (
            "test_short_target_exit_includes_spread_and_slippage",
            "test_final_close_applies_exit_costs",
            "test_forced_close_is_flagged_in_both_modes",
        )
    ) else "RECHAZADO"
    intrabar = "APROBADO" if all(
        passed(name)
        for name in (
            "test_intrabar_policy_standard_respects_priority",
            "test_intrabar_policy_conservative_overrides_priority",
        )
    ) else "RECHAZADO"
    timezone = "APROBADO" if all(
        passed(name)
        for name in (
            "test_real_project_data_keeps_sunday_reopen",
            "test_entry_open_index_respects_dst_offsets",
        )
    ) else "RECHAZADO"
    news_module = "APROBADO" if bool(news.get("enabled")) and news.get("module_verdict") == "APPROVED_OPERATIONAL" else "RECHAZADO"
    trade_logging = "APROBADO" if {"signal_price", "fill_price", "exit_signal_price", "exit_fill_price", "price_source_used"}.issubset(set(trade_log_sample.columns)) else "RECHAZADO"
    tests = "APROBADO" if test_status["status"].eq("PASS").all() else "RECHAZADO"
    overall = "NIVEL 3 APROBADO" if all(
        verdict in {"APROBADO", "APROBADO CON OBSERVACIONES"}
        for verdict in (data_prices, granularidad, bid_ask, execution_costs, fills, intrabar, timezone, news_module, trade_logging, tests)
    ) else "NIVEL 3 NO APROBADO"
    return {
        "data_prices": data_prices,
        "granularity": granularidad,
        "bid_ask": bid_ask,
        "execution_costs": execution_costs,
        "fills": fills,
        "intrabar": intrabar,
        "timezone": timezone,
        "news_module": news_module,
        "trade_logging": trade_logging,
        "tests": tests,
        "overall": overall,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Auditoria maxima Nivel 3.")
    parser.add_argument("--pair", default=DEFAULT_PAIR)
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--data-dirs", nargs="+", default=[str(path) for path in DEFAULT_DATA_DIRS])
    parser.add_argument("--results-dir", default=str(AUDIT_RESULTS_DIR))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_root = build_output_root(Path(args.results_dir))
    env = environment_summary(Path.cwd())
    raw_m5, frame_m15, data, compare_sample = data_summary(args.pair.upper().strip(), [Path(path) for path in args.data_dirs], args.start, args.end)
    prepared_frame = prepare_common_frame(raw_m5)
    execution, execution_comparison, trade_log_sample = execution_summary(prepared_frame)
    news, key_events = news_summary(frame_m15)
    test_status, test_output = run_test_suite()
    verdicts = compute_verdicts(data, news, test_status, trade_log_sample)
    report_text = build_report(env=env, data=data, execution=execution, news=news, verdicts=verdicts, test_status=test_status)

    (output_root / "audit_level3_report.md").write_text(report_text, encoding="utf-8")
    (output_root / "environment_summary.json").write_text(json.dumps(env, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_root / "data_summary.json").write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_root / "execution_summary.json").write_text(json.dumps(execution, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_root / "news_summary.json").write_text(json.dumps(news, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_root / "module_verdicts.json").write_text(json.dumps(verdicts, indent=2, ensure_ascii=False), encoding="utf-8")
    test_status.to_csv(output_root / "test_status.csv", index=False)
    (output_root / "test_output.txt").write_text(test_output, encoding="utf-8")
    pd.DataFrame(describe_available_price_data(args.pair.upper().strip(), [Path(path) for path in args.data_dirs])).to_csv(output_root / "price_source_catalog.csv", index=False)
    execution_comparison.to_csv(output_root / "execution_mode_comparison.csv", index=False)
    compare_sample.to_csv(output_root / "m15_resample_vs_prepared_sample.csv", index=False)
    key_events.to_csv(output_root / "news_key_event_validation.csv", index=False)
    trade_log_sample.to_csv(output_root / "trade_log_sample.csv", index=False)

    archive = sync_visible_chatgpt(output_root)
    print(f"[level3] Auditoria lista en {output_root}")
    print(f"[level3] ZIP visible listo en {archive}")


if __name__ == "__main__":
    main()

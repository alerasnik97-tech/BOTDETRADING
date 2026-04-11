from __future__ import annotations

import argparse
import io
import json
import shutil
import unittest
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from research_lab.config import (
    DEFAULT_DATA_DIRS,
    DEFAULT_PAIR,
    DEFAULT_RESULTS_DIR,
    EngineConfig,
    INITIAL_CAPITAL,
    NewsConfig,
    resolved_cost_profile,
    resolved_intrabar_policy,
    with_execution_mode,
)
from research_lab.data_loader import load_price_data, prepare_common_frame
from research_lab.engine import run_backtest
from research_lab.news_filter import load_news_events, news_result_payload
from research_lab.report import summarize_result, sync_visible_chatgpt
from research_lab.strategies import STRATEGY_REGISTRY


AUDIT_RESULTS_DIR = Path("results") / "research_lab_level2_audit"


def build_output_root(results_dir: Path) -> Path:
    timestamp = pd.Timestamp.now(tz="America/New_York").strftime("%Y%m%d_%H%M%S")
    output_root = results_dir / f"{timestamp}_level2_audit"
    output_root.mkdir(parents=True, exist_ok=True)
    return output_root


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
        "high_vol_range_atr": engine_config.high_vol_range_atr,
        "spread_high_vol_multiplier": engine_config.spread_high_vol_multiplier,
        "spread_late_session_multiplier": engine_config.spread_late_session_multiplier,
        "stress_spread_multiplier": engine_config.stress_spread_multiplier,
        "slippage_high_vol_multiplier": engine_config.slippage_high_vol_multiplier,
        "slippage_stop_multiplier": engine_config.slippage_stop_multiplier,
        "slippage_late_session_multiplier": engine_config.slippage_late_session_multiplier,
        "stress_slippage_multiplier": engine_config.stress_slippage_multiplier,
        "ambiguity_slippage_multiplier": engine_config.ambiguity_slippage_multiplier,
        "intrabar_exit_priority": engine_config.intrabar_exit_priority,
    }


def build_engine_config(execution_mode: str) -> EngineConfig:
    base = EngineConfig(
        pair="EURUSD",
        risk_pct=0.5,
        assumed_spread_pips=1.2,
        max_spread_pips=3.0,
        slippage_pips=0.2,
        commission_per_lot_roundturn_usd=7.0,
        shock_candle_atr_max=2.2,
        max_trades_per_day=2,
    )
    return with_execution_mode(base, execution_mode)


def run_strategy_smoke(frame: pd.DataFrame, execution_mode: str) -> dict[str, Any]:
    strategy = STRATEGY_REGISTRY["ema_trend_pullback"]
    engine_config = build_engine_config(execution_mode)
    news_config = NewsConfig(enabled=False)
    news_result = load_news_events(engine_config.pair, news_config)
    news_block = np.zeros(len(frame), dtype=bool)
    result = run_backtest(strategy, frame, baseline_params(), engine_config, news_block, news_result.enabled)
    summary, trades_export, monthly_stats, yearly_stats, equity_export = summarize_result(
        strategy.NAME,
        result.trades,
        result.equity_curve,
        baseline_params(),
        news_result.enabled,
        INITIAL_CAPITAL,
        None,
        costs_payload(engine_config),
        "M15",
        {"entry_start": "11:00", "entry_end": "19:00", "force_close": "19:00"},
        None,
    )
    return {
        "engine_config": engine_config,
        "summary": summary,
        "trades_export": trades_export,
        "monthly_stats": monthly_stats,
        "yearly_stats": yearly_stats,
        "equity_export": equity_export,
        "news_module": news_result_payload(news_result),
    }


def synthetic_trade_log_sample() -> pd.DataFrame:
    index = pd.DatetimeIndex(
        [
            pd.Timestamp("2022-01-03 10:45:00", tz="America/New_York"),
            pd.Timestamp("2022-01-03 11:00:00", tz="America/New_York"),
            pd.Timestamp("2022-01-03 11:15:00", tz="America/New_York"),
            pd.Timestamp("2022-01-03 11:30:00", tz="America/New_York"),
        ]
    )
    frame = pd.DataFrame(
        {
            "open": [1.1000, 1.1000, 1.1000, 1.1010],
            "high": [1.1002, 1.1002, 1.1015, 1.1012],
            "low": [1.0998, 1.0998, 1.0995, 1.1006],
            "close": [1.1000, 1.1001, 1.1010, 1.1009],
            "atr14": [0.0010, 0.0010, 0.0010, 0.0010],
            "range_atr": [0.5, 0.5, 0.5, 0.5],
        },
        index=index,
    )

    class SyntheticStrategy:
        NAME = "synthetic_level2_audit"
        WARMUP_BARS = 0

        @staticmethod
        def signal(frame: pd.DataFrame, i: int, params: dict) -> dict[str, Any] | None:
            if frame.index[i].strftime("%H:%M") == "11:00":
                return {
                    "direction": "long",
                    "stop_mode": "atr",
                    "stop_atr": 1.0,
                    "target_rr": 1.0,
                    "session_name": "light_fixed",
                    "trailing_atr": False,
                }
            return None

    engine_config = build_engine_config("conservative_mode")
    result = run_backtest(SyntheticStrategy, frame, {}, engine_config, np.zeros(len(frame), dtype=bool), False)
    summary, trades_export, _, _, _ = summarize_result(
        SyntheticStrategy.NAME,
        result.trades,
        result.equity_curve,
        {},
        False,
        INITIAL_CAPITAL,
        None,
        costs_payload(engine_config),
        "M15",
        {"entry_start": "11:00", "entry_end": "19:00", "force_close": "19:00"},
        None,
    )
    sample = trades_export.copy()
    sample.attrs["summary"] = summary
    return sample


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
    rows = [
        {"test_id": case_id, "status": "FAIL" if case_id in failure_ids else "PASS"}
        for case_id in case_ids
    ]
    return pd.DataFrame(rows), buffer.getvalue()


def build_mode_comparison(normal_payload: dict[str, Any], conservative_payload: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for label, payload in (("normal_mode", normal_payload), ("conservative_mode", conservative_payload)):
        summary = payload["summary"]
        rows.append(
            {
                "execution_mode": label,
                "cost_profile_used": payload["engine_config"].cost_profile,
                "intrabar_policy_used": payload["engine_config"].intrabar_policy,
                "total_trades": summary["total_trades"],
                "avg_trades_per_month": summary["avg_trades_per_month"],
                "win_rate": summary["win_rate"],
                "profit_factor": summary["profit_factor"],
                "expectancy_r": summary["expectancy_r"],
                "total_return_pct": summary["total_return_pct"],
                "max_drawdown_pct": summary["max_drawdown_pct"],
            }
        )
    return pd.DataFrame(rows)


def build_module_verdicts(test_status: pd.DataFrame, trade_log_sample: pd.DataFrame) -> dict[str, str]:
    def passed(name: str) -> bool:
        rows = test_status.loc[test_status["test_id"].str.endswith(name)]
        return not rows.empty and rows["status"].eq("PASS").all()

    baseline_verdict = "APROBADO" if passed("test_baseline_normal_mode_matches_existing_behavior") else "RECHAZADO"
    normal_mode_verdict = "APROBADO" if baseline_verdict == "APROBADO" else "RECHAZADO"
    conservative_verdict = "APROBADO" if passed("test_conservative_mode_marks_trade_and_is_more_expensive") else "RECHAZADO"
    costs_verdict = "APROBADO CON OBSERVACIONES" if all(
        passed(name)
        for name in (
            "test_spread_fields_are_logged_per_trade",
            "test_slippage_fields_are_logged_per_trade",
            "test_commission_is_separated_from_spread_and_slippage",
        )
    ) else "RECHAZADO"
    intrabar_verdict = "APROBADO" if all(
        passed(name)
        for name in (
            "test_intrabar_policy_standard_respects_priority",
            "test_intrabar_policy_conservative_overrides_priority",
        )
    ) else "RECHAZADO"
    logging_verdict = "APROBADO" if (
        passed("test_trade_export_contains_level2_execution_fields")
        and {"entry_side", "signal_time_ny", "fill_time_ny", "execution_mode_used"}.issubset(set(trade_log_sample.columns))
    ) else "RECHAZADO"
    tests_verdict = "APROBADO" if test_status["status"].eq("PASS").all() else "RECHAZADO"
    overall = "NIVEL 2 APROBADO" if all(
        verdict in {"APROBADO", "APROBADO CON OBSERVACIONES"}
        for verdict in (
            baseline_verdict,
            normal_mode_verdict,
            conservative_verdict,
            costs_verdict,
            intrabar_verdict,
            logging_verdict,
            tests_verdict,
        )
    ) else "NIVEL 2 NO APROBADO"
    return {
        "baseline_frozen": baseline_verdict,
        "normal_mode": normal_mode_verdict,
        "conservative_mode": conservative_verdict,
        "costs_by_regime": costs_verdict,
        "intrabar_policy": intrabar_verdict,
        "trade_logging": logging_verdict,
        "tests": tests_verdict,
        "overall": overall,
    }


def build_report(
    *,
    mode_comparison: pd.DataFrame,
    verdicts: dict[str, str],
    test_status: pd.DataFrame,
    normal_payload: dict[str, Any],
    conservative_payload: dict[str, Any],
) -> str:
    normal = normal_payload["summary"]
    conservative = conservative_payload["summary"]
    lines = [
        "# Auditoria Nivel 2",
        "",
        "## Base congelada",
        "- Baseline oficial: `research_lab/BASELINE_LEVEL1.md`",
        "- Noticias: `OFF`",
        "- Limitacion estructural: `OHLC BID sin ASK historico real`",
        "",
        "## Veredictos por modulo",
        f"- baseline congelada: **{verdicts['baseline_frozen']}**",
        f"- normal_mode: **{verdicts['normal_mode']}**",
        f"- conservative_mode: **{verdicts['conservative_mode']}**",
        f"- costos por regimen: **{verdicts['costs_by_regime']}**",
        f"- intrabar policy: **{verdicts['intrabar_policy']}**",
        f"- trade logging: **{verdicts['trade_logging']}**",
        f"- tests: **{verdicts['tests']}**",
        f"- estado general: **{verdicts['overall']}**",
        "",
        "## Comparacion operativa",
        f"- normal_mode: trades={normal['total_trades']}, PF={normal['profit_factor']:.4f}, expectancy={normal['expectancy_r']:.4f}R, DD={normal['max_drawdown_pct']:.2f}%",
        f"- conservative_mode: trades={conservative['total_trades']}, PF={conservative['profit_factor']:.4f}, expectancy={conservative['expectancy_r']:.4f}R, DD={conservative['max_drawdown_pct']:.2f}%",
        "",
        "## Criterio operativo",
        "1. El sistema queda mejor preparado para research serio porque ahora separa baseline, costo y politica intrabar.",
        "2. `conservative_mode` es usable como filtro de robustez, no como reemplazo del baseline.",
        "3. A partir de ahora conviene correr siempre ambos modos.",
        "4. Si una estrategia solo funciona en `normal_mode` y colapsa en `conservative_mode`, no debe pasar a la siguiente etapa.",
        "",
        "## Observaciones tecnicas",
        f"- tests PASS: {int(test_status['status'].eq('PASS').sum())}/{len(test_status)}",
        f"- modulo noticias sigue operativo en OFF: {normal_payload['news_module']['enabled']}",
        "- el limite BID-only sigue vigente; Nivel 2 mejora conservadurismo, no elimina esa limitacion",
    ]
    return "\n".join(lines)


def write_files(
    output_root: Path,
    *,
    report_text: str,
    verdicts: dict[str, str],
    mode_comparison: pd.DataFrame,
    test_status: pd.DataFrame,
    test_output: str,
    trade_log_sample: pd.DataFrame,
    normal_payload: dict[str, Any],
    conservative_payload: dict[str, Any],
) -> None:
    (output_root / "audit_level2_report.md").write_text(report_text, encoding="utf-8")
    (output_root / "module_verdicts.json").write_text(json.dumps(verdicts, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_root / "mode_comparison.csv").write_text(mode_comparison.to_csv(index=False), encoding="utf-8")
    (output_root / "test_status.csv").write_text(test_status.to_csv(index=False), encoding="utf-8")
    (output_root / "test_suite_output.txt").write_text(test_output, encoding="utf-8")
    (output_root / "trade_log_sample.csv").write_text(trade_log_sample.to_csv(index=False), encoding="utf-8")
    (output_root / "normal_mode_summary.json").write_text(json.dumps(normal_payload["summary"], indent=2, ensure_ascii=False), encoding="utf-8")
    (output_root / "conservative_mode_summary.json").write_text(json.dumps(conservative_payload["summary"], indent=2, ensure_ascii=False), encoding="utf-8")
    (output_root / "files_modified.txt").write_text(
        "\n".join(
            [
                "research_lab/config.py",
                "research_lab/engine.py",
                "research_lab/report.py",
                "research_lab/main.py",
                "research_lab/light_runner.py",
                "research_lab/BASELINE_LEVEL1.md",
                "research_lab/README.md",
                "research_lab/tests/test_level2_execution.py",
                "research_lab/audit_level2.py",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    shutil.copy2(Path("research_lab/BASELINE_LEVEL1.md"), output_root / "BASELINE_LEVEL1.md")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Auditoria Nivel 2 del motor de ejecucion.")
    parser.add_argument("--pair", default=DEFAULT_PAIR)
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--data-dirs", nargs="+", default=[str(path) for path in DEFAULT_DATA_DIRS])
    parser.add_argument("--results-dir", default=str(AUDIT_RESULTS_DIR))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_root = build_output_root(Path(args.results_dir))
    raw_frame = load_price_data(args.pair.upper().strip(), [Path(item) for item in args.data_dirs], args.start, args.end)
    frame = prepare_common_frame(raw_frame)

    normal_payload = run_strategy_smoke(frame, "normal_mode")
    conservative_payload = run_strategy_smoke(frame, "conservative_mode")
    mode_comparison = build_mode_comparison(normal_payload, conservative_payload)
    trade_log_sample = synthetic_trade_log_sample()
    test_status, test_output = run_test_suite()
    verdicts = build_module_verdicts(test_status, trade_log_sample)
    report_text = build_report(
        mode_comparison=mode_comparison,
        verdicts=verdicts,
        test_status=test_status,
        normal_payload=normal_payload,
        conservative_payload=conservative_payload,
    )
    write_files(
        output_root,
        report_text=report_text,
        verdicts=verdicts,
        mode_comparison=mode_comparison,
        test_status=test_status,
        test_output=test_output,
        trade_log_sample=trade_log_sample,
        normal_payload=normal_payload,
        conservative_payload=conservative_payload,
    )
    archive = sync_visible_chatgpt(output_root)
    print(f"[level2] Auditoria lista en {output_root}")
    print(f"[level2] ZIP visible listo en {archive}")


if __name__ == "__main__":
    main()

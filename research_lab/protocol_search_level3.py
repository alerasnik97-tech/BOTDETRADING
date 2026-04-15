from __future__ import annotations

import argparse
import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from research_lab.config import EngineConfig, INITIAL_CAPITAL, NY_TZ, with_execution_mode
from research_lab.data_loader import load_backtest_data_bundle, rsi, slice_high_precision_package_to_frame, ema
from research_lab.engine import run_backtest
from research_lab.report import summarize_result, sync_visible_chatgpt
from research_lab.strategies import (
    bollinger_stochastic_core as bollinger_stochastic_core_module,
)
from research_lab.strategies import (
    macd_3_10_16_pullback_core as macd_3_10_16_pullback_core_module,
)
from research_lab.strategies import (
    macd_histogram_pullback_continuation as macd_histogram_pullback_continuation_module,
)
from research_lab.strategies import ny_fix_momentum_core as ny_fix_momentum_core_module
from research_lab.strategies import (
    rsi_extreme_mean_reversion_core as rsi_extreme_mean_reversion_core_module,
)


RESULTS_DIR = Path("results") / "edge_search_level3"
MODES = ("normal_mode", "conservative_mode", "high_precision_mode")
PERIODS = {
    "development_2020_2023": ("2020-01-01", "2023-12-31"),
    "validation_2024": ("2024-01-01", "2024-12-31"),
    "holdout_2025": ("2025-01-01", "2025-12-31"),
}
MIN_TRADES = {
    "development_2020_2023": 60,
    "validation_2024": 12,
    "holdout_2025": 12,
}
COLLAPSE_PF_RATIO = 0.80
COLLAPSE_EXPECTANCY_RATIO = 0.70
MIN_PF = 1.00
MIN_EXPECTANCY = 0.0


@dataclass(frozen=True)
class CandidateSpec:
    family_name: str
    family_label: str
    module: Any
    phase1_combinations: int
    max_trades_per_day: int


CANDIDATES: tuple[CandidateSpec, ...] = (
    CandidateSpec("bollinger_stochastic", "Bollinger + Stochastic", bollinger_stochastic_core_module, 8, 2),
    CandidateSpec("ny_fix_momentum", "NY Fix Momentum", ny_fix_momentum_core_module, 6, 1),
    CandidateSpec("rsi_extremo_retorno_media", "RSI Extremo + Retorno a Media", rsi_extreme_mean_reversion_core_module, 6, 2),
    CandidateSpec("macd_3_10_16_pullback", "MACD 3-10-16 Pullback", macd_3_10_16_pullback_core_module, 6, 2),
    CandidateSpec("macd_histogram_pullback", "MACD + RSI + EMA50", macd_histogram_pullback_continuation_module, 6, 2),
)


def build_output_root(results_dir: Path) -> Path:
    timestamp = pd.Timestamp.now(tz=NY_TZ).strftime("%Y%m%d_%H%M%S")
    path = results_dir / f"{timestamp}_protocol_search_level3"
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_engine_config(execution_mode: str, max_trades_per_day: int) -> EngineConfig:
    return with_execution_mode(
        EngineConfig(
            pair="EURUSD",
            risk_pct=0.5,
            assumed_spread_pips=1.2,
            max_spread_pips=3.0,
            slippage_pips=0.2,
            commission_per_lot_roundturn_usd=7.0,
            shock_candle_atr_max=2.2,
            max_trades_per_day=max_trades_per_day,
            execution_mode=execution_mode,
        ),
        execution_mode,
    )


def stochastic_14_3_3(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    rolling_low = frame["low"].rolling(14, min_periods=14).min()
    rolling_high = frame["high"].rolling(14, min_periods=14).max()
    raw_k = ((frame["close"] - rolling_low) / (rolling_high - rolling_low).replace(0.0, np.nan)) * 100.0
    smooth_k = raw_k.rolling(3, min_periods=3).mean().clip(lower=0.0, upper=100.0).fillna(50.0)
    d_line = smooth_k.rolling(3, min_periods=3).mean().fillna(50.0)
    return smooth_k, d_line


def macd_histogram(series: pd.Series, fast: int, slow: int, signal: int) -> pd.Series:
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    return macd_line - signal_line


def enrich_protocol_frame(base_frame: pd.DataFrame) -> pd.DataFrame:
    frame = base_frame.copy()
    stoch_k, stoch_d = stochastic_14_3_3(frame)
    frame["stoch_k_14_3_3"] = stoch_k
    frame["stoch_d_14_3_3"] = stoch_d
    frame["macd_hist_3_10_16"] = macd_histogram(frame["close"], 3, 10, 16)
    for period in (7,):
        frame[f"rsi{period}"] = rsi(frame["close"], period)
    for macd_fast in (8, 12):
        for macd_slow in (17, 26):
            for macd_signal in (5, 9):
                frame[f"macd_hist_{macd_fast}_{macd_slow}_{macd_signal}"] = macd_histogram(
                    frame["close"],
                    macd_fast,
                    macd_slow,
                    macd_signal,
                )
    return frame


def load_mode_context(mode: str, pair: str, data_dirs: list[Path], start: str, end: str) -> dict[str, Any]:
    bundle = load_backtest_data_bundle(pair, data_dirs, start, end, mode)
    return {
        "frame": enrich_protocol_frame(bundle.frame),
        "precision_package": bundle.precision_package,
        "data_source_used": bundle.data_source_used,
    }


def normalize_params(params: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(params)
    normalized["session_name"] = "light_fixed"
    normalized.pop("use_h1_context", None)
    normalized.pop("use_h1_filter", None)
    return normalized


def unique_combos(combos: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for combo in combos:
        normalized = normalize_params(combo)
        signature = json.dumps(normalized, sort_keys=True, default=str)
        if signature in seen:
            continue
        seen.add(signature)
        deduped.append(normalized)
    return deduped


def phase1_combos(module: Any, max_combinations: int) -> list[dict[str, Any]]:
    signature = inspect.signature(module.parameter_grid)
    kwargs: dict[str, Any] = {}
    if "max_combinations" in signature.parameters:
        kwargs["max_combinations"] = max_combinations
    if "seed" in signature.parameters:
        kwargs["seed"] = 42
    return unique_combos(module.parameter_grid(**kwargs))


def period_slice(frame: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    start_ts = pd.Timestamp(start, tz=NY_TZ)
    end_ts = pd.Timestamp(end, tz=NY_TZ) + pd.Timedelta(days=1) - pd.Timedelta(minutes=15)
    return frame.loc[(frame.index >= start_ts) & (frame.index <= end_ts)].copy()


def share_best_year(yearly_stats: pd.DataFrame) -> float:
    if yearly_stats.empty:
        return 0.0
    yearly = yearly_stats.groupby("year")["total_pnl_r"].sum()
    positive_total = float(yearly[yearly > 0].sum())
    if positive_total <= 0:
        return 0.0
    return float(yearly.max() / positive_total)


def exit_reason_distribution(trades_export: pd.DataFrame) -> dict[str, int]:
    if trades_export.empty or "exit_reason" not in trades_export.columns:
        return {}
    counts = trades_export["exit_reason"].value_counts()
    return {str(key): int(value) for key, value in counts.items()}


def mode_score(summary: dict[str, Any]) -> float:
    trades_gap = abs(float(summary["avg_trades_per_month"]) - 15.0)
    return (
        float(summary["profit_factor"]) * 250.0
        + float(summary["expectancy_r"]) * 1400.0
        - float(summary["max_drawdown_pct"]) * 3.0
        - float(summary["negative_years"]) * 50.0
        - float(summary["negative_months"]) * 2.0
        - trades_gap * 4.0
    )


def collapse_against_normal(normal_summary: dict[str, Any], other_summary: dict[str, Any]) -> bool:
    if float(normal_summary["profit_factor"]) <= 0.0 or float(normal_summary["expectancy_r"]) <= 0.0:
        return True
    return (
        float(other_summary["profit_factor"]) < float(normal_summary["profit_factor"]) * COLLAPSE_PF_RATIO
        or float(other_summary["expectancy_r"]) < float(normal_summary["expectancy_r"]) * COLLAPSE_EXPECTANCY_RATIO
    )


def evaluate_period_mode(
    spec: CandidateSpec,
    params: dict[str, Any],
    mode: str,
    mode_context: dict[str, Any],
    period_name: str,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    start, end = PERIODS[period_name]
    frame = period_slice(mode_context["frame"], start, end)
    precision_package = slice_high_precision_package_to_frame(mode_context["precision_package"], frame.index)
    result = run_backtest(
        spec.module,
        frame,
        params,
        build_engine_config(mode, spec.max_trades_per_day),
        np.zeros(len(frame), dtype=bool),
        False,
        precision_package=precision_package,
        data_source_used=mode_context["data_source_used"],
    )
    return summarize_result(
        spec.module.NAME,
        result.trades,
        result.equity_curve,
        params,
        False,
        INITIAL_CAPITAL,
        None,
        timeframe="M15",
        schedule_used={"entry_start": "11:00", "entry_end": "19:00", "force_close": "19:00"},
        break_even_setting=params.get("break_even_at_r"),
    )


def development_reasons(mode_payloads: dict[str, tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]]) -> list[str]:
    reasons: list[str] = []
    for mode, payload in mode_payloads.items():
        summary = payload[0]
        yearly_stats = payload[3]
        if int(summary["total_trades"]) < MIN_TRADES["development_2020_2023"]:
            reasons.append(f"{mode}:sample_too_small")
        if float(summary["profit_factor"]) <= MIN_PF:
            reasons.append(f"{mode}:pf<=1")
        if float(summary["expectancy_r"]) <= MIN_EXPECTANCY:
            reasons.append(f"{mode}:expectancy<=0")
        if share_best_year(yearly_stats) > 0.60:
            reasons.append(f"{mode}:year_dependency>0.60")
    normal_summary = mode_payloads["normal_mode"][0]
    if collapse_against_normal(normal_summary, mode_payloads["conservative_mode"][0]):
        reasons.append("conservative_collapse")
    if collapse_against_normal(normal_summary, mode_payloads["high_precision_mode"][0]):
        reasons.append("high_precision_collapse")
    return reasons


def period_reasons(
    period_name: str,
    mode_payloads: dict[str, tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]],
) -> list[str]:
    reasons: list[str] = []
    for mode, payload in mode_payloads.items():
        summary = payload[0]
        if int(summary["total_trades"]) < MIN_TRADES[period_name]:
            reasons.append(f"{period_name}:{mode}:sample_too_small")
        if float(summary["profit_factor"]) <= MIN_PF:
            reasons.append(f"{period_name}:{mode}:pf<=1")
        if float(summary["expectancy_r"]) <= MIN_EXPECTANCY:
            reasons.append(f"{period_name}:{mode}:expectancy<=0")
    normal_summary = mode_payloads["normal_mode"][0]
    if collapse_against_normal(normal_summary, mode_payloads["conservative_mode"][0]):
        reasons.append(f"{period_name}:conservative_collapse")
    if collapse_against_normal(normal_summary, mode_payloads["high_precision_mode"][0]):
        reasons.append(f"{period_name}:high_precision_collapse")
    return reasons


def weighted_mode_score(mode_payloads: dict[str, tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]]) -> float:
    weights = {
        "normal_mode": 0.20,
        "conservative_mode": 0.35,
        "high_precision_mode": 0.45,
    }
    return float(sum(mode_score(mode_payloads[mode][0]) * weights[mode] for mode in MODES))


def evaluate_strategy_candidate(
    spec: CandidateSpec,
    params: dict[str, Any],
    mode_contexts: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    development_payloads = {
        mode: evaluate_period_mode(spec, params, mode, mode_contexts[mode], "development_2020_2023")
        for mode in MODES
    }
    dev_reasons = development_reasons(development_payloads)
    dev_score = weighted_mode_score(development_payloads)
    return {
        "params": params,
        "development_payloads": development_payloads,
        "development_reasons": dev_reasons,
        "development_score": dev_score,
    }


def selected_combo_row(
    spec: CandidateSpec,
    params: dict[str, Any],
    period_name: str,
    mode: str,
    payload: tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> dict[str, Any]:
    summary, trades_export, _monthly, yearly_stats, _equity = payload
    return {
        "family_name": spec.family_name,
        "family_label": spec.family_label,
        "strategy_module": spec.module.NAME,
        "period": period_name,
        "execution_mode": mode,
        "total_trades": summary["total_trades"],
        "avg_trades_per_month": summary["avg_trades_per_month"],
        "win_rate": summary["win_rate"],
        "profit_factor": summary["profit_factor"],
        "expectancy_r": summary["expectancy_r"],
        "total_return_pct": summary["total_return_pct"],
        "max_drawdown_pct": summary["max_drawdown_pct"],
        "negative_months": summary["negative_months"],
        "negative_years": summary["negative_years"],
        "share_best_year": share_best_year(yearly_stats),
        "data_source_used": trades_export["data_source_used"].iloc[0] if not trades_export.empty else "",
        "execution_mode_used": trades_export["execution_mode_used"].iloc[0] if not trades_export.empty else mode,
        "price_source_used": trades_export["price_source_used"].iloc[0] if not trades_export.empty else "",
        "exit_distribution": json.dumps(exit_reason_distribution(trades_export), ensure_ascii=False),
        "parameter_set_used": json.dumps(params, ensure_ascii=False),
    }


def export_selected_strategy_bundle(
    strategy_dir: Path,
    spec: CandidateSpec,
    params: dict[str, Any],
    payloads_by_period: dict[str, dict[str, tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]]],
) -> None:
    strategy_dir.mkdir(parents=True, exist_ok=True)
    (strategy_dir / "selected_params.json").write_text(json.dumps(params, indent=2, ensure_ascii=False), encoding="utf-8")
    period_mode_rows: list[dict[str, Any]] = []
    for period_name, mode_payloads in payloads_by_period.items():
        period_dir = strategy_dir / period_name
        period_dir.mkdir(parents=True, exist_ok=True)
        for mode, payload in mode_payloads.items():
            summary, trades_export, monthly_stats, yearly_stats, equity_export = payload
            period_mode_rows.append(selected_combo_row(spec, params, period_name, mode, payload))
            trades_export.to_csv(period_dir / f"{mode}_trades.csv", index=False)
            monthly_stats.to_csv(period_dir / f"{mode}_monthly_stats.csv", index=False)
            yearly_stats.to_csv(period_dir / f"{mode}_yearly_stats.csv", index=False)
            equity_export.to_csv(period_dir / f"{mode}_equity_curve.csv", index=False)
            (period_dir / f"{mode}_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(period_mode_rows).to_csv(strategy_dir / "period_mode_summary.csv", index=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Busqueda de edge con protocolo profesional development/validation/holdout en normal, conservative y high_precision.")
    parser.add_argument("--pair", default="EURUSD")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--data-dirs", nargs="+", default=["data_free_2020/prepared", "data_candidates_2022_2025/prepared"])
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_root = build_output_root(Path(args.results_dir))
    data_dirs = [Path(item) for item in args.data_dirs]

    protocol = {
        "pair": args.pair.upper().strip(),
        "modes": list(MODES),
        "periods": PERIODS,
        "min_trades": MIN_TRADES,
        "min_profit_factor": MIN_PF,
        "min_expectancy_r": MIN_EXPECTANCY,
        "collapse_pf_ratio": COLLAPSE_PF_RATIO,
        "collapse_expectancy_ratio": COLLAPSE_EXPECTANCY_RATIO,
    }
    (output_root / "protocol.json").write_text(json.dumps(protocol, indent=2, ensure_ascii=False), encoding="utf-8")

    mode_contexts = {
        mode: load_mode_context(mode, args.pair.upper().strip(), data_dirs, args.start, args.end)
        for mode in MODES
    }

    development_variant_rows: list[dict[str, Any]] = []
    strategy_rows: list[dict[str, Any]] = []

    for spec in CANDIDATES:
        combo_payloads = [
            evaluate_strategy_candidate(spec, params, mode_contexts)
            for params in phase1_combos(spec.module, spec.phase1_combinations)
        ]

        combo_rows: list[dict[str, Any]] = []
        for item in combo_payloads:
            dev_normal = item["development_payloads"]["normal_mode"][0]
            dev_conservative = item["development_payloads"]["conservative_mode"][0]
            dev_high_precision = item["development_payloads"]["high_precision_mode"][0]
            combo_rows.append(
                {
                    "family_name": spec.family_name,
                    "family_label": spec.family_label,
                    "strategy_module": spec.module.NAME,
                    "discarded": bool(item["development_reasons"]),
                    "discard_reasons": "|".join(item["development_reasons"]),
                    "development_score": item["development_score"],
                    "parameter_set_used": json.dumps(item["params"], ensure_ascii=False),
                    "dev_normal_trades": dev_normal["total_trades"],
                    "dev_normal_pf": dev_normal["profit_factor"],
                    "dev_normal_expectancy_r": dev_normal["expectancy_r"],
                    "dev_conservative_trades": dev_conservative["total_trades"],
                    "dev_conservative_pf": dev_conservative["profit_factor"],
                    "dev_conservative_expectancy_r": dev_conservative["expectancy_r"],
                    "dev_high_precision_trades": dev_high_precision["total_trades"],
                    "dev_high_precision_pf": dev_high_precision["profit_factor"],
                    "dev_high_precision_expectancy_r": dev_high_precision["expectancy_r"],
                }
            )

        combo_df = pd.DataFrame(combo_rows).sort_values(["discarded", "development_score"], ascending=[True, False]).reset_index(drop=True)
        combo_df.to_csv(output_root / f"{spec.family_name}_development_variants.csv", index=False)
        development_variant_rows.extend(combo_rows)

        surviving_items = [item for item in combo_payloads if not item["development_reasons"]]
        chosen_item = None
        strategy_discard_reasons: list[str] = []
        if surviving_items:
            chosen_item = max(surviving_items, key=lambda item: item["development_score"])
        elif combo_payloads:
            chosen_item = max(combo_payloads, key=lambda item: item["development_score"])
            strategy_discard_reasons.append("no_development_survivor")

        if chosen_item is None:
            continue

        selected_params = chosen_item["params"]
        payloads_by_period: dict[str, dict[str, tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]]] = {
            "development_2020_2023": chosen_item["development_payloads"]
        }
        for period_name in ("validation_2024", "holdout_2025"):
            payloads_by_period[period_name] = {
                mode: evaluate_period_mode(spec, selected_params, mode, mode_contexts[mode], period_name)
                for mode in MODES
            }
            strategy_discard_reasons.extend(period_reasons(period_name, payloads_by_period[period_name]))

        if chosen_item["development_reasons"]:
            strategy_discard_reasons.extend(chosen_item["development_reasons"])

        final_discarded = bool(strategy_discard_reasons)
        final_score = (
            weighted_mode_score(payloads_by_period["development_2020_2023"]) * 0.25
            + weighted_mode_score(payloads_by_period["validation_2024"]) * 0.35
            + weighted_mode_score(payloads_by_period["holdout_2025"]) * 0.40
        )
        if final_discarded:
            final_score -= 1000.0

        export_selected_strategy_bundle(output_root / spec.family_name, spec, selected_params, payloads_by_period)

        strategy_row = {
            "family_name": spec.family_name,
            "family_label": spec.family_label,
            "strategy_module": spec.module.NAME,
            "discarded": final_discarded,
            "discard_reasons": "|".join(dict.fromkeys(strategy_discard_reasons)),
            "selected_score": final_score,
            "parameter_set_used": json.dumps(selected_params, ensure_ascii=False),
        }
        for period_name, mode_payloads in payloads_by_period.items():
            for mode, payload in mode_payloads.items():
                summary = payload[0]
                prefix = f"{period_name}_{mode}"
                strategy_row[f"{prefix}_trades"] = summary["total_trades"]
                strategy_row[f"{prefix}_trades_per_month"] = summary["avg_trades_per_month"]
                strategy_row[f"{prefix}_win_rate"] = summary["win_rate"]
                strategy_row[f"{prefix}_profit_factor"] = summary["profit_factor"]
                strategy_row[f"{prefix}_expectancy_r"] = summary["expectancy_r"]
                strategy_row[f"{prefix}_total_return_pct"] = summary["total_return_pct"]
                strategy_row[f"{prefix}_max_drawdown_pct"] = summary["max_drawdown_pct"]
        strategy_rows.append(strategy_row)

    ranking = pd.DataFrame(strategy_rows).sort_values(["discarded", "selected_score"], ascending=[True, False]).reset_index(drop=True)
    dev_variants = pd.DataFrame(development_variant_rows).sort_values(["discarded", "development_score"], ascending=[True, False]).reset_index(drop=True)
    discarded = ranking.loc[ranking["discarded"]].copy() if not ranking.empty else pd.DataFrame()
    survivors = ranking.loc[~ranking["discarded"]].copy() if not ranking.empty else pd.DataFrame()

    ranking.to_csv(output_root / "strategy_ranking.csv", index=False)
    dev_variants.to_csv(output_root / "development_variants.csv", index=False)
    discarded.to_csv(output_root / "discarded_strategies.csv", index=False)
    survivors.to_csv(output_root / "surviving_strategies.csv", index=False)
    comparative = ranking.copy()
    comparative.to_csv(output_root / "strategy_comparative_table.csv", index=False)

    recommendation_lines = ["# Recomendacion final", ""]
    if survivors.empty:
        recommendation_lines.append("Ninguna estrategia sobrevivio al protocolo completo development/validation/holdout bajo normal_mode, conservative_mode y high_precision_mode.")
        recommendation_lines.append("")
        recommendation_lines.append("Conclusion: no conviene refinar todavia. Hace falta una hipotesis de entrada mejor.")
    else:
        winner = survivors.iloc[0]
        recommendation_lines.append(f"Mejor candidata actual: **{winner['family_label']}**")
        recommendation_lines.append(f"- score: {float(winner['selected_score']):.2f}")
        recommendation_lines.append(f"- params: `{winner['parameter_set_used']}`")
    (output_root / "recomendacion_final.md").write_text("\n".join(recommendation_lines), encoding="utf-8")

    mapping = [
        {
            "family_name": spec.family_name,
            "family_label": spec.family_label,
            "strategy_module": spec.module.NAME,
            "file": str(Path(inspect.getfile(spec.module)).resolve()),
            "phase1_combinations": spec.phase1_combinations,
            "max_trades_per_day": spec.max_trades_per_day,
        }
        for spec in CANDIDATES
    ]
    (output_root / "mapping_used.json").write_text(json.dumps(mapping, indent=2, ensure_ascii=False), encoding="utf-8")

    archive = sync_visible_chatgpt(output_root)
    print(f"[protocol] resultados en {output_root}")
    print(f"[protocol] zip visible en {archive}")
    if ranking.empty:
        print("[protocol] sin estrategias evaluables")
    else:
        print(ranking[["family_label", "discarded", "selected_score"]].to_string(index=False))


if __name__ == "__main__":
    main()

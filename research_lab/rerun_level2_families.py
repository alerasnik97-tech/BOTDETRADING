from __future__ import annotations

import argparse
import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from research_lab.config import DEFAULT_DATA_DIRS, DEFAULT_PAIR, NewsConfig
from research_lab.data_loader import load_price_data, prepare_common_frame
from research_lab.engine import run_backtest
from research_lab.news_filter import load_news_events
from research_lab.report import summarize_result, sync_visible_chatgpt
from research_lab.strategies import (
    bollinger_mean_reversion_adx_low,
    compression_breakout,
    donchian_breakout,
    ema_trend_pullback,
    keltner_volatility_expansion_simple,
    previous_day_level_m15,
    session_range_breakout,
    supertrend_ema_filter,
)
from research_lab.validation import run_walkforward
from research_lab.audit_level2 import build_engine_config, costs_payload


RESULTS_DIR = Path("results") / "research_lab_level2_rerun"
PHASE1_MAX_COMBINATIONS = 8
PHASE2_MAX_COMBINATIONS = 12
MIN_TRADES_PER_MONTH = 10.0
TARGET_TRADES_PER_MONTH = 20.0
COLLAPSE_PF_RATIO = 0.85
COLLAPSE_EXPECTANCY_RATIO = 0.70


@dataclass(frozen=True)
class FamilySpec:
    family_name: str
    family_label: str
    module: Any
    phase1_combinations: int = PHASE1_MAX_COMBINATIONS
    phase2_combinations: int = PHASE2_MAX_COMBINATIONS


FAMILY_SPECS: tuple[FamilySpec, ...] = (
    FamilySpec("session_range_breakout", "Session Range Breakout", session_range_breakout, 4, 4),
    FamilySpec("donchian_breakout", "Donchian Breakout", donchian_breakout, 4, 4),
    FamilySpec("compression_breakout", "Compression Breakout", compression_breakout, 8, 8),
    FamilySpec("previous_day_level", "Previous Day Level", previous_day_level_m15, 4, 4),
    FamilySpec("bollinger_mean_reversion", "Bollinger Mean Reversion", bollinger_mean_reversion_adx_low, 8, 12),
    FamilySpec("ema_trend_pullback", "EMA Trend Pullback", ema_trend_pullback, 8, 12),
    FamilySpec("supertrend_ema", "Supertrend + EMA", supertrend_ema_filter, 8, 12),
    FamilySpec("keltner_volatility_expansion", "Keltner / Volatility Expansion", keltner_volatility_expansion_simple, 8, 12),
)


def build_output_root(results_dir: Path) -> Path:
    timestamp = pd.Timestamp.now(tz="America/New_York").strftime("%Y%m%d_%H%M%S")
    path = results_dir / f"{timestamp}_level2_rerun"
    path.mkdir(parents=True, exist_ok=True)
    return path


def normalize_params(params: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(params)
    normalized["session_name"] = "light_fixed"
    if "use_h1_context" in normalized:
        normalized["use_h1_context"] = False
    if "use_h1_filter" in normalized:
        normalized["use_h1_filter"] = False
    if "break_even_enabled" in normalized and "break_even_at_r" not in normalized:
        normalized["break_even_at_r"] = 1.0 if bool(normalized["break_even_enabled"]) else None
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


def phase1_combos(module: Any, max_combinations: int, seed: int) -> list[dict[str, Any]]:
    if not hasattr(module, "parameter_grid"):
        if hasattr(module, "default_params"):
            return [normalize_params(module.default_params())]
        raise AttributeError(f"{module.NAME} no tiene parameter_grid ni default_params")
    signature = inspect.signature(module.parameter_grid)
    kwargs: dict[str, Any] = {}
    if "max_combinations" in signature.parameters:
        kwargs["max_combinations"] = max_combinations
    if "seed" in signature.parameters:
        kwargs["seed"] = seed
    combos = module.parameter_grid(**kwargs)
    return unique_combos(combos)


def neighbor_values(values: list[Any], center: Any) -> list[Any]:
    if center not in values:
        return [center]
    if isinstance(center, (str, bool)) or center is None:
        return [center]
    idx = values.index(center)
    start = max(0, idx - 1)
    end = min(len(values), idx + 2)
    return values[start:end]


def phase2_combos(module: Any, center_params: dict[str, Any], max_combinations: int, seed: int) -> list[dict[str, Any]]:
    if hasattr(module, "parameter_space"):
        param_space = module.parameter_space()
        local_space: dict[str, list[Any]] = {}
        for key, values in param_space.items():
            if key == "session_name":
                local_space[key] = ["light_fixed"]
            elif key == "use_h1_context":
                local_space[key] = [False]
            elif key == "break_even_at_r":
                local_space[key] = [center_params.get("break_even_at_r")]
            else:
                local_space[key] = neighbor_values(list(values), center_params.get(key))
        keys = list(local_space.keys())
        combos = [dict(zip(keys, values)) for values in __import__("itertools").product(*(local_space[key] for key in keys))]
        return unique_combos(combos)[:max_combinations]
    return unique_combos([center_params])


def run_mode(
    module: Any,
    frame: pd.DataFrame,
    params: dict[str, Any],
    execution_mode: str,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    engine_config = build_engine_config(execution_mode)
    news_config = NewsConfig(enabled=False)
    news_result = load_news_events(engine_config.pair, news_config)
    news_block = np.zeros(len(frame), dtype=bool)
    result = run_backtest(module, frame, params, engine_config, news_block, news_result.enabled)
    return summarize_result(
        module.NAME,
        result.trades,
        result.equity_curve,
        params,
        news_result.enabled,
        100_000.0,
        None,
        costs_payload(engine_config),
        "M15",
        {"entry_start": "11:00", "entry_end": "19:00", "force_close": "19:00"},
        params.get("break_even_at_r"),
    )


def clear_collapse(normal_summary: dict[str, Any], conservative_summary: dict[str, Any]) -> bool:
    if normal_summary["profit_factor"] <= 0 or normal_summary["expectancy_r"] <= 0:
        return True
    return (
        conservative_summary["profit_factor"] < normal_summary["profit_factor"] * COLLAPSE_PF_RATIO
        or conservative_summary["expectancy_r"] < normal_summary["expectancy_r"] * COLLAPSE_EXPECTANCY_RATIO
    )


def discard_reasons(normal_summary: dict[str, Any], conservative_summary: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if normal_summary["profit_factor"] <= 1.0 or conservative_summary["profit_factor"] <= 1.0:
        reasons.append("pf<=1")
    if normal_summary["expectancy_r"] <= 0.0 or conservative_summary["expectancy_r"] <= 0.0:
        reasons.append("expectancy<=0")
    if normal_summary["avg_trades_per_month"] < MIN_TRADES_PER_MONTH or conservative_summary["avg_trades_per_month"] < MIN_TRADES_PER_MONTH:
        reasons.append("trades_per_month<10")
    if normal_summary["negative_years"] > 3 or conservative_summary["negative_years"] > 3:
        reasons.append("negative_years>3")
    if clear_collapse(normal_summary, conservative_summary):
        reasons.append("conservative_collapse")
    return reasons


def score_variant(normal_summary: dict[str, Any], conservative_summary: dict[str, Any]) -> float:
    avg_drawdown = (float(normal_summary["max_drawdown_pct"]) + float(conservative_summary["max_drawdown_pct"])) / 2.0
    trades_gap = abs(float(conservative_summary["avg_trades_per_month"]) - TARGET_TRADES_PER_MONTH)
    score = (
        float(conservative_summary["profit_factor"]) * 450.0
        + float(conservative_summary["expectancy_r"]) * 1800.0
        + float(normal_summary["profit_factor"]) * 200.0
        + float(normal_summary["expectancy_r"]) * 900.0
        - avg_drawdown * 4.0
        - float(conservative_summary["negative_months"]) * 2.0
        - float(conservative_summary["negative_years"]) * 35.0
        - trades_gap * 8.0
    )
    if clear_collapse(normal_summary, conservative_summary):
        score -= 500.0
    return score


def strategy_row(
    spec: FamilySpec,
    params: dict[str, Any],
    normal_summary: dict[str, Any],
    conservative_summary: dict[str, Any],
    score: float,
    discarded: bool,
    reasons: list[str],
    phase: str,
) -> dict[str, Any]:
    return {
        "family_name": spec.family_name,
        "family_label": spec.family_label,
        "strategy_module": spec.module.NAME,
        "phase": phase,
        "discarded": discarded,
        "discard_reasons": "|".join(reasons),
        "selected_score": score,
        "parameter_set_used": json.dumps(params, ensure_ascii=False),
        "normal_total_trades": normal_summary["total_trades"],
        "normal_trades_per_month": normal_summary["avg_trades_per_month"],
        "normal_win_rate": normal_summary["win_rate"],
        "normal_profit_factor": normal_summary["profit_factor"],
        "normal_expectancy_r": normal_summary["expectancy_r"],
        "normal_total_return_pct": normal_summary["total_return_pct"],
        "normal_max_drawdown_pct": normal_summary["max_drawdown_pct"],
        "normal_negative_years": normal_summary["negative_years"],
        "conservative_total_trades": conservative_summary["total_trades"],
        "conservative_trades_per_month": conservative_summary["avg_trades_per_month"],
        "conservative_win_rate": conservative_summary["win_rate"],
        "conservative_profit_factor": conservative_summary["profit_factor"],
        "conservative_expectancy_r": conservative_summary["expectancy_r"],
        "conservative_total_return_pct": conservative_summary["total_return_pct"],
        "conservative_max_drawdown_pct": conservative_summary["max_drawdown_pct"],
        "conservative_negative_years": conservative_summary["negative_years"],
    }


def export_family_phase1(output_root: Path, spec: FamilySpec, family_rows: pd.DataFrame) -> None:
    family_dir = output_root / spec.family_name
    family_dir.mkdir(parents=True, exist_ok=True)
    family_rows.to_csv(family_dir / "phase1_variants.csv", index=False)


def export_family_final(
    output_root: Path,
    spec: FamilySpec,
    params: dict[str, Any],
    normal_payload: tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame],
    conservative_payload: tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    family_dir = output_root / spec.family_name
    family_dir.mkdir(parents=True, exist_ok=True)
    normal_summary, normal_trades, normal_monthly, normal_yearly, normal_equity = normal_payload
    conservative_summary, conservative_trades, conservative_monthly, conservative_yearly, conservative_equity = conservative_payload

    (family_dir / "selected_params.json").write_text(json.dumps(params, indent=2, ensure_ascii=False), encoding="utf-8")
    (family_dir / "normal_mode_summary.json").write_text(json.dumps(normal_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (family_dir / "conservative_mode_summary.json").write_text(json.dumps(conservative_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    normal_trades.to_csv(family_dir / "normal_mode_trades.csv", index=False)
    conservative_trades.to_csv(family_dir / "conservative_mode_trades.csv", index=False)
    normal_monthly.to_csv(family_dir / "normal_mode_monthly_stats.csv", index=False)
    conservative_monthly.to_csv(family_dir / "conservative_mode_monthly_stats.csv", index=False)
    normal_yearly.to_csv(family_dir / "normal_mode_yearly_stats.csv", index=False)
    conservative_yearly.to_csv(family_dir / "conservative_mode_yearly_stats.csv", index=False)
    normal_equity.to_csv(family_dir / "normal_mode_equity_curve.csv", index=False)
    conservative_equity.to_csv(family_dir / "conservative_mode_equity_curve.csv", index=False)


def build_recommendation(ranking: pd.DataFrame, survivors: pd.DataFrame, discarded: pd.DataFrame) -> str:
    lines = ["# Recomendacion final", ""]
    lines.append(f"- familias evaluadas: {ranking['family_name'].nunique() if not ranking.empty else 0}")
    lines.append(f"- supervivientes: {len(survivors)}")
    lines.append(f"- descartadas: {len(discarded)}")
    lines.append("")
    if survivors.empty:
        lines.append("Ninguna familia sirve bajo el sistema actual con los filtros de descarte definidos.")
        lines.append("")
        lines.append("Conclusión: no conviene seguir refinando esta tanda. Hay que cambiar la lógica de entrada, no seguir optimizando variantes muertas.")
        return "\n".join(lines)

    winner = survivors.iloc[0]
    lines.append(f"Siguiente ronda recomendada: **{winner['family_label']}**")
    lines.append("")
    lines.append(f"- score: {winner['selected_score']:.2f}")
    lines.append(f"- PF normal/conservative: {winner['normal_profit_factor']:.4f} / {winner['conservative_profit_factor']:.4f}")
    lines.append(f"- expectancy normal/conservative: {winner['normal_expectancy_r']:.4f}R / {winner['conservative_expectancy_r']:.4f}R")
    lines.append(f"- trades/mes normal/conservative: {winner['normal_trades_per_month']:.2f} / {winner['conservative_trades_per_month']:.2f}")
    lines.append("")
    lines.append("Criterio operativo:")
    lines.append("- continuar solo con familias que mantengan edge positivo en conservative_mode")
    lines.append("- descartar cualquier mejora que solo aparezca en normal_mode")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rerun general de familias bajo la infraestructura Nivel 2.")
    parser.add_argument("--pair", default=DEFAULT_PAIR)
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--data-dirs", nargs="+", default=[str(path) for path in DEFAULT_DATA_DIRS])
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_root = build_output_root(Path(args.results_dir))
    raw_frame = load_price_data(args.pair.upper().strip(), [Path(item) for item in args.data_dirs], args.start, args.end)
    frame = prepare_common_frame(raw_frame)

    phase1_rows: list[dict[str, Any]] = []
    family_best_rows: list[dict[str, Any]] = []

    for spec in FAMILY_SPECS:
        combos = phase1_combos(spec.module, spec.phase1_combinations, args.seed)
        family_variant_rows: list[dict[str, Any]] = []
        best_survivor: dict[str, Any] | None = None
        best_any: dict[str, Any] | None = None

        for combo_index, params in enumerate(combos):
            try:
                normal_payload = run_mode(spec.module, frame, params, "normal_mode")
                conservative_payload = run_mode(spec.module, frame, params, "conservative_mode")
            except Exception as exc:
                row = {
                    "family_name": spec.family_name,
                    "family_label": spec.family_label,
                    "strategy_module": spec.module.NAME,
                    "phase": "phase1",
                    "discarded": True,
                    "discard_reasons": f"runtime_error:{type(exc).__name__}",
                    "selected_score": -1_000_000.0,
                    "parameter_set_used": json.dumps(params, ensure_ascii=False),
                }
                family_variant_rows.append(row)
                phase1_rows.append(row)
                continue

            normal_summary = normal_payload[0]
            conservative_summary = conservative_payload[0]
            reasons = discard_reasons(normal_summary, conservative_summary)
            score = score_variant(normal_summary, conservative_summary)
            row = strategy_row(spec, params, normal_summary, conservative_summary, score, bool(reasons), reasons, "phase1")
            family_variant_rows.append(row)
            phase1_rows.append(row)
            if best_any is None or score > best_any["row"]["selected_score"]:
                best_any = {
                    "row": row,
                    "params": params,
                    "normal_payload": normal_payload,
                    "conservative_payload": conservative_payload,
                }
            if not reasons and (best_survivor is None or score > best_survivor["selected_score"]):
                best_survivor = {
                    "selected_score": score,
                    "row": row,
                    "params": params,
                }

        family_rows_df = pd.DataFrame(family_variant_rows).sort_values("selected_score", ascending=False).reset_index(drop=True)
        export_family_phase1(output_root, spec, family_rows_df)

        chosen = best_survivor if best_survivor is not None else best_any
        if chosen is None:
            continue

        chosen_row = dict(chosen["row"])
        chosen_params = chosen["params"]

        if best_survivor is not None:
            phase2_candidates = phase2_combos(spec.module, chosen_params, spec.phase2_combinations, args.seed + 100)
            refined_best: dict[str, Any] | None = None
            refined_rows: list[dict[str, Any]] = []
            for params in phase2_candidates:
                normal_payload = run_mode(spec.module, frame, params, "normal_mode")
                conservative_payload = run_mode(spec.module, frame, params, "conservative_mode")
                normal_summary = normal_payload[0]
                conservative_summary = conservative_payload[0]
                reasons = discard_reasons(normal_summary, conservative_summary)
                score = score_variant(normal_summary, conservative_summary)
                row = strategy_row(spec, params, normal_summary, conservative_summary, score, bool(reasons), reasons, "phase2")
                refined_rows.append(row)
                if not reasons and (refined_best is None or score > refined_best["row"]["selected_score"]):
                    refined_best = {
                        "row": row,
                        "params": params,
                        "normal_payload": normal_payload,
                        "conservative_payload": conservative_payload,
                    }
            refined_df = pd.DataFrame(refined_rows).sort_values("selected_score", ascending=False).reset_index(drop=True)
            refined_df.to_csv((output_root / spec.family_name / "phase2_variants.csv"), index=False)
            if refined_best is not None:
                chosen_row = dict(refined_best["row"])
                chosen_params = refined_best["params"]
                export_family_final(output_root, spec, chosen_params, refined_best["normal_payload"], refined_best["conservative_payload"])
            else:
                export_family_final(output_root, spec, chosen_params, chosen["normal_payload"], chosen["conservative_payload"])
        else:
            export_family_final(output_root, spec, chosen_params, chosen["normal_payload"], chosen["conservative_payload"])

        family_best_rows.append(chosen_row)

    ranking = pd.DataFrame(family_best_rows).sort_values(["discarded", "selected_score"], ascending=[True, False]).reset_index(drop=True)
    comparative_columns = [
        "family_label",
        "strategy_module",
        "normal_profit_factor",
        "conservative_profit_factor",
        "normal_expectancy_r",
        "conservative_expectancy_r",
        "normal_trades_per_month",
        "conservative_trades_per_month",
        "normal_max_drawdown_pct",
        "conservative_max_drawdown_pct",
        "discarded",
        "discard_reasons",
        "parameter_set_used",
    ]
    comparative = ranking[comparative_columns].copy() if not ranking.empty else pd.DataFrame(columns=comparative_columns)
    survivors = ranking.loc[~ranking["discarded"]].copy() if not ranking.empty else pd.DataFrame()
    discarded = ranking.loc[ranking["discarded"]].copy() if not ranking.empty else pd.DataFrame()

    yearly_rows: list[pd.DataFrame] = []
    for spec in FAMILY_SPECS:
        family_dir = output_root / spec.family_name
        for execution_mode in ("normal_mode", "conservative_mode"):
            yearly_path = family_dir / f"{execution_mode}_yearly_stats.csv"
            if yearly_path.exists():
                frame_yearly = pd.read_csv(yearly_path)
                frame_yearly.insert(0, "execution_mode", execution_mode)
                frame_yearly.insert(0, "family_name", spec.family_name)
                frame_yearly.insert(1, "family_label", spec.family_label)
                yearly_rows.append(frame_yearly)
    combined_yearly = pd.concat(yearly_rows, ignore_index=True) if yearly_rows else pd.DataFrame()

    phase1_variants = pd.DataFrame(phase1_rows).sort_values(["discarded", "selected_score"], ascending=[True, False]).reset_index(drop=True)
    phase1_variants.to_csv(output_root / "phase1_all_variants.csv", index=False)
    ranking.to_csv(output_root / "family_ranking.csv", index=False)
    comparative.to_csv(output_root / "family_comparative_table.csv", index=False)
    combined_yearly.to_csv(output_root / "yearly_stats_all_survivors.csv", index=False)
    survivors.to_csv(output_root / "surviving_families.csv", index=False)
    discarded.to_csv(output_root / "discarded_families.csv", index=False)
    (output_root / "recomendacion_final.md").write_text(build_recommendation(ranking, survivors, discarded), encoding="utf-8")
    (output_root / "mapping_used.json").write_text(
        json.dumps(
            [
                {
                    "family_name": spec.family_name,
                    "family_label": spec.family_label,
                    "strategy_module": spec.module.NAME,
                    "file": str(Path(inspect.getfile(spec.module)).resolve()),
                    "phase1_combinations": spec.phase1_combinations,
                    "phase2_combinations": spec.phase2_combinations,
                }
                for spec in FAMILY_SPECS
            ],
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    archive = sync_visible_chatgpt(output_root)
    print(f"[rerun] Ranking listo en {output_root}")
    print(f"[rerun] ZIP visible listo en {archive}")
    if ranking.empty:
        print("[rerun] Sin resultados comparables.")
    else:
        print(ranking[["family_label", "discarded", "selected_score", "normal_profit_factor", "conservative_profit_factor"]].to_string(index=False))


if __name__ == "__main__":
    main()

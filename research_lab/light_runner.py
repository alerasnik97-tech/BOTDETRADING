from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any
import sys

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from research_lab.config import (
    DEFAULT_DATA_DIRS,
    DEFAULT_EXECUTION_MODE,
    DEFAULT_NEWS_FILE,
    DEFAULT_PAIR,
    DEFAULT_SEED,
    EngineConfig,
    INITIAL_CAPITAL,
    NewsConfig,
    STRATEGY_NAMES,
    SUPPORTED_COST_PROFILES,
    SUPPORTED_EXECUTION_MODES,
    SUPPORTED_INTRABAR_POLICIES,
    time_to_minute,
    resolved_cost_profile,
    resolved_intrabar_policy,
    with_execution_mode,
)
from research_lab.data_loader import load_price_data, prepare_common_frame
from research_lab.engine import entry_open_index, estimate_spread_pips, run_backtest
from research_lab.news_filter import build_entry_block, load_news_events
from research_lab.report import export_strategy_bundle, summarize_result, sync_visible_chatgpt
from research_lab.strategies import STRATEGY_REGISTRY
from research_lab.strategies.common import cartesian_product, stratified_sample_combinations
from research_lab.validation import run_walkforward


DEFAULT_LIGHT_STRATEGIES = ("bollinger_mean_reversion_simple", "keltner_volatility_expansion_simple")
MIN_PHASE0_TRADES_PER_MONTH = 10.0
TARGET_TRADES_PER_MONTH_MIN = 15.0
TARGET_TRADES_PER_MONTH_MAX = 25.0


def build_output_root(results_dir: Path, label: str) -> Path:
    timestamp = pd.Timestamp.now(tz="America/New_York").strftime("%Y%m%d_%H%M%S")
    path = results_dir / f"{timestamp}_{label}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def costs_payload(engine_config: EngineConfig) -> dict[str, Any]:
    return {
        "assumed_spread_pips": engine_config.assumed_spread_pips if engine_config.assumed_spread_pips is not None else engine_config.max_spread_pips,
        "max_allowed_spread_pips": engine_config.max_spread_pips,
        "slippage_pips": engine_config.slippage_pips,
        "commission_per_lot_roundturn_usd": engine_config.commission_per_lot_roundturn_usd,
        "risk_pct": engine_config.risk_pct,
        "initial_capital": INITIAL_CAPITAL,
        "price_source": engine_config.price_source,
        "opening_session_end": engine_config.opening_session_end,
        "late_session_start": engine_config.late_session_start,
        "spread_opening_multiplier": engine_config.spread_opening_multiplier,
        "high_vol_range_atr": engine_config.high_vol_range_atr,
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
        "intrabar_exit_priority": engine_config.intrabar_exit_priority,
        "execution_mode": engine_config.execution_mode,
        "cost_profile_used": resolved_cost_profile(engine_config),
        "intrabar_policy_used": resolved_intrabar_policy(engine_config),
    }


def fixed_schedule() -> dict[str, str]:
    return {"entry_start": "11:00", "entry_end": "19:00", "force_close": "19:00"}


def share_best_year(yearly_stats: pd.DataFrame) -> float:
    if yearly_stats.empty:
        return 0.0
    yearly = yearly_stats.groupby("year")["total_pnl_r"].sum()
    positive_total = float(yearly[yearly > 0].sum())
    if positive_total <= 0:
        return 0.0
    return float(yearly.max() / positive_total)


def normalize_params(params: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(params)
    normalized["session_name"] = "light_fixed"
    normalized["use_h1_context"] = False
    normalized["break_even_at_r"] = normalized.get("break_even_at_r") if normalized.get("break_even_at_r") in {None, 1.0} else None
    return normalized


def dedupe_params(combos: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for combo in combos:
        key = json.dumps(combo, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(combo)
    return deduped


def phase0_combos(strategy_module: Any, max_evals: int, seed: int) -> list[dict[str, Any]]:
    reduced_space = dict(strategy_module.parameter_space())
    reduced_space["session_name"] = ["light_fixed"]
    reduced_space["use_h1_context"] = [False]
    reduced_space["break_even_at_r"] = [None, 1.0]
    sampled = [normalize_params(combo) for combo in stratified_sample_combinations(reduced_space, max_evals, seed)]
    return dedupe_params(sampled)


def neighbor_values(values: list[Any], center: Any) -> list[Any]:
    if center not in values:
        return [center]
    if isinstance(center, (bool, str)) or center is None:
        return [center]
    idx = values.index(center)
    start = max(0, idx - 1)
    end = min(len(values), idx + 2)
    return values[start:end]


def refine_combos(strategy_module: Any, center_params: dict[str, Any], max_evals: int, seed: int) -> list[dict[str, Any]]:
    param_space = dict(strategy_module.parameter_space())
    reduced_space: dict[str, list[Any]] = {}
    for key, values in param_space.items():
        if key == "session_name":
            reduced_space[key] = ["light_fixed"]
        elif key == "use_h1_context":
            reduced_space[key] = [False]
        elif key == "break_even_at_r":
            center_be = center_params.get("break_even_at_r")
            reduced_space[key] = [center_be] if center_be in {None, 1.0} else [None]
        else:
            reduced_space[key] = neighbor_values(list(values), center_params.get(key))
    combos = dedupe_params([normalize_params(combo) for combo in cartesian_product(reduced_space)])
    if len(combos) <= max_evals:
        return combos
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(combos), size=max_evals, replace=False)
    return [combos[int(idx)] for idx in indices]


def count_signal_candidates(
    strategy_module: Any,
    frame: pd.DataFrame,
    params: dict[str, Any],
    engine_config: EngineConfig,
    news_block: np.ndarray,
) -> tuple[int, float]:
    local_index = frame.index
    minute_values = (local_index.hour * 60 + local_index.minute).to_numpy()
    session_dates = np.array(local_index.date)
    range_atr = frame["range_atr"].to_numpy()

    start_minute = time_to_minute("11:00")
    force_close_minute = time_to_minute("19:00")
    count = 0
    opened_total_by_date: dict[Any, int] = {}

    for i in range(strategy_module.WARMUP_BARS, len(frame) - 1):
        session_date = session_dates[i]
        if minute_values[i] < start_minute or minute_values[i] >= force_close_minute:
            continue
        if news_block[i]:
            continue
        if opened_total_by_date.get(session_date, 0) >= engine_config.max_trades_per_day:
            continue
        if not np.isfinite(range_atr[i]) or range_atr[i] > engine_config.shock_candle_atr_max:
            continue
        if estimate_spread_pips(engine_config.pair, local_index[i], range_atr[i], engine_config) > engine_config.max_spread_pips:
            continue
        signal = strategy_module.signal(frame, i, params)
        if signal is None:
            continue
        count += 1
        opened_total_by_date[session_date] = opened_total_by_date.get(session_date, 0) + 1

    start_period = frame.index.min().tz_localize(None).to_period("M")
    end_period = frame.index.max().tz_localize(None).to_period("M")
    months = len(pd.period_range(start_period, end_period, freq="M"))
    avg_per_month = count / months if months else 0.0
    return count, avg_per_month


def discard_reasons(summary: dict[str, Any], yearly_stats: pd.DataFrame) -> tuple[list[str], float]:
    reasons: list[str] = []
    share = share_best_year(yearly_stats)
    if float(summary["profit_factor"]) <= 1.0:
        reasons.append("profit_factor<=1")
    if float(summary["expectancy_r"]) <= 0.0:
        reasons.append("expectancy<=0")
    if float(summary["avg_trades_per_month"]) < MIN_PHASE0_TRADES_PER_MONTH:
        reasons.append("frequency_too_low")
    if int(summary["negative_years"]) > 3:
        reasons.append("negative_years>3")
    if share > 0.60:
        reasons.append("year_dependency>0.60")
    return reasons, share


def phase1_score(summary: dict[str, Any], yearly_stats: pd.DataFrame) -> float:
    share = share_best_year(yearly_stats)
    trades_pm = float(summary["avg_trades_per_month"])
    score = (
        float(summary["profit_factor"]) * 250.0
        + float(summary["expectancy_r"]) * 900.0
        - float(summary["max_drawdown_pct"]) * 3.0
        - float(summary["negative_months"]) * 2.0
        - float(summary["negative_years"]) * 30.0
    )
    if trades_pm < TARGET_TRADES_PER_MONTH_MIN:
        score -= (TARGET_TRADES_PER_MONTH_MIN - trades_pm) * 30.0
    elif trades_pm > TARGET_TRADES_PER_MONTH_MAX:
        score -= (trades_pm - TARGET_TRADES_PER_MONTH_MAX) * 10.0
    if share > 0.60:
        score -= (share - 0.60) * 200.0
    if float(summary["profit_factor"]) <= 1.0:
        score -= 300.0
    if float(summary["expectancy_r"]) <= 0.0:
        score -= 300.0
    return score


def final_score(full_summary: dict[str, Any], oos_summary: dict[str, Any], yearly_stats: pd.DataFrame) -> float:
    share = share_best_year(yearly_stats)
    score = (
        float(oos_summary["profit_factor"]) * 300.0
        + float(oos_summary["expectancy_r"]) * 1000.0
        - float(oos_summary["max_drawdown_pct"]) * 3.0
        - float(oos_summary["negative_months"]) * 2.0
        - float(oos_summary["negative_years"]) * 35.0
    )
    if float(oos_summary["avg_trades_per_month"]) < MIN_PHASE0_TRADES_PER_MONTH:
        score -= (MIN_PHASE0_TRADES_PER_MONTH - float(oos_summary["avg_trades_per_month"])) * 40.0
    if share > 0.60:
        score -= (share - 0.60) * 200.0
    if float(oos_summary["profit_factor"]) <= 1.0:
        score -= 400.0
    if float(oos_summary["expectancy_r"]) <= 0.0:
        score -= 400.0
    return score


def export_phase1_bundle(
    strategy_dir: Path,
    *,
    summary: dict[str, Any],
    yearly_stats: pd.DataFrame,
    optimization_results: pd.DataFrame,
) -> None:
    strategy_dir.mkdir(parents=True, exist_ok=True)
    (strategy_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    yearly_stats.to_csv(strategy_dir / "yearly_stats.csv", index=False)
    optimization_results.to_csv(strategy_dir / "optimization_results.csv", index=False)


def export_root_light(
    output_root: Path,
    phase0_scan: pd.DataFrame,
    phase1_ranking: pd.DataFrame,
    comparative_table: pd.DataFrame,
    finalists_ranking: pd.DataFrame,
    recommendation: str,
) -> None:
    phase0_scan.to_csv(output_root / "phase0_scan.csv", index=False)
    if phase1_ranking.empty:
        fallback_ranking = phase0_scan.rename(columns={"phase0_pass": "passed_phase0"}).copy()
        fallback_ranking.to_csv(output_root / "strategy_ranking.csv", index=False)
    else:
        phase1_ranking.to_csv(output_root / "strategy_ranking.csv", index=False)
    comparative_table.to_csv(output_root / "comparative_table.csv", index=False)
    finalists_ranking.to_csv(output_root / "finalists_wfa.csv", index=False)
    (output_root / "recomendacion_final.md").write_text(recommendation, encoding="utf-8")


def evaluate_combo(
    strategy_name: str,
    strategy_module: Any,
    frame: pd.DataFrame,
    engine_config: EngineConfig,
    news_block: np.ndarray,
    news_filter_used: bool,
    params: dict[str, Any],
    selected_score: float | None = None,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    result = run_backtest(strategy_module, frame, params, engine_config, news_block, news_filter_used)
    return summarize_result(
        strategy_name,
        result.trades,
        result.equity_curve,
        params,
        news_filter_used,
        INITIAL_CAPITAL,
        selected_score,
        costs_payload(engine_config),
        "M15",
        fixed_schedule(),
        params.get("break_even_at_r"),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Runner liviano por tandas para EURUSD M15.")
    parser.add_argument("--pair", default=DEFAULT_PAIR)
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--data-dirs", nargs="+", default=[str(path) for path in DEFAULT_DATA_DIRS])
    parser.add_argument("--results-dir", default="results/research_lab_light")
    parser.add_argument("--news-file", default=str(DEFAULT_NEWS_FILE))
    parser.add_argument("--disable-news", action="store_true")
    parser.add_argument("--risk-pct", type=float, default=0.5)
    parser.add_argument("--shock-candle-atr-max", type=float, default=2.2)
    parser.add_argument("--assumed-spread-pips", type=float, default=1.2)
    parser.add_argument("--max-spread-pips", type=float, default=1.2)
    parser.add_argument("--slippage-pips", type=float, default=0.2)
    parser.add_argument("--commission-per-lot-roundturn-usd", type=float, default=7.0)
    parser.add_argument("--execution-mode", choices=list(SUPPORTED_EXECUTION_MODES), default=DEFAULT_EXECUTION_MODE)
    parser.add_argument("--cost-profile", choices=list(SUPPORTED_COST_PROFILES), default="auto")
    parser.add_argument("--intrabar-policy", choices=list(SUPPORTED_INTRABAR_POLICIES), default="auto")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--phase0-evals", type=int, default=8)
    parser.add_argument("--phase2-evals", type=int, default=12)
    parser.add_argument("--strategies", nargs="+", choices=STRATEGY_NAMES, default=list(DEFAULT_LIGHT_STRATEGIES))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    engine_config = with_execution_mode(
        EngineConfig(
        pair=args.pair.upper().strip(),
        risk_pct=args.risk_pct,
        shock_candle_atr_max=args.shock_candle_atr_max,
        assumed_spread_pips=args.assumed_spread_pips,
        max_spread_pips=args.max_spread_pips,
        commission_per_lot_roundturn_usd=args.commission_per_lot_roundturn_usd,
        slippage_pips=args.slippage_pips,
        execution_mode=args.execution_mode,
        cost_profile=args.cost_profile,
        intrabar_policy=args.intrabar_policy,
        ),
        args.execution_mode,
    )
    news_config = NewsConfig(
        enabled=not args.disable_news,
        file_path=Path(args.news_file),
        source_approved=True,
        pre_minutes=15,
        post_minutes=15,
    )

    start_time = time.time()
    raw_frame = load_price_data(engine_config.pair, [Path(item) for item in args.data_dirs], args.start, args.end)
    frame = prepare_common_frame(raw_frame)
    output_root = build_output_root(Path(args.results_dir), "light_batch")
    news_result = load_news_events(engine_config.pair, news_config)
    news_filter_used = news_result.enabled
    news_block = build_entry_block(entry_open_index(frame.index), news_result.events, news_config)

    strategy_names = list(args.strategies[:2])

    phase0_rows: list[dict[str, Any]] = []
    phase1_rows: list[dict[str, Any]] = []
    survivors: list[dict[str, Any]] = []

    for strategy_name in strategy_names:
        strategy_module = STRATEGY_REGISTRY[strategy_name]
        combos = phase0_combos(strategy_module, args.phase0_evals, args.seed)
        passing_phase0: list[dict[str, Any]] = []

        for params in combos:
            total_signals, avg_signals_pm = count_signal_candidates(strategy_module, frame, params, engine_config, news_block)
            phase0_rows.append(
                {
                    "strategy_name": strategy_name,
                    "phase0_pass": avg_signals_pm >= MIN_PHASE0_TRADES_PER_MONTH,
                    "total_signals": total_signals,
                    "avg_signals_per_month": avg_signals_pm,
                    "parameter_set_used": json.dumps(params, ensure_ascii=False),
                }
            )
            if avg_signals_pm >= MIN_PHASE0_TRADES_PER_MONTH:
                passing_phase0.append(params)

        if not passing_phase0:
            continue

        optimization_rows: list[dict[str, Any]] = []
        best_survivor_payload: dict[str, Any] | None = None
        best_survivor_score = -float("inf")

        for params in passing_phase0:
            summary, trades_export, monthly_stats, yearly_stats, equity_export = evaluate_combo(
                strategy_name, strategy_module, frame, engine_config, news_block, news_filter_used, params
            )
            reasons, share = discard_reasons(summary, yearly_stats)
            score = phase1_score(summary, yearly_stats)
            row = {
                "strategy_name": strategy_name,
                "discarded": bool(reasons),
                "discard_reasons": "|".join(reasons),
                "share_best_year": share,
                "selected_score": score,
                **summary,
                "parameter_set_used": json.dumps(params, ensure_ascii=False),
            }
            optimization_rows.append(row)
            if not reasons and score > best_survivor_score:
                best_survivor_score = score
                best_survivor_payload = {
                    "params": params,
                    "summary": summary,
                    "trades_export": trades_export,
                    "monthly_stats": monthly_stats,
                    "yearly_stats": yearly_stats,
                    "equity_export": equity_export,
                    "score": score,
                    "share_best_year": share,
                }

        optimization_df = pd.DataFrame(optimization_rows).sort_values("selected_score", ascending=False).reset_index(drop=True)
        phase1_rows.extend(optimization_rows)

        if best_survivor_payload is None:
            best_any = optimization_df.iloc[0].to_dict() if not optimization_df.empty else None
            if best_any is not None:
                export_phase1_bundle(
                    output_root / strategy_name / "PARA CHATGPT",
                    summary={
                        "strategy_name": strategy_name,
                        "phase": "phase1",
                        "discarded_phase1": True,
                        "selected_score": float(best_any["selected_score"]),
                        "parameter_set_used": json.loads(best_any["parameter_set_used"]),
                    },
                    yearly_stats=pd.DataFrame(),
                    optimization_results=optimization_df,
                )
            continue

        summary = dict(best_survivor_payload["summary"])
        summary["phase"] = "phase1"
        summary["discarded_phase1"] = False
        summary["share_best_year"] = best_survivor_payload["share_best_year"]
        summary["selected_score"] = best_survivor_payload["score"]
        export_phase1_bundle(
            output_root / strategy_name / "PARA CHATGPT",
            summary=summary,
            yearly_stats=best_survivor_payload["yearly_stats"],
            optimization_results=optimization_df,
        )
        survivors.append({"strategy_name": strategy_name, **best_survivor_payload})

    phase0_scan = pd.DataFrame(phase0_rows).sort_values(["phase0_pass", "avg_signals_per_month"], ascending=[False, False]).reset_index(drop=True)
    phase1_ranking = pd.DataFrame(phase1_rows).sort_values(["discarded", "selected_score"], ascending=[True, False]).reset_index(drop=True) if phase1_rows else pd.DataFrame()

    finalists_rows: list[dict[str, Any]] = []
    if survivors:
        top_candidates = sorted(survivors, key=lambda item: item["score"], reverse=True)[:3]
        for idx, candidate in enumerate(top_candidates, start=1):
            strategy_name = candidate["strategy_name"]
            strategy_module = STRATEGY_REGISTRY[strategy_name]
            refine_grid = refine_combos(strategy_module, candidate["params"], args.phase2_evals, args.seed + idx)
            best_payload: dict[str, Any] | None = None
            best_score = -float("inf")
            refine_rows: list[dict[str, Any]] = []

            for params in refine_grid:
                summary, trades_export, monthly_stats, yearly_stats, equity_export = evaluate_combo(
                    strategy_name, strategy_module, frame, engine_config, news_block, news_filter_used, params
                )
                reasons, share = discard_reasons(summary, yearly_stats)
                score = phase1_score(summary, yearly_stats)
                refine_rows.append(
                    {
                        "strategy_name": strategy_name,
                        "discarded": bool(reasons),
                        "discard_reasons": "|".join(reasons),
                        "share_best_year": share,
                        "selected_score": score,
                        **summary,
                        "parameter_set_used": json.dumps(params, ensure_ascii=False),
                    }
                )
                if not reasons and score > best_score:
                    best_score = score
                    best_payload = {
                        "params": params,
                        "summary": summary,
                        "trades_export": trades_export,
                        "monthly_stats": monthly_stats,
                        "yearly_stats": yearly_stats,
                        "equity_export": equity_export,
                    }

            if best_payload is None:
                continue

            wfa_result = run_walkforward(
                strategy_name=strategy_name,
                strategy_module=strategy_module,
                frame=frame,
                combos=[best_payload["params"]],
                engine_config=engine_config,
                news_config=news_config,
                is_months=24,
                oos_months=6,
            )
            summary = dict(best_payload["summary"])
            summary["phase"] = "phase2"
            summary["wfa_profit_factor"] = wfa_result.oos_summary["profit_factor"]
            summary["wfa_expectancy_r"] = wfa_result.oos_summary["expectancy_r"]
            summary["wfa_max_drawdown_pct"] = wfa_result.oos_summary["max_drawdown_pct"]
            summary["wfa_total_return_pct"] = wfa_result.oos_summary["total_return_pct"]
            summary["selected_score"] = final_score(summary, wfa_result.oos_summary, best_payload["yearly_stats"])

            export_strategy_bundle(
                output_root / strategy_name / "PARA CHATGPT",
                summary=summary,
                trades_export=best_payload["trades_export"],
                monthly_stats=best_payload["monthly_stats"],
                yearly_stats=best_payload["yearly_stats"],
                equity_export=best_payload["equity_export"],
                optimization_results=pd.DataFrame(refine_rows).sort_values("selected_score", ascending=False).reset_index(drop=True),
                extra_frames={"walkforward_summary.csv": wfa_result.fold_rows},
            )

            finalists_rows.append(
                {
                    "strategy_name": strategy_name,
                    "pf_oos": wfa_result.oos_summary["profit_factor"],
                    "expectancy_oos": wfa_result.oos_summary["expectancy_r"],
                    "dd_oos": wfa_result.oos_summary["max_drawdown_pct"],
                    "return_oos": wfa_result.oos_summary["total_return_pct"],
                    "trades_mes_oos": wfa_result.oos_summary["avg_trades_per_month"],
                    "selected_score": summary["selected_score"],
                    "parameter_set_used": json.dumps(best_payload["params"], ensure_ascii=False),
                }
            )

    finalists_wfa = pd.DataFrame(finalists_rows).sort_values("selected_score", ascending=False).reset_index(drop=True) if finalists_rows else pd.DataFrame()

    comparative_rows = []
    for strategy_name in strategy_names:
        subset = phase1_ranking[phase1_ranking["strategy_name"] == strategy_name] if not phase1_ranking.empty else pd.DataFrame()
        if subset.empty:
            comparative_rows.append({"strategy_name": strategy_name, "status": "phase0_failed"})
        else:
            best = subset.iloc[0].to_dict()
            comparative_rows.append(
                {
                    "strategy_name": strategy_name,
                    "status": "phase1_survivor" if not bool(best["discarded"]) else "phase1_failed",
                    "total_trades": best.get("total_trades"),
                    "avg_trades_per_month": best.get("avg_trades_per_month"),
                    "profit_factor": best.get("profit_factor"),
                    "expectancy_r": best.get("expectancy_r"),
                    "max_drawdown_pct": best.get("max_drawdown_pct"),
                    "negative_years": best.get("negative_years"),
                    "selected_score": best.get("selected_score"),
                    "discard_reasons": best.get("discard_reasons"),
                    "parameter_set_used": best.get("parameter_set_used"),
                }
            )
    comparative_table = pd.DataFrame(comparative_rows)

    if finalists_wfa.empty:
        recommendation = (
            "# Recomendacion final\n\n"
            "Ninguna variante llegó a ser finalista.\n\n"
            "Conclusión: esta tanda no sirve para seguir refinando. El problema no es la muestra; no apareció edge suficiente después del filtro de frecuencia."
        )
    else:
        winner = finalists_wfa.iloc[0]
        if float(winner["pf_oos"]) > 1.0 and float(winner["expectancy_oos"]) > 0.0:
            recommendation = (
                "# Recomendacion final\n\n"
                f"Seguir con **{winner['strategy_name']}**.\n\n"
                f"- PF OOS: {winner['pf_oos']:.4f}\n"
                f"- Expectancy OOS: {winner['expectancy_oos']:.4f}R\n"
                f"- DD OOS: {winner['dd_oos']:.2f}%\n"
            )
        else:
            recommendation = (
                "# Recomendacion final\n\n"
                f"La mejor finalista fue **{winner['strategy_name']}**, pero no tiene edge OOS suficiente.\n\n"
                f"- PF OOS: {winner['pf_oos']:.4f}\n"
                f"- Expectancy OOS: {winner['expectancy_oos']:.4f}R\n"
                f"- DD OOS: {winner['dd_oos']:.2f}%\n\n"
                "Conclusión: no conviene seguir refinando esta tanda."
            )

    export_root_light(output_root, phase0_scan, phase1_ranking, comparative_table, finalists_wfa, recommendation)
    sync_visible_chatgpt(output_root)

    print("\n=== PHASE 0 SCAN ===")
    if phase0_scan.empty:
        print("Sin resultados.")
    else:
        print(phase0_scan[["strategy_name", "phase0_pass", "total_signals", "avg_signals_per_month"]].to_string(index=False))

    print("\n=== PHASE 1 RANKING ===")
    if phase1_ranking.empty:
        print("Sin supervivientes.")
    else:
        print(phase1_ranking[["strategy_name", "discarded", "discard_reasons", "total_trades", "avg_trades_per_month", "profit_factor", "expectancy_r", "max_drawdown_pct", "negative_years", "selected_score"]].to_string(index=False))

    print("\n=== PHASE 2 FINALISTS ===")
    if finalists_wfa.empty:
        print("Sin finalistas.")
    else:
        print(finalists_wfa.to_string(index=False))
    print(f"\nruntime_seconds: {time.time() - start_time:.2f}")


if __name__ == "__main__":
    main()

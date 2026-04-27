from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .config import CONFIRMATION_MODE_LABELS, CONFIRMATION_PICK_LABELS, LEVEL_PROFILE_LABELS, NEWS_MODE_LABELS, CandidateConfig
from .ranking import apply_variant_verdicts, rank_variants


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Tipo no serializable: {type(value)!r}")


def _safe_round(value: float | int | None, digits: int = 4) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def _bucket_metrics(part: pd.DataFrame) -> dict[str, float]:
    if part.empty:
        return {"N": 0, "win_rate": 0.0, "pf": 0.0, "expectancy": 0.0, "avg_R": 0.0, "total_R": 0.0}
    pnl = part["pnl_r"].astype(float)
    wins = int((pnl > 0).sum())
    gross_profit = float(pnl[pnl > 0].sum())
    gross_loss = abs(float(pnl[pnl <= 0].sum()))
    pf = gross_profit / gross_loss if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0)
    return {
        "N": int(len(part)),
        "win_rate": _safe_round(wins / len(part), 4),
        "pf": _safe_round(pf, 4),
        "expectancy": _safe_round(pnl.mean(), 4),
        "avg_R": _safe_round(pnl.mean(), 4),
        "total_R": _safe_round(pnl.sum(), 4),
    }


def compute_trade_metrics(trades: pd.DataFrame, *, start_date: str, end_date: str) -> dict[str, object]:
    if trades.empty:
        empty = {}
        return {
            "sample_size": 0,
            "trades_per_month": 0.0,
            "win_rate": 0.0,
            "pf": 0.0,
            "expectancy": 0.0,
            "avg_R": 0.0,
            "max_drawdown_R": 0.0,
            "median_drawdown_R": 0.0,
            "avg_hold_minutes": 0.0,
            "timeout_exit_rate": 0.0,
            "result_by_year": empty,
            "year_positive_ratio": 0.0,
            "yearly_total_R_std": 0.0,
            "result_by_level_type": empty,
            "result_by_level_name": empty,
            "result_by_weekday": empty,
            "result_pre_news": empty,
            "result_post_news": empty,
        }

    frame = trades.copy()
    frame["entry_time"] = pd.to_datetime(frame["entry_time"], utc=True)
    frame["exit_time"] = pd.to_datetime(frame["exit_time"], utc=True)
    entry_time_naive = frame["entry_time"].dt.tz_localize(None)
    frame["entry_year"] = frame["entry_time"].dt.year.astype(str)
    frame["entry_month"] = entry_time_naive.dt.to_period("M").astype(str)
    frame["weekday"] = frame["entry_time"].dt.weekday.map({0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday", 4: "Friday", 5: "Saturday", 6: "Sunday"})
    frame["pnl_r"] = frame["pnl_r"].astype(float)
    frame["hold_minutes"] = frame["hold_minutes"].astype(float)
    frame["nearest_news_delta_minutes"] = pd.to_numeric(frame["nearest_news_delta_minutes"], errors="coerce")
    frame["minutes_since_previous_news"] = pd.to_numeric(frame["minutes_since_previous_news"], errors="coerce")
    frame["minutes_until_next_news"] = pd.to_numeric(frame["minutes_until_next_news"], errors="coerce")

    pnl = frame["pnl_r"]
    wins = int((pnl > 0).sum())
    gross_profit = float(pnl[pnl > 0].sum())
    gross_loss = abs(float(pnl[pnl <= 0].sum()))
    pf = gross_profit / gross_loss if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0)
    equity = pnl.cumsum()
    drawdown_series = equity - equity.cummax()
    month_span = max(1, len(pd.period_range(start=start_date, end=end_date, freq="M")))

    result_by_year = {year: _bucket_metrics(part) for year, part in frame.groupby("entry_year", sort=True)}
    result_by_level_type = {level: _bucket_metrics(part) for level, part in frame.groupby("level_group", sort=True)}
    result_by_level_name = {level: _bucket_metrics(part) for level, part in frame.groupby("level_name", sort=True)}
    result_by_weekday = {weekday: _bucket_metrics(part) for weekday, part in frame.groupby("weekday", sort=True)}

    result_pre_news = {
        "pre_news_15m": _bucket_metrics(frame.loc[(frame["minutes_until_next_news"] >= 0) & (frame["minutes_until_next_news"] <= 15)]),
        "pre_news_30m": _bucket_metrics(frame.loc[(frame["minutes_until_next_news"] >= 0) & (frame["minutes_until_next_news"] <= 30)]),
        "pre_news_60m": _bucket_metrics(frame.loc[(frame["minutes_until_next_news"] >= 0) & (frame["minutes_until_next_news"] <= 60)]),
    }
    result_post_news = {
        "post_news_15m": _bucket_metrics(frame.loc[(frame["minutes_since_previous_news"] >= 0) & (frame["minutes_since_previous_news"] <= 15)]),
        "post_news_30m": _bucket_metrics(frame.loc[(frame["minutes_since_previous_news"] >= 0) & (frame["minutes_since_previous_news"] <= 30)]),
        "post_news_60m": _bucket_metrics(frame.loc[(frame["minutes_since_previous_news"] >= 0) & (frame["minutes_since_previous_news"] <= 60)]),
    }

    yearly_totals = [bucket["total_R"] for bucket in result_by_year.values()]
    year_positive_ratio = (sum(1 for value in yearly_totals if value > 0) / len(yearly_totals)) if yearly_totals else 0.0

    return {
        "sample_size": int(len(frame)),
        "trades_per_month": _safe_round(len(frame) / month_span, 4),
        "win_rate": _safe_round(wins / len(frame), 4),
        "pf": _safe_round(pf, 4),
        "expectancy": _safe_round(float(pnl.mean()), 4),
        "avg_R": _safe_round(float(pnl.mean()), 4),
        "max_drawdown_R": _safe_round(float(drawdown_series.min()) if not drawdown_series.empty else 0.0, 4),
        "median_drawdown_R": _safe_round(float(drawdown_series.median()) if not drawdown_series.empty else 0.0, 4),
        "avg_hold_minutes": _safe_round(float(frame["hold_minutes"].mean()), 2),
        "timeout_exit_rate": _safe_round(float((frame["exit_reason"] == "timeout").mean()), 4),
        "result_by_year": result_by_year,
        "year_positive_ratio": _safe_round(year_positive_ratio, 4),
        "yearly_total_R_std": _safe_round(pd.Series(yearly_totals).std(ddof=0) if yearly_totals else 0.0, 4),
        "result_by_level_type": result_by_level_type,
        "result_by_level_name": result_by_level_name,
        "result_by_weekday": result_by_weekday,
        "result_pre_news": result_pre_news,
        "result_post_news": result_post_news,
    }


def build_variant_row(config: CandidateConfig, run_result: dict[str, object]) -> dict[str, object]:
    metrics = compute_trade_metrics(run_result["trades"], start_date=config.start_date, end_date=config.end_date)
    stats = run_result["stats"]
    return {
        "variant_id": config.variant_id,
        "profile_name": config.profile_name,
        "truth_model": config.truth_model,
        "research_status": config.research_status,
        "promotion_status": config.promotion_status,
        "experimental_variant": config.experimental_variant,
        "tp_r": config.tp_r,
        "timeout_hours": config.timeout_hours,
        "sl_buffer_pips": config.sl_buffer_pips,
        "long_entry_buffer_pips": config.long_entry_buffer_pips,
        "short_entry_buffer_pips": config.short_entry_buffer_pips,
        "confirmation_window": config.confirmation_window_label,
        "confirmation_mode": config.confirmation_mode,
        "confirmation_mode_label": CONFIRMATION_MODE_LABELS[config.confirmation_mode],
        "confirmation_pick": config.confirmation_pick,
        "confirmation_pick_label": CONFIRMATION_PICK_LABELS[config.confirmation_pick],
        "level_profile": config.level_profile,
        "level_profile_label": LEVEL_PROFILE_LABELS[config.level_profile],
        "news_mode": config.news_mode,
        "news_mode_label": NEWS_MODE_LABELS[config.news_mode],
        "sample_size": metrics["sample_size"],
        "trades_per_month": metrics["trades_per_month"],
        "win_rate": metrics["win_rate"],
        "pf": metrics["pf"],
        "expectancy": metrics["expectancy"],
        "avg_R": metrics["avg_R"],
        "max_drawdown_R": metrics["max_drawdown_R"],
        "median_drawdown_R": metrics["median_drawdown_R"],
        "avg_hold_minutes": metrics["avg_hold_minutes"],
        "timeout_exit_rate": metrics["timeout_exit_rate"],
        "year_positive_ratio": metrics["year_positive_ratio"],
        "yearly_total_R_std": metrics["yearly_total_R_std"],
        "sweeps_considered": int(stats.get("sweeps_considered", 0)),
        "trades_executed": int(stats.get("trades_executed", 0)),
        "news_blocked": int(stats.get("news_blocked", 0)),
        "daily_limit_skipped": int(stats.get("daily_limit_skipped", 0)),
        "no_scbi_window": int(stats.get("no_scbi_window", 0)),
        "no_scbi_found": int(stats.get("no_scbi_found", 0)),
        "no_entry_bar_after_scbi": int(stats.get("no_entry_bar_after_scbi", 0)),
        "invalid_risk": int(stats.get("invalid_risk", 0)),
        "level_filtered": int(stats.get("level_filtered", 0)),
        "result_by_year_json": json.dumps(metrics["result_by_year"], ensure_ascii=False, sort_keys=True),
        "result_by_level_type_json": json.dumps(metrics["result_by_level_type"], ensure_ascii=False, sort_keys=True),
        "result_by_level_name_json": json.dumps(metrics["result_by_level_name"], ensure_ascii=False, sort_keys=True),
        "result_by_weekday_json": json.dumps(metrics["result_by_weekday"], ensure_ascii=False, sort_keys=True),
        "result_pre_news_json": json.dumps(metrics["result_pre_news"], ensure_ascii=False, sort_keys=True),
        "result_post_news_json": json.dumps(metrics["result_post_news"], ensure_ascii=False, sort_keys=True),
    }


def build_baseline_payload(config: CandidateConfig, run_result: dict[str, object], coverage: dict[str, object]) -> dict[str, object]:
    return {
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "config": asdict(config),
        "coverage": coverage,
        "metrics": build_variant_row(config, run_result),
        "stats": run_result["stats"],
    }


def write_baseline_outputs(output_dir: Path, config: CandidateConfig, run_result: dict[str, object], coverage: dict[str, object]) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = build_baseline_payload(config, run_result, coverage)
    summary_json = output_dir / "baseline_summary.json"
    trades_csv = output_dir / "baseline_trades.csv"
    audit_csv = output_dir / "baseline_sweep_audit.csv"
    summary_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    run_result["trades"].to_csv(trades_csv, index=False)
    run_result["sweep_audit"].to_csv(audit_csv, index=False)
    return {
        "baseline_summary_json": summary_json,
        "baseline_trades_csv": trades_csv,
        "baseline_sweep_audit_csv": audit_csv,
    }


def _format_delta(value: float) -> str:
    prefix = "+" if value > 0 else ""
    return f"{prefix}{round(value, 4)}"


def _top3_robust(ranked: pd.DataFrame) -> pd.DataFrame:
    robust = ranked.loc[ranked["verdict"] == "ROBUST_RESEARCH_CANDIDATE"].copy()
    return robust.head(3)


def _build_baseline_vs_variants_md(ranked: pd.DataFrame) -> str:
    baseline = ranked.loc[ranked["truth_model"]].iloc[0].to_dict()
    lines = [
        "# Baseline vs Variants",
        "",
        "## Baseline exacta replicada",
        "",
        f"- variant_id: `{baseline['variant_id']}`",
        f"- sample_size: `{baseline['sample_size']}`",
        f"- PF: `{baseline['pf']}`",
        f"- expectancy: `{baseline['expectancy']}R`",
        f"- max_drawdown_R: `{baseline['max_drawdown_R']}R`",
        "",
        "## Top 10 variantes",
        "",
        "| variant_id | ranking_score | sample_size | PF | expectancy | max_drawdown_R | verdict |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in ranked.head(10).to_dict(orient="records"):
        lines.append(f"| {row['variant_id']} | {row['ranking_score']} | {row['sample_size']} | {row['pf']} | {row['expectancy']} | {row['max_drawdown_R']} | {row['verdict']} |")
    lines.extend(
        [
            "",
            "## Criterio de ranking",
            "",
            "- `year_positive_ratio`",
            "- `max_drawdown_R`",
            "- `yearly_total_R_std`",
            "- `sample_size`",
            "- `expectancy`",
            "- `PF`",
        ]
    )
    return "\n".join(lines) + "\n"


def _build_candidate_dossier(ranked: pd.DataFrame) -> str:
    baseline = ranked.loc[ranked["truth_model"]].iloc[0]
    top10 = ranked.head(10).copy()
    top3 = _top3_robust(ranked)
    shadow_candidate = top3.iloc[0] if not top3.empty else ranked.iloc[0]
    reject_rows = ranked.loc[ranked["verdict"] == "DO_NOT_PROMOTE"].head(10)
    lines = [
        "# Candidate Dossier",
        "",
        "## Baseline exacta replicada",
        "",
        "- Instrumento: `EURUSD`.",
        "- H1 para sweep y niveles; M5 para confirmacion, entrada y salida.",
        "- Niveles: `PDH/PDL`, `Asia H/L`, `London H/L`.",
        "- Confirmacion baseline: `+1h a +2h`, primera vela M5 cuyo `close` queda del lado correcto del nivel.",
        "- Entrada baseline: siguiente vela M5. Long `next_open + 0.3 pips`; short `next_open`.",
        "- Riesgo minimo: `2.0 pips`.",
        "- SL baseline: extremo del sweep `+-1 pip`.",
        "- TP baseline: `1.5R`.",
        "- Timeout baseline: `4 horas`.",
        "- Maximo `1` trade por dia.",
        "- Noticias baseline: filtro simplificado alrededor del sweep.",
        "",
        "## Cobertura de research",
        "",
        f"- Variantes evaluadas: `{len(ranked)}`.",
        f"- Baseline sample_size: `{baseline['sample_size']}`.",
        f"- Baseline PF: `{baseline['pf']}`.",
        f"- Baseline expectancy: `{baseline['expectancy']}R`.",
        f"- Baseline max_drawdown_R: `{baseline['max_drawdown_R']}R`.",
        "",
        "## Top 10 candidatos",
        "",
        "| rank | variant_id | ranking_score | PF | expectancy | max_drawdown_R | verdict |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for idx, row in enumerate(top10.to_dict(orient="records"), start=1):
        lines.append(f"| {idx} | {row['variant_id']} | {row['ranking_score']} | {row['pf']} | {row['expectancy']} | {row['max_drawdown_R']} | {row['verdict']} |")

    lines.extend(["", "## Top 3 candidatos robustos", ""])
    if top3.empty:
        lines.append("- No hubo `ROBUST_RESEARCH_CANDIDATE` segun el score actual.")
    else:
        for idx, row in enumerate(top3.to_dict(orient="records"), start=1):
            baseline_expectancy = float(baseline["expectancy"])
            baseline_drawdown = float(baseline["max_drawdown_R"])
            expectancy_delta = float(row["expectancy"]) - baseline_expectancy
            drawdown_delta = abs(float(row["max_drawdown_R"])) - abs(baseline_drawdown)
            lines.extend(
                [
                    f"### {idx}. {row['variant_id']}",
                    f"- Por que entra al top: `ranking_score={row['ranking_score']}`, `year_positive_ratio={row['year_positive_ratio']}`, `sample_size={row['sample_size']}`.",
                    f"- Mejora vs baseline: `PF { _format_delta(float(row['pf']) - float(baseline['pf'])) }`, `expectancy { _format_delta(expectancy_delta) }R`.",
                    f"- Empeora vs baseline: `drawdown absoluto { _format_delta(drawdown_delta) }R`, `timeout_exit_rate { _format_delta(float(row['timeout_exit_rate']) - float(baseline['timeout_exit_rate'])) }`.",
                    f"- Veredicto: `{row['verdict']}`.",
                    f"- Siguiente paso: {row['next_step']}",
                    "",
                ]
            )

    lines.extend(["## Candidatos que NO deben promoverse", ""])
    if reject_rows.empty:
        lines.append("- Ninguno bajo el criterio actual.")
    else:
        for row in reject_rows.to_dict(orient="records"):
            lines.append(f"- `{row['variant_id']}`: `PF={row['pf']}`, `expectancy={row['expectancy']}`, `sample_size={row['sample_size']}`, veredicto `{row['verdict']}`.")

    lines.extend(
        [
            "",
            "## Mejor candidato para una futura shadow line",
            "",
            f"- Candidato propuesto: `{shadow_candidate['variant_id']}`.",
            f"- Veredicto: `{shadow_candidate['verdict']}`.",
            f"- Razones principales: `PF={shadow_candidate['pf']}`, `expectancy={shadow_candidate['expectancy']}R`, `max_drawdown_R={shadow_candidate['max_drawdown_R']}R`, `sample_size={shadow_candidate['sample_size']}`.",
            "",
            "## Siguiente paso recomendado",
            "",
            "- Mantener todo en `RESEARCH_ONLY / NO_PRODUCTION` y, si se aprueba institucionalmente, usar solo el mejor candidato como futura shadow line externa.",
        ]
    )
    return "\n".join(lines) + "\n"


def _build_shadow_candidate_spec(candidate: pd.Series) -> str:
    lines = [
        "# Shadow Candidate Spec",
        "",
        "- Estado: `RESEARCH_ONLY` / `NO_PRODUCTION`.",
        f"- variant_id: `{candidate['variant_id']}`.",
        f"- levels: `{candidate['level_profile']}` ({candidate['level_profile_label']}).",
        f"- confirmation_window: `{candidate['confirmation_window']}`.",
        f"- confirmation_pick: `{candidate['confirmation_pick']}`.",
        f"- confirmation_mode: `{candidate['confirmation_mode']}`.",
        f"- long_entry_buffer_pips: `{candidate['long_entry_buffer_pips']}`.",
        f"- short_entry_buffer_pips: `{candidate['short_entry_buffer_pips']}`.",
        f"- sl_buffer_pips: `{candidate['sl_buffer_pips']}`.",
        f"- tp_r: `{candidate['tp_r']}`.",
        f"- timeout_hours: `{candidate['timeout_hours']}`.",
        f"- news_mode: `{candidate['news_mode']}`.",
        "",
        "## Reglas exactas",
        "",
        "- Sweep long: `low < nivel` y `close > nivel`.",
        "- Sweep short: `high > nivel` y `close < nivel`.",
        "- Confirmacion M5: dentro de la ventana indicada, segun el modo y pick indicados.",
        "- Entrada: apertura de la vela M5 siguiente a la confirmacion.",
        "- Riesgo minimo: `2.0 pips`.",
        "- SL: extremo del sweep mas el buffer configurado.",
        "- TP: multiple fijo `tp_r` sobre el riesgo.",
        "- Timeout: cierre por tiempo a `timeout_hours`.",
        "- Limite diario: `1 trade por dia`.",
        "- Noticias: filtro simplificado sobre `sweep_time` segun `news_mode`.",
        "",
        "No integrar a produccion sin aprobacion institucional posterior.",
    ]
    return "\n".join(lines) + "\n"


def write_matrix_outputs(output_dir: Path, *, ranked_results: pd.DataFrame, profile: str, baseline_payload: dict[str, object]) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    ranked_results = apply_variant_verdicts(rank_variants(ranked_results))
    matrix_csv = output_dir / "research_matrix_results.csv"
    top_csv = output_dir / "research_top_variants.csv"
    baseline_vs_md = output_dir / "research_baseline_vs_variants.md"
    summary_json = output_dir / "research_summary.json"
    dossier_md = output_dir / "candidate_dossier.md"
    shadow_spec_md = output_dir / "shadow_candidate_spec.md"

    ranked_results.to_csv(matrix_csv, index=False)
    ranked_results.head(25).to_csv(top_csv, index=False)
    baseline_vs_md.write_text(_build_baseline_vs_variants_md(ranked_results), encoding="utf-8")
    dossier_md.write_text(_build_candidate_dossier(ranked_results), encoding="utf-8")
    shadow_target = _top3_robust(ranked_results)
    chosen = shadow_target.iloc[0] if not shadow_target.empty else ranked_results.iloc[0]
    shadow_spec_md.write_text(_build_shadow_candidate_spec(chosen), encoding="utf-8")

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "profile": profile,
        "variant_count": int(len(ranked_results)),
        "baseline": baseline_payload,
        "top_variants": ranked_results.head(10).to_dict(orient="records"),
        "top_3_robust": _top3_robust(ranked_results).to_dict(orient="records"),
        "shadow_candidate": chosen.to_dict(),
    }
    summary_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    return {
        "research_matrix_results_csv": matrix_csv,
        "research_top_variants_csv": top_csv,
        "research_baseline_vs_variants_md": baseline_vs_md,
        "research_summary_json": summary_json,
        "candidate_dossier_md": dossier_md,
        "shadow_candidate_spec_md": shadow_spec_md,
    }


def summarize_existing_results(results_csv: Path, *, output_dir: Path, profile: str, baseline_payload: dict[str, object]) -> dict[str, Path]:
    frame = pd.read_csv(results_csv)
    if "truth_model" in frame.columns:
        frame["truth_model"] = frame["truth_model"].astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y"})
    return write_matrix_outputs(output_dir, ranked_results=frame, profile=profile, baseline_payload=baseline_payload)

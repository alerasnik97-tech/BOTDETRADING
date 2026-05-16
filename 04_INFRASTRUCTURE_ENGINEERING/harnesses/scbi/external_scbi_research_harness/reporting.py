from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .config import CONFIRMATION_MODE_LABELS, CONFIRMATION_PICK_LABELS, NEWS_MODE_LABELS, TruthModelConfig


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
        return {"N": 0, "win_rate": 0.0, "pf": 0.0, "expectancy": 0.0, "total_r": 0.0}
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
        "total_r": _safe_round(pnl.sum(), 4),
    }


def compute_trade_metrics(trades: pd.DataFrame, *, start_date: str, end_date: str) -> dict[str, object]:
    if trades.empty:
        empty_map = {}
        return {
            "sample_size": 0,
            "win_rate": 0.0,
            "pf": 0.0,
            "expectancy": 0.0,
            "avg_r": 0.0,
            "max_drawdown": 0.0,
            "avg_hold_minutes": 0.0,
            "timeout_exit_pct": 0.0,
            "trades_per_month": 0.0,
            "yearly": empty_map,
            "by_level": empty_map,
            "by_weekday": empty_map,
            "news_split": empty_map,
            "year_positive_ratio": 0.0,
            "worst_year_total_r": 0.0,
            "yearly_total_r_std": 0.0,
        }

    frame = trades.copy()
    frame["entry_time"] = pd.to_datetime(frame["entry_time"], utc=True)
    frame["exit_time"] = pd.to_datetime(frame["exit_time"], utc=True)
    entry_time_naive = frame["entry_time"].dt.tz_localize(None)
    frame["entry_year"] = frame["entry_time"].dt.year.astype(str)
    frame["entry_month"] = entry_time_naive.dt.to_period("M").astype(str)
    frame["weekday_num"] = frame["entry_time"].dt.weekday
    frame["weekday"] = frame["weekday_num"].map(
        {
            0: "Monday",
            1: "Tuesday",
            2: "Wednesday",
            3: "Thursday",
            4: "Friday",
            5: "Saturday",
            6: "Sunday",
        }
    )
    frame["pnl_r"] = frame["pnl_r"].astype(float)
    frame["hold_minutes"] = frame["hold_minutes"].astype(float)
    frame["nearest_news_delta_minutes"] = pd.to_numeric(frame["nearest_news_delta_minutes"], errors="coerce")
    frame["minutes_since_previous_news"] = pd.to_numeric(frame["minutes_since_previous_news"], errors="coerce")

    pnl = frame["pnl_r"]
    wins = int((pnl > 0).sum())
    gross_profit = float(pnl[pnl > 0].sum())
    gross_loss = abs(float(pnl[pnl <= 0].sum()))
    pf = gross_profit / gross_loss if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0)

    equity = pnl.cumsum()
    peak = equity.cummax()
    max_drawdown = float((equity - peak).min()) if not equity.empty else 0.0

    month_span = len(pd.period_range(start=start_date, end=end_date, freq="M"))
    month_span = max(month_span, 1)

    yearly = {
        year: _bucket_metrics(part)
        for year, part in frame.groupby("entry_year", sort=True)
    }
    by_level = {
        level: _bucket_metrics(part)
        for level, part in frame.groupby("level_name", sort=True)
    }
    by_weekday = {
        weekday: _bucket_metrics(part)
        for weekday, part in frame.groupby("weekday", sort=True)
    }

    news_split = {
        "near_news_30m": _bucket_metrics(frame.loc[frame["nearest_news_delta_minutes"] <= 30]),
        "outside_news_30m": _bucket_metrics(frame.loc[(frame["nearest_news_delta_minutes"] > 30) | frame["nearest_news_delta_minutes"].isna()]),
        "post_news_60m": _bucket_metrics(frame.loc[(frame["minutes_since_previous_news"] >= 0) & (frame["minutes_since_previous_news"] <= 60)]),
        "outside_post_news_60m": _bucket_metrics(frame.loc[(frame["minutes_since_previous_news"].isna()) | (frame["minutes_since_previous_news"] > 60) | (frame["minutes_since_previous_news"] < 0)]),
    }

    yearly_totals = [bucket["total_r"] for bucket in yearly.values()]
    year_positive_ratio = (sum(1 for value in yearly_totals if value > 0) / len(yearly_totals)) if yearly_totals else 0.0

    return {
        "sample_size": int(len(frame)),
        "win_rate": _safe_round(wins / len(frame), 4),
        "pf": _safe_round(pf, 4),
        "expectancy": _safe_round(float(pnl.mean()), 4),
        "avg_r": _safe_round(float(pnl.mean()), 4),
        "max_drawdown": _safe_round(max_drawdown, 4),
        "avg_hold_minutes": _safe_round(float(frame["hold_minutes"].mean()), 2),
        "timeout_exit_pct": _safe_round(float((frame["exit_reason"] == "timeout").mean()), 4),
        "trades_per_month": _safe_round(len(frame) / month_span, 4),
        "yearly": yearly,
        "by_level": by_level,
        "by_weekday": by_weekday,
        "news_split": news_split,
        "year_positive_ratio": _safe_round(year_positive_ratio, 4),
        "worst_year_total_r": _safe_round(min(yearly_totals) if yearly_totals else 0.0, 4),
        "yearly_total_r_std": _safe_round(pd.Series(yearly_totals).std(ddof=0) if yearly_totals else 0.0, 4),
    }


def build_variant_row(config: TruthModelConfig, run_result: dict[str, object]) -> dict[str, object]:
    trades = run_result["trades"]
    stats = run_result["stats"]
    metrics = compute_trade_metrics(trades, start_date=config.start_date, end_date=config.end_date)
    row = {
        "variant_id": config.variant_id,
        "profile_name": config.profile_name,
        "truth_model": config.truth_model,
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
        "body_strength_threshold": config.body_strength_threshold,
        "level_profile": config.level_profile,
        "news_mode": config.news_mode,
        "news_mode_label": NEWS_MODE_LABELS[config.news_mode],
        "sample_size": metrics["sample_size"],
        "win_rate": metrics["win_rate"],
        "pf": metrics["pf"],
        "expectancy": metrics["expectancy"],
        "avg_r": metrics["avg_r"],
        "max_drawdown": metrics["max_drawdown"],
        "avg_hold_minutes": metrics["avg_hold_minutes"],
        "timeout_exit_pct": metrics["timeout_exit_pct"],
        "trades_per_month": metrics["trades_per_month"],
        "year_positive_ratio": metrics["year_positive_ratio"],
        "worst_year_total_r": metrics["worst_year_total_r"],
        "yearly_total_r_std": metrics["yearly_total_r_std"],
        "news_blocked": int(stats.get("news_blocked", 0)),
        "daily_limit_skipped": int(stats.get("daily_limit_skipped", 0)),
        "no_scbi_window": int(stats.get("no_scbi_window", 0)),
        "no_scbi_found": int(stats.get("no_scbi_found", 0)),
        "no_entry_bar_after_scbi": int(stats.get("no_entry_bar_after_scbi", 0)),
        "invalid_risk": int(stats.get("invalid_risk", 0)),
        "level_filtered": int(stats.get("level_filtered", 0)),
        "yearly_json": json.dumps(metrics["yearly"], ensure_ascii=False, sort_keys=True),
        "level_json": json.dumps(metrics["by_level"], ensure_ascii=False, sort_keys=True),
        "weekday_json": json.dumps(metrics["by_weekday"], ensure_ascii=False, sort_keys=True),
        "news_split_json": json.dumps(metrics["news_split"], ensure_ascii=False, sort_keys=True),
    }
    return row


def rank_variants(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return results.copy()

    ranked = results.copy()
    ranked["drawdown_safety"] = -ranked["max_drawdown"].abs()
    ranked["consistency_safety"] = -ranked["yearly_total_r_std"]

    norm_columns = {
        "sample_size": "sample_size_norm",
        "year_positive_ratio": "year_positive_ratio_norm",
        "expectancy": "expectancy_norm",
        "drawdown_safety": "drawdown_safety_norm",
        "consistency_safety": "consistency_safety_norm",
    }

    for source, target in norm_columns.items():
        source_series = ranked[source].astype(float)
        min_value = float(source_series.min())
        max_value = float(source_series.max())
        if max_value == min_value:
            ranked[target] = 1.0
        else:
            ranked[target] = (source_series - min_value) / (max_value - min_value)

    ranked["ranking_score"] = (
        (ranked["year_positive_ratio_norm"] * 0.30)
        + (ranked["drawdown_safety_norm"] * 0.20)
        + (ranked["consistency_safety_norm"] * 0.20)
        + (ranked["sample_size_norm"] * 0.15)
        + (ranked["expectancy_norm"] * 0.15)
    )
    ranked["ranking_score"] = ranked["ranking_score"].round(6)
    ranked = ranked.sort_values(
        ["ranking_score", "year_positive_ratio", "expectancy", "sample_size"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    return ranked


def build_baseline_markdown(config: TruthModelConfig, row: dict[str, object], run_result: dict[str, object]) -> str:
    stats = run_result["stats"]
    lines = [
        "# Baseline Truth Model",
        "",
        "## Lo mas importante",
        "",
        "- Replica externa del runner real actual sin tocar produccion.",
        f"- Variante: `{config.variant_id}`.",
        f"- Periodo: `{config.start_date}` -> `{config.end_date}`.",
        "",
        "## Parametros replicados",
        "",
        f"- TP fijo: `{config.tp_r}R`.",
        f"- Timeout: `{config.timeout_hours}h`.",
        f"- Buffer SL: `{config.sl_buffer_pips} pips`.",
        f"- Buffer entrada long: `{config.long_entry_buffer_pips} pips`.",
        f"- Ventana confirmacion: `{config.confirmation_window_label}`.",
        f"- Confirmacion: `{CONFIRMATION_MODE_LABELS[config.confirmation_mode]}`.",
        f"- Seleccion de confirmacion: `{CONFIRMATION_PICK_LABELS[config.confirmation_pick]}`.",
        f"- Niveles habilitados: `{config.level_profile}`.",
        f"- Filtro noticias: `{NEWS_MODE_LABELS[config.news_mode]}`.",
        "",
        "## Metricas",
        "",
        f"- sample_size: `{row['sample_size']}`",
        f"- win_rate: `{row['win_rate']}`",
        f"- PF: `{row['pf']}`",
        f"- expectancy: `{row['expectancy']}R`",
        f"- avg_hold_minutes: `{row['avg_hold_minutes']}`",
        f"- timeout_exit_pct: `{row['timeout_exit_pct']}`",
        f"- max_drawdown: `{row['max_drawdown']}R`",
        "",
        "## Conteos de auditoria",
        "",
        f"- sweeps_considered: `{stats.get('sweeps_considered', 0)}`",
        f"- trades_executed: `{stats.get('trades_executed', 0)}`",
        f"- news_blocked: `{stats.get('news_blocked', 0)}`",
        f"- daily_limit_skipped: `{stats.get('daily_limit_skipped', 0)}`",
        f"- no_scbi_found: `{stats.get('no_scbi_found', 0)}`",
        f"- invalid_risk: `{stats.get('invalid_risk', 0)}`",
    ]
    return "\n".join(lines) + "\n"


def write_baseline_outputs(output_dir: Path, config: TruthModelConfig, run_result: dict[str, object]) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    row = build_variant_row(config, run_result)
    trades = run_result["trades"]
    audit = run_result["sweep_audit"]
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "config": asdict(config),
        "metrics": row,
        "stats": run_result["stats"],
    }

    trades_path = output_dir / "baseline_trades.csv"
    audit_path = output_dir / "baseline_sweep_audit.csv"
    json_path = output_dir / "baseline_summary.json"
    md_path = output_dir / "baseline_summary.md"

    trades.to_csv(trades_path, index=False)
    audit.to_csv(audit_path, index=False)
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    md_path.write_text(build_baseline_markdown(config, row, run_result), encoding="utf-8")

    return {
        "trades_csv": trades_path,
        "audit_csv": audit_path,
        "summary_json": json_path,
        "summary_md": md_path,
    }


def _markdown_table(rows: list[dict[str, object]]) -> str:
    headers = ["variant_id", "ranking_score", "sample_size", "pf", "expectancy", "max_drawdown", "year_positive_ratio"]
    table = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        table.append(
            "| "
            + " | ".join(str(row.get(header, "")) for header in headers)
            + " |"
        )
    return "\n".join(table)


def build_matrix_markdown(
    *,
    ranked_results: pd.DataFrame,
    baseline_row: dict[str, object] | None,
    profile: str,
) -> str:
    lines = [
        "# Baseline vs Variants",
        "",
        f"- perfil_matriz: `{profile}`",
        f"- variantes_evaluadas: `{len(ranked_results)}`",
        "",
    ]

    if baseline_row is not None:
        lines.extend(
            [
                "## Baseline",
                "",
                f"- variant_id: `{baseline_row['variant_id']}`",
                f"- sample_size: `{baseline_row['sample_size']}`",
                f"- PF: `{baseline_row['pf']}`",
                f"- expectancy: `{baseline_row['expectancy']}R`",
                f"- max_drawdown: `{baseline_row['max_drawdown']}R`",
                f"- year_positive_ratio: `{baseline_row['year_positive_ratio']}`",
                "",
            ]
        )

    top_rows = ranked_results.head(15).to_dict(orient="records")
    lines.extend(
        [
            "## Top Variants",
            "",
            _markdown_table(top_rows),
            "",
            "## Criterios de ranking",
            "",
            "- `year_positive_ratio`: mayor es mejor.",
            "- `max_drawdown`: menor drawdown absoluto es mejor.",
            "- `yearly_total_r_std`: menor dispersion anual es mejor.",
            "- `sample_size`: mayor es mejor.",
            "- `expectancy`: mayor es mejor.",
            "",
            "No se rankea por profit bruto solamente.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_matrix_outputs(
    output_dir: Path,
    *,
    ranked_results: pd.DataFrame,
    profile: str,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    results_csv = output_dir / "research_matrix_results.csv"
    top_csv = output_dir / "research_top_variants.csv"
    md_path = output_dir / "research_baseline_vs_variants.md"
    json_path = output_dir / "research_summary.json"

    ranked_results.to_csv(results_csv, index=False)
    ranked_results.head(25).to_csv(top_csv, index=False)

    baseline_row = None
    if not ranked_results.loc[ranked_results["truth_model"]].empty:
        baseline_row = ranked_results.loc[ranked_results["truth_model"]].iloc[0].to_dict()
    md_path.write_text(
        build_matrix_markdown(ranked_results=ranked_results, baseline_row=baseline_row, profile=profile),
        encoding="utf-8",
    )

    summary_payload = {
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "profile": profile,
        "variant_count": int(len(ranked_results)),
        "baseline_variant": baseline_row,
        "top_variants": ranked_results.head(10).to_dict(orient="records"),
        "ranking_fields": [
            "year_positive_ratio",
            "max_drawdown",
            "yearly_total_r_std",
            "sample_size",
            "expectancy",
        ],
        "files": {
            "research_matrix_results_csv": str(results_csv),
            "research_top_variants_csv": str(top_csv),
            "research_baseline_vs_variants_md": str(md_path),
        },
    }
    json_path.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")

    return {
        "research_matrix_results_csv": results_csv,
        "research_top_variants_csv": top_csv,
        "research_baseline_vs_variants_md": md_path,
        "research_summary_json": json_path,
    }


def summarize_existing_results(results_csv: Path, *, output_dir: Path, profile: str = "existing_results") -> dict[str, Path]:
    ranked_results = pd.read_csv(results_csv)
    if "truth_model" in ranked_results.columns:
        ranked_results["truth_model"] = (
            ranked_results["truth_model"]
            .astype(str)
            .str.strip()
            .str.lower()
            .isin({"1", "true", "yes", "y"})
        )
    ranked_results = rank_variants(ranked_results)
    return write_matrix_outputs(output_dir, ranked_results=ranked_results, profile=profile)

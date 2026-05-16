from __future__ import annotations

import pandas as pd


def rank_variants(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return results.copy()

    ranked = results.copy()
    ranked["drawdown_safety"] = -ranked["max_drawdown_R"].abs()
    ranked["consistency_safety"] = -ranked["yearly_total_R_std"]
    norm_columns = {
        "sample_size": "sample_size_norm",
        "year_positive_ratio": "year_positive_ratio_norm",
        "expectancy": "expectancy_norm",
        "pf": "pf_norm",
        "drawdown_safety": "drawdown_safety_norm",
        "consistency_safety": "consistency_safety_norm",
    }
    for source, target in norm_columns.items():
        series = ranked[source].astype(float)
        min_value = float(series.min())
        max_value = float(series.max())
        if max_value == min_value:
            ranked[target] = 1.0
        else:
            ranked[target] = (series - min_value) / (max_value - min_value)

    ranked["ranking_score"] = (
        (ranked["year_positive_ratio_norm"] * 0.25)
        + (ranked["drawdown_safety_norm"] * 0.20)
        + (ranked["consistency_safety_norm"] * 0.15)
        + (ranked["sample_size_norm"] * 0.15)
        + (ranked["expectancy_norm"] * 0.15)
        + (ranked["pf_norm"] * 0.10)
    ).round(6)
    ranked = ranked.sort_values(
        ["ranking_score", "year_positive_ratio", "pf", "expectancy", "sample_size"],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)
    return ranked


def apply_variant_verdicts(ranked: pd.DataFrame) -> pd.DataFrame:
    if ranked.empty:
        return ranked.copy()

    output = ranked.copy()
    baseline = output.loc[output["truth_model"]].iloc[0] if not output.loc[output["truth_model"]].empty else None
    robust_cut = float(output["ranking_score"].quantile(0.75)) if len(output) > 1 else 1.0

    verdicts: list[str] = []
    next_steps: list[str] = []
    for _, row in output.iterrows():
        if bool(row["truth_model"]):
            verdicts.append("BASELINE_TRUTH_MODEL")
            next_steps.append("Preservar como baseline institucional y comparar todo contra esta replica.")
            continue

        baseline_drawdown_abs = abs(float(baseline["max_drawdown_R"])) if baseline is not None else abs(float(row["max_drawdown_R"])) * 2.0
        robust_candidate = (
            row["ranking_score"] >= robust_cut
            and row["sample_size"] >= max(200, int((baseline["sample_size"] * 0.75) if baseline is not None else 200))
            and row["year_positive_ratio"] >= float(baseline["year_positive_ratio"] if baseline is not None else 0.5)
            and abs(float(row["max_drawdown_R"])) <= (baseline_drawdown_abs * 1.35)
            and float(row["expectancy"]) >= float((baseline["expectancy"] if baseline is not None else 0.0) * 0.95)
        )
        weak_candidate = (
            row["sample_size"] < 150
            or float(row["pf"]) < float((baseline["pf"] if baseline is not None else row["pf"]) * 0.80)
            or float(row["expectancy"]) < float((baseline["expectancy"] if baseline is not None else row["expectancy"]) * 0.80)
        )

        if robust_candidate:
            verdicts.append("ROBUST_RESEARCH_CANDIDATE")
            next_steps.append("Mantener como RESEARCH_ONLY y dejarlo listo para futura shadow line.")
        elif weak_candidate:
            verdicts.append("DO_NOT_PROMOTE")
            next_steps.append("No promover; conservar solo como evidencia comparativa de research.")
        else:
            verdicts.append("RESEARCH_ONLY_MONITOR")
            next_steps.append("Seguir en research only y re-evaluar si aparece una familia mejor.")

    output["verdict"] = verdicts
    output["next_step"] = next_steps
    return output

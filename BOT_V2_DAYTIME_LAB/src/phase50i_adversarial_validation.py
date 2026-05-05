from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

ROOT = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo")
HIST = ROOT / "BOT_V2_DAYTIME_LAB" / "reports" / "manipulante_tick_historical"
REP = ROOT / "BOT_V2_DAYTIME_LAB" / "reports"
TRADE_LEVEL = HIST / "PHASE50H_MULTI_MONTH_TRADE_LEVEL_RESULTS.csv"
MONTHLY = HIST / "PHASE50H_MULTI_MONTH_MONTHLY_METRICS.csv"
PH50H_JSON = REP / "PHASE50H_MULTI_MONTH_TICK_VALIDATION_STAGE_REPORT.json"
PH50H_CP = HIST / "PHASE50H_EXECUTION_CHECKPOINT.json"
RAW_TRADES = ROOT / "BOT_V2_DAYTIME_LAB" / "outputs" / "phase38_manipulante_deep_explainer" / "csv" / "phase38_raw_trades_enriched.csv"

NO_AUDITABLES_OUT = HIST / "PHASE50I_NO_AUDITABLES_LIST.csv"
STRESS_OUT = HIST / "PHASE50I_STRESS_TEST_RESULTS.csv"
CONC_OUT = HIST / "PHASE50I_PROFIT_CONCENTRATION_AUDIT.csv"
JUL_MD = REP / "PHASE50I_2025_07_FRAGILITY_AUDIT.md"
JUL_JSON = REP / "PHASE50I_2025_07_FRAGILITY_AUDIT.json"
FINAL_MD = REP / "PHASE50I_ADVERSARIAL_VALIDATION_REPORT.md"
FINAL_JSON = REP / "PHASE50I_ADVERSARIAL_VALIDATION_REPORT.json"


def pf(series: pd.Series) -> float:
    wins = series[series > 0].sum()
    losses = series[series < 0].sum()
    if losses == 0:
        return math.inf
    return abs(float(wins / losses))


def dd_seq(series: pd.Series) -> float:
    eq = series.cumsum()
    return float((eq - eq.cummax()).min()) if len(eq) else math.nan


def metrics(df: pd.DataFrame, tick_col: str = "tick_R") -> dict:
    s = df[tick_col].dropna()
    total = len(df)
    aud = len(s)
    return {
        "total_trades": int(total),
        "audited_trades": int(aud),
        "non_auditable_trades": int(total - aud),
        "PF_tick": pf(s) if aud else math.nan,
        "expectancy_tick": float(s.mean()) if aud else math.nan,
        "DD_tick": dd_seq(s) if aud else math.nan,
        "winrate_tick": float((s > 0).mean() * 100) if aud else math.nan,
        "total_R_tick": float(s.sum()) if aud else math.nan,
        "TP_tick": int((df["tick_outcome"] == "TP").sum()),
        "BE_tick": int((df["tick_outcome"] == "BE").sum()),
        "SL_tick": int((df["tick_outcome"] == "SL").sum()),
        "match_rate": float((df["match_status"] == "MATCH").mean() * 100) if total else math.nan,
        "reliability_score": float(aud / total) if total else math.nan,
    }


def pick_reason(row: pd.Series) -> str:
    c = str(row.get("classification", "")).upper()
    to = str(row.get("tick_outcome", "")).upper()
    if "GAP" in c:
        return "TRUE_DATA_GAP"
    if "LOW_TICK" in c:
        return "LOW_TICK_DENSITY"
    if "EXIT_TIME" in c:
        return "EXIT_TIME_MISSING"
    if to in {"NONE", "NO_TICK_DATA"}:
        return "TICK_WINDOW_TOO_SHORT"
    if "FORCED" in str(row.get("bar_outcome", "")).upper():
        return "FORCED_CLOSE_MODEL_GAP"
    return "UNKNOWN_REQUIRES_REVIEW"


def to_safe(v):
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return str(v)
    if isinstance(v, dict):
        return {k: to_safe(x) for k, x in v.items()}
    if isinstance(v, list):
        return [to_safe(x) for x in v]
    return v


def main() -> None:
    df = pd.read_csv(TRADE_LEVEL)
    df["entry_time"] = pd.to_datetime(df["entry_time"], errors="coerce", utc=True).dt.tz_convert("America/New_York")
    df["exit_time"] = pd.to_datetime(df["exit_time"], errors="coerce", utc=True).dt.tz_convert("America/New_York")
    df["tick_R"] = pd.to_numeric(df["tick_R"], errors="coerce")
    df["bar_R"] = pd.to_numeric(df["bar_R"], errors="coerce")
    df["month"] = df["month"].astype(str)
    monthly = pd.read_csv(MONTHLY)
    raw = pd.read_csv(RAW_TRADES)
    raw["entry_time"] = pd.to_datetime(raw["entry_time"], errors="coerce", utc=True).dt.tz_convert("America/New_York")
    raw["exit_time"] = pd.to_datetime(raw["exit_time"], errors="coerce", utc=True).dt.tz_convert("America/New_York")
    if "trade_id" not in raw.columns:
        raw["trade_id"] = list(range(1, len(raw) + 1))
    raw = raw[["trade_id", "entry_time", "exit_time", "type", "year_month"]].rename(
        columns={"entry_time": "entry_time_raw", "exit_time": "exit_time_raw", "type": "direction_raw", "year_month": "month_raw"}
    )
    hjson = json.loads(PH50H_JSON.read_text(encoding="utf-8")) if PH50H_JSON.exists() else {}
    cp = json.loads(PH50H_CP.read_text(encoding="utf-8")) if PH50H_CP.exists() else {}

    base = metrics(df)
    month_contrib = df.groupby("month", dropna=False)["tick_R"].sum(min_count=1).sort_values(ascending=False)
    best_month = month_contrib.index[0]
    worst_month = month_contrib.index[-1]

    no_aud = df[df["tick_R"].isna() | df["classification"].str.contains("NOT_AUDITABLE|DATA_GAP|LOW_TICK_DENSITY|NONE|EXIT_TIME", na=False, case=False)].copy()
    no_aud = no_aud.merge(raw, on="trade_id", how="left")
    no_aud["entry_time"] = no_aud["entry_time"].fillna(no_aud["entry_time_raw"].astype(str))
    no_aud["exit_time"] = no_aud["exit_time"].fillna(no_aud["exit_time_raw"].astype(str))
    no_aud["direction"] = no_aud["direction"].fillna(no_aud["direction_raw"])
    no_aud["month"] = no_aud["month"].fillna(no_aud["month_raw"])
    no_aud["date"] = no_aud["entry_time"].dt.date.astype(str)
    no_aud["entry_time_ny"] = no_aud["entry_time"].astype(str)
    no_aud["exit_time_ny"] = no_aud["exit_time"].astype(str)
    no_aud["reason"] = no_aud.apply(pick_reason, axis=1)
    no_aud["data_gap_yes_no"] = no_aud["reason"].isin(["TRUE_DATA_GAP", "LOW_TICK_DENSITY"]).map({True: "YES", False: "NO"})
    no_aud["can_be_reaudited_yes_no"] = no_aud["reason"].isin(["TRUE_DATA_GAP", "LOW_TICK_DENSITY", "TICK_WINDOW_TOO_SHORT"]).map({True: "YES", False: "NO"})
    no_aud["conservative_assignment"] = -1.0
    no_aud["notes"] = "Adversarial conservative assignment as SL when stressed"
    no_aud_out = no_aud.rename(columns={"entry_time": "entry_time_raw", "exit_time": "exit_time_raw"})[
        ["trade_id", "month", "date", "direction", "entry_time_ny", "exit_time_ny", "bar_outcome", "tick_outcome", "bar_R", "tick_R", "classification", "reason", "data_gap_yes_no", "can_be_reaudited_yes_no", "conservative_assignment", "notes"]
    ].sort_values(["month", "trade_id"])
    no_aud_out.to_csv(NO_AUDITABLES_OUT, index=False)

    # Scenarios
    rows = []
    def add_scenario(name: str, sdf: pd.DataFrame, notes: str):
        m = metrics(sdf)
        passes = (m["PF_tick"] >= 1.25) and (m["expectancy_tick"] >= 0.06)
        rows.append({
            "scenario": name,
            "total_trades": m["total_trades"],
            "audited_trades": m["audited_trades"],
            "PF_tick": m["PF_tick"],
            "expectancy_tick": m["expectancy_tick"],
            "DD_tick": m["DD_tick"],
            "winrate_tick": m["winrate_tick"],
            "total_R_tick": m["total_R_tick"],
            "passes_edge_threshold_yes_no": "YES" if passes else "NO",
            "notes": notes
        })

    add_scenario("BASE", df, "PHASE50H trade-level tal cual")
    add_scenario("EXCLUDE_NON_AUDITABLES", df[df["tick_R"].notna()].copy(), "excluye tick_R nulo")

    b = df.copy()
    b.loc[b["tick_R"].isna(), "tick_R"] = -1.0
    b.loc[b["tick_R"].isna(), "tick_outcome"] = "SL"
    add_scenario("ALL_NON_AUDITABLES_AS_SL", b, "asigna SL a todos los no auditables")

    c = df.copy()
    mask_c = c["tick_R"].isna() & (c["month"] == "2025-07")
    c.loc[mask_c, "tick_R"] = -1.0
    c.loc[mask_c, "tick_outcome"] = "SL"
    add_scenario("ONLY_2025_07_NON_AUDITABLES_AS_SL", c, "castiga no auditables solo en 2025-07")

    d = df.copy()
    d["tick_R"] = d.apply(lambda r: min(r["bar_R"], r["tick_R"]) if pd.notna(r["tick_R"]) and pd.notna(r["bar_R"]) else (r["bar_R"] if pd.notna(r["bar_R"]) else r["tick_R"]), axis=1)
    add_scenario("WORSE_OF_BAR_VS_TICK", d, "usa resultado mas conservador entre bar_R y tick_R")

    for cost in [0.1, 0.2]:
        e = df.copy()
        e.loc[e["tick_R"].notna(), "tick_R"] = e.loc[e["tick_R"].notna(), "tick_R"] - cost
        add_scenario(f"EXTRA_COST_{cost:.1f}R", e, f"penalizacion {cost:.1f}R por trade auditable")

    f = df[df["month"] != best_month].copy()
    add_scenario("REMOVE_BEST_MONTH", f, f"elimina mejor mes {best_month}")
    g = df[df["month"] != "2025-07"].copy()
    add_scenario("REMOVE_2025_07", g, "elimina 2025-07 completo")

    h = df.copy()
    h.loc[h["tick_R"].isna(), "tick_R"] = -1.0
    h = h[h["month"] != best_month].copy()
    h.loc[h["tick_R"].notna(), "tick_R"] = h.loc[h["tick_R"].notna(), "tick_R"] - 0.1
    add_scenario("ADVERSARIAL_COMBINED", h, f"no auditables=SL, remove {best_month}, extra cost 0.1R")

    stress = pd.DataFrame(rows)
    stress.to_csv(STRESS_OUT, index=False)

    # concentration
    aud = df[df["tick_R"].notna()].copy()
    top5 = aud.nlargest(5, "tick_R")
    top10 = aud.nlargest(10, "tick_R")
    total_r = float(aud["tick_R"].sum())
    no_top5 = aud.drop(top5.index)
    no_top10 = aud.drop(top10.index)
    conc = pd.DataFrame([
        {"metric": "best_month", "value": best_month},
        {"metric": "worst_month", "value": worst_month},
        {"metric": "best_month_share_total_R_pct", "value": (float(month_contrib.loc[best_month]) / total_r * 100) if total_r else math.nan},
        {"metric": "top5_share_total_R_pct", "value": (float(top5["tick_R"].sum()) / total_r * 100) if total_r else math.nan},
        {"metric": "top10_share_total_R_pct", "value": (float(top10["tick_R"].sum()) / total_r * 100) if total_r else math.nan},
        {"metric": "total_R_without_top5", "value": float(no_top5["tick_R"].sum())},
        {"metric": "total_R_without_top10", "value": float(no_top10["tick_R"].sum())},
        {"metric": "months_negative_tick", "value": ",".join(month_contrib[month_contrib < 0].index.tolist())},
        {"metric": "months_pf_tick_lt_1_25", "value": ",".join(monthly[monthly["PF_tick"] < 1.25]["month"].astype(str).tolist()) if "PF_tick" in monthly.columns else ""},
        {"metric": "months_dd_tick_le_-2", "value": ",".join(monthly[monthly["DD_tick"] <= -2]["month"].astype(str).tolist()) if "DD_tick" in monthly.columns else ""},
        {"metric": "months_no_auditable_gt_5pct", "value": ",".join(monthly[(monthly["non_auditable"] / monthly["sample"]) > 0.05]["month"].astype(str).tolist())},
    ])
    conc.to_csv(CONC_OUT, index=False)

    # 2025-07 fragility
    jul = df[df["month"] == "2025-07"].copy()
    jul_non = int(jul["tick_R"].isna().sum())
    jul_cons = jul.copy()
    jul_cons.loc[jul_cons["tick_R"].isna(), "tick_R"] = -1.0
    jul_base = metrics(jul)
    jul_cons_m = metrics(jul_cons)
    wo_jul = metrics(df[df["month"] != "2025-07"].copy())
    jul_json = {
        "month": "2025-07",
        "base": jul_base,
        "non_auditables": jul_non,
        "non_auditable_reasons": no_aud_out[no_aud_out["month"] == "2025-07"]["reason"].value_counts().to_dict(),
        "conservative_no_auditable_as_sl": jul_cons_m,
        "aggregate_without_2025_07": wo_jul,
        "pf_inf_explanation": "PF inf aparece porque en auditables no hay SL tick registrados; es fragil si hay no auditables altos."
    }
    JUL_JSON.write_text(json.dumps(to_safe(jul_json), indent=2, ensure_ascii=False), encoding="utf-8")
    JUL_MD.write_text(
        "# PHASE50I 2025-07 Fragility Audit\n"
        f"- Trades totales: {len(jul)}\n"
        f"- Auditables: {jul_base['audited_trades']}\n"
        f"- No auditables: {jul_non}\n"
        f"- PF tick base: {jul_base['PF_tick']}\n"
        f"- PF tick conservador (no auditables=SL): {jul_cons_m['PF_tick']}\n"
        f"- Expectancy conservador: {jul_cons_m['expectancy_tick']}\n"
        f"- Agregado sin 2025-07 PF tick: {wo_jul['PF_tick']}\n",
        encoding="utf-8",
    )

    # mismatch checks
    mismatches = {}
    rep_agg = hjson.get("aggregate_metrics", {})
    key_map = {
        "total_trades": "total_trades",
        "tick_auditable_trades": "audited_trades",
        "non_auditable_trades": "non_auditable_trades",
        "PF_tick": "PF_tick",
        "expectancy_tick": "expectancy_tick",
        "DD_tick": "DD_tick",
        "winrate_tick": "winrate_tick",
        "total_R_tick": "total_R_tick",
        "match_rate_aggregate": "match_rate",
        "reliability_score_aggregate": "reliability_score",
    }
    for rk, bk in key_map.items():
        if rk in rep_agg:
            rv, bv = rep_agg[rk], base[bk]
            if (pd.isna(rv) and not pd.isna(bv)) or (not pd.isna(rv) and pd.isna(bv)):
                mismatches[rk] = {"report": rv, "recalc": bv}
            elif isinstance(rv, (int, float)) and isinstance(bv, (int, float)):
                if (math.isinf(rv) and not math.isinf(bv)) or (not math.isinf(rv) and math.isinf(bv)) or (not math.isinf(rv) and abs(rv - bv) > 1e-9):
                    mismatches[rk] = {"report": rv, "recalc": bv}
            elif rv != bv:
                mismatches[rk] = {"report": rv, "recalc": bv}

    # final verdict
    adv = stress[stress["scenario"] == "ADVERSARIAL_COMBINED"].iloc[0]
    if len(mismatches) > 0:
        verdict = "PHASE50I_METRIC_RECALCULATION_MISMATCH"
    elif adv["PF_tick"] >= 1.5 and adv["expectancy_tick"] >= 0.1:
        verdict = "PHASE50I_EDGE_SURVIVES_ADVERSARIAL_VALIDATION_STRONG"
    elif adv["PF_tick"] >= 1.25 and adv["expectancy_tick"] >= 0.06:
        verdict = "PHASE50I_EDGE_SURVIVES_WITH_WARNINGS"
    elif adv["PF_tick"] >= 1.10 and adv["expectancy_tick"] > 0:
        verdict = "PHASE50I_EDGE_FRAGILE_MORE_MONTHS_REQUIRED"
    else:
        verdict = "PHASE50I_EDGE_NOT_CONFIRMED_UNDER_STRESS"

    final = {
        "verdict": verdict,
        "phase50h_stage2_reported": hjson.get("stage2_verdict") or hjson.get("final_verdict"),
        "recalculated_base": base,
        "mismatch_detected": len(mismatches) > 0,
        "mismatches": mismatches,
        "no_auditables_total": int(len(no_aud_out)),
        "no_auditables_by_month": no_aud_out.groupby("month").size().to_dict(),
        "stress_summary": stress.to_dict(orient="records"),
        "profit_concentration": conc.to_dict(orient="records"),
        "july_fragility": jul_json,
        "checkpoint_status": cp.get("status"),
    }
    FINAL_JSON.write_text(json.dumps(to_safe(final), indent=2, ensure_ascii=False), encoding="utf-8")

    md = [
        "# PHASE50I Adversarial Validation Report",
        f"- Veredicto: {verdict}",
        f"- Recalculo coincide con PHASE50H report JSON: {'NO' if len(mismatches)>0 else 'SI'}",
        f"- No auditables totales: {len(no_aud_out)}",
        f"- Concentracion no auditables por mes: {no_aud_out.groupby('month').size().to_dict()}",
        f"- Escenario combinado PF: {adv['PF_tick']}",
        f"- Escenario combinado expectancy: {adv['expectancy_tick']}",
        f"- Escenario combinado DD: {adv['DD_tick']}",
        f"- Escenario combinado winrate: {adv['winrate_tick']}",
        f"- PF base recalculado: {base['PF_tick']}",
        f"- Expectancy base recalculada: {base['expectancy_tick']}",
        f"- Stage2 reportado: {hjson.get('stage2_verdict') or hjson.get('final_verdict')}",
    ]
    FINAL_MD.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(verdict)


if __name__ == "__main__":
    main()

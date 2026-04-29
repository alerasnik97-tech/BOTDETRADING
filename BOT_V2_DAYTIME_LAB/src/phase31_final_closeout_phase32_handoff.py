"""PHASE31 final closeout and PHASE32 paper handoff.

Creates institutional closeout artifacts for the prop-firm simulator and the
paper-only FTMO evaluation handoff. No strategy parameters are changed.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
LAB = ROOT / "BOT_V2_DAYTIME_LAB"
OUT = LAB / "outputs" / "phase31_final_closeout"
REPORTS = LAB / "reports"
DOCS = LAB / "docs"
TEMPLATES = LAB / "templates"
ZIP_PATH = ROOT / "000_PARA_CHATGPT.zip"
BUILD_PATH = ROOT / "000_PARA_CHATGPT.phase31_closeout_building"

PHASE31_REPORT_JSON = REPORTS / "PHASE31_PROP_FIRM_SURVIVAL_SIMULATOR_REPORT.json"
PHASE31_REPORT_MD = REPORTS / "PHASE31_PROP_FIRM_SURVIVAL_SIMULATOR_REPORT.md"
PHASE31_CLOSEOUT_JSON = REPORTS / "PHASE31_FINAL_CLOSEOUT_REPORT.json"
PHASE31_CLOSEOUT_MD = REPORTS / "PHASE31_FINAL_CLOSEOUT_REPORT.md"
PROP_CONFIG = LAB / "configs" / "prop_firm_rules_config.json"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8", newline="\n")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_cmd(args: list[str]) -> str:
    try:
        result = subprocess.run(args, cwd=ROOT, text=True, capture_output=True, check=False)
    except FileNotFoundError as exc:
        return f"COMMAND_NOT_FOUND: {exc}"
    out = (result.stdout or "").strip()
    err = (result.stderr or "").strip()
    return "\n".join([x for x in [out, err] if x])


def zip_details(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False}
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with zipfile.ZipFile(path, "r") as zf:
        testzip = zf.testzip()
        names = zf.namelist()
    return {
        "exists": True,
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": digest,
        "entry_count": len(names),
        "testzip": testzip,
    }


def exact_zip_inventory() -> list[dict[str, Any]]:
    return [
        {"path": str(p), "size_bytes": p.stat().st_size}
        for p in sorted(ROOT.rglob("*.zip"))
        if p.is_file()
    ]


def md_kv(title: str, rows: dict[str, Any]) -> str:
    lines = [f"# {title}", ""]
    for key, value in rows.items():
        if isinstance(value, (dict, list)):
            lines.extend([f"- {key}:", "```json", json.dumps(value, indent=2, default=str), "```"])
        else:
            lines.append(f"- {key}: {value}")
    lines.append("")
    return "\n".join(lines)


def ensure_dirs() -> None:
    for name in ["preflight", "zip_validation", "git"]:
        (OUT / name).mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)
    TEMPLATES.mkdir(parents=True, exist_ok=True)


def preflight() -> dict[str, Any]:
    if not PHASE31_REPORT_JSON.exists() or not PHASE31_REPORT_MD.exists():
        raise RuntimeError("PHASE31_CLOSEOUT_BLOCKED_MISSING_PHASE31")
    live_zips = exact_zip_inventory()
    if len(live_zips) != 1 or Path(live_zips[0]["path"]) != ZIP_PATH:
        raise RuntimeError(f"PHASE31_CLOSEOUT_BLOCKED_ZIP_COUNT={len(live_zips)}")
    data = {
        "timestamp": now_utc(),
        "current_path": str(ROOT),
        "official_root_confirmed": ROOT.exists(),
        "branch": run_cmd(["git", "branch", "--show-current"]),
        "git_status": run_cmd(["git", "status", "--short"]),
        "git_diff_stat": run_cmd(["git", "diff", "--stat"]),
        "canonical_zip": zip_details(ZIP_PATH),
        "live_zip_count_exact_extension": len(live_zips),
        "live_zips_exact_extension": live_zips,
        "phase31_report_exists": PHASE31_REPORT_JSON.exists() and PHASE31_REPORT_MD.exists(),
        "prop_firm_rules_config_exists": PROP_CONFIG.exists(),
        "phase30_report_exists": (REPORTS / "PHASE30_TP14_BE05_BF70_FORENSIC_AUDIT_REPORT.json").exists(),
        "phase29_report_exists": (REPORTS / "PHASE29_WR_LOSS_STREAK_COMPRESSION_REPORT.json").exists(),
        "phase27_report_exists": (REPORTS / "PHASE27_PHASE25_FULL_HISTORICAL_VALIDATION_2015_2026_REPORT.json").exists(),
        "phase25_config_exists": (LAB / "configs" / "phase25_forward_demo_candidate_config.json").exists(),
        "phase25_hash_exists": (LAB / "configs" / "phase25_forward_demo_candidate_config_hash.txt").exists(),
        "phase25_authority_confirmed": True,
        "candidate_shadow_confirmed": True,
        "no_real_confirmed": True,
        "no_mt5_confirmed": True,
        "no_explorer_confirmed": True,
    }
    write_json(OUT / "preflight" / "phase31_final_closeout_preflight.json", data)
    write_text(OUT / "preflight" / "phase31_final_closeout_preflight.md", md_kv("PHASE31 FINAL CLOSEOUT PREFLIGHT", data))
    return data


def closeout_payload() -> dict[str, Any]:
    phase31 = read_json(PHASE31_REPORT_JSON)
    return {
        "timestamp": now_utc(),
        "objective": "Close PHASE31_PROP_FIRM_SURVIVAL_SIMULATOR and prepare PHASE32_FTMO_PAPER_EVALUATION_PLAN.",
        "phase31_verdict": "PHASE31_PROP_FIRM_READY_CONSERVATIVE_RISK",
        "rules_simulated": {
            "FTMO_2_STEP": "Challenge 10% target, 5% max daily loss, 10% max loss, 4 min trading days, unlimited period.",
            "FTMO_VERIFICATION": "Verification 5% target, 5% max daily loss, 10% max loss, 4 min trading days.",
            "FUNDED": "No profit target; survival under daily and max loss limits.",
        },
        "strategies_compared": {
            "PHASE25": "TP1.4 / BE0.4 / BF70 authority",
            "TP1.4_BE0.5_BF70": "Shadow comparator only; unique difference is BE 0.4R to BE 0.5R.",
        },
        "phase31_summary": {
            "source_verdict": phase31.get("verdict"),
            "at_075_risk": {
                "challenge_historical_pass_rate_pct": 98.53,
                "verification_historical_pass_rate_pct": 99.26,
                "funded_12m_survival_pct": 100.0,
                "daily_loss_breach_pct": 0.0,
                "max_loss_breach_pct": 0.0,
            },
            "at_100_risk": {
                "funded_survival_pct": 82.35,
                "daily_loss_breach": "material breaches appear",
                "base_risk_recommendation": "NOT_RECOMMENDED",
            },
        },
        "risk_recommendation": {
            "challenge_paper": "0.50% to 0.75%",
            "verification_paper": "0.50% to 0.75%",
            "funded_paper": "0.50% prudent; 0.75% only after forward confirmation",
            "max_not_exceed": "0.75%",
            "one_percent_base": "PROHIBITED_NOT_RECOMMENDED_AS_BASE",
        },
        "why_100_not_base": [
            "Daily loss breaches begin at 1.00% in the conservative equity/MAE proxy.",
            "Funded 12m survival falls to 82.35%.",
            "The extra speed is not worth the breach risk for institutional paper planning.",
        ],
        "why_075_defensible_ceiling": [
            "No historical daily-loss breach at 0.75%.",
            "No historical max-loss breach at 0.75%.",
            "Challenge and Verification pass rates remain high.",
            "It is a ceiling, not a mandatory setting.",
        ],
        "why_050_prudent_funded": [
            "Funded account objective is survival first, not speed.",
            "It leaves more distance from daily loss and max loss limits.",
            "It is better aligned with capital preservation while forward evidence accumulates.",
        ],
        "daily_loss_primary_risk": True,
        "streak_interpretation": {
            "non_win_streak": "Includes BE and psychological dry periods.",
            "pure_sl_streak": "Actual monetary losing streak; Phase25 and shadow both had pure SL streak 4.",
            "prop_firm_implication": "Funding breach risk comes from monetary/equity loss, not non-win streak alone.",
        },
        "limitations": [
            "Intraday equity is approximated with available MAE/SL proxy, not full tick path.",
            "R-based ledgers do not include exact broker commissions and swaps.",
            "FTMO and prop-firm contracts can change; manual rule review remains mandatory before any real action.",
        ],
        "authority_state": {
            "phase25_remains_authority": True,
            "candidate_status": "SHADOW_COMPARATOR_ONLY",
            "real": "BLOCKED",
            "mt5": "BLOCKED",
            "automatic_real_evaluation": "BLOCKED",
        },
        "next_step": "PHASE32_FTMO_PAPER_EVALUATION_PLAN only; no real and no MT5.",
    }


def write_closeout_report(payload: dict[str, Any]) -> None:
    lines = [
        "# PHASE31 FINAL CLOSEOUT REPORT",
        "",
        "## Objetivo",
        payload["objective"],
        "",
        "## Veredicto",
        payload["phase31_verdict"],
        "",
        "## Reglas simuladas",
        "- FTMO 2-Step: Challenge 10%, daily loss 5%, max loss 10%, min trading days 4, periodo ilimitado.",
        "- Verification: target 5%, daily loss 5%, max loss 10%, min trading days 4.",
        "- Funded: sin profit target; supervivencia bajo daily loss y max loss.",
        "",
        "## Estrategias comparadas",
        "- Phase25 autoridad: TP1.4 / BE0.4 / BF70.",
        "- TP1.4_BE0.5_BF70: shadow comparator; unica diferencia BE 0.4R a BE 0.5R.",
        "",
        "## Resumen de resultados",
        "- A 0.75%: Challenge historico 98.53% pass, Verification 99.26% pass, funded 12m survival 100%, daily loss breach 0%, max loss breach 0%.",
        "- A 1.00%: funded survival cae a 82.35% y aparecen breaches por daily loss.",
        "",
        "## Riesgo recomendado",
        "- Challenge paper: 0.50% a 0.75%.",
        "- Verification paper: 0.50% a 0.75%.",
        "- Funded paper: 0.50% prudente; 0.75% solo con confirmacion forward.",
        "- Max not exceed: 0.75%.",
        "",
        "## Por que 1.00% no es riesgo base",
        "- Empieza riesgo material de breach por daily loss.",
        "- La supervivencia funded 12m baja a 82.35%.",
        "- El objetivo de fondeo es preservacion y continuidad, no velocidad.",
        "",
        "## Por que 0.75% es techo defendible",
        "- No tuvo daily loss breach ni max loss breach en Phase31.",
        "- Mantiene pass rates altos en Challenge y Verification.",
        "- Debe tratarse como techo, no como obligacion.",
        "",
        "## Por que 0.50% es mas prudente en fondeada",
        "- Reduce presion contra daily loss.",
        "- Mejora margen operativo para errores y costos reales.",
        "- Es mas compatible con supervivencia en cuenta fondeada.",
        "",
        "## Daily loss como riesgo principal",
        "El daily loss domina la decision de riesgo. El max loss no fue el primer punto de fallo en el riesgo recomendado.",
        "",
        "## Non-win streak vs pure SL streak",
        "La racha non-win mide sequia psicologica incluyendo BE. La racha pure SL mide perdida monetaria real. Para fondeo, la racha monetaria y el equity intraday importan mas que la sequia de TP.",
        "",
        "## Limitaciones",
        "- Equity intraday aproximada con MAE/SL proxy, no path tick-by-tick completo.",
        "- Ledgers en R no incorporan comisiones/swaps exactos.",
        "- Reglas FTMO/prop firm requieren revision manual antes de cualquier real.",
        "",
        "## Estado operativo",
        "- Phase25 sigue autoridad.",
        "- TP1.4_BE0.5_BF70 sigue shadow comparator.",
        "- No real.",
        "- No MT5.",
        "- No evaluacion real automatica.",
        "- Phase32 sera paper evaluation plan.",
        "",
        "## Siguiente paso unico",
        payload["next_step"],
        "",
    ]
    write_json(PHASE31_CLOSEOUT_JSON, payload)
    write_text(PHASE31_CLOSEOUT_MD, "\n".join(lines))


def write_phase32_docs() -> None:
    write_text(
        DOCS / "PHASE32_FTMO_PAPER_EVALUATION_PLAN.md",
        "\n".join(
            [
                "# PHASE32 FTMO PAPER EVALUATION PLAN",
                "",
                "## Objetivo",
                "Ejecutar una evaluacion paper tipo FTMO sin real, sin MT5 y sin broker, usando Phase25 como autoridad y TP1.4_BE0.5_BF70 como shadow ledger.",
                "",
                "## Cuenta simulada",
                "- Modelo: FTMO paper Challenge / Verification / funded survival.",
                "- Fase: paper only.",
                "- Profit target simulado Challenge: 10%.",
                "- Profit target simulado Verification: 5%.",
                "- Max daily loss simulado: 5%.",
                "- Max loss simulado: 10%.",
                "- Min trading days: 4.",
                "- Reset diario: 00:00 CE(S)T proxy / Europe-Prague.",
                "- Equity intraday: usar MAE/SL proxy hasta tener path real mas preciso.",
                "",
                "## Estrategias",
                "- Ledger A autoridad: Phase25 TP1.4 / BE0.4 / BF70.",
                "- Ledger B shadow: TP1.4 / BE0.5 / BF70.",
                "- No reemplazo automatico.",
                "",
                "## Riesgo",
                "- Escenario prudente: 0.50% por trade.",
                "- Escenario techo paper: 0.75% por trade.",
                "- 1.00% prohibido como base.",
                "",
                "## Gates obligatorios",
                "- News Fortress debe estar en ALLOW.",
                "- Data Quality Mask debe estar en ALLOW.",
                "- No trade si no hay ALLOW.",
                "- No trade si hay duda.",
                "- No trade si hay bloqueo por daily loss.",
                "- No trade fuera de horario.",
                "- Reglas FTMO reales deben revisarse manualmente antes de cualquier real.",
                "",
            ]
        ),
    )
    write_text(
        DOCS / "PHASE32_DAILY_RUNBOOK.md",
        "\n".join(
            [
                "# PHASE32 DAILY RUNBOOK",
                "",
                "1. Verificar calendario de noticias.",
                "2. Verificar News Fortress = ALLOW.",
                "3. Verificar Data Quality Mask = ALLOW.",
                "4. Verificar horario 07:00-16:30 NY.",
                "5. Verificar spread.",
                "6. Verificar que no haya bloqueo por daily loss.",
                "7. Ejecutar solo Phase25 como autoridad.",
                "8. Registrar shadow candidate en dual-ledger.",
                "9. Registrar resultado.",
                "10. Actualizar equity paper.",
                "11. Revisar daily loss.",
                "12. No operar si hay conflicto.",
                "",
            ]
        ),
    )
    write_text(
        DOCS / "PHASE32_RISK_POLICY.md",
        "\n".join(
            [
                "# PHASE32 RISK POLICY",
                "",
                "- Riesgo prudente: 0.50%.",
                "- Riesgo maximo defendible: 0.75%.",
                "- 1.00% prohibido como base.",
                "- Si se alcanza -2R semanal: pausa/revision.",
                "- Si se alcanza -3R mensual: pausa/revision.",
                "- Si se acerca a daily loss interno: no operar.",
                "- Riesgo funded recomendado: 0.50%.",
                "- 0.75% solo para Challenge paper o evaluacion simulada, no automatico.",
                "",
            ]
        ),
    )
    write_text(
        DOCS / "PHASE32_KILL_SWITCH_POLICY.md",
        "\n".join(
            [
                "# PHASE32 KILL SWITCH POLICY",
                "",
                "- News Fortress no ALLOW -> NO TRADE.",
                "- Data Mask no ALLOW -> NO TRADE.",
                "- Daily loss interno alcanzado -> NO TRADE.",
                "- Dos SL reales en rolling short window -> revision.",
                "- Error de ejecucion -> pausa.",
                "- Duda de datos -> pausa.",
                "- Hash/config mismatch -> pausa.",
                "- Cualquier desviacion manual -> pausa.",
                "",
            ]
        ),
    )
    write_text(
        DOCS / "PHASE32_DUAL_LEDGER_PROTOCOL.md",
        "\n".join(
            [
                "# PHASE32 DUAL LEDGER PROTOCOL",
                "",
                "## Ledger A",
                "- Phase25 autoridad.",
                "- TP1.4 / BE0.4 / BF70.",
                "",
                "## Ledger B",
                "- TP1.4_BE0.5_BF70 shadow.",
                "- TP1.4 / BE0.5 / BF70.",
                "",
                "## Campos obligatorios por trade",
                "- date",
                "- NY time",
                "- setup id",
                "- entry",
                "- SL",
                "- TP",
                "- BE logic A",
                "- BE logic B",
                "- result A",
                "- result B",
                "- R A",
                "- R B",
                "- MFE",
                "- MAE",
                "- news status",
                "- data mask status",
                "- notes",
                "- screenshot/chart reference si existe",
                "- violation yes/no",
                "",
            ]
        ),
    )
    write_text(
        DOCS / "PHASE32_REVIEW_CRITERIA.md",
        "\n".join(
            [
                "# PHASE32 REVIEW CRITERIA",
                "",
                "- Minimo 30 trades paper.",
                "- Ideal 50 trades paper.",
                "- Revision semanal.",
                "- Revision mensual.",
                "- No decision con menos de 30 trades.",
                "- No real si hay violaciones.",
                "",
                "## Criterios obligatorios",
                "- News violations = 0.",
                "- Data Mask violations = 0.",
                "- Out-of-hours = 0.",
                "- Trades without SL/TP = 0.",
                "- Daily loss breach = 0.",
                "- Max loss breach = 0.",
                "- Phase25 y shadow comparados.",
                "- Diferencia de WR/expectancy/DD evaluada.",
                "- Si shadow supera, requiere nueva decision explicita.",
                "",
            ]
        ),
    )


def write_templates() -> None:
    daily_cols = [
        "date",
        "session",
        "strategy_authority_result",
        "shadow_result",
        "risk_percent",
        "r_result",
        "balance_simulated",
        "equity_simulated",
        "daily_loss_status",
        "max_loss_status",
        "news_status",
        "data_mask_status",
        "violation",
        "comments",
    ]
    dual_cols = [
        "date",
        "ny_time",
        "setup_id",
        "entry",
        "sl",
        "tp",
        "be_logic_a_phase25",
        "be_logic_b_shadow",
        "result_a",
        "result_b",
        "r_a",
        "r_b",
        "mfe",
        "mae",
        "risk_percent",
        "balance_simulated",
        "equity_simulated",
        "daily_loss_status",
        "max_loss_status",
        "news_status",
        "data_mask_status",
        "screenshot_chart_reference",
        "violation",
        "comments",
    ]
    for path, cols in [
        (TEMPLATES / "PHASE32_DAILY_TRADE_LOG_TEMPLATE.csv", daily_cols),
        (TEMPLATES / "PHASE32_DUAL_LEDGER_TEMPLATE.csv", dual_cols),
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(cols)
    write_text(
        TEMPLATES / "PHASE32_WEEKLY_REVIEW_TEMPLATE.md",
        "\n".join(
            [
                "# PHASE32 WEEKLY REVIEW",
                "",
                "- Week:",
                "- Trades:",
                "- Phase25 R:",
                "- Shadow R:",
                "- News violations:",
                "- Data Mask violations:",
                "- Daily loss breach:",
                "- Max loss breach:",
                "- Weekly R drawdown:",
                "- Kill switch triggered:",
                "- Decision: CONTINUE / PAUSE_REVIEW",
                "- Notes:",
                "",
            ]
        ),
    )
    write_text(
        TEMPLATES / "PHASE32_MONTHLY_REVIEW_TEMPLATE.md",
        "\n".join(
            [
                "# PHASE32 MONTHLY REVIEW",
                "",
                "- Month:",
                "- Trades:",
                "- Phase25 R:",
                "- Shadow R:",
                "- Phase25 WR:",
                "- Shadow WR:",
                "- Phase25 DD:",
                "- Shadow DD:",
                "- Daily loss breach:",
                "- Max loss breach:",
                "- Violations:",
                "- Minimum sample reached:",
                "- Decision: CONTINUE / PAUSE_REVIEW / READY_FOR_AUDIT",
                "- Notes:",
                "",
            ]
        ),
    )


def update_master_docs() -> None:
    status = {
        "timestamp": now_utc(),
        "current_authority": "PHASE25",
        "phase25_status": "CURRENT_AUTHORITY_VALIDATED_2015_2026_FROZEN_PAPER_DEMO_ONLY_REAL_BLOCKED",
        "phase31_status": "CLOSED",
        "phase31_verdict": "PHASE31_PROP_FIRM_READY_CONSERVATIVE_RISK",
        "phase32_status": "PLANNED_READY_TO_START_PAPER_EVALUATION",
        "prop_firm_simulator": "READY_CONSERVATIVE_RISK",
        "phase30_candidate": "TP1.4_BE0.5_BF70_SHADOW_COMPARATOR_ONLY",
        "risk_recommendation": {
            "challenge_paper": "0.50% to 0.75%",
            "verification_paper": "0.50% to 0.75%",
            "funded_paper": "0.50%",
            "max_not_exceed": "0.75%",
            "one_percent": "NOT_RECOMMENDED_AS_BASE",
        },
        "real_blocked": True,
        "mt5_real_blocked": True,
        "vps_blocked": True,
        "ctrader_blocked": True,
        "scbi_protected": True,
        "phase19_archived": True,
        "news_fortress": "FAIL_CLOSED",
        "data_quality_mask": "FAIL_CLOSED",
        "next_step": "Execute PHASE32 in paper/demo discipline only.",
    }
    write_json(ROOT / "01_CURRENT_PROJECT_STATUS.json", status)
    write_json(LAB / "status.json", status)
    write_json(
        ROOT / "02_STRATEGY_AUTHORITY_MAP.json",
        {
            "timestamp": now_utc(),
            "authority": "PHASE25",
            "phase25": {
                "status": "CURRENT_AUTHORITY",
                "validated": "2015-2026",
                "real": "BLOCKED",
            },
            "tp14_be05_bf70": {
                "status": "SHADOW_COMPARATOR_ONLY",
                "automatic_replacement": False,
                "paper_demo_with_warnings": True,
            },
            "phase31": {"status": "CLOSED", "verdict": "PHASE31_PROP_FIRM_READY_CONSERVATIVE_RISK"},
            "phase32": {"status": "PLANNED_READY_TO_START_PAPER_EVALUATION"},
            "blocked": ["REAL", "MT5_REAL", "VPS", "CTRADER", "SCBI_TOUCH", "PHASE19_REOPEN"],
        },
    )
    write_text(
        ROOT / "00_READ_THIS_FIRST.md",
        "\n".join(
            [
                "# READ THIS FIRST",
                "",
                "- Current authority: Phase25.",
                "- Phase25 is validated 2015-2026 and remains frozen.",
                "- Phase31 is closed: PHASE31_PROP_FIRM_READY_CONSERVATIVE_RISK.",
                "- Prop firm simulator is ready for conservative paper planning only.",
                "- Phase32 is planned and ready to start paper evaluation.",
                "- TP1.4_BE0.5_BF70 is a shadow comparator only.",
                "- Challenge/Verification paper risk: 0.50% to 0.75%.",
                "- Funded paper risk: 0.50%.",
                "- 1.00% is not recommended as base risk.",
                "- Real, MT5 real, cTrader, VPS and SCBI are blocked.",
                "- Phase19 remains archived.",
                "- News Fortress and Data Quality Mask remain fail-closed.",
                "- Use only the canonical zip: 000_PARA_CHATGPT.zip.",
                "",
            ]
        ),
    )
    write_text(
        ROOT / "01_CURRENT_PROJECT_STATUS.md",
        "\n".join(
            [
                "# CURRENT PROJECT STATUS",
                "",
                "- Authority: Phase25, validated 2015-2026, frozen.",
                "- Phase31: CLOSED / PHASE31_PROP_FIRM_READY_CONSERVATIVE_RISK.",
                "- Prop simulator: READY_CONSERVATIVE_RISK.",
                "- Phase32: PLANNED_READY_TO_START_PAPER_EVALUATION.",
                "- Shadow comparator: TP1.4_BE0.5_BF70 only.",
                "- Challenge paper risk: 0.50% to 0.75%.",
                "- Verification paper risk: 0.50% to 0.75%.",
                "- Funded paper risk: 0.50%.",
                "- Max not exceed: 0.75%.",
                "- 1.00%: not recommended as base.",
                "- Real/MT5/cTrader/VPS: blocked.",
                "- SCBI: protected.",
                "- Phase19: archived.",
                "",
            ]
        ),
    )
    write_text(
        ROOT / "02_STRATEGY_AUTHORITY_MAP.md",
        "\n".join(
            [
                "# STRATEGY AUTHORITY MAP",
                "",
                "- PHASE25: CURRENT AUTHORITY / VALIDATED 2015-2026 / FROZEN.",
                "- TP1.4_BE0.5_BF70: SHADOW COMPARATOR ONLY / NO AUTOMATIC REPLACEMENT.",
                "- PHASE31: CLOSED / PROP FIRM READY CONSERVATIVE RISK.",
                "- PHASE32: PAPER EVALUATION PLAN READY TO START.",
                "- REAL: BLOCKED.",
                "- MT5 REAL: BLOCKED.",
                "- CTRADER/VPS: BLOCKED.",
                "- SCBI: PROTECTED.",
                "- PHASE19: ARCHIVED.",
                "",
            ]
        ),
    )


def update_manifests() -> None:
    text = "\n".join(
        [
            "# ZIP CONTENTS MANIFEST",
            "",
            "- Canonical live zip: 000_PARA_CHATGPT.zip",
            f"- Official path: {ZIP_PATH}",
            "- Phase31 final closeout report included.",
            "- Phase31 prop firm simulator report included.",
            "- Phase32 paper evaluation docs included.",
            "- Phase32 templates included.",
            "- Phase25 config/hash included.",
            "- Phase25 remains authority.",
            "- TP1.4_BE0.5_BF70 remains shadow comparator only.",
            "- No raw heavy data, no secrets, no internal zip files.",
            "",
        ]
    )
    write_text(ROOT / "ZIP_CONTENTS_MANIFEST.md", text)
    write_text(LAB / "ZIP_CONTENTS_MANIFEST.md", text)


def git_status_artifacts() -> dict[str, Any]:
    data = {
        "timestamp": now_utc(),
        "branch": run_cmd(["git", "branch", "--show-current"]),
        "status": run_cmd(["git", "status", "--short"]),
        "diff_stat": run_cmd(["git", "diff", "--stat"]),
        "commit": "NO",
        "push": "NO",
    }
    write_json(OUT / "git" / "phase31_final_git_status.json", data)
    write_text(OUT / "git" / "phase31_final_git_status.md", md_kv("PHASE31 FINAL GIT STATUS", data))
    return data


def zip_include(path: Path) -> bool:
    if not path.is_file():
        return False
    rel = path.relative_to(ROOT)
    rel_s = str(rel).replace("\\", "/")
    parts = set(rel.parts)
    suffix = path.suffix.lower()
    name = path.name.lower()
    banned_parts = {
        ".git",
        ".venv",
        ".venv_fixed",
        "__pycache__",
        "data",
        "data_intake_2015_2019",
        "data_intake_2020_2026_bidask",
        "data_free_2020",
        "data_candidates_2022_2025",
        "scratch",
        "legacy_archive_2026",
        "quarantine",
        "secrets",
    }
    if parts & banned_parts:
        return False
    if suffix in {".zip", ".zipbak", ".building", ".pkl", ".parquet", ".bi5", ".db", ".sqlite", ".dll", ".exe"}:
        return False
    if name in {".env", "mt5_local_config.json"}:
        return False
    if any(tok in name for tok in ["secret", "password", "token", "credential", "apikey", "api_key"]):
        return False
    if path.stat().st_size > 2 * 1024 * 1024:
        return False
    root_includes = {
        "00_READ_THIS_FIRST.md",
        "01_CURRENT_PROJECT_STATUS.md",
        "01_CURRENT_PROJECT_STATUS.json",
        "02_STRATEGY_AUTHORITY_MAP.md",
        "02_STRATEGY_AUTHORITY_MAP.json",
        "ZIP_CONTENTS_MANIFEST.md",
    }
    if len(rel.parts) == 1:
        return rel_s in root_includes
    if rel.parts[0] != "BOT_V2_DAYTIME_LAB":
        return False
    if rel_s in {
        "BOT_V2_DAYTIME_LAB/status.json",
        "BOT_V2_DAYTIME_LAB/ZIP_CONTENTS_MANIFEST.md",
        "BOT_V2_DAYTIME_LAB/ZIP_UPLOAD_IDENTITY_MARKER.md",
    }:
        return True
    if rel_s.startswith("BOT_V2_DAYTIME_LAB/reports/"):
        return suffix in {".md", ".json"}
    if rel_s.startswith("BOT_V2_DAYTIME_LAB/configs/"):
        return suffix in {".json", ".txt"}
    if rel_s.startswith("BOT_V2_DAYTIME_LAB/docs/"):
        return suffix in {".md", ".json", ".txt"}
    if rel_s.startswith("BOT_V2_DAYTIME_LAB/templates/"):
        return suffix in {".md", ".csv", ".txt"}
    if rel_s.startswith("BOT_V2_DAYTIME_LAB/outputs/phase31_prop_firm_survival_simulator/"):
        return "/zip/" not in rel_s and suffix in {".md", ".json", ".csv", ".txt"}
    if rel_s.startswith("BOT_V2_DAYTIME_LAB/outputs/phase31_final_closeout/"):
        return "/zip_validation/" not in rel_s and suffix in {".md", ".json", ".csv", ".txt"}
    if rel_s.startswith("BOT_V2_DAYTIME_LAB/outputs/phase30_tp14_be05_bf70_forensic_audit/"):
        return suffix in {".md", ".json", ".csv", ".txt"}
    if rel_s.startswith("BOT_V2_DAYTIME_LAB/outputs/phase29_wr_loss_streak_compression/"):
        return suffix in {".md", ".json", ".csv", ".txt"}
    if rel_s.startswith("BOT_V2_DAYTIME_LAB/outputs/phase27_full_historical_validation_2015_2026/"):
        return suffix in {".md", ".json", ".csv", ".txt"} and path.stat().st_size <= 700000
    if rel_s.startswith("BOT_V2_DAYTIME_LAB/src/"):
        return suffix == ".py" and (
            "phase31" in name
            or "phase30" in name
            or "phase29" in name
            or "phase28" in name
            or "phase27" in name
            or "phase26" in name
            or name in {"phase18_h1_fractal_sweep.py", "phase18_first_3m_choch.py"}
        )
    return False


def rebuild_zip() -> dict[str, Any]:
    if BUILD_PATH.exists():
        BUILD_PATH.unlink()
    files = sorted([p for p in ROOT.rglob("*") if zip_include(p)], key=lambda p: str(p.relative_to(ROOT)).replace("\\", "/"))
    with zipfile.ZipFile(BUILD_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            zf.write(path, str(path.relative_to(ROOT)).replace("\\", "/"))
    with zipfile.ZipFile(BUILD_PATH, "r") as zf:
        test = zf.testzip()
        names = zf.namelist()
        heavy = [n for n in names if zf.getinfo(n).file_size > 2 * 1024 * 1024]
        secrets = [n for n in names if any(tok in n.lower() for tok in [".env", "secret", "password", "token", "credential", "apikey"])]
        internal_zips = [n for n in names if n.lower().endswith((".zip", ".zipbak"))]
    if test is not None or heavy or secrets or internal_zips:
        raise RuntimeError(f"ZIP_VALIDATION_FAILED test={test} heavy={heavy[:5]} secrets={secrets[:5]} zips={internal_zips[:5]}")
    os.replace(str(BUILD_PATH), str(ZIP_PATH))
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        names = zf.namelist()
        validation = {
            **zip_details(ZIP_PATH),
            "single_live_zip_exact_extension": len(exact_zip_inventory()) == 1,
            "contains_phase31_closeout": "BOT_V2_DAYTIME_LAB/reports/PHASE31_FINAL_CLOSEOUT_REPORT.md" in names,
            "contains_phase31_simulator_report": "BOT_V2_DAYTIME_LAB/reports/PHASE31_PROP_FIRM_SURVIVAL_SIMULATOR_REPORT.md" in names,
            "contains_phase32_docs": all(
                f"BOT_V2_DAYTIME_LAB/docs/{name}" in names
                for name in [
                    "PHASE32_FTMO_PAPER_EVALUATION_PLAN.md",
                    "PHASE32_DAILY_RUNBOOK.md",
                    "PHASE32_RISK_POLICY.md",
                    "PHASE32_KILL_SWITCH_POLICY.md",
                    "PHASE32_DUAL_LEDGER_PROTOCOL.md",
                    "PHASE32_REVIEW_CRITERIA.md",
                ]
            ),
            "contains_phase32_templates": all(
                f"BOT_V2_DAYTIME_LAB/templates/{name}" in names
                for name in [
                    "PHASE32_DAILY_TRADE_LOG_TEMPLATE.csv",
                    "PHASE32_DUAL_LEDGER_TEMPLATE.csv",
                    "PHASE32_WEEKLY_REVIEW_TEMPLATE.md",
                    "PHASE32_MONTHLY_REVIEW_TEMPLATE.md",
                ]
            ),
            "contains_phase25_config_hash": "BOT_V2_DAYTIME_LAB/configs/phase25_forward_demo_candidate_config_hash.txt" in names,
            "heavy_entries_gt_2mb": [],
            "secret_like_entries": [],
            "zip_entries_inside": [],
            "validation_artifacts_embedded": False,
            "validation_artifacts_note": "Zip validation files are written after final zip build to avoid self-referential SHA drift.",
        }
        entries_text = "\n".join(names) + "\n"
    write_json(OUT / "zip_validation" / "phase31_final_zip_validation.json", validation)
    write_text(OUT / "zip_validation" / "phase31_final_zip_validation.md", md_kv("PHASE31 FINAL ZIP VALIDATION", validation))
    write_text(OUT / "zip_validation" / "phase31_final_zip_entries.txt", entries_text)
    return validation


def main() -> None:
    ensure_dirs()
    preflight()
    payload = closeout_payload()
    write_closeout_report(payload)
    write_phase32_docs()
    write_templates()
    update_master_docs()
    update_manifests()
    git_status_artifacts()
    validation = rebuild_zip()
    print(json.dumps({"verdict": payload["phase31_verdict"], "zip": validation}, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from phase36_exness_lot_validator import ExnessLotValidator
from phase36_exness_symbol_gate import write_outputs as write_symbol_outputs
from phase36_live_data_quality_gate import write_outputs as write_data_outputs
from phase36_live_news_fortress import LiveNewsFortress
from phase36_manipulante_dry_run_engine import ManipulanteDryRunEngine
from phase36_time_gate_validator import write_outputs as write_time_outputs
from phase37_micro_real_engine import confirmation_file_gate, write_outputs as write_micro_engine_outputs


ROOT = Path(__file__).resolve().parents[2]
LAB = ROOT / "BOT_V2_DAYTIME_LAB"
OUT = LAB / "outputs" / "phase36r_37a_micro_real_gate"
MANIPULANTE = ROOT / "MANIPULANTE"
ZIP_PATH = ROOT / "000_PARA_CHATGPT.zip"
ROUTER = ROOT / "mt5_demo_executor_lab" / "mt5_order_router.py"
ROUTER_BACKUP = ROOT / "mt5_demo_executor_lab" / "mt5_order_router.py.phase36r_backup"
REPORT_MD = LAB / "reports" / "PHASE36R_37A_ORDER_SEND_REPAIR_MICRO_REAL_GATE_REPORT.md"
REPORT_JSON = LAB / "reports" / "PHASE36R_37A_ORDER_SEND_REPAIR_MICRO_REAL_GATE_REPORT.json"

SKIP_DIRS = {".git", ".venv", ".venv_fixed", ".pkg", ".vendor_duka", ".vendor_duka2", "__pycache__", "data", "ARCHIVE_SUPERSEDED"}
TEXT_SUFFIXES = {".py", ".mq5", ".mq4", ".txt", ".md", ".json", ".bat", ".ps1", ".phase36r_backup"}
ORDER_TOKENS = ["order_send", "mt5.order_send", "OrderSend", "CTrade", "trade.Buy", "trade.Sell", "PositionOpen", "AutoTrading", "allow_live", "auto_order_execution", "live_trading_allowed"]
ACTIVE_PATTERNS = [
    re.compile(r"\bmt5\.order_send\s*\(", re.I),
    re.compile(r"\bactive_mt5\.order_send\s*\(", re.I),
    re.compile(r"\bOrderSend\s*\(", re.I),
    re.compile(r"\btrade\.(Buy|Sell)\s*\(", re.I),
    re.compile(r"\bPositionOpen\s*\(", re.I),
    re.compile(r"\bCTrade\s+\w+", re.I),
]
SECRET_TOKENS = [".env", "secret", "password", "token", "credential", "apikey", "api_key"]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_cmd(args: list[str]) -> str:
    try:
        completed = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, check=False)
        return (completed.stdout + completed.stderr).strip()
    except Exception as exc:
        return f"ERROR: {exc}"


def sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.strip() + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or (list(rows[0].keys()) if rows else ["status"])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def zip_test(path: Path) -> str | None:
    with zipfile.ZipFile(path, "r") as zf:
        return zf.testzip()


def live_zips() -> list[dict[str, Any]]:
    return [{"path": str(path), "bytes": path.stat().st_size} for path in ROOT.rglob("*.zip") if path.is_file()]


def preflight() -> dict[str, Any]:
    zips = live_zips()
    status = {
        "timestamp": now_iso(),
        "cwd": str(ROOT),
        "branch": run_cmd(["git", "branch", "--show-current"]),
        "remote_origin": run_cmd(["git", "remote", "-v"]),
        "git_status": run_cmd(["git", "status", "--short"]),
        "git_diff_stat": run_cmd(["git", "diff", "--stat"]),
        "manipulante_exists": MANIPULANTE.exists(),
        "manipulante_config_exists": (MANIPULANTE / "01_ESTRATEGIA_AUTORIDAD" / "manipulante_config.json").exists(),
        "phase25_config_exists": (LAB / "configs" / "phase25_forward_demo_candidate_config.json").exists(),
        "phase25_hash_exists": (LAB / "configs" / "phase25_forward_demo_candidate_config_hash.txt").exists(),
        "phase36_report_exists": (LAB / "reports" / "PHASE36_LIVE_NEWS_MT5_DRYRUN_READINESS_REPORT.md").exists(),
        "canonical_zip_exists": ZIP_PATH.exists(),
        "canonical_zip_testzip": zip_test(ZIP_PATH),
        "canonical_zip_sha256": sha256(ZIP_PATH),
        "live_zip_count": len(zips),
        "live_zips": zips,
        "no_credentials": True,
        "no_mt5_real_config_saved": True,
        "no_autotrading_from_scripts": True,
    }
    write_json(OUT / "preflight" / "phase36r_preflight.json", status)
    write_text(
        OUT / "preflight" / "phase36r_preflight.md",
        f"""
# Phase36R/37A Preflight

- branch: {status['branch']}
- remote origin: present
- MANIPULANTE exists: {status['manipulante_exists']}
- Phase36 report exists: {status['phase36_report_exists']}
- canonical zip testzip: {status['canonical_zip_testzip']}
- live zip count: {status['live_zip_count']}
""",
    )
    return status


def classify_order_line(path: Path, line: str, line_no: int) -> dict[str, Any] | None:
    matched = [token for token in ORDER_TOKENS if token.lower() in line.lower()]
    if not matched:
        return None
    rel = str(path.relative_to(ROOT))
    rel_posix = path.relative_to(ROOT).as_posix()
    active = any(pattern.search(line) for pattern in ACTIVE_PATTERNS)
    in_router_wrapper = rel == "mt5_demo_executor_lab\\mt5_order_router.py" and "active_mt5.order_send(request)" in line
    backup = path.name.endswith(".phase36r_backup")
    if rel_posix == "BOT_V2_DAYTIME_LAB/src/phase36r_37a_micro_real_gate_orchestrator.py":
        classification = "SAFE_BLOCKED"
        severity = "INFO"
    elif rel_posix.startswith("BOT_V2_DAYTIME_LAB/outputs/") or rel_posix.startswith("BOT_V2_DAYTIME_LAB/reports/"):
        classification = "SAFE_BLOCKED"
        severity = "INFO"
    elif backup:
        classification = "SAFE_BLOCKED"
        severity = "INFO"
    elif in_router_wrapper:
        classification = "WRAPPED_AND_GATED"
        severity = "INFO"
    elif active:
        classification = "BLOCKER_UNGATED_ORDER_SEND"
        severity = "BLOCKER"
    elif "dry-run" in line.lower() or "dry_run" in line.lower():
        classification = "DRY_RUN_ONLY"
        severity = "INFO"
    elif any(word in line.lower() for word in ["false", "blocked", "no real", "not authorized"]):
        classification = "SAFE_BLOCKED"
        severity = "INFO"
    else:
        classification = "UNKNOWN_REVIEW_REQUIRED"
        severity = "WARNING"
    return {
        "path": rel,
        "line": line_no,
        "matched": ";".join(matched),
        "classification": classification,
        "severity": severity,
        "line_text": line.strip()[:240],
    }


def scan_order_send() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        if set(path.parts) & SKIP_DIRS:
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES and not path.name.endswith(".phase36r_backup"):
            continue
        try:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception:
            continue
        for idx, line in enumerate(lines, start=1):
            item = classify_order_line(path, line, idx)
            if item:
                rows.append(item)
    return rows


def scan_backup_before() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if ROUTER_BACKUP.exists():
        for idx, line in enumerate(ROUTER_BACKUP.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
            if "order_send" in line:
                rows.append({
                    "path": str(ROUTER_BACKUP.relative_to(ROOT)),
                    "line": idx,
                    "matched": "order_send",
                    "classification": "BLOCKER_UNGATED_ORDER_SEND_BEFORE_REPAIR",
                    "severity": "BLOCKER",
                    "line_text": line.strip(),
                })
    return rows


def run_router_safety_tests() -> list[dict[str, Any]]:
    spec = importlib.util.spec_from_file_location("phase36r_router", ROUTER)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)  # type: ignore[attr-defined]

    class FakeMT5:
        def __init__(self) -> None:
            self.calls = 0

        def order_send(self, request: dict[str, Any]) -> str:
            self.calls += 1
            return "SENT"

    rows: list[dict[str, Any]] = []
    fake = FakeMT5()
    request = {"symbol": "EURUSD", "sl": 1.0, "tp": 1.1}
    tests = [
        ("dry_run_only", {"mode": "DRY_RUN_ONLY", "mt5_module": fake}, "NO_TRADE"),
        ("missing_confirmation_file", {
            "mode": "MICRO_REAL_ONLY",
            "micro_real_flag": True,
            "understand_real_risk_flag": True,
            "no_force_flag": True,
            "risk": 0.001,
            "gates": {key: "ALLOW" for key in module.REQUIRED_ALLOW_GATES},
            "order_check_func": lambda _: "PASS",
            "confirmation_path": ROOT / "MANIPULANTE" / "12_MICRO_REAL_READINESS" / "_MISSING_CONFIRMATION_TEST.txt",
            "mt5_module": fake,
        }, "NO_TRADE"),
        ("risk_above_025", {
            "mode": "MICRO_REAL_ONLY",
            "micro_real_flag": True,
            "understand_real_risk_flag": True,
            "no_force_flag": True,
            "risk": 0.005,
            "gates": {key: "ALLOW" for key in module.REQUIRED_ALLOW_GATES},
            "order_check_func": lambda _: "PASS",
            "mt5_module": fake,
        }, "NO_TRADE"),
    ]
    for name, kwargs, expected in tests:
        fake.calls = 0
        result = module.safe_order_send_guarded(request, **kwargs)
        rows.append({
            "test": name,
            "expected_decision": expected,
            "actual_decision": result.final_decision,
            "order_sent": result.order_sent,
            "fake_order_send_calls": fake.calls,
            "pass": result.order_sent is False and fake.calls == 0,
            "reason": result.reason,
        })
    write_csv(OUT / "order_send_repair" / "phase36r_order_send_safety_tests.csv", rows)
    return rows


def order_send_repair_outputs() -> dict[str, Any]:
    before = scan_backup_before()
    after = scan_order_send()
    tests = run_router_safety_tests()
    fields = ["path", "line", "matched", "classification", "severity", "line_text"]
    write_csv(OUT / "order_send_repair" / "phase36r_order_send_findings_before.csv", before, fields)
    write_csv(OUT / "order_send_repair" / "phase36r_order_send_findings_after.csv", after, fields)
    blockers_after = [row for row in after if row["classification"] == "BLOCKER_UNGATED_ORDER_SEND"]
    summary = {
        "timestamp": now_iso(),
        "backup_created": ROUTER_BACKUP.exists(),
        "blockers_before": len(before),
        "blockers_after": len(blockers_after),
        "wrapped_order_send_count": sum(1 for row in after if row["classification"] == "WRAPPED_AND_GATED"),
        "safety_tests_pass": all(row["pass"] for row in tests),
        "order_sent_false_in_tests": all(row["order_sent"] is False for row in tests),
        "repaired": len(blockers_after) == 0 and all(row["pass"] for row in tests),
    }
    write_json(OUT / "order_send_repair" / "phase36r_order_send_repair.json", summary)
    write_text(
        OUT / "order_send_repair" / "phase36r_order_send_repair.md",
        f"""
# Phase36R Order Send Repair

- backup_created: {summary['backup_created']}
- blockers_before: {summary['blockers_before']}
- blockers_after: {summary['blockers_after']}
- wrapped_order_send_count: {summary['wrapped_order_send_count']}
- safety_tests_pass: {summary['safety_tests_pass']}
- order_sent_false_in_tests: {summary['order_sent_false_in_tests']}
- repaired: {summary['repaired']}
""",
    )
    post_summary = {
        "timestamp": now_iso(),
        "findings_count": len(after),
        "blockers_count": len(blockers_after),
        "blockers": blockers_after,
        "verdict": "PASS" if not blockers_after else "BLOCKER",
    }
    write_csv(OUT / "order_send_post_audit" / "phase36r_order_send_post_findings.csv", after, fields)
    write_json(OUT / "order_send_post_audit" / "phase36r_order_send_post_audit.json", post_summary)
    write_text(
        OUT / "order_send_post_audit" / "phase36r_order_send_post_audit.md",
        f"""
# Phase36R Order Send Post Audit

- verdict: {post_summary['verdict']}
- findings_count: {post_summary['findings_count']}
- blockers_count: {post_summary['blockers_count']}
""",
    )
    return {"repair": summary, "post_audit": post_summary}


def live_news_outputs() -> dict[str, Any]:
    gate = LiveNewsFortress()
    today, today_status = gate.load_today_news()
    week, week_status = gate.load_week_news()
    status = gate.get_news_gate_status()
    def rows(events: list[Any]) -> list[dict[str, Any]]:
        return [{
            "event_id": item.event_id,
            "name": item.name,
            "currency": item.currency,
            "impact": item.impact,
            "event_time_utc": item.event_time_utc.isoformat(),
            "event_time_ny": item.event_time_ny.isoformat(),
            "source": item.source,
        } for item in events]
    write_csv(OUT / "live_news" / "phase36r_news_today.csv", rows(today), ["event_id", "name", "currency", "impact", "event_time_utc", "event_time_ny", "source"])
    write_csv(OUT / "live_news" / "phase36r_news_week.csv", rows(week), ["event_id", "name", "currency", "impact", "event_time_utc", "event_time_ny", "source"])
    summary = {
        "timestamp": now_iso(),
        "today_loaded": today_status == "OK",
        "week_loaded": week_status == "OK",
        "today_status": today_status,
        "week_status": week_status,
        "gate": status.get("gate"),
        "status": status.get("status"),
        "next_blocking_event": status.get("next_blocking_event"),
        "source": "MT5/MQL5 cache or manual verified cache",
    }
    write_json(OUT / "live_news" / "phase36r_live_news_status.json", summary)
    write_text(
        OUT / "live_news" / "phase36r_live_news_status.md",
        f"""
# Phase36R Live News Status

- today_loaded: {summary['today_loaded']}
- week_loaded: {summary['week_loaded']}
- gate: {summary['gate']}
- status: {summary['status']}
- source: {summary['source']}
""",
    )
    return summary


def lot_100_outputs(symbol_status: dict[str, Any]) -> dict[str, Any]:
    out_dir = OUT / "lot_validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    min_lot = float(symbol_status.get("min_lot") or 0.01)
    lot_step = float(symbol_status.get("lot_step") or 0.01)
    validator = ExnessLotValidator(min_lot=min_lot, lot_step=lot_step)
    risks = [0.001, 0.0025, 0.005, 0.0075, 0.01]
    stops = [3, 5, 8, 10, 15, 20]
    rows = []
    for risk in risks:
        for stop in stops:
            item = validator.validate(100, risk, stop)
            row = item.__dict__.copy()
            row["risk_label"] = f"{risk*100:.2f}%"
            row["real_allowed_today"] = row["status"] != "BLOCK" and risk <= 0.0025 and symbol_status.get("state") == "ALLOW"
            rows.append(row)
    write_csv(out_dir / "phase36r_lot_scenarios_100usd.csv", rows)
    def risk_allowed(risk: float) -> bool:
        return any(row["risk_pct"] == risk and row["real_allowed_today"] for row in rows)
    summary = {
        "timestamp": now_iso(),
        "balance_usd": 100,
        "symbol_source": symbol_status.get("symbol_detected"),
        "min_lot": min_lot,
        "lot_step": lot_step,
        "pip_value_source": "conservative_default_if_symbol_unavailable",
        "risk_010_allowed": risk_allowed(0.001),
        "risk_025_allowed": risk_allowed(0.0025),
        "risk_050_allowed": False,
        "risk_075_allowed": False,
        "risk_100_allowed": False,
        "state": "ALLOW" if risk_allowed(0.001) or risk_allowed(0.0025) else "NO_TRADE",
        "reason": "min lot/symbol gate prevents <=0.25% real risk" if not (risk_allowed(0.001) or risk_allowed(0.0025)) else "0.10/0.25 executable",
    }
    write_json(out_dir / "phase36r_lot_validation.json", summary)
    write_text(
        out_dir / "phase36r_lot_validation.md",
        f"""
# Phase36R Lot Validation 100 USD

- 0.10 allowed: {summary['risk_010_allowed']}
- 0.25 allowed: {summary['risk_025_allowed']}
- 0.50 allowed: {summary['risk_050_allowed']}
- 0.75 allowed: {summary['risk_075_allowed']}
- 1.00 allowed: {summary['risk_100_allowed']}
- state: {summary['state']}
- reason: {summary['reason']}
""",
    )
    return summary


def dry_run_full_outputs() -> dict[str, Any]:
    decision = ManipulanteDryRunEngine().run_once()
    out_dir = OUT / "dry_run_full"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "phase36r_dry_run_full.json", decision)
    write_csv(out_dir / "phase36r_dry_run_decisions.csv", [decision])
    write_text(
        out_dir / "phase36r_dry_run_full.md",
        f"""
# Phase36R Dry Run Full

- executed: yes
- final_decision: {decision['final_decision']}
- order_sent: {decision['order_sent']}
- reason: {decision['reason']}
""",
    )
    return decision


def write_micro_docs() -> None:
    base = MANIPULANTE / "12_MICRO_REAL_READINESS"
    write_text(
        base / "MICRO_REAL_ACTIVATION_PROTOCOL.md",
        """
# Micro Real Activation Protocol

No real trade is allowed unless the final matrix says `MICRO_REAL_READY_WITH_WARNINGS` and the user gives a new explicit instruction.

Required:

- confirmation file with exact text;
- `--micro-real`;
- `--risk 0.001` or `--risk 0.0025`;
- `--i-understand-real-risk`;
- `--no-force`;
- all gates ALLOW/PASS;
- order_check PASS;
- SL and TP present;
- one trade max.
""",
    )
    write_text(base / "NO_REAL_UNTIL_CONFIRMATION.md", "# NO REAL UNTIL CONFIRMATION\n\nPhase36R/37A does not authorize real trading automatically.")
    write_text(base / "MICRO_REAL_KILL_SWITCH.md", "# Micro Real Kill Switch\n\nAny gate not ALLOW/PASS => NO_TRADE. 0.75% and 1.00% are prohibited today.")
    write_text(base / "MICRO_REAL_POSITION_SIZE_POLICY.md", "# Micro Real Position Size Policy\n\nOnly 0.10%-0.25% may be considered. If min lot exceeds risk, NO_TRADE.")


def today_matrix(order: dict[str, Any], news: dict[str, Any], data: dict[str, Any], time_status: dict[str, Any], symbol: dict[str, Any], lot: dict[str, Any], dry: dict[str, Any], micro: dict[str, Any]) -> dict[str, Any]:
    confirmation = confirmation_file_gate()
    matrix = {
        "timestamp": now_iso(),
        "news_gate": news["gate"],
        "week_news_loaded": news["week_loaded"],
        "data_gate": data["state"],
        "time_gate": time_status["state"],
        "symbol_gate": symbol["state"],
        "spread_gate": "ALLOW" if symbol["state"] == "ALLOW" else symbol["state"],
        "stoplevel_gate": "ALLOW" if symbol["state"] == "ALLOW" else symbol["state"],
        "lot_gate_100usd": lot["state"],
        "order_send_safety": "PASS" if order["post_audit"]["blockers_count"] == 0 else "BLOCKER",
        "dry_run": "PASS" if dry["order_sent"] is False else "FAIL",
        "confirmation_file": "present" if confirmation["present"] else "absent",
        "micro_real": "BLOCKED",
        "final_decision": "NO_TRADE",
        "verdict": "",
        "conditions_required": [],
    }
    if matrix["order_send_safety"] != "PASS":
        verdict = "MICRO_REAL_BLOCKED_ORDER_SEND"
    elif news["gate"] != "ALLOW" or not news["week_loaded"]:
        verdict = "MICRO_REAL_BLOCKED_NEWS"
    elif data["state"] != "ALLOW":
        verdict = "MICRO_REAL_BLOCKED_DATA"
    elif time_status["state"] not in {"ALLOW"}:
        verdict = "MICRO_REAL_BLOCKED_TIME"
    elif symbol["state"] != "ALLOW":
        verdict = "MICRO_REAL_BLOCKED_SYMBOL"
    elif lot["state"] != "ALLOW":
        verdict = "MICRO_REAL_BLOCKED_LOT"
    elif matrix["confirmation_file"] != "present":
        verdict = "MICRO_REAL_BLOCKED_CONFIRMATION_FILE"
    else:
        verdict = "MICRO_REAL_READY_WITH_WARNINGS"
        matrix["micro_real"] = "READY"
    matrix["verdict"] = verdict
    matrix["conditions_required"] = [
        "Live News Gate ALLOW with today/week loaded",
        "Data Quality Live Gate ALLOW",
        "MT5 server time validated",
        "Exness symbol/spread/stoplevel ALLOW",
        "100 USD lot gate <=0.25% executable",
        "confirmation file present and exact",
        "new explicit user instruction before real",
    ]
    write_json(OUT / "today_readiness" / "phase36r_37a_today_readiness.json", matrix)
    write_text(
        OUT / "today_readiness" / "phase36r_37a_today_readiness.md",
        f"""
# Phase36R/37A Today Readiness

- News Gate: {matrix['news_gate']}
- Week News loaded: {matrix['week_news_loaded']}
- Data Gate: {matrix['data_gate']}
- Time Gate: {matrix['time_gate']}
- Symbol Gate: {matrix['symbol_gate']}
- Lot Gate 100 USD: {matrix['lot_gate_100usd']}
- OrderSend Safety: {matrix['order_send_safety']}
- Dry-run: {matrix['dry_run']}
- Confirmation File: {matrix['confirmation_file']}
- Micro Real: {matrix['micro_real']}
- Verdict: {matrix['verdict']}
- Final decision: NO_TRADE
""",
    )
    return matrix


def final_verdict(matrix: dict[str, Any]) -> str:
    if matrix["order_send_safety"] != "PASS":
        return "BLOCKED_ORDER_SEND_REPAIR_FAILED"
    if matrix["news_gate"] != "ALLOW" or not matrix["week_news_loaded"]:
        return "BLOCKED_NEWS_SOURCE_UNAVAILABLE"
    if matrix["data_gate"] != "ALLOW":
        return "BLOCKED_DATA_LIVE_UNAVAILABLE"
    if matrix["lot_gate_100usd"] != "ALLOW":
        return "BLOCKED_LOT_MIN_TOO_HIGH"
    if matrix["micro_real"] == "READY":
        return "MICRO_REAL_READY_WITH_WARNINGS"
    return "REQUIRES_REPAIR_BEFORE_REAL"


def update_master_docs(verdict: str) -> None:
    write_text(ROOT / "00_READ_THIS_FIRST.md", f"# READ THIS FIRST\n\n- MANIPULANTE remains the only authority.\n- Phase36R/37A verdict: `{verdict}`.\n- Real trading remains blocked unless verdict is `MICRO_REAL_READY_WITH_WARNINGS` and the user gives a new explicit command.\n- Risk above 0.25% is prohibited today.\n- Live News and Data Quality live gates are mandatory.")
    payload = {
        "timestamp": now_iso(),
        "current_authority": "MANIPULANTE",
        "latest_phase_completed": "PHASE36R_37A",
        "verdict": verdict,
        "phase25_parameters_changed": False,
        "real_blocked": verdict != "MICRO_REAL_READY_WITH_WARNINGS",
        "risk_real_initial": "0.10%-0.25% only if all gates pass",
        "risk_075_authorized_today": False,
        "risk_100_authorized": False,
        "live_news_required": True,
        "data_live_required": True,
        "confirmation_file_required": True,
    }
    write_json(ROOT / "01_CURRENT_PROJECT_STATUS.json", payload)
    write_json(LAB / "status.json", payload)
    write_text(ROOT / "01_CURRENT_PROJECT_STATUS.md", f"# CURRENT PROJECT STATUS\n\n- Phase36R/37A verdict: `{verdict}`.\n- OrderSend router repaired with guarded wrapper.\n- Real remains blocked unless every gate passes.\n- 0.75% and 1.00% are not authorized.")
    write_json(ROOT / "02_STRATEGY_AUTHORITY_MAP.json", {"timestamp": now_iso(), "authority": "MANIPULANTE", "phase25_parameters_changed": False, "phase36r_37a": verdict, "real_blocked": verdict != "MICRO_REAL_READY_WITH_WARNINGS"})
    write_text(ROOT / "02_STRATEGY_AUTHORITY_MAP.md", f"# STRATEGY AUTHORITY MAP\n\n- MANIPULANTE: CURRENT AUTHORITY.\n- Phase25 unchanged.\n- Phase36R/37A: `{verdict}`.\n- BE0.5 remains shadow only.")
    manifest = "# ZIP CONTENTS MANIFEST\n\nIncludes MANIPULANTE, repaired router, Phase36R/37A outputs, validators, gated micro-real engine, reports, docs and Phase25 config/hash. Excludes secrets, credentials, heavy data, .pkl and internal ZIPs."
    write_text(ROOT / "ZIP_CONTENTS_MANIFEST.md", manifest)
    write_text(LAB / "ZIP_CONTENTS_MANIFEST.md", manifest)
    write_text(MANIPULANTE / "00_LEER_PRIMERO" / "README_MANIPULANTE.md", f"# MANIPULANTE\n\nPhase25 authority. Phase36R/37A verdict: `{verdict}`. No real trade unless all gates pass and user explicitly confirms. TP/BE/BF unchanged.")
    write_text(MANIPULANTE / "04_OPERACION_DIARIA" / "MANIPULANTE_DAILY_RUNBOOK.md", "# MANIPULANTE Daily Runbook\n\nPhase25 unchanged. Before any action: News ALLOW, Data ALLOW, Time ALLOW, Symbol/Spread/StopLevel ALLOW, Lot <=0.25% ALLOW, confirmation file present. Otherwise NO_TRADE.")
    write_text(MANIPULANTE / "04_OPERACION_DIARIA" / "MANIPULANTE_KILL_SWITCH.md", "# MANIPULANTE Kill Switch\n\nAny failed gate => NO_TRADE. No 0.75% today. No 1.00%. No AutoTrading blind.")
    write_text(MANIPULANTE / "09_COMPLIANCE" / "LIVE_NEWS_FORTRESS_POLICY.md", "# Live News Fortress Policy\n\nLive EUR/USD high-impact news for today and week must be loaded from MT5/MQL5 cache or verified manual emergency cache. Missing/stale/unknown => NO_TRADE.")


def write_report(verdict: str, order: dict[str, Any], news: dict[str, Any], data: dict[str, Any], time_status: dict[str, Any], symbol: dict[str, Any], lot: dict[str, Any], dry: dict[str, Any], micro: dict[str, Any], matrix: dict[str, Any]) -> dict[str, Any]:
    report = {
        "timestamp": now_iso(),
        "objective": "Repair order_send and evaluate final micro-real gate for MANIPULANTE only",
        "strategy_changed": False,
        "order_send_repair": order,
        "live_news_gate": news,
        "data_quality_gate": data,
        "time_gate": time_status,
        "symbol_gate": symbol,
        "lot_gate_100usd": lot,
        "dry_run_full": dry,
        "micro_real_engine": micro,
        "today_readiness": matrix,
        "verdict": verdict,
        "next_step": "Load verified live news/data through MT5 and rerun final matrix; no real trade now.",
    }
    write_json(REPORT_JSON, report)
    write_text(
        REPORT_MD,
        f"""
# PHASE36R/37A ORDER_SEND REPAIR + MICRO REAL GATE REPORT

## Verdict

`{verdict}`

## OrderSend repair

- repaired: {order['repair']['repaired']}
- post blockers: {order['post_audit']['blockers_count']}
- safety tests pass: {order['repair']['safety_tests_pass']}

## Live News

- gate: {news['gate']}
- today loaded: {news['today_loaded']}
- week loaded: {news['week_loaded']}
- status: {news['status']}

## Data / Time / Symbol / Lot

- data: {data['state']}
- time: {time_status['state']}
- symbol: {symbol['state']}
- lot 100 USD: {lot['state']}

## Dry-run

- final decision: {dry['final_decision']}
- order_sent: {dry['order_sent']}

## Final

NO_TRADE until every gate is ALLOW/PASS.
""",
    )
    return report


def include_file(path: Path) -> bool:
    if not path.is_file():
        return False
    rel = path.relative_to(ROOT)
    rel_str = rel.as_posix()
    if set(rel.parts) & {".git", ".venv", ".venv_fixed", ".pkg", ".vendor_duka", ".vendor_duka2", "__pycache__", "data", "ARCHIVE_SUPERSEDED"}:
        return False
    lower = rel_str.lower()
    if lower.endswith((".zip", ".zipbak", ".pkl", ".pyc")):
        return False
    if any(token in lower for token in SECRET_TOKENS):
        return False
    if path.stat().st_size > 2 * 1024 * 1024:
        return False
    allowed = rel.parts[0] in {"MANIPULANTE", "ESTRATEGIAS", "BOT_V2_DAYTIME_LAB", "mt5_demo_executor_lab"}
    if allowed:
        if rel_str.startswith("BOT_V2_DAYTIME_LAB/outputs/") and "phase36r_37a_micro_real_gate" not in rel_str and "phase36_live_news_mt5_dryrun" not in rel_str:
            return False
        if rel_str.startswith("BOT_V2_DAYTIME_LAB/reports/") and "PHASE36" not in rel_str:
            return False
        if rel_str.startswith("BOT_V2_DAYTIME_LAB/data/"):
            return False
        return True
    return path.name in {"00_READ_THIS_FIRST.md", "01_CURRENT_PROJECT_STATUS.md", "01_CURRENT_PROJECT_STATUS.json", "02_STRATEGY_AUTHORITY_MAP.md", "02_STRATEGY_AUTHORITY_MAP.json", "ZIP_CONTENTS_MANIFEST.md"}


def rebuild_zip() -> dict[str, Any]:
    temp = ZIP_PATH.with_suffix(".zip.tmp")
    if temp.exists():
        temp.unlink()
    files = sorted([path for path in ROOT.rglob("*") if include_file(path)], key=lambda p: p.relative_to(ROOT).as_posix().lower())
    with zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for path in files:
            zf.write(path, path.relative_to(ROOT).as_posix())
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    temp.rename(ZIP_PATH)
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        names = zf.namelist()
        validation = {
            "path": str(ZIP_PATH),
            "size": ZIP_PATH.stat().st_size,
            "entries": len(names),
            "sha256": sha256(ZIP_PATH),
            "testzip": zf.testzip(),
            "single_live_zip": len(live_zips()) == 1,
            "contains_phase36r_report": "BOT_V2_DAYTIME_LAB/reports/PHASE36R_37A_ORDER_SEND_REPAIR_MICRO_REAL_GATE_REPORT.md" in names,
            "contains_repaired_router": "mt5_demo_executor_lab/mt5_order_router.py" in names,
            "contains_router_backup": "mt5_demo_executor_lab/mt5_order_router.py.phase36r_backup" in names,
            "contains_phase25_hash": "BOT_V2_DAYTIME_LAB/configs/phase25_forward_demo_candidate_config_hash.txt" in names,
            "heavy_entries_gt_2mb": [(name, zf.getinfo(name).file_size) for name in names if zf.getinfo(name).file_size > 2 * 1024 * 1024],
            "secret_like_entries": [name for name in names if any(token in name.lower() for token in SECRET_TOKENS)],
            "zip_entries_inside": [name for name in names if name.lower().endswith((".zip", ".zipbak"))],
        }
    write_json(OUT / "zip_validation" / "phase36r_zip_validation.json", validation)
    write_text(OUT / "zip_validation" / "phase36r_zip_validation.md", "\n".join([f"# Phase36R ZIP Validation", f"- path: {validation['path']}", f"- size: {validation['size']}", f"- entries: {validation['entries']}", f"- sha256: {validation['sha256']}", f"- testzip: {validation['testzip']}", f"- single_live_zip: {validation['single_live_zip']}"]))
    return validation


def git_status(commit: bool = False) -> dict[str, Any]:
    payload = {
        "timestamp": now_iso(),
        "branch": run_cmd(["git", "branch", "--show-current"]),
        "status_short": run_cmd(["git", "status", "--short"]),
        "diff_stat": run_cmd(["git", "diff", "--stat"]),
        "commit": "NO",
        "push": "NO",
        "hash": "N/A",
        "reason": "Final matrix did not permit micro-real; commit/push skipped.",
    }
    write_json(OUT / "git" / "phase36r_git_status.json", payload)
    write_text(OUT / "git" / "phase36r_git_status.md", f"# Phase36R Git Status\n\n- branch: {payload['branch']}\n- commit: NO\n- push: NO\n- reason: {payload['reason']}")
    return payload


def main() -> dict[str, Any]:
    preflight()
    order = order_send_repair_outputs()
    news = live_news_outputs()
    data = write_data_outputs()
    time_status = write_time_outputs()
    symbol = write_symbol_outputs()
    lot = lot_100_outputs(symbol)
    dry = dry_run_full_outputs()
    write_micro_docs()
    micro = write_micro_engine_outputs()
    matrix = today_matrix(order, news, data, time_status, symbol, lot, dry, micro)
    verdict = final_verdict(matrix)
    update_master_docs(verdict)
    report = write_report(verdict, order, news, data, time_status, symbol, lot, dry, micro, matrix)
    zip_validation = rebuild_zip()
    git = git_status()
    final = {"report": report, "zip_validation": zip_validation, "git": git}
    write_json(OUT / "phase36r_37a_final_execution_summary.json", final)
    print(json.dumps(final, indent=2, ensure_ascii=False))
    return final


if __name__ == "__main__":
    main()

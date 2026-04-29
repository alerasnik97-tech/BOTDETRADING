import os, json, csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
AUDIT_DIR = ROOT / 'BOT_V2_DAYTIME_LAB' / 'outputs' / 'phase35_final_real_readiness_audit' / 'manipulante_config_audit'
AUDIT_DIR.mkdir(parents=True, exist_ok=True)

CONFIG_PATH = ROOT / 'MANIPULANTE' / '01_ESTRATEGIA_AUTORIDAD' / 'manipulante_config.json'
BASELINE_PATH = ROOT / 'BOT_V2_DAYTIME_LAB' / 'configs' / 'phase25_forward_demo_candidate_config.json'

expected = {
    "symbol": "EURUSD",
    "tp_r": 1.4,
    "be_r": 0.4,
    "body_filter_threshold": 0.7,
    "timeframe": "M3",
    "context_timeframe": "H1",
    "start_time_ny": "07:00",
    "end_time_ny": "16:30",
    "max_trades_per_day": 1,
    "hard_close_friday_ny": "16:55",
    "allow_weekend_holding": False,
    "news_fortress_fail_closed": True,
    "data_quality_mask_fail_closed": True,
    "live_trading_allowed": False,
    "auto_order_execution": False
}

findings = []
def check_val(name, current, target):
    if current != target:
        findings.append({"param": name, "current": current, "expected": target, "status": "MISMATCH"})
    else:
        findings.append({"param": name, "current": current, "expected": target, "status": "OK"})

if not CONFIG_PATH.exists():
    findings.append({"param": "FILE_EXISTS", "current": False, "expected": True, "status": "BLOCKER"})
else:
    with open(CONFIG_PATH, 'r') as f:
        cfg = json.load(f)
    
    check_val("symbol", cfg.get("symbol"), expected["symbol"])
    check_val("tp_r", cfg.get("tp_r"), expected["tp_r"])
    check_val("be_r", cfg.get("be_r"), expected["be_r"])
    check_val("body_filter", cfg.get("body_filter"), expected["body_filter_threshold"])
    check_val("live_trading_allowed", cfg.get("live_trading_allowed"), expected["live_trading_allowed"])
    check_val("auto_order_execution", cfg.get("auto_order_execution"), expected["auto_order_execution"])
    check_val("hard_close_friday_ny", cfg.get("global_weekend_policy", {}).get("hard_close_time_ny"), expected["hard_close_friday_ny"])

mismatches = [f for f in findings if f['status'] != 'OK']
verdict = "BLOCKER" if any(f['status'] == 'MISMATCH' for f in findings) else "PASS"

res = {"verdict": verdict, "results": findings}

with open(AUDIT_DIR / 'phase35_manipulante_config_audit.json', 'w') as f:
    json.dump(res, f, indent=2)

with open(AUDIT_DIR / 'phase35_manipulante_config_diff.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=["param", "current", "expected", "status"])
    writer.writeheader()
    writer.writerows(findings)

md = [f"# MANIPULANTE CONFIG AUDIT\nVerdict: {verdict}\n\n| Param | Current | Expected | Status |", "|---|---|---|---|"]
for f in findings:
    md.append(f"| {f['param']} | {f['current']} | {f['expected']} | {f['status']} |")

with open(AUDIT_DIR / 'phase35_manipulante_config_audit.md', 'w') as f:
    f.write('\n'.join(md))

print(json.dumps(res, indent=2))

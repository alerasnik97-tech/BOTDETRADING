import sys
import pandas as pd
from pathlib import Path

# Correct path to lab sub-root
project_root = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\03_RESEARCH_LAB\BOT_V2_DAYTIME_LAB")
sys.path.append(str(project_root))

from src.v7_engine.engine import TestLeakageGuard, TestLeakageViolation

def test_guard():
    print("Testing Anti-Leakage Guard for Real V50B...")
    guard = TestLeakageGuard(active_phase="train", test_start_year=2025)
    
    dates_to_test = [
        ("2022-05-01", True),
        ("2023-01-01", True),
        ("2024-12-31", True),
        ("2025-01-01", False),
        ("2026-06-01", False)
    ]
    
    results = []
    for dt_str, expected in dates_to_test:
        ts = pd.Timestamp(dt_str)
        try:
            guard.verify_timestamp(ts)
            passed = True
        except TestLeakageViolation:
            passed = False
            
        status = "PASSED" if passed == expected else "FAILED"
        results.append({"date": dt_str, "expected_allow": expected, "actual_allow": passed, "status": status})
        print(f"Date: {dt_str} | Expected Allow: {expected} | Actual Allow: {passed} | Status: {status}")

    all_passed = all(r["status"] == "PASSED" for r in results)
    
    report_path = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\03_RESEARCH_LAB\BOT_V2_DAYTIME_LAB\reports\v50b_real_implementation_precheck\audits\V50B_REAL_ANTI_LEAKAGE_TEST.md")
    with open(report_path, "w") as f:
        f.write("# V50B REAL ANTI-LEAKAGE TEST REPORT\n\n")
        f.write(f"**Result**: {'PASS' if all_passed else 'FAIL'}\n\n")
        f.write("| Date | Expected Allow | Actual Allow | Status |\n")
        f.write("| --- | --- | --- | --- |\n")
        for r in results:
            f.write(f"| {r['date']} | {r['expected_allow']} | {r['actual_allow']} | {r['status']} |\n")
            
    return all_passed

if __name__ == "__main__":
    if test_guard():
        sys.exit(0)
    else:
        sys.exit(1)

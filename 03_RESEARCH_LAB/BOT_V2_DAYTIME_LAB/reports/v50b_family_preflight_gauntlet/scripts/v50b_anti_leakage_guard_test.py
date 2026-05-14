import sys
import pandas as pd
from pathlib import Path
# Add laboratory root to path so 'src' is discoverable
lab_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(lab_root))
from src.v7_engine.engine import TestLeakageGuard, TestLeakageViolation

def test_guard():
    print("Testing Anti-Leakage Guard for V50B...")
    # Initialize guard with train phase and 2025 as test start year
    guard = TestLeakageGuard(active_phase="train", test_start_year=2025)
    
    # 2024 should PASS
    try:
        guard.verify_timestamp(pd.Timestamp("2024-12-31"))
        print("2024-12-31: PASSED (Correct)")
    except TestLeakageViolation:
        print("2024-12-31: FAILED (Incorrectly Blocked)")
        return False

    # 2025 should FAIL
    try:
        guard.verify_timestamp(pd.Timestamp("2025-01-01"))
        print("2025-01-01: PASSED (Incorrectly Allowed)")
        return False
    except TestLeakageViolation:
        print("2025-01-01: BLOCKED (Correct)")

    print("Guard is ACTIVE and CORRECT.")
    return True

if __name__ == "__main__":
    if test_guard():
        sys.exit(0)
    else:
        sys.exit(1)

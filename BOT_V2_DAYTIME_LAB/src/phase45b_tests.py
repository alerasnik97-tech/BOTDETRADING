# -*- coding: utf-8 -*-
"""
Phase 45B - Runner Recovery Logic Tests
"""
import sys
from pathlib import Path

# Add src to path
ROOT = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo")
sys.path.append(str(ROOT / "BOT_V2_DAYTIME_LAB" / "src"))

from phase37_ftmo_trial_support import start_safety_preflight, acquire_start_lock, release_start_lock
import phase45b_runner_recovery as recovery

def run_tests():
    print("EJECUTANDO TESTS DE RECUPERACION PHASE 45B")
    print("=" * 60)
    
    # Test 1: Start with no runner, no lock
    print("Test 1: START con no runner, no lock")
    preflight = start_safety_preflight()
    print(f"  Decision: {preflight['decision']}")
    print(f"  Can Start: {preflight['can_start']}")
    
    # Test 2: Status with no lock
    print("\nTest 2: STATUS sin lock")
    rep = recovery.diagnose()
    print(f"  Estado: {rep['status']}")
    
    # Test 3: Create fake lock and check diagnosis
    print("\nTest 3: STATUS con lock stale")
    lock_file = ROOT / "MANIPULANTE" / "10_LOGS_PAPER" / "ftmo_trial_bot" / "runner.lock"
    lock_file.write_text("999999") # Fake PID
    rep = recovery.diagnose()
    print(f"  Estado: {rep['status']}")
    print(f"  Lock PID: {rep['lock_pid']}")
    
    # Test 4: Clean stale lock
    print("\nTest 4: Limpieza de lock stale")
    success = recovery.clean_stale_lock(rep)
    print(f"  Exito: {success}")
    print(f"  Lock exists: {lock_file.exists()}")
    
    print("\n" + "=" * 60)
    print("TESTS COMPLETADOS")

if __name__ == "__main__":
    run_tests()

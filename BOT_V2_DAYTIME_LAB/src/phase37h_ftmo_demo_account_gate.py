import os
import json
import MetaTrader5 as mt5
from pathlib import Path

# --- CONFIGURATION ---
ROOT = Path(r"c:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo")
ALLOWLIST_PATH = ROOT / "MANIPULANTE" / "13_FTMO_TRIAL_AUTOMATION" / "FTMO_DEMO_ALLOWLIST.json"
OUTPUT_DIR = ROOT / "BOT_V2_DAYTIME_LAB" / "outputs" / "phase37h_demo_only_terminal_isolation" / "account_gate"

def run_account_gate():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    status = {"gate_result": "ERROR_FAIL_CLOSED", "details": []}
    
    if not ALLOWLIST_PATH.exists():
        status["details"].append("Allowlist missing")
        return status
        
    with open(ALLOWLIST_PATH, "r") as f:
        allowlist = json.load(f)

    if not mt5.initialize():
        status["gate_result"] = "ERROR_MT5_CONNECT"
        return status

    acc = mt5.account_info()
    if not acc:
        status["gate_result"] = "ERROR_ACCOUNT_INFO"
        mt5.shutdown()
        return status

    # 1. Forbidden brokers check
    for fb in allowlist["forbidden_brokers"]:
        if fb.upper() in acc.company.upper() or fb.upper() in acc.server.upper():
            status["gate_result"] = f"BLOCKED_{fb.upper()}_DETECTED"
            mt5.shutdown()
            return status

    # 2. Allowed check
    passed = False
    for allowed in allowlist["allowed_accounts"]:
        if allowed["broker_contains"].upper() in acc.company.upper() and \
           allowed["server_contains"].upper() in acc.server.upper():
            
            # Mode check (mt5.ACCOUNT_TRADE_MODE_DEMO is 0)
            # trade_mode 0: demo, 1: contest, 2: trial, 3: real
            mode_map = {0: "DEMO", 1: "CONTEST", 2: "TRIAL", 3: "REAL"}
            current_mode = mode_map.get(acc.trade_mode, "UNKNOWN")
            
            if current_mode in allowed["mode_allowed"]:
                if allowed["balance_min"] <= acc.balance <= allowed["balance_max"]:
                    if acc.currency == allowed["currency"]:
                        # Symbol check
                        if mt5.symbol_select(allowed["symbol_required"], True):
                            passed = True
                            break
                        else:
                            status["details"].append(f"Symbol {allowed['symbol_required']} not found")
                    else:
                        status["details"].append(f"Currency mismatch: {acc.currency}")
                else:
                    status["details"].append(f"Balance out of range: {acc.balance}")
            else:
                status["details"].append(f"Trade mode not allowed: {current_mode}")

    if passed:
        status["gate_result"] = "FTMO_DEMO_TRIAL_CONFIRMED"
    else:
        status["gate_result"] = "BLOCKED_ACCOUNT_NOT_IN_ALLOWLIST"

    mt5.shutdown()
    with open(OUTPUT_DIR / "phase37h_account_gate.json", "w") as f:
        json.dump(status, f, indent=2)
        
    return status

if __name__ == "__main__":
    run_account_gate()

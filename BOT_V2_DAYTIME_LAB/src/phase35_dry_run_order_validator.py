import json, csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
AUDIT_DIR = ROOT / 'BOT_V2_DAYTIME_LAB' / 'outputs' / 'phase35_final_real_readiness_audit' / 'dry_run_order_simulation'
AUDIT_DIR.mkdir(parents=True, exist_ok=True)

def simulate_order(symbol, direction, entry, sl, tp, balance, risk_pct):
    sl_pips = abs(entry - sl) * 10000 # EURUSD 4 digits for pips approx
    # Risk calculation
    risk_amt = balance * risk_pct
    lots = risk_amt / (sl_pips * 10) # 10 USD per pip at 1 lot
    lots = round(int(lots / 0.01) * 0.01, 2)
    
    rr = abs(tp - entry) / abs(sl - entry)
    
    return {
        "symbol": symbol,
        "dir": direction,
        "entry": entry,
        "sl": sl,
        "tp": tp,
        "lots": lots,
        "rr": round(rr, 2),
        "status": "DRY_RUN_ONLY_SUCCESS"
    }

orders = []
for i in range(1, 11):
    orders.append(simulate_order("EURUSD", "BUY", 1.0850, 1.0840, 1.0864, 1000, 0.0025))

with open(AUDIT_DIR / 'phase35_dry_run_orders.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=orders[0].keys())
    writer.writeheader()
    writer.writerows(orders)

with open(AUDIT_DIR / 'phase35_dry_run_order_simulation.json', 'w') as f:
    json.dump({"orders": orders}, f, indent=2)

md = ["# DRY RUN ORDER SIMULATION\n\n| Symbol | Dir | Entry | SL | TP | Lots | RR | Status |", "|---|---|---|---|---|---|---|---|"]
for o in orders:
    md.append(f"| {o['symbol']} | {o['dir']} | {o['entry']} | {o['sl']} | {o['tp']} | {o['lots']} | {o['rr']} | {o['status']} |")

with open(AUDIT_DIR / 'phase35_dry_run_order_simulation.md', 'w') as f:
    f.write('\n'.join(md))

print("Dry-run order simulation completed.")

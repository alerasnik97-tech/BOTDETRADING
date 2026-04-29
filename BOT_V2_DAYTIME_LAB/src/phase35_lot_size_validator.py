import json, csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
AUDIT_DIR = ROOT / 'BOT_V2_DAYTIME_LAB' / 'outputs' / 'phase35_final_real_readiness_audit' / 'risk_lot_audit'
AUDIT_DIR.mkdir(parents=True, exist_ok=True)

def calculate_lot(balance, risk_pct, sl_pips, contract_size=100000, pip_value_at_1_lot=10, min_lot=0.01, lot_step=0.01):
    risk_amt = balance * risk_pct
    # Risk = Lots * sl_pips * (pip_value_at_1_lot)
    # Lots = Risk / (sl_pips * pip_value_at_1_lot)
    if sl_pips <= 0: return 0
    raw_lots = risk_amt / (sl_pips * pip_value_at_1_lot)
    
    # Conservative rounding down
    lots = int(raw_lots / lot_step) * lot_step
    lots = round(lots, 2)
    
    if lots < min_lot:
        return 0
    return lots

balances = [100, 200, 500, 1000]
risks = [0.0010, 0.0025, 0.0050] # 0.10%, 0.25%, 0.50%
sls = [3, 5, 8, 10, 15, 20]

scenarios = []
for b in balances:
    for r in risks:
        for sl in sls:
            lot = calculate_lot(b, r, sl)
            scenarios.append({
                "balance": b,
                "risk_pct": f"{r*100:.2f}%",
                "sl_pips": sl,
                "lots": lot,
                "risk_usd": round(b * r, 2),
                "actual_risk_usd": round(lot * sl * 10, 2) if lot > 0 else 0
            })

with open(AUDIT_DIR / 'phase35_lot_scenarios.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=["balance", "risk_pct", "sl_pips", "lots", "risk_usd", "actual_risk_usd"])
    writer.writeheader()
    writer.writerows(scenarios)

with open(AUDIT_DIR / 'phase35_risk_lot_audit.json', 'w') as f:
    json.dump({"scenarios": scenarios}, f, indent=2)

md = ["# RISK AND LOT AUDIT\n\n| Balance | Risk% | SL Pips | Lots | Risk USD | Actual Risk |", "|---|---|---|---|---|---|"]
for s in scenarios:
    md.append(f"| {s['balance']} | {s['risk_pct']} | {s['sl_pips']} | {s['lots']} | {s['risk_usd']} | {s['actual_risk_usd']} |")

with open(AUDIT_DIR / 'phase35_risk_lot_audit.md', 'w') as f:
    f.write('\n'.join(md))

print("Lot size validation completed.")

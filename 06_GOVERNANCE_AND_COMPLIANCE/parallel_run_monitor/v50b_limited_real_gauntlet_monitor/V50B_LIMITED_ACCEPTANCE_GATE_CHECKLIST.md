# V50B LIMITED REAL GAUNTLET — ACCEPTANCE GATE CHECKLIST

## Required Audits for Final Acceptance
- [ ] **Rowcount Audit**: Total trades vs configurations vs months.
- [ ] **Date Split Audit**: Confirmation of Train (2020-2022) and Val (2023-2024).
- [ ] **TEST Leakage Audit**: Targeted search for 2025/2026.
- [ ] **No Synthetic Audit**: Verification of physical tick source for all trades.
- [ ] **Metric Recalc Audit**: PF calculation from raw trades matches ranking.
- [ ] **Family Coverage Audit**: All families (F06, F08, F12) correctly sampled.
- [ ] **Slippage Stress Audit**: Robustness check at 0.3/0.5 pips.
- [ ] **Engine Proof Audit**: One engine ID per trade confirmation.
- [ ] **Root Hygiene**: Zero ZIPs or temp files created.
- [ ] **GitHub Sync**: Clean push of all final reports.

## Gate Decision
**PENDING_RUN_COMPLETION**

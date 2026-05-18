# NEXT PROMPT — AUDIT M2 TRAIN-ONLY STRUCTURAL EXECUTION V1

This prompt is to be executed in **READ-ONLY AUDIT MODE**.
Under blocker penalty, the following are strictly **PROHIBITED** during this audit:
- NO executing Python scripts or commands.
- NO executing helper scripts (such as `safety_scan.py`).
- NO M2 execution.
- NO loading of market data.
- NO modifying strategy code, engine, runner, or test files.
- NO backtesting or formal training.
- NO validation or holdout partition access.
- NO 2025 or 2026 data loading.
- NO optimization sweeps, grid searches, or walk-forward parameters.

---

## 1. Audit Objective
Verify the integrity, compliance, and abort lineage of the M2 Conservative Train-Only Structural Evaluation execution.

---

## 2. Verification Checklist

### 2.1 Abort State and Lineage Verification
- Confirm that the base branch is `audit/m2-conservative-execution-prompt-draft-v1-20260518`.
- Confirm that the execution branch is `research/m2-conservative-structural-bo01-mr02-v1-20260518`.
- Verify the physical local HEAD SHA is `ba2993199086659c1d15def3de02ddebba82fddf`.
- Verify that the execution aborted with status `BLOCKED_M2_RUNNER_NOT_AUDITED_OR_NOT_FOUND` because no audited runner `research_lab.runners.m2_structural_runner` or equivalent exists.
- Verify that **no** code, tests, or market data files were modified.

### 2.2 Local Output Policy Compliance
- Verify that **no** local outputs were created or modified under `03_RESEARCH_LAB/`.
- Confirm that **no** forbidden output files (e.g., `trades.csv`, `equity_curve.csv`, `pnl.csv`, `performance_report`) were generated or staged.
- Confirm that **no** local output root files are staged or committed.

---

## 3. Allowed Methods
The auditor is permitted to use **ONLY** read-only text commands:
- Git inspection commands (`git status`, `git branch`, `git log`, `git diff`).
- Text search commands (`rg` or native PowerShell search commands).
- Reading markdown files using file viewers.

---

## 4. Final Audit Decision
The auditor must report a final safety status:
- **STATUS = PASS:** If all checks comply, the abort is documented correctly, no incorrect files are committed, and no data leakages exist.
- **STATUS = BLOCKER:** If any python script execution is allowed, any performance metrics are permitted, or any execution is possible without the exact activation gate phrase.

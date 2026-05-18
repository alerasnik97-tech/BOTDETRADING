# NEXT PROMPT AUDIT M1 TRAIN-ONLY MICRORUN EXECUTION V1

This prompt is to be executed in **EXTERNAL AUDITOR MODE (READ-ONLY)**. It authorizes no code changes, no execution, and no data mutations. The objective is to verify the physical evidence generated during the M1 train-only microrun.

---

## 1. Context and Branch Configuration
- **Branch to Audit:** `research/m1-train-only-bo01-mr02-v1-20260518`
- **Pushed Commit to Audit:** `<COMMIT_SHA>` *(Update with final commit hash after push)*
- **State to Validate:** `M1_TRAIN_ONLY_PLUMBING_COMPLETED_READY_FOR_EXTERNAL_AUDIT`

---

## 2. Audit Scope & Verification Steps

### Step 2.1: Worktree & Branch Integrity Check
1. Run `git status --short` to ensure the workspace has no staged files.
2. Confirm current branch is `research/m1-train-only-bo01-mr02-v1-20260518`.
3. Confirm that the base commit matches `a59557d11aace57326183f3b35e3beb7ca7def46`.

### Step 2.2: Governance Document Verification
Verify the existence and content of the following two files in the repository:
1. `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/M1_TRAIN_ONLY_MICRORUN_EXECUTION_REPORT_V1.md`
2. `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/NEXT_PROMPT_AUDIT_M1_TRAIN_ONLY_MICRORUN_EXECUTION_V1.md`

### Step 2.3: Local Physical Evidence Audit
Check that the gitignored local output folder exists and contains all required artifacts:
- **Root Path:** `03_RESEARCH_LAB/research_lab/local_outputs_do_not_commit/m1_train_only_bo01_mr02/M1_TRAIN_ONLY_BO01_MR02_20260518_112700/`
- **Artifacts:**
  1. `m1_temporary_runner.py` (Verify it contains no hardcoded data, no secrets, no 2025/2026 references).
  2. `M1_TRAIN_ONLY_MICRORUN_REPORT.md` (Verify row counts `864`, BO01 calls `864` / valid signals `14`, MR02 calls `864` / valid signals `0`).
  3. `command_log.txt` (Verify console trace and zero exit code).
  4. `data_access_log.txt` (Verify timestamps read, file paths, and no 2025/2026 years parsed).
  5. `output_manifest.json` (Verify the JSON matches the schema, contains exact file hashes, and validates that no validation/holdout was used).

### Step 2.4: Zero Leaks Audit
1. Confirm that no files from `local_outputs_do_not_commit` have been committed or staged.
2. Confirm that no modifications have occurred to the core strategy files (`BO01Strategy.py`, `MR02Strategy.py`), unit tests, engine, runner, or vault data files.
3. Confirm that pre-existing backlogs W-01 and W-02 remain completely untouched.

---

## 3. Auditor Response Format
The auditor must respond with the standard handoff layout:
1. **STATUS:** READY / BLOCKED / RED / INCONCLUSIVE
2. **GITHUB:** Repo, Branch, Commit, Pushed, main_touched.
3. **SCOPE:** Objective, Touched, Untouched.
4. **EVIDENCIA:** Rowcounts, Hashes, Tests, Limits, Safety checks.
5. **RESULTADO:** Technical summary, Verdict, next step.
6. **PROHIBICIONES RESPETADAS:** Standard checklist.

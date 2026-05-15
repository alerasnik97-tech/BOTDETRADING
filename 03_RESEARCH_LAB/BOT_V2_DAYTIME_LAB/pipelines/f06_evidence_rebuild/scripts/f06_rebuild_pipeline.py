#!/usr/bin/env python3
# ============================================================
# F06 EVIDENCE REBUILD - FAIL-CLOSED PIPELINE SCAFFOLD
# ============================================================
# SCOPE: scaffold + guards + dry_run ONLY.
#   - DOES NOT run any strategy.
#   - DOES NOT run any backtest.
#   - DOES NOT read raw data / tick data.
#   - DOES NOT touch validation / holdout / 2025 / 2026.
#   - DOES NOT read quarantined folders as productive input.
# Every guard FAILS CLOSED: on doubt -> BLOCKED, never a silent pass.
# F06/F08/F12 = NOT CERTIFIED. This file certifies nothing.
# ============================================================
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import uuid
from datetime import datetime, timezone

# ---- Institutional constants (the blockers, encoded) -------------------------
QUARANTINE_TOKENS = [
    "QUARANTINED",
    "DO_NOT_USE",
    "v50b_limited_real_gauntlet_rerun_sw",
    "V50B_RERUN_TRADES.csv",
    "V50B_RERUN_MASTER_RANKING.csv",
]
VALIDATION_COLUMNS = ["N_val", "PF_val", "Total_R_val", "WR_val", "val_pass", "combined_pass"]
FORBIDDEN_YEARS = ["2025", "2026"]
DEFAULT_SAMPLE_FLOOR = 100
COST_COMPONENTS = ["spread_component", "slippage_component", "round_turn_commission"]

# Column-name substrings that mark a column as temporal (W2).
TEMPORAL_HINTS = ("time", "date", "datetime", "timestamp", "_ts", "ts_",
                  "epoch", "month", "year", "signal", "fill", "exit", "entry")
# Unambiguous date patterns -> blocked even in non-temporal columns.
_ISO_2025_26 = re.compile(r"20(25|26)-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])")
_SLASH_2025_26 = re.compile(r"20(25|26)[/.](0[1-9]|1[0-2])[/.](0[1-9]|[12]\d|3[01])")
_COMPACT_2025_26 = re.compile(r"(?<!\d)20(25|26)(0[1-9]|1[0-2])([0-2]\d|3[01])(?!\d)")
# Bare year / year-month -> only blocked inside temporal columns.
_YEARISH_2025_26 = re.compile(r"(?<!\d)20(25|26)(?!\d)|20(25|26)-(0[1-9]|1[0-2])")


# ---- Hashing -----------------------------------------------------------------
def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ---- CSV helper (stdlib only) ------------------------------------------------
def read_csv(path: str):
    with open(path, "r", newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        header = list(reader.fieldnames or [])
        rows = [dict(r) for r in reader]
    return header, rows


# ============================================================================
# GUARDS  -- pure functions, return (ok: bool, errors: list[str])
# ============================================================================
def check_no_quarantined_path(path) -> tuple[bool, list[str]]:
    p = str(path)
    low = p.lower()
    errs = [f"quarantined token '{t}' in path: {p}"
            for t in QUARANTINE_TOKENS if t.lower() in low]
    return (len(errs) == 0, errs)


def check_no_validation_columns(header) -> tuple[bool, list[str]]:
    bad = [c for c in header if c in VALIDATION_COLUMNS]
    return (len(bad) == 0, [f"forbidden validation column in train-only: {c}" for c in bad])


def _blank(v) -> bool:
    return str(v if v is not None else "").strip() == ""


def _as_int(v):
    try:
        if isinstance(v, bool):
            return None
        return int(v)
    except Exception:
        return None


def _as_float(v):
    try:
        if isinstance(v, bool):
            return None
        return float(v)
    except Exception:
        return None


def check_single_run_id(rows, col: str = "run_id") -> tuple[bool, list[str]]:
    ids = sorted({(r.get(col) or "").strip() for r in rows if (r.get(col) or "").strip()})
    if not ids:
        return (False, [f"no '{col}' values found (fail-closed)"])
    if len(ids) > 1:
        return (False, [f"multiple run_ids found ({len(ids)}): {ids}"])
    return (True, [])


def check_no_2025_2026(values) -> tuple[bool, list[str]]:
    errs = []
    for v in values:
        s = str(v)
        if (_ISO_2025_26.search(s) or _SLASH_2025_26.search(s)
                or _COMPACT_2025_26.search(s) or _YEARISH_2025_26.search(s)):
            errs.append(f"forbidden 2025/2026 date value: {s}")
    return (len(errs) == 0, errs)


def check_ledger_no_2025_2026(header, rows) -> tuple[bool, list[str]]:
    return check_temporal_no_2025_2026(header, rows)


def _is_temporal_col(name: str) -> bool:
    n = str(name).lower()
    return any(h in n for h in TEMPORAL_HINTS)


def _epoch_year(value: str):
    """Return year if value is a plausible epoch (s or ms), else None."""
    s = str(value).strip()
    if not re.fullmatch(r"\d{10}|\d{13}", s):
        return None
    try:
        n = int(s)
        if len(s) == 13:           # milliseconds
            n //= 1000
        if not (0 < n < 4102444800):  # < year 2100
            return None
        return datetime.fromtimestamp(n, tz=timezone.utc).year
    except Exception:
        return None


def check_temporal_no_2025_2026(header, rows) -> tuple[bool, list[str]]:
    """W2: robust 2025/2026 guard.
    - Unambiguous ISO / slash / compact full dates -> blocked in ANY column.
    - Bare year / year-month / epoch(s|ms) -> blocked only in TEMPORAL columns
      (avoids false positives on run_id hashes / non-temporal numerics)."""
    errs: list[str] = []
    for r in rows:
        for c in header:
            raw = r.get(c, "")
            s = str(raw)
            if (_ISO_2025_26.search(s) or _SLASH_2025_26.search(s)
                    or _COMPACT_2025_26.search(s)):
                errs.append(f"forbidden 2025/2026 date in column '{c}': {s}")
                continue
            if _is_temporal_col(c):
                if _YEARISH_2025_26.search(s):
                    errs.append(f"forbidden 2025/2026 in temporal column '{c}': {s}")
                    continue
                ey = _epoch_year(s)
                if ey in (2025, 2026):
                    errs.append(f"forbidden epoch -> year {ey} in temporal "
                                f"column '{c}': {s}")
    return (len(errs) == 0, errs)


def check_script_tracked(path, tracked_paths=None) -> tuple[bool, list[str]]:
    """If tracked_paths is provided (tests), do membership check (no git).
    Otherwise query git ls-files --error-unmatch (fail-closed on any error)."""
    norm = str(path).replace("\\", "/")
    if tracked_paths is not None:
        tp = {str(t).replace("\\", "/") for t in tracked_paths}
        ok = any(norm == t or norm.endswith("/" + t) or t.endswith("/" + os.path.basename(norm))
                 for t in tp) or norm in tp
        return (ok, [] if ok else [f"script not tracked by git: {path}"])
    try:
        root_r = subprocess.run(["git", "rev-parse", "--show-toplevel"],
                                capture_output=True, text=True)
        if root_r.returncode != 0 or not root_r.stdout.strip():
            return (False, ["git repo root unavailable for script tracking check"])
        repo_root = os.path.abspath(root_r.stdout.strip())
        candidates = {norm}
        raw = str(path)
        abs_path = os.path.abspath(raw)
        try:
            if os.path.commonpath([repo_root, abs_path]) == repo_root:
                candidates.add(os.path.relpath(abs_path, repo_root).replace("\\", "/"))
        except Exception:
            pass
        for cand in sorted(candidates):
            r = subprocess.run(["git", "-C", repo_root, "ls-files",
                                "--error-unmatch", "--", cand],
                               capture_output=True, text=True)
            if r.returncode == 0:
                return (True, [])
        listed = subprocess.run(["git", "-C", repo_root, "ls-files"],
                                capture_output=True, text=True)
        if listed.returncode == 0:
            tracked = [x.strip().replace("\\", "/") for x in listed.stdout.splitlines()]
            if any(t == norm or t.endswith("/" + norm) for t in tracked):
                return (True, [])
        return (False, [f"script not tracked by git: {path}"])
    except Exception as e:  # fail-closed
        return (False, [f"git tracking check failed (fail-closed): {e}"])


def check_config_uniqueness(rows, result_cols=None, config_col="config_id",
                            dedup_flag_col="deduplicated") -> tuple[bool, list[str]]:
    """W3 (hardened, NOT a substitute for real parameter-sensitivity).
    Blocks obvious degeneration: many config_id but few unique
    parameter_hash and/or result_signature, unless deduplicated=true is
    explicitly set. Reports metrics so the degeneracy is visible."""
    if not rows:
        return (False, ["empty ranking (fail-closed)"])

    def col_present(name):
        return any(name in r for r in rows)

    dedup_values = [_truthy(r.get(dedup_flag_col, "")) for r in rows]
    deduped = bool(dedup_values) and all(dedup_values)
    n_cfg = len({(r.get(config_col) or "").strip()
                 for r in rows if (r.get(config_col) or "").strip()})

    if col_present("result_signature"):
        sigs = {str(r.get("result_signature", "")).strip() for r in rows}
    elif result_cols:
        sigs = {tuple(str(r.get(c, "")) for c in result_cols) for r in rows}
    else:
        meta = {config_col, "family_id", dedup_flag_col,
                "parameter_hash", "result_signature"}
        cols = [c for c in rows[0].keys() if c not in meta]
        sigs = {tuple(str(r.get(c, "")) for c in cols) for r in rows}
    n_sig = len(sigs)

    has_ph = col_present("parameter_hash")
    n_ph = len({str(r.get("parameter_hash", "")).strip()
                for r in rows}) if has_ph else None
    dup_ratio = round(1 - (n_sig / n_cfg), 4) if n_cfg else 1.0
    metrics = (f"total_configs={n_cfg} unique_result_signatures={n_sig} "
               f"unique_parameter_hashes={n_ph} duplicate_ratio={dup_ratio}")

    if deduped:
        return (True, [])  # explicit institutional acknowledgement

    errs = []
    if n_cfg > 1 and n_sig == 1:
        errs.append(f"degenerate ranking: {n_cfg} configs -> 1 result "
                    f"signature [{metrics}]")
    if has_ph and n_cfg > 1 and n_ph == 1:
        errs.append(f"degenerate ranking: {n_cfg} configs -> 1 parameter_hash "
                    f"[{metrics}]")
    if n_cfg >= 5 and n_sig < math.ceil(0.5 * n_cfg):
        errs.append(f"degenerate ranking: only {n_sig}/{n_cfg} unique result "
                    f"signatures (<50%) [{metrics}]")
    if has_ph and n_cfg >= 5 and n_ph < math.ceil(0.5 * n_cfg):
        errs.append(f"degenerate ranking: only {n_ph}/{n_cfg} unique "
                    f"parameter_hashes (<50%) [{metrics}]")
    return (len(errs) == 0, errs)


def validate_ledger_schema(rows, header=None) -> tuple[bool, list[str]]:
    """Runtime ledger contract (stdlib-only).
    This mirrors the JSON schema basics without depending on jsonschema in
    the audit environment."""
    errs = []
    if not isinstance(rows, list) or not rows:
        return (False, ["ledger has no rows (fail-closed)"])
    if header is None:
        header = list(rows[0].keys())
    header = list(header or [])
    required = ["run_id", "family_id", "config_id"]
    missing = [c for c in required if c not in header]
    errs += [f"ledger missing required column: {c}" for c in missing]
    temporal_cols = [c for c in header if _is_temporal_col(c)]
    if not temporal_cols:
        errs.append("ledger requires at least one datetime/month/timestamp column")
    ok_val, e_val = check_no_validation_columns(header)
    errs += e_val
    for i, row in enumerate(rows, start=2):
        for c in required:
            if c in header and _blank(row.get(c)):
                errs.append(f"ledger row {i} empty critical field: {c}")
        for c in ("gross_r", "net_r", "sl_pips"):
            if c in header and not _blank(row.get(c)) and _as_float(row.get(c)) is None:
                errs.append(f"ledger row {i} non-numeric {c}: {row.get(c)!r}")
        if "month" in header:
            m = str(row.get("month", "")).strip()
            if m and not re.fullmatch(r"20(20|21|22|23|24)-(0[1-9]|1[0-2])", m):
                errs.append(f"ledger row {i} month outside train scope: {m!r}")
    return (len(errs) == 0, errs)


def validate_ranking_schema(rows, train_only=True) -> tuple[bool, list[str]]:
    errs = []
    if not isinstance(rows, list) or not rows:
        return (False, ["ranking has no rows (fail-closed)"])
    header = list(rows[0].keys())
    required = ["family_id", "config_id", "N_train", "PF_train",
                "Total_R_train", "WR_train"]
    missing = [c for c in required if c not in header]
    errs += [f"ranking missing required column: {c}" for c in missing]
    if train_only:
        ok_val, e_val = check_no_validation_columns(header)
        errs += e_val
    for i, row in enumerate(rows, start=2):
        for c in ("family_id", "config_id"):
            if c in header and _blank(row.get(c)):
                errs.append(f"ranking row {i} empty critical field: {c}")
        for c in ("N_train", "PF_train", "Total_R_train", "WR_train"):
            if c in header and (_blank(row.get(c)) or _as_float(row.get(c)) is None):
                errs.append(f"ranking row {i} invalid numeric {c}: {row.get(c)!r}")
    if not missing:
        ok_u, e_u = check_config_uniqueness(
            rows,
            result_cols=["N_train", "PF_train", "Total_R_train", "WR_train"],
        )
        errs += e_u
    return (len(errs) == 0, errs)


def validate_cost_report_schema(obj) -> tuple[bool, list[str]]:
    errs = []
    if not isinstance(obj, dict):
        return (False, ["cost report is not a JSON object (fail-closed)"])
    comps = obj.get("components_applied", obj)
    ok_cm, e_cm = check_cost_model_components(comps if isinstance(comps, dict) else {})
    errs += e_cm
    if _blank(obj.get("input_ledger_run_id")):
        errs.append("cost report missing input_ledger_run_id")
    if obj.get("input_is_quarantined_path") is not False:
        errs.append("cost report input_is_quarantined_path must be false")
    scenarios = obj.get("scenarios", [])
    if not isinstance(scenarios, list) or len(scenarios) == 0:
        errs.append("cost report scenarios must be a non-empty list")
    elif len(scenarios) < 3:
        errs.append("cost report must include at least 3 scenarios")
    for i, sc in enumerate(scenarios, start=1):
        if not isinstance(sc, dict):
            errs.append(f"cost scenario {i} is not an object")
            continue
        for c in ("name", "spread_pips", "slippage_pips",
                  "commission_round_turn_usd"):
            if c not in sc or _blank(sc.get(c)):
                errs.append(f"cost scenario {i} missing {c}")
        for c in ("spread_pips", "slippage_pips", "commission_round_turn_usd"):
            if c in sc and not _blank(sc.get(c)):
                n = _as_float(sc.get(c))
                if n is None or n < 0:
                    errs.append(f"cost scenario {i} invalid {c}: {sc.get(c)!r}")
    return (len(errs) == 0, errs)


def check_sample_size_floor(n: int, floor: int = DEFAULT_SAMPLE_FLOOR) -> tuple[bool, list[str]]:
    try:
        n = int(n)
    except Exception:
        return (False, [f"sample size not an int: {n!r} (fail-closed)"])
    if n < floor:
        return (False, [f"sample size {n} below institutional floor {floor} "
                        f"(N=10-style certification is forbidden)"])
    return (True, [])


def check_output_dir_absent(path) -> tuple[bool, list[str]]:
    """Fail-closed: refuse to (re)write into an existing output dir
    (prevents the multi-writer / mixed-output failure mode)."""
    if not path:
        return (False, ["output_dir is empty (fail-closed)"])
    if os.path.exists(path):
        return (False, [f"output_dir already exists (fail-closed): {path}"])
    return (True, [])


def _truthy(v) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes", "y")


def check_cost_model_components(cost_model: dict) -> tuple[bool, list[str]]:
    """Require spread + slippage + round-turn commission. Accepts explicit
    component booleans, 'require_*' booleans, or a 'components' list."""
    if not isinstance(cost_model, dict):
        return (False, ["cost_model missing/not a mapping (fail-closed)"])
    comps = set()
    cl = cost_model.get("components")
    if isinstance(cl, (list, tuple)):
        comps |= {str(x).strip() for x in cl}
    alias = {
        "spread_component": ["spread_component", "require_real_spread_component", "spread"],
        "slippage_component": ["slippage_component", "require_slippage_component", "slippage"],
        "round_turn_commission": ["round_turn_commission", "require_round_turn_commission",
                                   "commission_round_turn"],
    }
    for canonical, keys in alias.items():
        if canonical in comps:
            continue
        if any(k in cost_model and _truthy(cost_model[k]) for k in keys):
            comps.add(canonical)
    missing = [c for c in COST_COMPONENTS if c not in comps]
    return (len(missing) == 0, [f"cost model missing component: {m}" for m in missing])


def _walk_strings(obj, prefix=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{prefix}.{k}" if prefix else str(k)
            yield from _walk_strings(v, p)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from _walk_strings(v, f"{prefix}[{i}]")
    elif isinstance(obj, str):
        yield prefix, obj


# ---- Manifest validation (lightweight; no jsonschema dependency) -------------
_MANIFEST_REQUIRED = [
    "run_id", "generated_at", "git_branch", "git_commit_sha", "generator_pid",
    "script_path", "script_sha256", "script_is_tracked", "config_path",
    "config_sha256", "input_dataset_path", "input_dataset_sha256_or_reference",
    "input_is_quarantined_path", "symbol", "families", "exact_months",
    "train_only", "validation_evaluated", "holdout_touched", "allow_2025",
    "allow_2026", "row_count_input", "trade_count", "rejected_count",
    "output_hashes", "safety_flags", "cost_model", "sample_size_floor", "status",
]
_MANIFEST_CONST = {
    "script_is_tracked": True, "input_is_quarantined_path": False,
    "train_only": True, "validation_evaluated": False, "holdout_touched": False,
    "allow_2025": False, "allow_2026": False,
}
_STATUS_ENUM = ["DRY_RUN_SCHEMA_VALIDATED", "BLOCKED_GUARD_FAILED",
                "READY_FOR_CLEAN_TRAIN_RERUN", "NOT_READY"]


def validate_manifest(manifest: dict, schema: dict | None = None) -> tuple[bool, list[str]]:
    errs = []
    if not isinstance(manifest, dict):
        return (False, ["manifest is not a JSON object (fail-closed)"])
    required = _MANIFEST_REQUIRED
    if schema and isinstance(schema.get("required"), list):
        required = schema["required"]
    for k in required:
        if k not in manifest:
            errs.append(f"missing required manifest field: {k}")
    for k, want in _MANIFEST_CONST.items():
        if k in manifest and manifest[k] != want:
            errs.append(f"manifest.{k} must be {want} (got {manifest[k]!r})")

    if manifest.get("symbol") not in (None, "EURUSD"):
        errs.append(f"manifest.symbol must be EURUSD (got {manifest.get('symbol')!r})")
    if "families" in manifest and manifest.get("families") != ["F06"]:
        errs.append(f"manifest.families must be exactly ['F06'] "
                    f"(got {manifest.get('families')!r})")
    for nf in ("generator_pid", "row_count_input", "trade_count", "rejected_count"):
        if nf in manifest:
            n = _as_int(manifest.get(nf))
            if n is None or n < 0:
                errs.append(f"manifest.{nf} must be a non-negative integer "
                            f"(got {manifest.get(nf)!r})")

    def _is_sha256(x) -> bool:
        s = str(x).lower()
        return len(s) == 64 and all(c in "0123456789abcdef" for c in s)

    oh = manifest.get("output_hashes")
    if "output_hashes" in manifest and (not isinstance(oh, dict) or len(oh) == 0):
        errs.append("manifest.output_hashes must be a non-empty object")
    elif isinstance(oh, dict):
        for hk, hv in oh.items():
            if not _is_sha256(hv):
                errs.append(f"manifest.output_hashes['{hk}'] is not a sha256 "
                            f"(fake/short hash forbidden): {hv!r}")
    for hf in ("script_sha256", "config_sha256"):
        if hf in manifest and not _is_sha256(manifest[hf]):
            errs.append(f"manifest.{hf} is not a valid sha256: {manifest[hf]!r}")
    st = manifest.get("status")
    if "status" in manifest and st not in _STATUS_ENUM:
        errs.append(f"manifest.status invalid: {st!r} (allowed {_STATUS_ENUM})")
    if "exact_months" in manifest:
        months = manifest.get("exact_months")
        if not isinstance(months, list) or not months:
            errs.append("manifest.exact_months must be a non-empty list")
        else:
            ok_my, e_my = check_no_2025_2026(months)
            errs += e_my
            for m in months:
                if not re.fullmatch(r"20(20|21|22|23|24)-(0[1-9]|1[0-2])", str(m)):
                    errs.append(f"manifest.exact_months invalid train month: {m!r}")
    ssf = manifest.get("sample_size_floor")
    if "sample_size_floor" in manifest:
        try:
            if int(ssf) < DEFAULT_SAMPLE_FLOOR:
                errs.append(f"sample_size_floor {ssf} < {DEFAULT_SAMPLE_FLOOR}")
        except Exception:
            errs.append(f"sample_size_floor not int: {ssf!r}")
    cm = manifest.get("cost_model")
    if "cost_model" in manifest:
        ok_cm, e_cm = check_cost_model_components(cm if isinstance(cm, dict) else {})
        errs += e_cm
    sf = manifest.get("safety_flags")
    if "safety_flags" in manifest:
        if not isinstance(sf, dict):
            errs.append("safety_flags must be an object")
        else:
            for fk in ["test_touched", "validation_touched", "holdout_touched",
                       "raw_data_mutated", "sweep_run", "optimization_run"]:
                if sf.get(fk) is not False:
                    errs.append(f"safety_flags.{fk} must be false (got {sf.get(fk)!r})")
    for where, text in _walk_strings(manifest):
        ok_p, e_p = check_no_quarantined_path(text)
        errs += [f"manifest.{where}: {x}" for x in e_p]
    return (len(errs) == 0, errs)


# ---- Minimal YAML loader (template subset; tests do NOT depend on this) ------
def _coerce(v: str):
    s = v.strip()
    if (len(s) >= 2) and ((s[0] == s[-1] == '"') or (s[0] == s[-1] == "'")):
        return s[1:-1]
    low = s.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


def _mini_yaml_load(text: str):
    """Constrained YAML loader for our fixed template shape (maps, nested
    maps, sequences of scalars). Correctly converts a 'key:' whose first
    indented child is a '- ' item into a LIST (the prior version mis-nested
    it, which under the no-PyYAML fallback turned the 2025/2026 guard into a
    fail-OPEN path). Tested directly by test_config_parsing."""
    root: dict = {}
    # frame = [indent, container, key_in_parent, parent_container]
    stack = [[-1, root, None, None]]
    for raw in text.splitlines():
        if raw.strip().startswith("#") or not raw.strip():
            continue
        line = raw.split(" #", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        content = line.strip()
        while len(stack) > 1 and indent <= stack[-1][0]:
            stack.pop()
        cont = stack[-1][1]
        if content.startswith("- "):
            val = _coerce(content[2:])
            if not isinstance(cont, list):
                # the just-opened child must actually be a list: replace it
                _, _, pkey, pparent = stack[-1]
                newlist: list = []
                if pkey is not None and isinstance(pparent, dict):
                    pparent[pkey] = newlist
                stack[-1][1] = newlist
                cont = newlist
            cont.append(val)
            continue
        if ":" in content:
            key, _, rest = content.partition(":")
            key = key.strip()
            rest = rest.strip()
            if not isinstance(cont, dict):
                continue  # fail-closed: malformed structure -> drop
            if rest == "":
                child: dict = {}
                cont[key] = child
                stack.append([indent, child, key, cont])
            else:
                cont[key] = _coerce(rest)
    return root


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    try:
        import yaml  # type: ignore
        data = yaml.safe_load(text)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    if path.lower().endswith(".json"):
        return json.loads(text)
    return _mini_yaml_load(text)


# ============================================================================
# CLI COMMANDS
# ============================================================================
def _config_invariants(cfg: dict) -> list[str]:
    e = []
    if str(cfg.get("mode")) != "TRAIN_ONLY":
        e.append("mode must be TRAIN_ONLY")
    ds = cfg.get("data_scope", {}) or {}
    for k in ("allow_2025", "allow_2026", "validation_enabled", "holdout_enabled"):
        if ds.get(k) not in (False, "false", "False"):
            e.append(f"data_scope.{k} must be false")
    fam = cfg.get("families")
    if fam != ["F06"]:
        e.append(f"families must be exactly [F06] (got {fam!r})")
    months = ds.get("exact_months", []) or []
    ok_m, e_m = check_no_2025_2026(months)
    e += e_m
    ir = cfg.get("input_rules", {}) or {}
    for k in ("forbid_quarantined_paths", "forbid_legacy_v50b_outputs",
              "forbid_old_master_ranking", "forbid_old_trades_csv"):
        if ir.get(k) not in (True, "true", "True"):
            e.append(f"input_rules.{k} must be true")
    ok_cm, e_cm = check_cost_model_components(cfg.get("cost_model", {}) or {})
    e += e_cm
    ss = cfg.get("sample_size", {}) or {}
    try:
        if int(ss.get("min_trades_per_family", 0)) < DEFAULT_SAMPLE_FLOOR:
            e.append(f"sample_size.min_trades_per_family must be >= {DEFAULT_SAMPLE_FLOOR}")
    except Exception:
        e.append("sample_size.min_trades_per_family must be an int")
    return e


def cmd_validate_config(args) -> int:
    try:
        cfg = load_config(args.config)
    except Exception as ex:
        print(f"FAIL\nconfig load error (fail-closed): {ex}")
        return 2
    errs = _config_invariants(cfg)
    if errs:
        print("FAIL")
        for x in errs:
            print(" - " + x)
        return 2
    print("PASS\nconfig invariants satisfied (TRAIN_ONLY, no 2025/2026, "
          "single family F06, cost components present, sample floor ok)")
    return 0


def cmd_dry_run(args) -> int:
    print("== F06 EVIDENCE REBUILD :: DRY RUN (no strategy, no backtest, no data) ==")
    try:
        cfg = load_config(args.config)
    except Exception as ex:
        print(f"BLOCKED_GUARD_FAILED\nconfig load error (fail-closed): {ex}")
        return 2
    errors = list(_config_invariants(cfg))

    # output dir must NOT exist
    out_dir = args.output_dir
    if not out_dir:
        out_dir = ("03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/reports/"
                   "v50b_f06_evidence_rebuild_" + datetime.now().strftime("%Y%m%d_%H%M"))
    ok_od, e_od = check_output_dir_absent(out_dir)
    errors += e_od

    # quarantined token guard on any declared input-ish path
    for p in (args.config, out_dir):
        ok, e = check_no_quarantined_path(p)
        errors += e

    # script tracked guard (best-effort; fail-closed)
    script_self = os.path.relpath(__file__).replace("\\", "/")
    ok_t, e_t = check_script_tracked(script_self)
    script_tracked = ok_t  # recorded; not hard-fail in dry_run if git unavailable
    if not ok_t:
        print(" warn: " + "; ".join(e_t) + " (recorded; must be tracked before FASE 3)")

    status = "DRY_RUN_SCHEMA_VALIDATED" if not errors else "BLOCKED_GUARD_FAILED"
    cfg_sha = sha256_file(args.config)
    self_sha = sha256_file(__file__)
    run_id = "RB" + uuid.uuid4().hex[:8]
    manifest = {
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_branch": _git("rev-parse --abbrev-ref HEAD"),
        "git_commit_sha": _git("rev-parse HEAD"),
        "generator_pid": os.getpid(),
        "script_path": script_self,
        "script_sha256": self_sha,
        "script_is_tracked": bool(script_tracked),
        "config_path": str(args.config).replace("\\", "/"),
        "config_sha256": cfg_sha,
        "input_dataset_path": "DRY_RUN_NO_INPUT",
        "input_dataset_sha256_or_reference": "DRY_RUN_NO_INPUT",
        "input_is_quarantined_path": False,
        "symbol": cfg.get("symbol", "EURUSD"),
        "families": cfg.get("families", ["F06"]),
        "exact_months": (cfg.get("data_scope", {}) or {}).get("exact_months", []),
        "train_only": True,
        "validation_evaluated": False,
        "holdout_touched": False,
        "allow_2025": False,
        "allow_2026": False,
        "row_count_input": 0,
        "trade_count": 0,
        "rejected_count": 0,
        "output_hashes": {"DRY_RUN": self_sha},
        "safety_flags": {
            "test_touched": False, "validation_touched": False,
            "holdout_touched": False, "raw_data_mutated": False,
            "sweep_run": False, "optimization_run": False,
        },
        "cost_model": {"spread_component": True, "slippage_component": True,
                        "round_turn_commission": True},
        "sample_size_floor": DEFAULT_SAMPLE_FLOOR,
        "status": status,
    }
    ok_m, e_m = validate_manifest(manifest)
    if not ok_m:
        manifest["status"] = "BLOCKED_GUARD_FAILED"
        status = "BLOCKED_GUARD_FAILED"
        errors += e_m

    emit = args.emit or ("03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/reports/"
                          "f06_evidence_rebuild_foundation/DRYRUN_MANIFEST.json")
    try:
        os.makedirs(os.path.dirname(emit), exist_ok=True)
        with open(emit, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2)
        print(f" dry-run manifest written: {emit}")
    except Exception as ex:
        print(f" warn: could not write manifest: {ex}")

    print(f"STATUS: {status}")
    if errors:
        print("ERRORS (fail-closed):")
        for x in errors:
            print(" - " + x)
        return 2
    print("dry_run OK: schema validated, no strategy/backtest/data touched.")
    return 0


def cmd_validate_outputs(args) -> int:
    res = validate_output_dir(args.output_dir, args.manifest, args.config)
    print(res["text"])
    return 0 if res["ok"] else 2


def _discover_manifest(output_dir: str) -> str | None:
    if not output_dir or not os.path.isdir(output_dir):
        return None
    names = sorted(n for n in os.listdir(output_dir)
                   if n.lower().startswith("manifest") and n.lower().endswith(".json"))
    if len(names) == 1:
        return os.path.join(output_dir, names[0])
    return None


def _artifact_declared(manifest: dict, *names):
    art = manifest.get("artifacts")
    if isinstance(art, dict):
        for n in names:
            if art.get(n):
                return art.get(n)
    outputs = manifest.get("outputs")
    if isinstance(outputs, dict):
        for n in names:
            if outputs.get(n):
                return outputs.get(n)
    for n in names:
        for key in (n, f"{n}_path"):
            if manifest.get(key):
                return manifest.get(key)
    return None


def _infer_artifact_from_hashes(manifest: dict, kind: str):
    oh = manifest.get("output_hashes")
    if not isinstance(oh, dict):
        return None
    for key in oh:
        k = str(key).replace("\\", "/").lower()
        if kind == "ledger" and k.endswith(".csv") and ("ledger" in k or "trade" in k):
            return key
        if kind == "ranking" and k.endswith(".csv") and "ranking" in k:
            return key
        if kind == "cost" and k.endswith(".json") and "cost" in k:
            return key
        if kind == "hashes" and ("hash" in k or k.endswith(".txt")):
            return key
    return None


def _resolve_under(base_dir: str, rel_path: str) -> tuple[str | None, list[str]]:
    errs = []
    if _blank(rel_path):
        return None, ["empty artifact path"]
    raw = str(rel_path)
    ok_p, e_p = check_no_quarantined_path(raw)
    errs += e_p
    if os.path.isabs(raw):
        candidate = os.path.abspath(raw)
    else:
        candidate = os.path.abspath(os.path.join(base_dir, raw))
    base_abs = os.path.abspath(base_dir)
    try:
        if os.path.commonpath([base_abs, candidate]) != base_abs:
            errs.append(f"artifact path escapes output_dir: {raw}")
    except Exception:
        errs.append(f"artifact path cannot be resolved safely: {raw}")
    return candidate, errs


def _load_json(path: str):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _repo_root_or_none():
    try:
        r = subprocess.run(["git", "rev-parse", "--show-toplevel"],
                           capture_output=True, text=True)
        if r.returncode == 0 and r.stdout.strip():
            return os.path.abspath(r.stdout.strip())
    except Exception:
        pass
    return None


def _resolve_declared_file(path: str) -> str | None:
    if _blank(path):
        return None
    raw = str(path)
    candidates = []
    if os.path.isabs(raw):
        candidates.append(os.path.abspath(raw))
    else:
        candidates.append(os.path.abspath(raw))
        repo_root = _repo_root_or_none()
        if repo_root:
            candidates.append(os.path.abspath(os.path.join(repo_root, raw)))
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


def validate_output_dir(output_dir: str, manifest_path: str | None,
                         config_path: str | None) -> dict:
    errors, warnings = [], []
    if not output_dir or not os.path.isdir(output_dir):
        errors.append(f"output_dir does not exist: {output_dir}")
    ok, e = check_no_quarantined_path(output_dir or "")
    errors += e
    manifest = None
    if not manifest_path:
        manifest_path = _discover_manifest(output_dir or "")
    if not manifest_path or not os.path.isfile(manifest_path):
        errors.append(f"manifest not found: {manifest_path}")
    else:
        try:
            manifest = _load_json(manifest_path)
        except Exception as ex:
            errors.append(f"manifest unreadable (fail-closed): {ex}")
    if manifest is not None:
        ok_m, e_m = validate_manifest(manifest)
        errors += e_m
        run_id = str(manifest.get("run_id", "")).strip()
        if config_path:
            ok_cp, e_cp = check_no_quarantined_path(config_path)
            errors += e_cp
            if not os.path.isfile(config_path):
                errors.append(f"provided config not found: {config_path}")
            else:
                cfg_actual = sha256_file(config_path)
                cfg_expected = str(manifest.get("config_sha256", "")).lower()
                if cfg_actual.lower() != cfg_expected:
                    errors.append(f"provided config sha mismatch: "
                                  f"manifest={cfg_expected} disk={cfg_actual}")
        for path_field, hash_field in (("script_path", "script_sha256"),
                                       ("config_path", "config_sha256")):
            declared = manifest.get(path_field, "")
            full = _resolve_declared_file(declared)
            if not full:
                errors.append(f"manifest.{path_field} not found on disk: {declared}")
                continue
            actual = sha256_file(full)
            expected = str(manifest.get(hash_field, "")).lower()
            if actual.lower() != expected:
                errors.append(f"manifest.{hash_field} mismatch for {declared}: "
                              f"manifest={expected} disk={actual}")
        output_hashes = manifest.get("output_hashes", {})
        resolved_hash_paths: dict[str, str] = {}
        if isinstance(output_hashes, dict):
            for rel, expected_hash in output_hashes.items():
                full, e_r = _resolve_under(output_dir, rel)
                errors += e_r
                if not full:
                    continue
                resolved_hash_paths[str(rel)] = full
                if not os.path.isfile(full):
                    errors.append(f"output_hashes file missing on disk: {rel}")
                    continue
                actual = sha256_file(full)
                if actual.lower() != str(expected_hash).lower():
                    errors.append(f"output_hashes mismatch for {rel}: "
                                  f"manifest={expected_hash} disk={actual}")

        artifact_specs = {
            "ledger": _artifact_declared(manifest, "ledger", "ledger_csv",
                                         "trades", "trades_csv")
                      or _infer_artifact_from_hashes(manifest, "ledger"),
            "ranking": _artifact_declared(manifest, "ranking", "ranking_csv",
                                          "master_ranking")
                       or _infer_artifact_from_hashes(manifest, "ranking"),
            "cost": _artifact_declared(manifest, "cost_report", "cost",
                                       "cost_report_json")
                    or _infer_artifact_from_hashes(manifest, "cost"),
        }
        artifact_paths: dict[str, str] = {}
        for kind, rel in artifact_specs.items():
            if not rel:
                errors.append(f"manifest does not declare required {kind} artifact")
                continue
            full, e_r = _resolve_under(output_dir, rel)
            errors += e_r
            if full:
                artifact_paths[kind] = full
                if not os.path.isfile(full):
                    errors.append(f"{kind} artifact missing on disk: {rel}")

        ledger_rows = []
        if "ledger" in artifact_paths and os.path.isfile(artifact_paths["ledger"]):
            try:
                header, ledger_rows = read_csv(artifact_paths["ledger"])
                ok_l, e_l = validate_ledger_schema(ledger_rows, header)
                errors += e_l
                ok_rid, e_rid = check_single_run_id(ledger_rows)
                errors += e_rid
                if ok_rid:
                    ledger_run_id = sorted({r.get("run_id", "").strip()
                                            for r in ledger_rows if r.get("run_id")})[0]
                    if ledger_run_id != run_id:
                        errors.append(f"ledger run_id {ledger_run_id!r} "
                                      f"does not match manifest.run_id {run_id!r}")
                ok_d, e_d = check_ledger_no_2025_2026(header, ledger_rows)
                errors += e_d
                ssf = int(manifest.get("sample_size_floor", DEFAULT_SAMPLE_FLOOR))
                ok_s, e_s = check_sample_size_floor(len(ledger_rows), ssf)
                errors += e_s
                tc = _as_int(manifest.get("trade_count"))
                if tc is not None and tc != len(ledger_rows):
                    errors.append(f"manifest.trade_count {tc} does not match "
                                  f"ledger rows {len(ledger_rows)}")
            except Exception as ex:
                errors.append(f"ledger unreadable/invalid (fail-closed): {ex}")

        if "ranking" in artifact_paths and os.path.isfile(artifact_paths["ranking"]):
            try:
                r_header, ranking_rows = read_csv(artifact_paths["ranking"])
                ok_rk, e_rk = validate_ranking_schema(
                    ranking_rows, train_only=bool(manifest.get("train_only", True)))
                errors += e_rk
                ok_rd, e_rd = check_temporal_no_2025_2026(r_header, ranking_rows)
                errors += e_rd
            except Exception as ex:
                errors.append(f"ranking unreadable/invalid (fail-closed): {ex}")

        if "cost" in artifact_paths and os.path.isfile(artifact_paths["cost"]):
            try:
                cost_obj = _load_json(artifact_paths["cost"])
                ok_c, e_c = validate_cost_report_schema(cost_obj)
                errors += e_c
                c_run = str(cost_obj.get("input_ledger_run_id", "")).strip()
                if c_run and c_run != run_id:
                    errors.append(f"cost report input_ledger_run_id {c_run!r} "
                                  f"does not match manifest.run_id {run_id!r}")
            except Exception as ex:
                errors.append(f"cost report unreadable/invalid (fail-closed): {ex}")

        sp = manifest.get("script_path", "")
        ok_t, e_t = check_script_tracked(sp) if sp else (False, ["manifest.script_path empty"])
        errors += e_t
    decision = "READY_FOR_CLAUDE_AUDIT" if not errors else "BLOCKED_GUARD_FAILED"
    head = "PASS" if not errors else "FAIL"
    lines = [f"{head}  decision={decision}",
             f"  errors={len(errors)} warnings={len(warnings)}"]
    lines += [f"  ERROR: {x}" for x in errors]
    lines += [f"  WARN:  {x}" for x in warnings]
    return {"ok": not errors, "decision": decision, "errors": errors,
            "warnings": warnings, "text": "\n".join(lines)}


def _git(args: str) -> str:
    try:
        r = subprocess.run(["git"] + args.split(), capture_output=True, text=True)
        return r.stdout.strip() or "UNKNOWN"
    except Exception:
        return "UNKNOWN"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="f06_rebuild_pipeline",
        description="F06 evidence rebuild scaffold (NO strategy / NO backtest).")
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("validate_config")
    a.add_argument("--config", required=True)
    a.set_defaults(func=cmd_validate_config)
    b = sub.add_parser("dry_run")
    b.add_argument("--config", required=True)
    b.add_argument("--output-dir", dest="output_dir", default=None)
    b.add_argument("--emit", default=None)
    b.set_defaults(func=cmd_dry_run)
    c = sub.add_parser("validate_outputs")
    c.add_argument("--output-dir", dest="output_dir", required=True)
    c.add_argument("--manifest", default=None)
    c.add_argument("--config", default=None)
    c.set_defaults(func=cmd_validate_outputs)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

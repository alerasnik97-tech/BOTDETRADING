#!/usr/bin/env python3
# ============================================================
# F06 EVIDENCE REBUILD - OUTPUT VALIDATOR (standalone CLI)
# ============================================================
# Validates a FUTURE output dir (real or synthetic). Read-only.
# Does NOT run strategy/backtest. Fail-closed.
#   PASS -> decision READY_FOR_CLAUDE_AUDIT
#   FAIL -> decision BLOCKED_GUARD_FAILED
# ============================================================
from __future__ import annotations

import argparse
import importlib.util
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PIPE = os.path.join(_HERE, "f06_rebuild_pipeline.py")

_spec = importlib.util.spec_from_file_location("f06_rebuild_pipeline", _PIPE)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)  # type: ignore


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="validate_rebuild_outputs",
        description="Validate an F06 evidence rebuild output dir (fail-closed).")
    p.add_argument("--output-dir", dest="output_dir", required=True)
    p.add_argument("--manifest", default=None)
    p.add_argument("--config", default=None)
    args = p.parse_args(argv)

    res = _mod.validate_output_dir(args.output_dir, args.manifest, args.config)
    print("=" * 56)
    print("F06 EVIDENCE REBUILD OUTPUT VALIDATION")
    print("=" * 56)
    print(res["text"])
    print("-" * 56)
    print(f"DECISION: {res['decision']}")
    return 0 if res["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

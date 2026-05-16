from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from research_lab.eurusd_ltf_objective_entry_replacement_ecb_autopilot import (
    _ensure_canonical_root,
    _failure_report,
    _status_payload,
    _write_status,
    build_paths,
    main,
)


if __name__ == "__main__":
    try:
        result = main()
        print(json.dumps(result, indent=2, ensure_ascii=False))
    except Exception as exc:
        _ensure_canonical_root()
        paths = build_paths()
        failure_path = _failure_report(paths, "wrapper", exc)
        _write_status(
            paths,
            _status_payload(
                phase="FAILED",
                current_segment="wrapper",
                last_checkpoint=str(failure_path),
                next_action="manual_audit_required",
                completed_segments=[],
                decision="BLOCKED_FOR_SAFETY",
                details={"error": str(exc)},
            ),
        )
        raise

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scratch.post_hardening_drift_lib import (
    BASELINE_JSON,
    REPORT_JSON,
    VALIDATION_JSON,
    build_baselines,
    load_json,
    monitor_live_forward,
    write_json,
)


def main() -> None:
    baselines = load_json(BASELINE_JSON) if BASELINE_JSON.exists() else build_baselines()
    validation = load_json(VALIDATION_JSON) if VALIDATION_JSON.exists() else None
    report = monitor_live_forward(baselines, validation)
    write_json(REPORT_JSON, report)
    print(
        json.dumps(
            {
                "report_path": str(REPORT_JSON),
                "tribunal_integration_mode": report["tribunal_integration_mode"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

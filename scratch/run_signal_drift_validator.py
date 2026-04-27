from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scratch.post_hardening_drift_lib import (
    BASELINE_JSON,
    VALIDATION_JSON,
    build_baselines,
    load_json,
    validate_monitor,
    write_json,
)


def run_validation() -> dict:
    baselines = load_json(BASELINE_JSON) if BASELINE_JSON.exists() else build_baselines()
    results = validate_monitor(baselines)
    write_json(VALIDATION_JSON, results)
    return results


if __name__ == "__main__":
    print(json.dumps(run_validation(), indent=2))

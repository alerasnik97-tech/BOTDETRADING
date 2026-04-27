from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scratch.post_hardening_drift_lib import BASELINE_JSON, build_baselines, write_json


def main() -> None:
    baselines = build_baselines()
    write_json(BASELINE_JSON, baselines)
    print(
        json.dumps(
            {
                "baseline_path": str(BASELINE_JSON),
                "baseline_version": baselines["baseline_version"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
